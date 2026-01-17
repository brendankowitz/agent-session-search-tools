# Implementation Plan: Vector Search with ONNX Embeddings

**Feature**: agent-session-search-tool  
**Status**: Planning  
**Created**: 2026-01-17

## Overview

This document outlines the implementation plan for adding semantic vector search capabilities to agent-journal, including:

1. **AJVI Vector Index** - Custom binary format for storing embeddings
2. **ONNX Embeddings** - MiniLM model for semantic embeddings
3. **Hash-based Fallback** - FNV-1a hash embeddings when model unavailable
4. **Hybrid Search** - Reciprocal Rank Fusion combining lexical + semantic

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SEARCH ORCHESTRATION                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │  SearchCommand  │───▶│ HybridSearcher  │───▶│  SearchResult   │         │
│  │  (--mode)       │    │                 │    │  (unified)      │         │
│  └─────────────────┘    └────────┬────────┘    └─────────────────┘         │
│                                  │                                          │
│                    ┌─────────────┴─────────────┐                           │
│                    ▼                           ▼                           │
│         ┌─────────────────┐         ┌─────────────────┐                    │
│         │ LuceneSearch    │         │ VectorSearch    │                    │
│         │ (lexical)       │         │ (semantic)      │                    │
│         └─────────────────┘         └────────┬────────┘                    │
│                                              │                              │
│                                    ┌─────────┴─────────┐                   │
│                                    ▼                   ▼                   │
│                          ┌─────────────────┐ ┌─────────────────┐           │
│                          │ OnnxEmbedder    │ │ HashEmbedder    │           │
│                          │ (MiniLM-384)    │ │ (FNV-1a-384)    │           │
│                          └─────────────────┘ └─────────────────┘           │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                           STORAGE LAYER                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │ SQLite          │    │ Lucene Index    │    │ AJVI Index      │         │
│  │ (metadata)      │    │ (full-text)     │    │ (vectors)       │         │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Embedding Infrastructure (3-4 days)

### 1.1 IEmbeddingProvider Interface

**File**: `AgentJournal.Core/Embeddings/IEmbeddingProvider.cs`

```csharp
public interface IEmbeddingProvider
{
    /// <summary>Number of dimensions in output vectors</summary>
    int Dimensions { get; }
    
    /// <summary>Whether this provider uses a real ML model</summary>
    bool IsSemanticModel { get; }
    
    /// <summary>Generate embedding for a single text</summary>
    Task<float[]> EmbedAsync(string text, CancellationToken ct = default);
    
    /// <summary>Generate embeddings for multiple texts (batched)</summary>
    Task<float[][]> EmbedBatchAsync(IReadOnlyList<string> texts, CancellationToken ct = default);
    
    /// <summary>L2-normalize a vector in-place</summary>
    void Normalize(Span<float> vector);
}
```

### 1.2 ONNX Embedding Provider

**File**: `AgentJournal.Core/Embeddings/OnnxEmbeddingProvider.cs`

**Dependencies**:
```xml
<PackageReference Include="Microsoft.ML.OnnxRuntime" Version="1.20.0" />
<PackageReference Include="Microsoft.ML.Tokenizers" Version="1.0.0" />
```

**Model Files** (user downloads to `~/.agent-journal/models/`):
- `model.onnx` - MiniLM-L6-v2 (~90MB)
- `tokenizer.json` - Tokenizer config
- `config.json` - Model config

```csharp
public class OnnxEmbeddingProvider : IEmbeddingProvider, IDisposable
{
    private readonly InferenceSession _session;
    private readonly Tokenizer _tokenizer;
    
    public int Dimensions => 384;  // MiniLM-L6-v2
    public bool IsSemanticModel => true;
    
    public static async Task<OnnxEmbeddingProvider?> TryCreateAsync(string modelsPath)
    {
        var modelPath = Path.Combine(modelsPath, "model.onnx");
        if (!File.Exists(modelPath))
            return null;  // Model not installed
            
        // Load ONNX model and tokenizer
        var session = new InferenceSession(modelPath, new SessionOptions
        {
            ExecutionMode = ExecutionMode.ORT_PARALLEL,
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL
        });
        
        var tokenizer = await LoadTokenizerAsync(modelsPath);
        return new OnnxEmbeddingProvider(session, tokenizer);
    }
    
    public async Task<float[]> EmbedAsync(string text, CancellationToken ct = default)
    {
        // 1. Tokenize
        var encoded = _tokenizer.Encode(text, maxLength: 512);
        
        // 2. Create input tensors
        var inputIds = new DenseTensor<long>(encoded.InputIds, new[] { 1, encoded.Length });
        var attentionMask = new DenseTensor<long>(encoded.AttentionMask, new[] { 1, encoded.Length });
        
        // 3. Run inference
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input_ids", inputIds),
            NamedOnnxValue.CreateFromTensor("attention_mask", attentionMask)
        };
        
        using var results = await Task.Run(() => _session.Run(inputs), ct);
        
        // 4. Mean pooling over token embeddings
        var embeddings = results.First().AsTensor<float>();
        var pooled = MeanPool(embeddings, attentionMask);
        
        // 5. Normalize
        Normalize(pooled);
        return pooled;
    }
    
    public async Task<float[][]> EmbedBatchAsync(IReadOnlyList<string> texts, CancellationToken ct = default)
    {
        // Process in batches of 32 for memory efficiency
        const int batchSize = 32;
        var results = new List<float[]>();
        
        foreach (var batch in texts.Chunk(batchSize))
        {
            ct.ThrowIfCancellationRequested();
            var batchResults = await EmbedBatchInternalAsync(batch, ct);
            results.AddRange(batchResults);
        }
        
        return results.ToArray();
    }
}
```

### 1.3 Hash-based Fallback Provider

**File**: `AgentJournal.Core/Embeddings/HashEmbeddingProvider.cs`

Uses FNV-1a hash to create deterministic pseudo-embeddings when ONNX model isn't available:

```csharp
public class HashEmbeddingProvider : IEmbeddingProvider
{
    public int Dimensions => 384;  // Match MiniLM for index compatibility
    public bool IsSemanticModel => false;
    
    public Task<float[]> EmbedAsync(string text, CancellationToken ct = default)
    {
        var vector = new float[Dimensions];
        
        // Tokenize into words
        var words = Tokenize(text);
        
        foreach (var word in words)
        {
            // FNV-1a hash of each word
            var hash = Fnv1aHash(word);
            
            // Use hash to determine which dimensions to activate
            var random = new Random((int)hash);
            for (int i = 0; i < 8; i++)  // 8 activations per word
            {
                var dim = random.Next(Dimensions);
                vector[dim] += 1.0f / words.Count;  // TF-like weighting
            }
        }
        
        // Normalize to unit length
        Normalize(vector);
        return Task.FromResult(vector);
    }
    
    private static ulong Fnv1aHash(string text)
    {
        const ulong FnvPrime = 0x100000001B3;
        const ulong FnvOffsetBasis = 0xCBF29CE484222325;
        
        var hash = FnvOffsetBasis;
        foreach (var c in text.ToLowerInvariant())
        {
            hash ^= c;
            hash *= FnvPrime;
        }
        return hash;
    }
    
    private static IReadOnlyList<string> Tokenize(string text)
    {
        // Simple word tokenization
        return Regex.Split(text.ToLowerInvariant(), @"\W+")
            .Where(w => w.Length > 1)
            .ToList();
    }
}
```

### 1.4 Embedding Provider Factory

**File**: `AgentJournal.Core/Embeddings/EmbeddingProviderFactory.cs`

```csharp
public static class EmbeddingProviderFactory
{
    public static async Task<IEmbeddingProvider> CreateAsync(AgentJournalConfig config)
    {
        // Try ONNX first
        if (config.VectorSearchEnabled)
        {
            var onnxProvider = await OnnxEmbeddingProvider.TryCreateAsync(config.ModelsPath);
            if (onnxProvider != null)
            {
                return onnxProvider;
            }
        }
        
        // Fall back to hash-based
        return new HashEmbeddingProvider();
    }
}
```

---

## Phase 2: AJVI Vector Index (3-4 days)

### 2.1 AJVI Index Format Implementation

**File**: `AgentJournal.Core/Search/AjviIndex.cs`

```csharp
/// <summary>
/// Memory-mapped binary vector index for fast similarity search.
/// Format: 32-byte header + N entries of fixed size.
/// </summary>
public sealed class AjviIndex : IDisposable
{
    private const uint MagicNumber = 0x494A5641;  // "AJVI"
    private const byte FormatVersion = 1;
    private const int HeaderSize = 32;
    
    private readonly string _filePath;
    private readonly int _dimensions;
    private readonly VectorPrecision _precision;
    private readonly int _entrySize;
    
    private MemoryMappedFile? _mmf;
    private MemoryMappedViewAccessor? _accessor;
    private long _entryCount;
    
    public enum VectorPrecision : byte
    {
        Float32 = 0,
        Float16 = 1
    }
    
    public long EntryCount => _entryCount;
    public int Dimensions => _dimensions;
    
    public static AjviIndex Create(string filePath, int dimensions, VectorPrecision precision = VectorPrecision.Float16)
    {
        var index = new AjviIndex(filePath, dimensions, precision);
        index.WriteHeader();
        return index;
    }
    
    public static AjviIndex Open(string filePath)
    {
        using var fs = File.OpenRead(filePath);
        using var reader = new BinaryReader(fs);
        
        var magic = reader.ReadUInt32();
        if (magic != MagicNumber)
            throw new InvalidDataException("Not a valid AJVI file");
            
        var version = reader.ReadByte();
        var precision = (VectorPrecision)reader.ReadByte();
        var dimensions = reader.ReadUInt16();
        var entryCount = reader.ReadInt64();
        
        return new AjviIndex(filePath, dimensions, precision, entryCount);
    }
    
    /// <summary>Add a vector entry to the index</summary>
    public void AddEntry(byte[] contentHash, Guid messageId, byte agentType, long timestamp, ReadOnlySpan<float> vector)
    {
        if (vector.Length != _dimensions)
            throw new ArgumentException($"Vector must have {_dimensions} dimensions");
            
        EnsureCapacity(_entryCount + 1);
        
        var offset = HeaderSize + (_entryCount * _entrySize);
        
        // Write entry
        _accessor!.WriteArray(offset, contentHash, 0, 32);
        offset += 32;
        
        _accessor.Write(offset, messageId);
        offset += 16;
        
        _accessor.Write(offset, agentType);
        offset += 1;
        
        _accessor.Write(offset, timestamp);
        offset += 8;
        
        WriteVector(offset, vector);
        
        _entryCount++;
        UpdateEntryCount();
    }
    
    /// <summary>Read vector at index (zero-copy for F32)</summary>
    public ReadOnlySpan<float> GetVector(long index)
    {
        var offset = HeaderSize + (index * _entrySize) + 57;  // Skip to vector
        
        if (_precision == VectorPrecision.Float16)
        {
            // Convert F16 to F32
            var result = new float[_dimensions];
            for (int i = 0; i < _dimensions; i++)
            {
                var half = _accessor!.ReadUInt16(offset + i * 2);
                result[i] = (float)BitConverter.UInt16BitsToHalf(half);
            }
            return result;
        }
        
        // F32: Could use unsafe pointer for true zero-copy
        var f32Result = new float[_dimensions];
        _accessor!.ReadArray(offset, f32Result, 0, _dimensions);
        return f32Result;
    }
    
    /// <summary>Get message ID at index</summary>
    public Guid GetMessageId(long index)
    {
        var offset = HeaderSize + (index * _entrySize) + 32;
        Span<byte> guidBytes = stackalloc byte[16];
        _accessor!.ReadArray(offset, guidBytes.ToArray(), 0, 16);
        return new Guid(guidBytes);
    }
    
    /// <summary>Check if content hash exists (for deduplication)</summary>
    public bool ContainsHash(ReadOnlySpan<byte> contentHash)
    {
        // Linear scan - could add hash index for O(1) lookup
        for (long i = 0; i < _entryCount; i++)
        {
            var offset = HeaderSize + (i * _entrySize);
            Span<byte> existing = stackalloc byte[32];
            _accessor!.ReadArray(offset, existing.ToArray(), 0, 32);
            
            if (existing.SequenceEqual(contentHash))
                return true;
        }
        return false;
    }
    
    /// <summary>SIMD-accelerated similarity search</summary>
    public IReadOnlyList<(long Index, float Score)> Search(ReadOnlySpan<float> queryVector, int topK = 20)
    {
        var heap = new PriorityQueue<long, float>();
        
        for (long i = 0; i < _entryCount; i++)
        {
            var vector = GetVector(i);
            var score = TensorPrimitives.Dot(queryVector, vector);
            
            if (heap.Count < topK)
            {
                heap.Enqueue(i, -score);  // Min-heap trick
            }
            else if (score > -heap.Peek())
            {
                heap.DequeueEnqueue(i, -score);
            }
        }
        
        // Extract results in descending score order
        var results = new List<(long, float)>(topK);
        while (heap.Count > 0)
        {
            heap.TryDequeue(out var index, out var negScore);
            results.Add((index, -negScore));
        }
        results.Reverse();
        
        return results;
    }
}
```

### 2.2 Vector Search Engine Implementation

**File**: `AgentJournal.Core/Search/VectorSearchEngine.cs` (replace stub)

```csharp
public class VectorSearchEngine : ISearchEngine, IDisposable
{
    private readonly string _indexPath;
    private readonly IEmbeddingProvider _embedder;
    private readonly ConcurrentDictionary<string, Session> _sessionCache = new();
    
    private AjviIndex? _index;
    private bool _initialized;
    
    public IReadOnlyList<SearchMode> SupportedModes { get; } = new[] 
    { 
        SearchMode.Semantic, 
        SearchMode.Hybrid 
    };
    
    public VectorSearchEngine(string indexPath, IEmbeddingProvider embedder)
    {
        _indexPath = indexPath;
        _embedder = embedder;
    }
    
    public async Task InitializeAsync(CancellationToken ct = default)
    {
        var ajviPath = Path.Combine(_indexPath, "index.ajvi");
        
        if (File.Exists(ajviPath))
        {
            _index = AjviIndex.Open(ajviPath);
        }
        else
        {
            Directory.CreateDirectory(_indexPath);
            _index = AjviIndex.Create(ajviPath, _embedder.Dimensions);
        }
        
        _initialized = true;
    }
    
    public async Task IndexSessionAsync(Session session, CancellationToken ct = default)
    {
        EnsureInitialized();
        _sessionCache[session.Id] = session;
        
        // Collect texts for batch embedding
        var messages = session.Messages.ToList();
        var texts = messages.Select(m => m.Content ?? "").ToList();
        
        // Generate embeddings
        var embeddings = await _embedder.EmbedBatchAsync(texts, ct);
        
        // Index each message
        for (int i = 0; i < messages.Count; i++)
        {
            var message = messages[i];
            var contentHash = SHA256.HashData(Encoding.UTF8.GetBytes(message.Content ?? ""));
            
            // Skip if already indexed (deduplication)
            if (_index!.ContainsHash(contentHash))
                continue;
            
            var agentType = session.AgentType switch
            {
                "claude-code" => (byte)1,
                "copilot-cli" => (byte)0,
                _ => (byte)255
            };
            
            _index.AddEntry(
                contentHash,
                Guid.Parse(message.Id),
                agentType,
                message.Timestamp.ToUnixTimeMilliseconds(),
                embeddings[i]
            );
        }
    }
    
    public async Task<IReadOnlyList<SearchResult>> SearchAsync(
        string query,
        SearchMode mode = SearchMode.Semantic,
        int maxResults = 10,
        CancellationToken ct = default)
    {
        EnsureInitialized();
        
        // Embed query
        var queryVector = await _embedder.EmbedAsync(query, ct);
        
        // Search vector index
        var vectorResults = _index!.Search(queryVector, maxResults);
        
        // Build results with session data
        var results = new List<SearchResult>();
        var seenSessions = new HashSet<string>();
        
        foreach (var (index, score) in vectorResults)
        {
            var messageId = _index.GetMessageId(index);
            
            // Find session containing this message
            foreach (var (sessionId, session) in _sessionCache)
            {
                if (seenSessions.Contains(sessionId))
                    continue;
                    
                var message = session.Messages.FirstOrDefault(m => m.Id == messageId.ToString());
                if (message != null)
                {
                    seenSessions.Add(sessionId);
                    results.Add(new SearchResult(
                        Session: session,
                        Score: score,
                        MatchingMessages: new[] { message }.ToList(),
                        Highlight: message.Content?.Substring(0, Math.Min(200, message.Content.Length ?? 0))
                    ));
                    break;
                }
            }
        }
        
        return results;
    }
}
```

---

## Phase 3: Hybrid Search with RRF (2-3 days)

### 3.1 Reciprocal Rank Fusion Algorithm

**File**: `AgentJournal.Core/Search/HybridSearcher.cs`

```csharp
/// <summary>
/// Combines lexical (BM25) and semantic (vector) search results using
/// Reciprocal Rank Fusion for robust ranking across modalities.
/// </summary>
public class HybridSearcher : ISearchEngine
{
    private readonly LuceneSearchEngine _lexicalEngine;
    private readonly VectorSearchEngine _vectorEngine;
    private readonly float _lexicalWeight;
    private readonly float _semanticWeight;
    private readonly int _rrfK;  // RRF constant (typically 60)
    
    public IReadOnlyList<SearchMode> SupportedModes { get; } = new[] 
    { 
        SearchMode.Lexical, 
        SearchMode.Semantic, 
        SearchMode.Hybrid 
    };
    
    public HybridSearcher(
        LuceneSearchEngine lexicalEngine,
        VectorSearchEngine vectorEngine,
        float lexicalWeight = 0.5f,
        float semanticWeight = 0.5f,
        int rrfK = 60)
    {
        _lexicalEngine = lexicalEngine;
        _vectorEngine = vectorEngine;
        _lexicalWeight = lexicalWeight;
        _semanticWeight = semanticWeight;
        _rrfK = rrfK;
    }
    
    public async Task<IReadOnlyList<SearchResult>> SearchAsync(
        string query,
        SearchMode mode = SearchMode.Hybrid,
        int maxResults = 10,
        CancellationToken ct = default)
    {
        return mode switch
        {
            SearchMode.Lexical => await _lexicalEngine.SearchAsync(query, mode, maxResults, ct),
            SearchMode.Semantic => await _vectorEngine.SearchAsync(query, mode, maxResults, ct),
            SearchMode.Hybrid => await HybridSearchAsync(query, maxResults, ct),
            _ => throw new ArgumentOutOfRangeException(nameof(mode))
        };
    }
    
    private async Task<IReadOnlyList<SearchResult>> HybridSearchAsync(
        string query, 
        int maxResults, 
        CancellationToken ct)
    {
        // Fetch more results from each engine for better fusion
        var fetchCount = maxResults * 3;
        
        // Run both searches in parallel
        var lexicalTask = _lexicalEngine.SearchAsync(query, SearchMode.Lexical, fetchCount, ct);
        var semanticTask = _vectorEngine.SearchAsync(query, SearchMode.Semantic, fetchCount, ct);
        
        await Task.WhenAll(lexicalTask, semanticTask);
        
        var lexicalResults = await lexicalTask;
        var semanticResults = await semanticTask;
        
        // Apply RRF
        var fusedScores = new Dictionary<string, (float Score, SearchResult Result)>();
        
        // Score lexical results
        for (int rank = 0; rank < lexicalResults.Count; rank++)
        {
            var result = lexicalResults[rank];
            var rrfScore = _lexicalWeight / (_rrfK + rank + 1);
            
            if (fusedScores.TryGetValue(result.Session.Id, out var existing))
            {
                fusedScores[result.Session.Id] = (existing.Score + rrfScore, result);
            }
            else
            {
                fusedScores[result.Session.Id] = (rrfScore, result);
            }
        }
        
        // Score semantic results
        for (int rank = 0; rank < semanticResults.Count; rank++)
        {
            var result = semanticResults[rank];
            var rrfScore = _semanticWeight / (_rrfK + rank + 1);
            
            if (fusedScores.TryGetValue(result.Session.Id, out var existing))
            {
                fusedScores[result.Session.Id] = (existing.Score + rrfScore, existing.Result);
            }
            else
            {
                fusedScores[result.Session.Id] = (rrfScore, result);
            }
        }
        
        // Sort by fused score and return top results
        return fusedScores.Values
            .OrderByDescending(x => x.Score)
            .Take(maxResults)
            .Select(x => x.Result with { Score = x.Score })
            .ToList();
    }
    
    public async Task InitializeAsync(CancellationToken ct = default)
    {
        await Task.WhenAll(
            _lexicalEngine.InitializeAsync(ct),
            _vectorEngine.InitializeAsync(ct)
        );
    }
    
    public async Task IndexSessionAsync(Session session, CancellationToken ct = default)
    {
        await Task.WhenAll(
            _lexicalEngine.IndexSessionAsync(session, ct),
            _vectorEngine.IndexSessionAsync(session, ct)
        );
    }
    
    public async Task IndexSessionsAsync(IEnumerable<Session> sessions, CancellationToken ct = default)
    {
        var sessionList = sessions.ToList();
        await Task.WhenAll(
            _lexicalEngine.IndexSessionsAsync(sessionList, ct),
            _vectorEngine.IndexSessionsAsync(sessionList, ct)
        );
    }
}
```

---

## Phase 4: CLI & Configuration (1-2 days)

### 4.1 Config Updates

**File**: `AgentJournal/Configuration/AgentJournalConfig.cs`

```csharp
public class AgentJournalConfig
{
    // ... existing properties ...
    
    // Vector search settings
    public bool VectorSearchEnabled { get; set; } = true;
    public string VectorPrecision { get; set; } = "f16";  // f16 or f32
    
    // Hybrid search settings
    public float LexicalWeight { get; set; } = 0.5f;
    public float SemanticWeight { get; set; } = 0.5f;
    
    // Computed paths
    public string ModelsPath => Path.Combine(DataPath, "models");
    public string VectorIndexPath => Path.Combine(DataPath, "vector-index");
}
```

### 4.2 DI Registration

**File**: `AgentJournal/Program.cs`

```csharp
// Create embedding provider (ONNX or hash fallback)
var embedder = await EmbeddingProviderFactory.CreateAsync(config);
services.AddSingleton(embedder);

// Create search engines
var luceneEngine = new LuceneSearchEngine(config.LuceneIndexPath);
var vectorEngine = new VectorSearchEngine(config.VectorIndexPath, embedder);
var hybridSearcher = new HybridSearcher(
    luceneEngine, 
    vectorEngine,
    config.LexicalWeight,
    config.SemanticWeight
);

services.AddSingleton<ISearchEngine>(hybridSearcher);
```

### 4.3 Model Download Command

**File**: `AgentJournal/Commands/ModelsCommand.cs`

```bash
# Download MiniLM model
agent-journal models download minilm

# List installed models
agent-journal models list

# Remove model
agent-journal models remove minilm
```

---

## Phase 5: Testing (2 days)

### Test Cases

1. **Embedding Tests**
   - OnnxEmbeddingProvider produces 384-dim normalized vectors
   - HashEmbeddingProvider produces consistent outputs for same input
   - Batch embedding matches single embedding results
   
2. **AJVI Index Tests**
   - Create/Open/Close lifecycle
   - Add entries and retrieve by index
   - Content hash deduplication
   - Similarity search returns correct top-K
   - F16 vs F32 precision accuracy
   
3. **Hybrid Search Tests**
   - RRF correctly merges rankings
   - Results appear in both lexical and semantic (boosted)
   - Unique results from each engine included
   
4. **Integration Tests**
   - Index real Claude/Copilot sessions
   - Search with each mode: lexical, semantic, hybrid
   - Performance benchmarks (<50ms for 10K messages)

---

## Timeline Summary

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| Phase 1: Embeddings | 3-4 days | None |
| Phase 2: AJVI Index | 3-4 days | Phase 1 |
| Phase 3: Hybrid Search | 2-3 days | Phase 1, 2 |
| Phase 4: CLI/Config | 1-2 days | Phase 3 |
| Phase 5: Testing | 2 days | All |

**Total: ~12-15 days**

---

## Package Dependencies

```xml
<!-- ONNX Runtime for embeddings -->
<PackageReference Include="Microsoft.ML.OnnxRuntime" Version="1.20.0" />
<PackageReference Include="Microsoft.ML.Tokenizers" Version="1.0.0" />

<!-- SIMD operations for vector math -->
<!-- Already included in .NET 10: System.Numerics.Tensors -->
```

---

## Open Questions

1. **Model distribution**: Host MiniLM on GitHub releases or point to HuggingFace?
2. **GPU acceleration**: Support DirectML for Windows GPU inference?
3. **Incremental indexing**: Append-only AJVI or rebuild on change?
4. **Approximate search**: Add HNSW for >100K vectors?
