# Investigation: .NET 10 Global Tool Architecture

**Feature**: agent-session-search-tool
**Status**: In Progress
**Created**: 2026-01-15

## Approach

Build a .NET 10 global tool (`dotnet tool install -g chats`) that provides:

### Core Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    agent-session-search CLI                      │
├─────────────────────────────────────────────────────────────────┤
│  Commands: index | search | export | watch | config             │
├─────────────────────────────────────────────────────────────────┤
│                      Core Services                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Indexer    │  │   Searcher   │  │   Exporter   │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
├─────────────────────────────────────────────────────────────────┤
│                    Search Engines                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Lucene.NET   │  │ Vector Store │  │   Hybrid     │          │
│  │  (BM25)      │  │  (ONNX/Hash) │  │   (RRF)      │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
├─────────────────────────────────────────────────────────────────┤
│                    Agent Connectors                              │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐  │
│  │ Copilot │ │ Claude  │ │ Cursor  │ │  Aider  │ │  Cline  │  │
│  │  CLI    │ │  Code   │ │         │ │         │ │         │  │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                    Storage Layer                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  SQLite DB   │  │ Lucene Index │  │ Vector Index │          │
│  │  (metadata)  │  │  (full-text) │  │   (.cvvi)    │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

### Key Components

#### 1. Unified Data Model
```csharp
public record Session(
    string Id,
    string AgentType,         // "copilot-cli", "claude-code", etc.
    string? ProjectPath,
    DateTime StartedAt,
    DateTime? EndedAt,
    IReadOnlyList<Message> Messages
);

public record Message(
    string Id,
    string SessionId,
    MessageRole Role,         // User, Assistant, System, Tool
    string Content,
    DateTime Timestamp,
    IReadOnlyList<ToolCall>? ToolCalls
);

public enum MessageRole { User, Assistant, System, Tool }
```

#### 2. Agent Connectors (Plugin Architecture)
```csharp
public interface IAgentConnector
{
    string AgentType { get; }
    IEnumerable<string> GetSessionPaths();
    IAsyncEnumerable<Session> ParseSessionsAsync(CancellationToken ct);
}
```

#### 3. Search Modes
- **Lexical**: Lucene.NET with BM25 scoring and edge n-gram tokenization
- **Semantic**: ONNX Runtime with MiniLM (384-dim vectors) or FNV-1a hash fallback
- **Hybrid**: Reciprocal Rank Fusion combining both

#### 4. Export Formats
- **HTML**: Self-contained with embedded CSS/JS, dark mode, collapsible tool calls
- **Markdown**: Clean formatting for documentation/sharing
- **JSON**: Structured output for agent consumption (robot mode)

### CLI Interface

```bash
# Index all supported agents
chats index [--agent <type>] [--watch]

# Search with context
chats search "authentication" [--mode lexical|semantic|hybrid] 
    [--context 3] [--agent <type>] [--project <path>]
    [--robot]  # JSON output for agents

# Navigate results
chats search "auth" --context 3 --offset 0
chats search "auth" --context 3 --offset 3  # next page

# Export conversation
chats export <session-id> [--format html|md|json] [--output <path>]

# Summarize search results
chats search "error handling" --summarize [--model <name>]

# Configuration
chats config show
chats config set index.path ~/.chats/index
chats config agents list
```

### Technology Stack

| Component | Library | Rationale |
|-----------|---------|-----------|
| CLI Framework | System.CommandLine | Official .NET CLI framework |
| Full-Text Search | Lucene.NET 4.8 | Mature, .NET native, BM25 support |
| Vector Embeddings | Microsoft.ML.OnnxRuntime | Cross-platform, local inference |
| Database | Microsoft.Data.Sqlite | Lightweight, embedded, cross-platform |
| JSON Parsing | System.Text.Json | Built-in, fast, AOT-compatible |
| HTML Templates | Scriban | Lightweight, logic-less templates |
| File Watching | System.IO.FileSystemWatcher | Built-in, cross-platform |

### .NET 10 Features Leveraged

1. **Native AOT**: Single-file deployment, fast startup
2. **One-shot tool execution**: `dotnet tool run agent-session-search search "query"`
3. **Platform-specific packaging**: Optimized binaries per platform
4. **Improved trimming**: Smaller binaries with better tree-shaking
5. **ONNX Runtime improvements**: Better ARM64 support

### Data Storage

```
~/.chats/
├── config.toml                 # User configuration
├── agent-search.db            # SQLite: sessions, messages metadata
├── lucene-index/              # Full-text search index
├── vector-index/
│   └── index-minilm-384.cvvi  # Vector embeddings (if ONNX model present)
└── models/                    # Optional ONNX models
    ├── model.onnx
    ├── tokenizer.json
    ├── config.json
    ├── special_tokens_map.json
    └── tokenizer_config.json
```

### CVVI Vector Index Format (Deep Dive)

The `.cvvi` (Chats Vector Index) format is a custom binary format for storing semantic embeddings. It's inspired by the `cass` project but adapted for .NET.

#### Why a Custom Format?

| Alternative | Why Not |
|-------------|---------|
| **Pinecone/Qdrant/Milvus** | Cloud dependency, overkill for <1M vectors |
| **FAISS** | C++ library, complex .NET interop |
| **SQLite + BLOB** | Poor performance for similarity search |
| **JSON/MessagePack** | Slow parsing, no memory-mapping |

**CVVI advantages:**
- **Zero dependencies**: No external vector DB needed
- **Memory-mapped**: Load 100K vectors in <10ms without copying to RAM
- **Single file**: Easy backup, sync, portable
- **Optimized**: Fixed-size entries enable O(1) random access

#### Binary Structure

```
┌─────────────────────────────────────────────────────────────────────────┐
│ HEADER (32 bytes)                                                        │
├─────────────────────────────────────────────────────────────────────────┤
│ Offset │ Size  │ Field           │ Description                          │
├────────┼───────┼─────────────────┼──────────────────────────────────────┤
│ 0      │ 4     │ Magic           │ "CVVI" ASCII (0x43565649)            │
│ 4      │ 1     │ Version         │ Format version (currently 1)         │
│ 5      │ 1     │ Precision       │ 0=F32 (4 bytes), 1=F16 (2 bytes)     │
│ 6      │ 2     │ Dimension       │ Vector dimension (384 for MiniLM)    │
│ 8      │ 8     │ EntryCount      │ Number of vectors stored (u64)       │
│ 16     │ 8     │ CreatedAt       │ Unix timestamp in milliseconds       │
│ 24     │ 4     │ CRC32           │ Checksum of all entry data           │
│ 28     │ 4     │ Reserved        │ Future use (zero-padded)             │
├─────────────────────────────────────────────────────────────────────────┤
│ ENTRIES (repeating, variable total size)                                 │
├─────────────────────────────────────────────────────────────────────────┤
│ Offset │ Size  │ Field           │ Description                          │
├────────┼───────┼─────────────────┼──────────────────────────────────────┤
│ 0      │ 32    │ ContentHash     │ SHA-256 hash of message content      │
│ 32     │ 16    │ MessageId       │ GUID linking to SQLite message       │
│ 48     │ 1     │ AgentType       │ Enum: 0=Copilot, 1=Claude, 2=Cursor… │
│ 49     │ 8     │ Timestamp       │ Message timestamp (Unix ms)          │
│ 57     │ D×P   │ Vector          │ D=dimension, P=bytes per component   │
└─────────────────────────────────────────────────────────────────────────┘

Entry sizes for 384-dimensional MiniLM:
├── F32 precision: 32 + 16 + 1 + 8 + (384 × 4) = 1,593 bytes/entry
└── F16 precision: 32 + 16 + 1 + 8 + (384 × 2) =   825 bytes/entry
```

#### Search Algorithm Flow

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    SEMANTIC SEARCH PIPELINE                              │
└──────────────────────────────────────────────────────────────────────────┘

Step 1: QUERY EMBEDDING
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│ User Query      │────▶│ Embedding Engine │────▶│ Query Vector        │
│ "auth error"    │     │                  │     │ [0.12, 0.87, ...]   │
└─────────────────┘     │  ┌────────────┐  │     │ (384 dimensions)    │
                        │  │ ONNX Model │  │     └─────────────────────┘
                        │  │ (MiniLM)   │  │
                        │  └────────────┘  │
                        │       OR         │
                        │  ┌────────────┐  │
                        │  │ Hash Embed │  │  ◀── Fallback if no model
                        │  │ (FNV-1a)   │  │
                        │  └────────────┘  │
                        └──────────────────┘

Step 2: SIMILARITY COMPUTATION (SIMD-accelerated)
┌─────────────────────────────────────────────────────────────────────────┐
│  Memory-mapped CVVI file                                                │
│  ┌─────────┬─────────┬─────────┬─────────┬─────────┬─────────┐        │
│  │ Entry 0 │ Entry 1 │ Entry 2 │ Entry 3 │   ...   │ Entry N │        │
│  └────┬────┴────┬────┴────┬────┴────┬────┴─────────┴────┬────┘        │
│       │         │         │         │                   │              │
│       ▼         ▼         ▼         ▼                   ▼              │
│    cosine    cosine    cosine    cosine              cosine            │
│    (Q, V₀)   (Q, V₁)   (Q, V₂)   (Q, V₃)            (Q, Vₙ)           │
│       │         │         │         │                   │              │
│       ▼         ▼         ▼         ▼                   ▼              │
│     0.91      0.34      0.87      0.12               0.76              │
└─────────────────────────────────────────────────────────────────────────┘

  Cosine similarity (for L2-normalized vectors):
  similarity = Q · V = Σ(qᵢ × vᵢ)   ◀── Simple dot product!

Step 3: TOP-K SELECTION (Min-Heap)
┌─────────────────────────────────────────────────────────────────────────┐
│  PriorityQueue<(MessageId, Score), Score>                               │
│                                                                         │
│  Maintain K best results (default K=20):                                │
│  ┌──────────────────────────────────────┐                              │
│  │ (msg_42,  0.91) ◀── highest          │                              │
│  │ (msg_2,   0.87)                      │                              │
│  │ (msg_99,  0.84)                      │                              │
│  │ (msg_17,  0.81)                      │                              │
│  │    ...                               │                              │
│  │ (msg_203, 0.76) ◀── K-th best        │                              │
│  └──────────────────────────────────────┘                              │
└─────────────────────────────────────────────────────────────────────────┘

Step 4: JOIN WITH METADATA
┌─────────────────────────────────────────────────────────────────────────┐
│  For each result:                                                       │
│    msg_id ──▶ SQLite lookup ──▶ Full message content + surrounding     │
│                                  context (N messages before/after)      │
└─────────────────────────────────────────────────────────────────────────┘
```

#### C# Implementation

```csharp
/// <summary>
/// Memory-mapped CVVI vector index for fast similarity search.
/// </summary>
public sealed class CvviIndex : IDisposable
{
    private const uint MagicNumber = 0x49565643; // "CVVI" little-endian
    private const int HeaderSize = 32;
    
    private readonly MemoryMappedFile _mmf;
    private readonly MemoryMappedViewAccessor _accessor;
    private readonly int _dimension;
    private readonly int _entrySize;
    private readonly long _entryCount;
    private readonly bool _isF16;
    
    public static CvviIndex Open(string path)
    {
        var mmf = MemoryMappedFile.CreateFromFile(path, FileMode.Open);
        return new CvviIndex(mmf);
    }
    
    /// <summary>
    /// Get vector at index without allocation (zero-copy read).
    /// </summary>
    public ReadOnlySpan<float> GetVector(long index)
    {
        var offset = HeaderSize + (index * _entrySize) + 57; // Skip to vector
        
        if (_isF16)
        {
            // Convert F16 to F32 on the fly
            Span<float> result = stackalloc float[_dimension];
            for (int i = 0; i < _dimension; i++)
            {
                var half = _accessor.ReadUInt16(offset + i * 2);
                result[i] = (float)BitConverter.UInt16BitsToHalf(half);
            }
            return result.ToArray(); // Must copy for F16
        }
        
        // F32: Direct memory access
        unsafe
        {
            byte* ptr = null;
            _accessor.SafeMemoryMappedViewHandle.AcquirePointer(ref ptr);
            return new ReadOnlySpan<float>(ptr + offset, _dimension);
        }
    }
    
    /// <summary>
    /// SIMD-accelerated cosine similarity using .NET 8+ TensorPrimitives.
    /// Vectors must be L2-normalized.
    /// </summary>
    public static float CosineSimilarity(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        // Uses AVX2/AVX-512/NEON automatically
        return TensorPrimitives.Dot(a, b);
    }
    
    /// <summary>
    /// Brute-force search - fast enough for &lt;100K vectors (~50ms).
    /// </summary>
    public IReadOnlyList<SearchResult> Search(
        ReadOnlySpan<float> queryVector, 
        int topK = 20)
    {
        var heap = new PriorityQueue<SearchResult, float>();
        
        for (long i = 0; i < _entryCount; i++)
        {
            var vector = GetVector(i);
            var score = CosineSimilarity(queryVector, vector);
            
            if (heap.Count < topK)
            {
                heap.Enqueue(CreateResult(i, score), -score); // Min-heap trick
            }
            else if (score > -heap.Peek())
            {
                heap.DequeueEnqueue(CreateResult(i, score), -score);
            }
        }
        
        return heap.UnorderedItems
            .Select(x => x.Element)
            .OrderByDescending(x => x.Score)
            .ToList();
    }
    
    public record SearchResult(Guid MessageId, float Score, byte AgentType);
}
```

#### Performance Characteristics

| Metric | Value | Notes |
|--------|-------|-------|
| **Index Size** | | |
| 10K messages (F32) | ~16 MB | 1,593 bytes × 10,000 |
| 10K messages (F16) | ~8 MB | 825 bytes × 10,000 |
| 100K messages (F32) | ~160 MB | Still fits in memory |
| **Load Time** | | |
| Memory-map 100K vectors | <10ms | No data copying |
| **Search Time** | | |
| 10K vectors | ~5ms | SIMD dot product |
| 50K vectors | ~25ms | Linear scan |
| 100K vectors | ~50ms | Still interactive |
| **Embedding Time** | | |
| ONNX MiniLM (per query) | ~20ms | GPU: ~5ms |
| Hash fallback (per query) | <1ms | Deterministic |

#### Content Deduplication

The `ContentHash` (SHA-256) field enables smart deduplication:

```csharp
// Before embedding, check if content already indexed
var hash = SHA256.HashData(Encoding.UTF8.GetBytes(message.Content));

if (await _index.ContainsHashAsync(hash))
{
    // Skip embedding - just add reference
    await _index.AddReferenceAsync(hash, message.Id);
}
else
{
    // Compute embedding and store
    var vector = await _embedder.EmbedAsync(message.Content);
    await _index.AddEntryAsync(hash, message.Id, vector);
}
```

**Benefits:**
- Identical messages across agents share one vector
- Reduces index size when content repeats
- Skips expensive ONNX inference for duplicates

#### Future Scaling (If Needed)

For >100K vectors, we can add approximate search without changing the file format:

| Method | Complexity | Accuracy | Implementation |
|--------|------------|----------|----------------|
| **HNSW** | O(log n) | ~95% | Add graph layer on top |
| **IVF** | O(√n) | ~90% | Cluster vectors, search subset |
| **PQ** | O(n) faster | ~85% | Quantize to 8-bit sub-vectors |

The CVVI format stores raw vectors, so any index structure can be built on top.

## Tradeoffs

| Pros | Cons |
|------|------|
| **.NET 10 ecosystem**: Mature tooling, great IDE support, cross-platform | Larger runtime footprint than Rust (~20MB vs ~5MB) |
| **Lucene.NET**: Battle-tested, feature-rich search | More complex than simpler search libraries |
| **ONNX Runtime**: Official Microsoft support, optimized | Model files add ~90MB when present |
| **Global tool distribution**: Simple `dotnet tool install` | Requires .NET runtime (or AOT compilation) |
| **C# productivity**: Fast development, strong typing | Performance slightly lower than Rust for indexing |
| **Plugin architecture**: Easy to add new agents | More complex initial implementation |
| **SQLite metadata**: ACID, queryable, portable | Additional dependency (though very lightweight) |
| **HTML export with templates**: Maintainable, customizable | Requires template engine dependency |

## Alignment

- [x] Follows architectural layering rules (clean separation of concerns)
- [x] Developer Experience (single `dotnet tool install`, minimal config)
- [x] Specification compliance (ONNX standard, SQLite standard)
- [x] Consistent with existing patterns (similar to cass architecture)

## Evidence

### Prior Art Analysis

#### 1. coding_agent_session_search (cass) - Rust
**Strengths adopted**:
- Unified data model (Conversation → Message → Snippet)
- Three search modes (lexical, semantic, hybrid)
- Hash-based vector fallback when ML models absent
- CVVI binary format for vector storage
- Robot mode for agent consumption
- Edge n-gram indexing for instant search

**Differences**:
- Rust's Tantivy vs .NET's Lucene.NET (both BM25-capable)
- TUI-first vs CLI-first (we focus on CLI with optional TUI later)
- Async concurrency model differences

#### 2. claude-run - TypeScript/Node.js
**Strengths adopted**:
- Beautiful HTML rendering of conversations
- Dark mode support
- Collapsible tool calls
- Session filtering by project
- Real-time updates (file watching)

**Differences**:
- Web server vs CLI tool (we're CLI-first)
- Single agent (Claude) vs multi-agent support
- Node.js runtime vs .NET runtime

### .NET Library Research

#### Lucene.NET 4.8
- Full BM25 support with configurable parameters
- Edge n-gram tokenizer available
- Memory-mapped directory for large indexes
- Active maintenance, .NET 8+ compatible

#### Microsoft.ML.OnnxRuntime
- MiniLM model support confirmed
- Cross-platform (Windows, Linux, macOS, ARM64)
- ~20ms inference time for 384-dim embeddings
- DirectML support for GPU acceleration (optional)

#### System.CommandLine
- Official Microsoft CLI framework
- Automatic help generation
- Tab completion support
- Middleware for common concerns (logging, config)

### Copilot CLI Session Format Research

Location: `~/.copilot-cli/sessions/` (needs verification)
Format: JSONL with message events

### Claude Code Session Format
Location: `~/.claude/projects/<project-hash>/`
Format: JSONL with typed events:
- `session_start`
- `message` (user/assistant)
- `tool_use`, `tool_result`

## Alternative Approaches Worth Investigating

1. **rust-based-tool**: Pure Rust implementation (like cass) for maximum performance
2. **node-based-tool**: Node.js/TypeScript for easier HTML/web integration
3. **hybrid-approach**: .NET core with Rust FFI for search performance
4. **mcp-server**: Build as MCP server rather than CLI tool for direct agent integration

## Verdict

**Recommended**: Proceed with .NET 10 global tool architecture.

**Rationale**:
1. .NET 10's enhanced tooling features (one-shot execution, better AOT) address deployment concerns
2. Lucene.NET provides production-grade search without reinventing the wheel
3. ONNX Runtime is first-party Microsoft with excellent cross-platform support
4. C# productivity enables faster iteration on features
5. Plugin architecture allows incremental agent support
6. Easy distribution via NuGet (`dotnet tool install -g chats`)

**Next Steps**:
1. Create project structure with `dotnet new tool`
2. Implement core data model and SQLite storage
3. Build first connector (Claude Code - best documented format)
4. Add Lucene.NET indexing
5. Implement basic CLI commands (index, search, export)
6. Add vector search with ONNX/hash fallback
7. Build HTML export templates
8. Add remaining agent connectors
