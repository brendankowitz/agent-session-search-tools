# AJVI Architecture Overview

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    VectorSearchEngine                       │
│                   (ISearchEngine impl)                      │
├─────────────────────────────────────────────────────────────┤
│  - IndexSessionAsync()                                      │
│  - SearchAsync()                                            │
│  - InitializeAsync()                                        │
│  - ClearIndexAsync()                                        │
└────────────┬────────────────────────────────────────────────┘
             │ uses
             ▼
┌─────────────────────────────────────────────────────────────┐
│                       AjviIndex                             │
│               (Memory-Mapped Vector Index)                  │
├─────────────────────────────────────────────────────────────┤
│  Properties:                                                │
│    - EntryCount: long                                       │
│    - Dimensions: int                                        │
│    - Precision: VectorPrecision                             │
│                                                             │
│  Factory Methods:                                           │
│    + Create(path, dims, precision)                          │
│    + Open(path, readOnly)                                   │
│                                                             │
│  Core Operations:                                           │
│    + AddEntry(hash, id, type, timestamp, vector)            │
│    + Search(queryVector, topK) → List<(Index, Score)>       │
│    + ContainsHash(hash) → bool                              │
│                                                             │
│  Getters:                                                   │
│    + GetVector(index) → float[]                             │
│    + GetMessageId(index) → Guid                             │
│    + GetContentHash(index) → byte[]                         │
│    + GetAgentType(index) → byte                             │
│    + GetTimestamp(index) → long                             │
└────────────┬────────────────────────────────────────────────┘
             │ uses
             ▼
┌─────────────────────────────────────────────────────────────┐
│                  System Components                          │
├─────────────────────────────────────────────────────────────┤
│  • MemoryMappedFile (System.IO.MemoryMappedFiles)           │
│  • TensorPrimitives (System.Numerics.Tensors)               │
│  • PriorityQueue (System.Collections.Generic)               │
│  • SHA256 (System.Security.Cryptography)                    │
└─────────────────────────────────────────────────────────────┘
```

## File Format Structure

```
╔══════════════════════════════════════════════════════════════╗
║                     AJVI INDEX FILE                          ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  ┌──────────────────────────────────────────────────────┐   ║
║  │              HEADER (32 bytes)                       │   ║
║  ├──────────────────────────────────────────────────────┤   ║
║  │  Magic:      0x494A5641 ("AJVI")        [4 bytes]   │   ║
║  │  Version:    1                          [1 byte]    │   ║
║  │  Precision:  0=F32, 1=F16               [1 byte]    │   ║
║  │  Dimensions: 1-65535                    [2 bytes]   │   ║
║  │  EntryCount: 0-2^63-1                   [8 bytes]   │   ║
║  │  Reserved:   (future use)               [16 bytes]  │   ║
║  └──────────────────────────────────────────────────────┘   ║
║                                                              ║
║  ┌──────────────────────────────────────────────────────┐   ║
║  │           ENTRY 0 (variable size)                    │   ║
║  ├──────────────────────────────────────────────────────┤   ║
║  │  ContentHash: SHA256                    [32 bytes]  │   ║
║  │  MessageId:   GUID                      [16 bytes]  │   ║
║  │  AgentType:   0=copilot, 1=claude       [1 byte]    │   ║
║  │  Timestamp:   Unix ms                   [8 bytes]   │   ║
║  │  Vector:      Float16/32 array          [dims×2/4]  │   ║
║  └──────────────────────────────────────────────────────┘   ║
║                                                              ║
║  ┌──────────────────────────────────────────────────────┐   ║
║  │           ENTRY 1 (variable size)                    │   ║
║  └──────────────────────────────────────────────────────┘   ║
║                                                              ║
║  ...                                                         ║
║                                                              ║
║  ┌──────────────────────────────────────────────────────┐   ║
║  │           ENTRY N-1 (variable size)                  │   ║
║  └──────────────────────────────────────────────────────┘   ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

## Data Flow

### Indexing Flow

```
Session Messages
     │
     ▼
┌─────────────────┐
│  Extract Text   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Generate Hash   │  ◄── SHA256
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Check Duplicate │  ◄── AjviIndex.ContainsHash()
└────────┬────────┘
         │ (if new)
         ▼
┌─────────────────┐
│ Get Embedding   │  ◄── Embedding Model (384-dim)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Normalize Vec   │  ◄── L2 normalization
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Add Entry     │  ◄── AjviIndex.AddEntry()
└─────────────────┘
```

### Search Flow

```
User Query
    │
    ▼
┌─────────────────┐
│ Get Embedding   │  ◄── Embedding Model
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Normalize Vec   │  ◄── L2 normalization
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Top-K Search  │  ◄── AjviIndex.Search(query, topK)
└────────┬────────┘
         │
         ├─── For each entry:
         │    1. Read vector from memory-mapped file
         │    2. Compute cosine similarity (SIMD)
         │    3. Update PriorityQueue if score > min
         │
         ▼
┌─────────────────┐
│ Get Message IDs │  ◄── AjviIndex.GetMessageId()
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Fetch Messages  │  ◄── SessionStore
└────────┬────────┘
         │
         ▼
  Search Results
```

## Performance Characteristics

### Time Complexity

| Operation              | Complexity | Notes                           |
|------------------------|------------|---------------------------------|
| Create index           | O(1)       | Just writes header              |
| Add entry              | O(1)       | Append-only, may resize file    |
| Search (linear)        | O(N·D)     | N=entries, D=dimensions         |
| ContainsHash (linear)  | O(N)       | Linear scan through hashes      |
| GetVector              | O(D)       | Read from memory-mapped file    |
| GetMessageId           | O(1)       | Direct offset access            |

### Space Complexity

| Component        | Size                  | Notes                           |
|------------------|-----------------------|---------------------------------|
| Header           | 32 bytes              | Fixed size                      |
| Entry (F16)      | 57 + dims×2           | 384-dim = 825 bytes             |
| Entry (F32)      | 57 + dims×4           | 384-dim = 1,593 bytes           |
| Memory footprint | ~File handle only     | OS manages memory-mapped pages  |

### SIMD Acceleration

```
Vector Dot Product:
  Without SIMD: 384 multiplies + 383 adds = 767 ops
  With SIMD:    384 / 8 = 48 ops (AVX2, 8 floats per instruction)
  Speedup:      ~16x theoretical, ~4-8x practical
```

## Integration Example

### Complete VectorSearchEngine Implementation

```csharp
public class VectorSearchEngine : ISearchEngine
{
    private readonly string _indexPath;
    private readonly IEmbeddingProvider _embeddings;
    private readonly ISessionStore _sessionStore;
    private AjviIndex? _index;
    
    public VectorSearchEngine(
        string indexPath, 
        IEmbeddingProvider embeddings,
        ISessionStore sessionStore)
    {
        _indexPath = indexPath;
        _embeddings = embeddings;
        _sessionStore = sessionStore;
    }
    
    public async Task InitializeAsync(CancellationToken ct = default)
    {
        // Create or open index
        _index = File.Exists(_indexPath)
            ? AjviIndex.Open(_indexPath)
            : AjviIndex.Create(_indexPath, 384, AjviIndex.VectorPrecision.Float16);
    }
    
    public async Task IndexSessionAsync(Session session, CancellationToken ct = default)
    {
        _index ??= AjviIndex.Open(_indexPath);
        
        foreach (var message in session.Messages)
        {
            // Skip empty messages
            if (string.IsNullOrWhiteSpace(message.Content)) continue;
            
            // Check for duplicates
            var hash = SHA256.HashData(Encoding.UTF8.GetBytes(message.Content));
            if (_index.ContainsHash(hash)) continue;
            
            // Generate and normalize embedding
            var embedding = await _embeddings.GetEmbeddingAsync(message.Content, ct);
            embedding = NormalizeVector(embedding);
            
            // Add to index
            _index.AddEntry(
                hash,
                Guid.Parse(message.Id),
                GetAgentTypeCode(session.AgentType),
                new DateTimeOffset(message.Timestamp).ToUnixTimeMilliseconds(),
                embedding
            );
        }
    }
    
    public async Task<IReadOnlyList<SearchResult>> SearchAsync(
        string query, 
        SearchMode mode, 
        int maxResults, 
        CancellationToken ct = default)
    {
        _index ??= AjviIndex.Open(_indexPath);
        
        // Generate and normalize query embedding
        var queryEmbedding = await _embeddings.GetEmbeddingAsync(query, ct);
        queryEmbedding = NormalizeVector(queryEmbedding);
        
        // Search index
        var indexResults = _index.Search(queryEmbedding, maxResults);
        
        // Convert to search results
        var results = new List<SearchResult>();
        foreach (var (idx, score) in indexResults)
        {
            var messageId = _index.GetMessageId(idx);
            var message = await _sessionStore.GetMessageAsync(messageId.ToString(), ct);
            var session = await _sessionStore.GetSessionAsync(message.SessionId, ct);
            
            results.Add(new SearchResult(
                session,
                score,
                new[] { message },
                null
            ));
        }
        
        return results;
    }
    
    private static float[] NormalizeVector(float[] vector)
    {
        var sumSquares = vector.Sum(v => v * v);
        var magnitude = MathF.Sqrt(sumSquares);
        return vector.Select(v => v / magnitude).ToArray();
    }
    
    private static byte GetAgentTypeCode(string agentType) => agentType.ToLowerInvariant() switch
    {
        "copilot" => 0,
        "claude" => 1,
        _ => 255
    };
}
```

## Testing Strategy

### Unit Tests
- ✅ Index creation and opening
- ✅ Entry addition and retrieval
- ✅ Hash deduplication
- ✅ Search functionality
- ✅ File format validation
- ✅ Error handling

### Integration Tests
- 🔲 Full indexing pipeline
- 🔲 Search with real embeddings
- 🔲 Multi-session indexing
- 🔲 Large dataset performance

### Performance Tests
- 🔲 Search latency benchmarks
- 🔲 Indexing throughput
- 🔲 Memory usage profiling
- 🔲 SIMD acceleration validation

## Limitations and Future Work

### Current Limitations
1. **Linear Search**: O(N) search complexity
2. **No Deletion**: Append-only design
3. **Single Writer**: No concurrent write support
4. **No Compression**: Full precision storage
5. **No Partitioning**: Single monolithic file

### Future Enhancements
1. **ANN Indexing**: Add HNSW for sub-linear search
2. **Quantization**: PQ/SQ for 4-8x compression
3. **Segmentation**: Split into multiple segments
4. **Filtering**: Metadata-based filtering
5. **Distributed**: Multi-node support
6. **Incremental Updates**: Support deletion/updates

## See Also
- `AjviIndex.cs` - Implementation
- `AJVI_SPECIFICATION.md` - Binary format specification
- `README_AJVI.md` - Usage guide
- `AJVI_QUICK_REFERENCE.md` - Quick reference
- `AjviIndexTest.cs` - Test examples
