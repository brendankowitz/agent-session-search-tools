# AJVI (Agent Journal Vector Index) Specification

## Overview

AJVI is a memory-mapped binary vector index format designed for efficient semantic search over agent conversation messages. It provides fast, SIMD-accelerated similarity search with support for both Float32 and Float16 precision vectors.

## Features

- **Memory-mapped I/O**: Efficient file access without loading entire index into memory
- **Dual precision**: Support for Float32 (4 bytes/dim) and Float16 (2 bytes/dim) storage
- **SIMD acceleration**: Uses `System.Numerics.Tensors.TensorPrimitives` for fast cosine similarity
- **Deduplication**: Built-in content hash checking to avoid duplicate entries
- **Dynamic growth**: Automatically resizes index file as entries are added
- **Cross-platform**: Works on Windows, Linux, and macOS with .NET 10

## Binary Format

### File Structure

```
┌─────────────────────────────────────────────┐
│ Header (32 bytes)                           │
├─────────────────────────────────────────────┤
│ Entry 0                                     │
│   - Content Hash (32 bytes)                 │
│   - Message ID (16 bytes)                   │
│   - Agent Type (1 byte)                     │
│   - Timestamp (8 bytes)                     │
│   - Vector (dimensions * precision bytes)   │
├─────────────────────────────────────────────┤
│ Entry 1                                     │
│   ...                                       │
├─────────────────────────────────────────────┤
│ Entry N-1                                   │
│   ...                                       │
└─────────────────────────────────────────────┘
```

### Header Format (32 bytes)

| Offset | Size | Type    | Description                                      |
|--------|------|---------|--------------------------------------------------|
| 0      | 4    | uint32  | Magic number: 0x494A5641 ("AJVI" in little-endian) |
| 4      | 1    | byte    | Version: 1                                       |
| 5      | 1    | byte    | Precision: 0=Float32, 1=Float16                  |
| 6      | 2    | uint16  | Dimensions (1-65535)                             |
| 8      | 8    | int64   | Entry count                                      |
| 16     | 16   | -       | Reserved for future use                          |

### Entry Format (variable size)

| Field         | Size                      | Type    | Description                        |
|---------------|---------------------------|---------|------------------------------------|
| ContentHash   | 32 bytes                  | byte[]  | SHA256 hash of message content     |
| MessageId     | 16 bytes                  | GUID    | Unique message identifier          |
| AgentType     | 1 byte                    | byte    | Agent type (0=copilot, 1=claude)   |
| Timestamp     | 8 bytes                   | int64   | Unix timestamp in milliseconds     |
| Vector        | dimensions * (2 or 4)     | float[] | Embedding vector (normalized)      |

**Entry Size Calculation**:
- Float32: 57 + (dimensions × 4) bytes
- Float16: 57 + (dimensions × 2) bytes

**Example Sizes** (for common embedding dimensions):
- 384 dimensions (all-MiniLM-L6-v2):
  - Float16: 825 bytes/entry
  - Float32: 1,593 bytes/entry
- 768 dimensions (BERT-base):
  - Float16: 1,593 bytes/entry
  - Float32: 3,129 bytes/entry
- 1536 dimensions (OpenAI text-embedding-ada-002):
  - Float16: 3,129 bytes/entry
  - Float32: 6,201 bytes/entry

## Usage Examples

### Creating a New Index

```csharp
using AgentJournal.Core.Search;

// Create index with 384 dimensions using Float16 precision
var index = AjviIndex.Create("embeddings.ajvi", dimensions: 384, 
    AjviIndex.VectorPrecision.Float16);
```

### Adding Entries

```csharp
using System.Security.Cryptography;

// Prepare entry data
var contentHash = SHA256.HashData(Encoding.UTF8.GetBytes(messageContent));
var messageId = Guid.Parse(message.Id);
byte agentType = message.AgentType == "copilot" ? (byte)0 : (byte)1;
long timestamp = new DateTimeOffset(message.Timestamp).ToUnixTimeMilliseconds();

// Get embedding vector (assumed to be normalized)
float[] vector = await embeddingModel.GetEmbeddingAsync(messageContent);

// Add to index
index.AddEntry(contentHash, messageId, agentType, timestamp, vector);
```

### Searching the Index

```csharp
// Get query embedding
float[] queryVector = await embeddingModel.GetEmbeddingAsync(searchQuery);

// Search for top 20 most similar entries
var results = index.Search(queryVector, topK: 20);

foreach (var (entryIndex, similarity) in results)
{
    var messageId = index.GetMessageId(entryIndex);
    var timestamp = index.GetTimestamp(entryIndex);
    
    Console.WriteLine($"Message {messageId}: {similarity:F4}");
}
```

### Deduplication Check

```csharp
var contentHash = SHA256.HashData(Encoding.UTF8.GetBytes(messageContent));

if (!index.ContainsHash(contentHash))
{
    // Add new entry
    index.AddEntry(contentHash, messageId, agentType, timestamp, vector);
}
else
{
    Console.WriteLine("Duplicate content detected, skipping...");
}
```

### Opening Existing Index

```csharp
// Open in read-write mode
using var index = AjviIndex.Open("embeddings.ajvi");

// Open in read-only mode (allows concurrent readers)
using var readOnlyIndex = AjviIndex.Open("embeddings.ajvi", readOnly: true);

Console.WriteLine($"Loaded index with {index.EntryCount} entries");
Console.WriteLine($"Dimensions: {index.Dimensions}");
Console.WriteLine($"Precision: {index.Precision}");
```

## Performance Characteristics

### Memory Usage

- **File handle only**: The index uses memory-mapped files, so the OS manages memory
- **No in-memory cache**: All data is accessed directly from the memory-mapped file
- **Minimal heap allocation**: Only active operations allocate temporary buffers

### Search Performance

- **Linear scan**: O(N) where N is the number of entries
- **SIMD acceleration**: 2-4x speedup on modern CPUs with AVX2/AVX-512
- **Typical throughput**: 
  - ~1M similarity computations/second (384-dim vectors on modern CPU)
  - For 10K entries: ~10ms search time
  - For 100K entries: ~100ms search time
  - For 1M entries: ~1s search time

### Storage Efficiency

**Float16 vs Float32 comparison** (384 dimensions):
- Float16: 825 bytes/entry → ~1.2M entries/GB
- Float32: 1,593 bytes/entry → ~656K entries/GB
- **Space savings: ~48%** with minimal accuracy loss

### Precision Trade-offs

Float16 precision is generally sufficient for semantic search:
- Accuracy loss: < 1% for most embedding models
- Cosine similarity differences: typically < 0.001
- Recommended for production use unless extreme precision is required

## Best Practices

### 1. Always Normalize Vectors

The index assumes vectors are normalized (unit length). Always normalize before adding:

```csharp
float[] NormalizeVector(float[] vector)
{
    float sumSquares = vector.Sum(v => v * v);
    float magnitude = MathF.Sqrt(sumSquares);
    return vector.Select(v => v / magnitude).ToArray();
}
```

### 2. Use Content Hashing for Deduplication

Always check for duplicates before adding entries:

```csharp
var hash = SHA256.HashData(Encoding.UTF8.GetBytes(content));
if (!index.ContainsHash(hash))
{
    index.AddEntry(hash, messageId, agentType, timestamp, vector);
}
```

### 3. Choose Appropriate Precision

- **Float16**: Default choice for most use cases (48% space savings)
- **Float32**: Only when precision is critical or debugging

### 4. Dispose Properly

Always use `using` statements or call `Dispose()`:

```csharp
using (var index = AjviIndex.Create(...))
{
    // Use index
} // Automatically disposed
```

### 5. Batch Operations

For bulk indexing, keep the index open and add entries in batches:

```csharp
using var index = AjviIndex.Create("embeddings.ajvi", 384);

foreach (var batch in messages.Chunk(1000))
{
    foreach (var message in batch)
    {
        // Add entry
        index.AddEntry(...);
    }
    
    Console.WriteLine($"Indexed {index.EntryCount} entries so far...");
}
```

### 6. Read-Only Access for Concurrent Queries

Multiple processes can read the same index simultaneously in read-only mode:

```csharp
// Process 1
using var index1 = AjviIndex.Open("embeddings.ajvi", readOnly: true);
var results1 = index1.Search(query1);

// Process 2 (can run concurrently)
using var index2 = AjviIndex.Open("embeddings.ajvi", readOnly: true);
var results2 = index2.Search(query2);
```

## Limitations

### Current Limitations

1. **No incremental updates**: Index only supports appending entries
2. **No deletion**: Cannot remove entries without rebuilding the index
3. **No compression**: Vectors are stored uncompressed
4. **Linear search only**: No indexing structures (HNSW, IVF, etc.)
5. **Single writer**: Only one process can write to the index at a time

### When NOT to Use AJVI

Consider alternative solutions if you need:
- **> 1M entries**: Use specialized vector databases (Pinecone, Weaviate, etc.)
- **Real-time updates**: AJVI is optimized for write-once, read-many scenarios
- **Sub-linear search**: For very large datasets, use ANN indices (HNSW, etc.)
- **Distributed search**: AJVI is designed for single-machine use

## Integration with VectorSearchEngine

The AJVI index integrates with `VectorSearchEngine` for semantic search:

```csharp
public class VectorSearchEngine : ISearchEngine
{
    private readonly string _indexPath;
    private readonly IEmbeddingProvider _embeddings;
    private AjviIndex? _index;

    public async Task IndexSessionAsync(Session session, CancellationToken ct)
    {
        _index ??= OpenOrCreateIndex();
        
        foreach (var message in session.Messages)
        {
            // Generate embedding
            var embedding = await _embeddings.GetEmbeddingAsync(message.Content, ct);
            
            // Add to index
            var hash = SHA256.HashData(Encoding.UTF8.GetBytes(message.Content));
            if (!_index.ContainsHash(hash))
            {
                _index.AddEntry(
                    hash,
                    Guid.Parse(message.Id),
                    GetAgentTypeCode(session.AgentType),
                    new DateTimeOffset(message.Timestamp).ToUnixTimeMilliseconds(),
                    embedding
                );
            }
        }
    }

    public async Task<IReadOnlyList<SearchResult>> SearchAsync(
        string query, SearchMode mode, int maxResults, CancellationToken ct)
    {
        _index ??= OpenOrCreateIndex();
        
        // Generate query embedding
        var queryEmbedding = await _embeddings.GetEmbeddingAsync(query, ct);
        
        // Search index
        var results = _index.Search(queryEmbedding, maxResults);
        
        // Convert to SearchResult objects
        // ... implementation details
    }
}
```

## Future Enhancements

Potential improvements for future versions:

1. **Approximate Nearest Neighbor (ANN)**: Add HNSW or IVF indexing
2. **Compression**: Support for product quantization (PQ) or scalar quantization
3. **Filtering**: Add support for metadata filtering (agent type, date range, etc.)
4. **Incremental updates**: Support for updating/deleting entries
5. **Multi-threading**: Parallel search across multiple segments
6. **Mmap pooling**: Reuse memory-mapped views for better performance
7. **Statistics**: Track index statistics (utilization, query latency, etc.)

## Version History

### Version 1 (Current)
- Initial release
- Float32 and Float16 precision support
- Memory-mapped file storage
- SIMD-accelerated cosine similarity
- Linear scan search

## License

This format is part of the Agent Journal project and follows the project's license terms.
