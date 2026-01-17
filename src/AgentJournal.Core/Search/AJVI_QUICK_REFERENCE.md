# AJVI Quick Reference

## File Format

| Component      | Size (bytes)          | Description                           |
|----------------|-----------------------|---------------------------------------|
| **Header**     | 32                    | Magic, version, precision, dimensions |
| **Entry**      | 57 + dims×(2 or 4)    | Hash, ID, type, timestamp, vector     |

**Magic**: `0x494A5641` ("AJVI" in little-endian)  
**Version**: `1`  
**Precision**: `0` = Float32, `1` = Float16  

## API Quick Reference

### Factory Methods

```csharp
// Create new index
AjviIndex.Create(string path, int dimensions, VectorPrecision precision = Float16)

// Open existing index
AjviIndex.Open(string path, bool readOnly = false)
```

### Properties

```csharp
long EntryCount      // Number of entries in the index
int Dimensions       // Vector dimensions
VectorPrecision Precision  // Float32 or Float16
```

### Core Operations

```csharp
// Add entry (throws if read-only)
void AddEntry(byte[] contentHash, Guid messageId, byte agentType, 
              long timestamp, ReadOnlySpan<float> vector)

// Search (returns top-K by cosine similarity)
IReadOnlyList<(long Index, float Score)> Search(
    ReadOnlySpan<float> queryVector, int topK = 20)

// Check for duplicate content
bool ContainsHash(ReadOnlySpan<byte> contentHash)
```

### Getters

```csharp
ReadOnlySpan<float> GetVector(long index)
Guid GetMessageId(long index)
byte[] GetContentHash(long index)
byte GetAgentType(long index)
long GetTimestamp(long index)
```

## Usage Pattern

```csharp
// 1. Create/Open
using var index = File.Exists(path) 
    ? AjviIndex.Open(path) 
    : AjviIndex.Create(path, 384);

// 2. Add entries
var hash = SHA256.HashData(Encoding.UTF8.GetBytes(content));
if (!index.ContainsHash(hash)) {
    var vector = NormalizeVector(GetEmbedding(content));
    index.AddEntry(hash, messageId, agentType, timestamp, vector);
}

// 3. Search
var queryVector = NormalizeVector(GetEmbedding(query));
var results = index.Search(queryVector, topK: 20);

// 4. Process results
foreach (var (idx, score) in results) {
    var msgId = index.GetMessageId(idx);
    Console.WriteLine($"{msgId}: {score:F4}");
}
```

## Important Notes

⚠️ **Always normalize vectors** (unit length) before adding/searching  
⚠️ **Content hash must be 32 bytes** (SHA256)  
⚠️ **Timestamp in Unix milliseconds**, not seconds  
⚠️ **Single writer only**, multiple readers OK in read-only mode  
⚠️ **No deletion support**, index is append-only  

## Agent Type Codes

| Code | Agent       |
|------|-------------|
| 0    | Copilot     |
| 1    | Claude      |
| 2+   | Custom      |

## Storage Sizes (384 dimensions)

| Precision | Entry Size | Entries/GB |
|-----------|------------|------------|
| Float16   | 825 bytes  | ~1.2M      |
| Float32   | 1,593 bytes| ~656K      |

## Performance (384-dim, modern CPU)

| Entries | Search Time |
|---------|-------------|
| 10K     | ~10ms       |
| 100K    | ~100ms      |
| 1M      | ~1s         |

## Vector Normalization

```csharp
float[] NormalizeVector(float[] vector) {
    float sumSq = 0;
    foreach (var v in vector) sumSq += v * v;
    float mag = MathF.Sqrt(sumSq);
    for (int i = 0; i < vector.Length; i++)
        vector[i] /= mag;
    return vector;
}
```

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `InvalidOperationException: file exists` | Creating over existing | Use `Open()` instead |
| `ArgumentException: hash must be 32 bytes` | Wrong hash size | Use SHA256 |
| `ArgumentException: vector dimensions` | Wrong vector size | Match index dimensions |
| `ObjectDisposedException` | Using after dispose | Create new instance |
| `InvalidOperationException: read-only` | Adding to read-only | Open without readOnly flag |

## File Operations

```csharp
// Check if index exists
bool exists = File.Exists(indexPath);

// Get index info without opening
using var fs = File.OpenRead(indexPath);
// Read header manually if needed

// Delete index
File.Delete(indexPath);

// Copy index
File.Copy(indexPath, backupPath);

// Get file size
long size = new FileInfo(indexPath).Length;
long estimatedEntries = (size - 32) / entrySize;
```

## Thread Safety

- ✅ **Single writer** or **multiple read-only readers**
- ❌ **NOT thread-safe** for concurrent writes
- ❌ **NOT safe** to mix read-write and read-only

```csharp
// OK: Single writer
using var writer = AjviIndex.Open(path);

// OK: Multiple concurrent readers
using var reader1 = AjviIndex.Open(path, readOnly: true);
using var reader2 = AjviIndex.Open(path, readOnly: true);

// NOT OK: Concurrent writers
// Multiple AjviIndex.Open(path) without readOnly
```

## Troubleshooting

**Slow search?**
- Check vector dimensions (higher = slower)
- Verify SIMD support (AVX2/AVX-512)
- Consider indexing only recent messages

**Large file size?**
- Switch from Float32 to Float16 (48% savings)
- Reduce vector dimensions if possible
- Implement entry filtering/pruning

**Duplicates being indexed?**
- Always use `ContainsHash()` before `AddEntry()`
- Ensure consistent content hashing

**Out of memory?**
- Memory-mapped files use virtual memory
- Check OS-level memory limits
- Consider read-only mode for queries

## See Also

- **AJVI_SPECIFICATION.md** - Full format documentation
- **README_AJVI.md** - Detailed usage guide
- **AjviIndexTest.cs** - Example usage code
