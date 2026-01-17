# AJVI API Usage Examples

## Table of Contents
1. [Basic Usage](#basic-usage)
2. [Advanced Patterns](#advanced-patterns)
3. [Error Handling](#error-handling)
4. [Performance Optimization](#performance-optimization)
5. [Common Pitfalls](#common-pitfalls)

---

## Basic Usage

### 1. Creating a New Index

```csharp
using AgentJournal.Core.Search;

// Create with default Float16 precision (recommended)
using var index = AjviIndex.Create(
    "embeddings.ajvi",
    dimensions: 384  // all-MiniLM-L6-v2
);

// Or with Float32 for maximum precision
using var indexF32 = AjviIndex.Create(
    "embeddings_f32.ajvi",
    dimensions: 384,
    AjviIndex.VectorPrecision.Float32
);
```

### 2. Opening an Existing Index

```csharp
// Open for read/write
using var index = AjviIndex.Open("embeddings.ajvi");

// Open read-only (allows multiple concurrent readers)
using var readIndex = AjviIndex.Open("embeddings.ajvi", readOnly: true);

// Check properties
Console.WriteLine($"Entries: {index.EntryCount}");
Console.WriteLine($"Dimensions: {index.Dimensions}");
Console.WriteLine($"Precision: {index.Precision}");
```

### 3. Adding Entries

```csharp
using System.Security.Cryptography;
using System.Text;

// Prepare data
var content = "How do I implement error handling in C#?";
var contentHash = SHA256.HashData(Encoding.UTF8.GetBytes(content));
var messageId = Guid.NewGuid();
byte agentType = 0; // 0 = copilot, 1 = claude
long timestamp = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();

// Get embedding (from your embedding provider)
float[] embedding = await embeddingProvider.GetEmbeddingAsync(content);

// IMPORTANT: Normalize the vector
embedding = NormalizeVector(embedding);

// Add to index
index.AddEntry(contentHash, messageId, agentType, timestamp, embedding);
```

### 4. Searching

```csharp
// Get query embedding
var query = "error handling best practices";
var queryEmbedding = await embeddingProvider.GetEmbeddingAsync(query);
queryEmbedding = NormalizeVector(queryEmbedding);

// Search for top 20 results
var results = index.Search(queryEmbedding, topK: 20);

// Process results
foreach (var (entryIndex, similarity) in results)
{
    var messageId = index.GetMessageId(entryIndex);
    var timestamp = index.GetTimestamp(entryIndex);
    var agentType = index.GetAgentType(entryIndex);
    
    var date = DateTimeOffset.FromUnixTimeMilliseconds(timestamp);
    var agent = agentType == 0 ? "Copilot" : "Claude";
    
    Console.WriteLine($"{messageId} ({agent}, {date:g}): {similarity:F4}");
}
```

### 5. Checking for Duplicates

```csharp
var contentHash = SHA256.HashData(Encoding.UTF8.GetBytes(content));

if (!index.ContainsHash(contentHash))
{
    // New content, add it
    var embedding = await embeddingProvider.GetEmbeddingAsync(content);
    embedding = NormalizeVector(embedding);
    index.AddEntry(contentHash, messageId, agentType, timestamp, embedding);
    Console.WriteLine("Added new entry");
}
else
{
    Console.WriteLine("Duplicate detected, skipping");
}
```

---

## Advanced Patterns

### 1. Batch Indexing with Progress Reporting

```csharp
async Task IndexSessionsAsync(
    IEnumerable<Session> sessions, 
    IEmbeddingProvider embeddings,
    AjviIndex index,
    IProgress<int> progress)
{
    int processed = 0, added = 0, skipped = 0;
    
    foreach (var session in sessions)
    {
        foreach (var message in session.Messages)
        {
            // Skip empty messages
            if (string.IsNullOrWhiteSpace(message.Content))
            {
                skipped++;
                continue;
            }
            
            // Check for duplicates
            var hash = SHA256.HashData(Encoding.UTF8.GetBytes(message.Content));
            if (index.ContainsHash(hash))
            {
                skipped++;
                continue;
            }
            
            // Generate embedding
            var embedding = await embeddings.GetEmbeddingAsync(message.Content);
            embedding = NormalizeVector(embedding);
            
            // Add to index
            index.AddEntry(
                hash,
                Guid.Parse(message.Id),
                GetAgentTypeCode(session.AgentType),
                new DateTimeOffset(message.Timestamp).ToUnixTimeMilliseconds(),
                embedding
            );
            
            added++;
            processed++;
            
            // Report progress every 100 items
            if (processed % 100 == 0)
            {
                progress.Report(processed);
            }
        }
    }
    
    Console.WriteLine($"Done! Added {added}, skipped {skipped} duplicates");
}
```

### 2. Concurrent Search with Read-Only Indexes

```csharp
async Task<Dictionary<string, IReadOnlyList<SearchResult>>> 
    MultiQuerySearchAsync(string[] queries, string indexPath)
{
    var results = new ConcurrentDictionary<string, IReadOnlyList<SearchResult>>();
    
    // Parallel search with read-only indexes
    await Parallel.ForEachAsync(queries, async (query, ct) =>
    {
        // Each task gets its own read-only index instance
        using var index = AjviIndex.Open(indexPath, readOnly: true);
        
        var embedding = await GetEmbeddingAsync(query);
        var searchResults = index.Search(embedding, topK: 10);
        
        var converted = await ConvertToSearchResults(searchResults, index);
        results[query] = converted;
    });
    
    return results;
}
```

### 3. Index with Automatic Backup

```csharp
class BackupAjviIndex : IDisposable
{
    private readonly AjviIndex _index;
    private readonly string _indexPath;
    private readonly string _backupPath;
    private int _entriesSinceBackup;
    
    public BackupAjviIndex(string indexPath, int backupThreshold = 1000)
    {
        _indexPath = indexPath;
        _backupPath = indexPath + ".backup";
        _index = File.Exists(indexPath) 
            ? AjviIndex.Open(indexPath) 
            : AjviIndex.Create(indexPath, 384);
    }
    
    public void AddEntry(byte[] hash, Guid id, byte type, long ts, ReadOnlySpan<float> vec)
    {
        _index.AddEntry(hash, id, type, ts, vec);
        _entriesSinceBackup++;
        
        if (_entriesSinceBackup >= 1000)
        {
            CreateBackup();
            _entriesSinceBackup = 0;
        }
    }
    
    private void CreateBackup()
    {
        // Close index to flush all changes
        _index.Dispose();
        
        // Copy file
        File.Copy(_indexPath, _backupPath, overwrite: true);
        Console.WriteLine($"Backup created: {_backupPath}");
        
        // Reopen index
        _index = AjviIndex.Open(_indexPath);
    }
    
    public void Dispose() => _index?.Dispose();
}
```

### 4. Filtering Search Results by Metadata

```csharp
IReadOnlyList<(long Index, float Score)> SearchWithFilter(
    AjviIndex index,
    ReadOnlySpan<float> queryVector,
    Func<long, bool> filter,
    int topK = 20)
{
    var allResults = index.Search(queryVector, topK: index.EntryCount);
    
    return allResults
        .Where(r => filter(r.Index))
        .Take(topK)
        .ToList();
}

// Example: Search only messages from last 7 days
var sevenDaysAgo = DateTimeOffset.UtcNow.AddDays(-7).ToUnixTimeMilliseconds();
var recentResults = SearchWithFilter(
    index,
    queryVector,
    idx => index.GetTimestamp(idx) >= sevenDaysAgo,
    topK: 20
);

// Example: Search only Claude messages
var claudeResults = SearchWithFilter(
    index,
    queryVector,
    idx => index.GetAgentType(idx) == 1,
    topK: 20
);
```

### 5. Progressive Loading for Large Result Sets

```csharp
async IAsyncEnumerable<SearchResult> SearchStreamAsync(
    AjviIndex index,
    float[] queryVector,
    int batchSize = 10)
{
    var results = index.Search(queryVector, topK: 100);
    
    foreach (var batch in results.Chunk(batchSize))
    {
        foreach (var (idx, score) in batch)
        {
            var messageId = index.GetMessageId(idx);
            var message = await LoadMessageAsync(messageId);
            var session = await LoadSessionAsync(message.SessionId);
            
            yield return new SearchResult(session, score, new[] { message }, null);
        }
    }
}

// Usage
await foreach (var result in SearchStreamAsync(index, queryVector))
{
    Console.WriteLine($"Found: {result.Session.Id} (Score: {result.Score:F4})");
}
```

---

## Error Handling

### 1. Graceful Handling of Corrupt Indexes

```csharp
AjviIndex? TryOpenOrRecreate(string indexPath, int dimensions)
{
    try
    {
        return AjviIndex.Open(indexPath);
    }
    catch (InvalidDataException ex)
    {
        Console.WriteLine($"Index corrupted: {ex.Message}");
        Console.WriteLine("Creating new index...");
        
        // Backup corrupted file
        var backupPath = indexPath + $".corrupted.{DateTimeOffset.UtcNow:yyyyMMddHHmmss}";
        File.Move(indexPath, backupPath);
        
        // Create new index
        return AjviIndex.Create(indexPath, dimensions);
    }
    catch (FileNotFoundException)
    {
        Console.WriteLine("Index not found, creating new one...");
        return AjviIndex.Create(indexPath, dimensions);
    }
}
```

### 2. Handling Dimension Mismatches

```csharp
async Task<bool> TryAddEntryAsync(
    AjviIndex index,
    byte[] hash,
    Guid id,
    byte type,
    long timestamp,
    float[] vector)
{
    if (vector.Length != index.Dimensions)
    {
        Console.WriteLine(
            $"Dimension mismatch: expected {index.Dimensions}, got {vector.Length}");
        return false;
    }
    
    try
    {
        index.AddEntry(hash, id, type, timestamp, vector);
        return true;
    }
    catch (ArgumentException ex)
    {
        Console.WriteLine($"Failed to add entry: {ex.Message}");
        return false;
    }
}
```

### 3. Handling Read-Only Mode Violations

```csharp
bool TryAddEntry(AjviIndex index, /* ... parameters ... */)
{
    try
    {
        index.AddEntry(contentHash, messageId, agentType, timestamp, vector);
        return true;
    }
    catch (InvalidOperationException ex) when (ex.Message.Contains("read-only"))
    {
        Console.WriteLine("Cannot modify read-only index");
        return false;
    }
}
```

---

## Performance Optimization

### 1. Vector Normalization

```csharp
// Efficient normalization with SIMD
static float[] NormalizeVector(float[] vector)
{
    float sumSquares = 0;
    for (int i = 0; i < vector.Length; i++)
    {
        sumSquares += vector[i] * vector[i];
    }
    
    float magnitude = MathF.Sqrt(sumSquares);
    if (magnitude < 1e-10f) // Avoid division by zero
    {
        return vector; // Already zero vector
    }
    
    var normalized = new float[vector.Length];
    for (int i = 0; i < vector.Length; i++)
    {
        normalized[i] = vector[i] / magnitude;
    }
    
    return normalized;
}

// Or using System.Numerics.Tensors
static float[] NormalizeVectorSIMD(float[] vector)
{
    var sumSquares = TensorPrimitives.Dot(vector, vector);
    var magnitude = MathF.Sqrt(sumSquares);
    
    var normalized = new float[vector.Length];
    TensorPrimitives.Divide(vector, magnitude, normalized);
    
    return normalized;
}
```

### 2. Batch Processing

```csharp
async Task BatchIndexAsync(
    IEnumerable<Message> messages,
    IEmbeddingProvider embeddings,
    AjviIndex index,
    int batchSize = 32)
{
    foreach (var batch in messages.Chunk(batchSize))
    {
        // Generate embeddings in parallel
        var embeddings = await Task.WhenAll(
            batch.Select(m => embeddings.GetEmbeddingAsync(m.Content))
        );
        
        // Add to index sequentially (single writer)
        for (int i = 0; i < batch.Length; i++)
        {
            var message = batch[i];
            var embedding = NormalizeVector(embeddings[i]);
            var hash = SHA256.HashData(Encoding.UTF8.GetBytes(message.Content));
            
            if (!index.ContainsHash(hash))
            {
                index.AddEntry(
                    hash,
                    Guid.Parse(message.Id),
                    GetAgentTypeCode(message),
                    GetTimestamp(message),
                    embedding
                );
            }
        }
    }
}
```

### 3. Caching Frequently Accessed Vectors

```csharp
class CachedAjviIndex
{
    private readonly AjviIndex _index;
    private readonly LruCache<long, float[]> _vectorCache;
    
    public CachedAjviIndex(AjviIndex index, int cacheSize = 1000)
    {
        _index = index;
        _vectorCache = new LruCache<long, float[]>(cacheSize);
    }
    
    public ReadOnlySpan<float> GetVector(long index)
    {
        if (_vectorCache.TryGet(index, out var cached))
        {
            return cached;
        }
        
        var vector = _index.GetVector(index).ToArray();
        _vectorCache.Add(index, vector);
        return vector;
    }
}
```

---

## Common Pitfalls

### ❌ Pitfall 1: Not Normalizing Vectors

```csharp
// WRONG - vectors not normalized
var embedding = await embeddingProvider.GetEmbeddingAsync(content);
index.AddEntry(hash, id, type, timestamp, embedding); // ❌

// CORRECT - always normalize
var embedding = await embeddingProvider.GetEmbeddingAsync(content);
embedding = NormalizeVector(embedding); // ✅
index.AddEntry(hash, id, type, timestamp, embedding);
```

### ❌ Pitfall 2: Wrong Hash Size

```csharp
// WRONG - MD5 is only 16 bytes
var hash = MD5.HashData(Encoding.UTF8.GetBytes(content)); // ❌
index.AddEntry(hash, id, type, timestamp, vector); // Throws ArgumentException

// CORRECT - use SHA256 (32 bytes)
var hash = SHA256.HashData(Encoding.UTF8.GetBytes(content)); // ✅
index.AddEntry(hash, id, type, timestamp, vector);
```

### ❌ Pitfall 3: Forgetting to Dispose

```csharp
// WRONG - leaks file handles
var index = AjviIndex.Open("embeddings.ajvi"); // ❌
// ... use index ...
// File handle not released!

// CORRECT - use using statement
using (var index = AjviIndex.Open("embeddings.ajvi")) // ✅
{
    // ... use index ...
} // Automatically disposed
```

### ❌ Pitfall 4: Mixing Precision Types

```csharp
// WRONG - creating with one precision, expecting another
var indexF16 = AjviIndex.Create("test.ajvi", 384, VectorPrecision.Float16);
// Later...
var index = AjviIndex.Open("test.ajvi");
if (index.Precision != VectorPrecision.Float32) // ❌ Wrong expectation
{
    // This will be true, but code assumes Float32
}

// CORRECT - check precision after opening
var index = AjviIndex.Open("test.ajvi");
Console.WriteLine($"Index uses {index.Precision} precision"); // ✅
```

### ❌ Pitfall 5: Concurrent Writers

```csharp
// WRONG - multiple writers
var index1 = AjviIndex.Open("test.ajvi"); // ❌
var index2 = AjviIndex.Open("test.ajvi"); // ❌
index1.AddEntry(...); // Corruption risk!
index2.AddEntry(...); // Corruption risk!

// CORRECT - single writer or multiple read-only
var writer = AjviIndex.Open("test.ajvi"); // ✅
var reader1 = AjviIndex.Open("test.ajvi", readOnly: true); // ✅
var reader2 = AjviIndex.Open("test.ajvi", readOnly: true); // ✅
```

---

## Helper Functions

### Normalize Vector

```csharp
static float[] NormalizeVector(float[] vector)
{
    float sumSquares = vector.Sum(v => v * v);
    float magnitude = MathF.Sqrt(sumSquares);
    return vector.Select(v => v / magnitude).ToArray();
}
```

### Get Agent Type Code

```csharp
static byte GetAgentTypeCode(string agentType) => agentType.ToLowerInvariant() switch
{
    "copilot" => 0,
    "claude" => 1,
    _ => 255
};
```

### Get Unix Timestamp

```csharp
static long GetUnixTimestamp(DateTime dateTime)
{
    return new DateTimeOffset(dateTime).ToUnixTimeMilliseconds();
}
```

### Check Index Exists

```csharp
static bool IndexExists(string indexPath)
{
    return File.Exists(indexPath);
}
```

### Get Index File Size

```csharp
static long GetIndexSize(string indexPath)
{
    return new FileInfo(indexPath).Length;
}
```

---

## See Also

- **AjviIndex.cs** - Full implementation
- **AJVI_SPECIFICATION.md** - Binary format specification  
- **README_AJVI.md** - Comprehensive usage guide
- **AJVI_QUICK_REFERENCE.md** - Quick API reference
