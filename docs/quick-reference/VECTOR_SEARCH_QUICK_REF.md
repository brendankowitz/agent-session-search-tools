# Vector Search Quick Reference

## Quick Start

### 1. Create Search Engines

```csharp
using AgentJournal.Core.Embeddings;
using AgentJournal.Core.Search;

// Create embedding provider (choose one)
var embedder = new HashEmbeddingProvider();  // Fast, not semantic
// var embedder = new OnnxEmbeddingProvider("model.onnx");  // Semantic

// Create search engines
var lexicalEngine = new LuceneSearchEngine("./lucene-index");
var vectorEngine = new VectorSearchEngine("./vector-index", embedder);
var hybridSearcher = new HybridSearcher(lexicalEngine, vectorEngine);

// Initialize
await hybridSearcher.InitializeAsync();
```

### 2. Index Sessions

```csharp
// Single session
await hybridSearcher.IndexSessionAsync(session);

// Multiple sessions (bulk)
await hybridSearcher.IndexSessionsAsync(sessions);
```

### 3. Search

```csharp
// Lexical search (BM25, keyword-based)
var results = await hybridSearcher.SearchAsync(
    query: "error handling",
    mode: SearchMode.Lexical,
    maxResults: 10
);

// Semantic search (vector-based, meaning-aware)
var results = await hybridSearcher.SearchAsync(
    query: "how to handle exceptions",
    mode: SearchMode.Semantic,
    maxResults: 10
);

// Hybrid search (best of both worlds)
var results = await hybridSearcher.SearchAsync(
    query: "async await patterns",
    mode: SearchMode.Hybrid,
    maxResults: 10
);
```

### 4. Use Results

```csharp
foreach (var result in results)
{
    Console.WriteLine($"Session: {result.Session.Id}");
    Console.WriteLine($"Score: {result.Score:F4}");
    Console.WriteLine($"Agent: {result.Session.AgentType}");
    Console.WriteLine($"Messages: {result.MatchingMessages?.Count ?? 0}");
    
    if (result.Highlight != null)
    {
        Console.WriteLine($"Preview: {result.Highlight}");
    }
    
    Console.WriteLine();
}
```

### 5. Cleanup

```csharp
// Clear index
await hybridSearcher.ClearIndexAsync();

// Dispose resources
hybridSearcher.Dispose();
```

## Search Modes

| Mode | Engine | Best For | Speed |
|------|--------|----------|-------|
| **Lexical** | Lucene BM25 | Exact keywords, technical terms | ⚡⚡⚡ Fast |
| **Semantic** | Vector similarity | Concepts, paraphrases, related terms | ⚡⚡ Medium |
| **Hybrid** | Both + RRF | Best overall results | ⚡ Slower |

## Tuning Hybrid Search

```csharp
var hybridSearcher = new HybridSearcher(
    lexicalEngine,
    vectorEngine,
    lexicalWeight: 0.7f,   // Higher = favor exact matches
    semanticWeight: 0.3f,  // Higher = favor semantic similarity
    rrfK: 60               // RRF constant (typical: 60)
);
```

## Performance Tips

1. **Bulk Indexing**: Use `IndexSessionsAsync()` for large datasets
2. **Result Limits**: Request only what you need (default: 10)
3. **Caching**: Sessions are cached in memory after indexing
4. **Initialization**: Call `InitializeAsync()` once at startup

## Common Patterns

### Search with Filters

```csharp
var results = await hybridSearcher.SearchAsync("query", SearchMode.Hybrid, 50);

// Filter by agent type
var copilotResults = results.Where(r => r.Session.AgentType == "copilot-cli");

// Filter by date
var recentResults = results.Where(r => 
    r.Session.StartedAt > DateTime.Now.AddDays(-7)
);

// Filter by project
var projectResults = results.Where(r => 
    r.Session.ProjectPath?.Contains("my-project") == true
);
```

### Pagination

```csharp
int pageSize = 10;
int pageNumber = 0;

var allResults = await hybridSearcher.SearchAsync("query", SearchMode.Hybrid, 100);
var page = allResults.Skip(pageNumber * pageSize).Take(pageSize);
```

### Error Handling

```csharp
try
{
    var results = await hybridSearcher.SearchAsync(query, SearchMode.Hybrid);
}
catch (NotSupportedException ex)
{
    // Unsupported search mode
}
catch (InvalidOperationException ex)
{
    // Not initialized
}
catch (ObjectDisposedException ex)
{
    // Already disposed
}
```

## File Structure

```
index-directory/
├── lucene-index/           # Lexical search index
│   ├── segments_*
│   └── _*.cfs
└── vector-index/           # Semantic search index
    └── index.ajvi          # AJVI vector index (~800 bytes/message)
```

## Architecture

```
┌─────────────────────────────────────────┐
│         HybridSearcher                  │
│  (Supports all 3 modes)                 │
└─────────────┬───────────────────────────┘
              │
       ┌──────┴──────┐
       │             │
       ▼             ▼
┌─────────────┐ ┌──────────────┐
│  Lucene     │ │   Vector     │
│  Search     │ │   Search     │
│  Engine     │ │   Engine     │
│             │ │              │
│  (BM25)     │ │  (Cosine)    │
└─────────────┘ └──────┬───────┘
                       │
                       ▼
                ┌──────────────┐
                │  Embedding   │
                │  Provider    │
                │              │
                │ (Hash/ONNX)  │
                └──────────────┘
```

## Troubleshooting

### Index Files Locked
- Ensure only one writer per index
- Call `Dispose()` when done
- Check for zombie processes

### Poor Semantic Results
- Try a better embedding model (ONNX vs Hash)
- Increase result count for better coverage
- Adjust hybrid weights

### Slow Searches
- Reduce maxResults parameter
- Use Lexical mode for simple queries
- Index fewer sessions initially
- Enable async/await properly

## API Reference

### ISearchEngine Interface

```csharp
public interface ISearchEngine
{
    IReadOnlyList<SearchMode> SupportedModes { get; }
    
    Task InitializeAsync(CancellationToken ct = default);
    Task IndexSessionAsync(Session session, CancellationToken ct = default);
    Task IndexSessionsAsync(IEnumerable<Session> sessions, CancellationToken ct = default);
    Task<IReadOnlyList<SearchResult>> SearchAsync(string query, SearchMode mode, int maxResults, CancellationToken ct = default);
    Task ClearIndexAsync(CancellationToken ct = default);
}
```

### SearchResult Model

```csharp
public record SearchResult(
    Session Session,                           // Matched session
    double Score,                              // Relevance score (higher = better)
    IReadOnlyList<Message>? MatchingMessages,  // Messages that matched
    string? Highlight                          // Preview snippet
);
```

## Best Practices

1. ✅ **Always initialize** before use
2. ✅ **Dispose** when done (implements IDisposable)
3. ✅ **Use bulk indexing** for multiple sessions
4. ✅ **Start with Hybrid mode** for best results
5. ✅ **Handle cancellation** with CancellationToken
6. ✅ **Tune weights** based on your use case
7. ✅ **Cache frequently used** search results
8. ✅ **Monitor index size** and clean periodically

## Next Steps

- Read [VECTOR_SEARCH_IMPLEMENTATION.md](VECTOR_SEARCH_IMPLEMENTATION.md) for full details
- Check [AJVI_SPECIFICATION.md](src/AgentJournal.Core/Search/AJVI_SPECIFICATION.md) for index format
- See test files for more examples
- Review [ARCHITECTURE_DIAGRAM.md](ARCHITECTURE_DIAGRAM.md) for system design
