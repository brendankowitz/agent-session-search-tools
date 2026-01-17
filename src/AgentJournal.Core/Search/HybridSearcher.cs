using AgentJournal.Core.Models;

namespace AgentJournal.Core.Search;

/// <summary>
/// Combines lexical (BM25) and semantic (vector) search using Reciprocal Rank Fusion
/// </summary>
public class HybridSearcher : ISearchEngine, IDisposable
{
    private readonly LuceneSearchEngine _lexicalEngine;
    private readonly VectorSearchEngine _vectorEngine;
    private readonly float _lexicalWeight;
    private readonly float _semanticWeight;
    private readonly int _rrfK;  // RRF constant, typically 60
    private bool _disposed;
    
    public IReadOnlyList<SearchMode> SupportedModes { get; } = 
        [SearchMode.Lexical, SearchMode.Semantic, SearchMode.Hybrid];
    
    /// <summary>
    /// Creates a new hybrid searcher combining lexical and semantic search
    /// </summary>
    /// <param name="lexicalEngine">Lexical search engine (BM25)</param>
    /// <param name="vectorEngine">Semantic search engine (vector-based)</param>
    /// <param name="lexicalWeight">Weight for lexical search in fusion (default 0.5)</param>
    /// <param name="semanticWeight">Weight for semantic search in fusion (default 0.5)</param>
    /// <param name="rrfK">RRF constant (default 60)</param>
    public HybridSearcher(
        LuceneSearchEngine lexicalEngine,
        VectorSearchEngine vectorEngine,
        float lexicalWeight = 0.5f,
        float semanticWeight = 0.5f,
        int rrfK = 60)
    {
        _lexicalEngine = lexicalEngine ?? throw new ArgumentNullException(nameof(lexicalEngine));
        _vectorEngine = vectorEngine ?? throw new ArgumentNullException(nameof(vectorEngine));
        _lexicalWeight = lexicalWeight;
        _semanticWeight = semanticWeight;
        _rrfK = rrfK;
    }

    /// <summary>
    /// Initializes both lexical and semantic search engines
    /// </summary>
    public async Task InitializeAsync(CancellationToken ct = default)
    {
        await Task.WhenAll(
            _lexicalEngine.InitializeAsync(ct),
            _vectorEngine.InitializeAsync(ct)
        );
    }

    /// <summary>
    /// Indexes a session in both lexical and semantic engines
    /// </summary>
    public async Task IndexSessionAsync(Session session, CancellationToken ct = default)
    {
        await Task.WhenAll(
            _lexicalEngine.IndexSessionAsync(session, ct),
            _vectorEngine.IndexSessionAsync(session, ct)
        );
    }

    /// <summary>
    /// Indexes multiple sessions in both engines
    /// </summary>
    public async Task IndexSessionsAsync(IEnumerable<Session> sessions, CancellationToken ct = default)
    {
        var sessionList = sessions.ToList(); // Materialize to avoid multiple enumeration
        
        await Task.WhenAll(
            _lexicalEngine.IndexSessionsAsync(sessionList, ct),
            _vectorEngine.IndexSessionsAsync(sessionList, ct)
        );
    }

    /// <summary>
    /// Searches using the specified mode: lexical, semantic, or hybrid (RRF fusion)
    /// </summary>
    public async Task<IReadOnlyList<SearchResult>> SearchAsync(
        string query,
        SearchMode mode = SearchMode.Hybrid,
        int maxResults = 10,
        int contextCount = 0,
        CancellationToken ct = default)
    {
        // Validate input
        ArgumentNullException.ThrowIfNull(query);
        if (maxResults <= 0 || maxResults > 1000)
        {
            throw new ArgumentOutOfRangeException(nameof(maxResults), 
                "maxResults must be between 1 and 1000.");
        }
        
        if (string.IsNullOrWhiteSpace(query))
        {
            return Array.Empty<SearchResult>();
        }

        return mode switch
        {
            SearchMode.Lexical => await _lexicalEngine.SearchAsync(query, SearchMode.Lexical, maxResults, contextCount, ct),
            SearchMode.Semantic => await _vectorEngine.SearchAsync(query, SearchMode.Semantic, maxResults, contextCount, ct),
            SearchMode.Hybrid => await HybridSearchAsync(query, maxResults, contextCount, ct),
            _ => throw new NotSupportedException($"Search mode {mode} is not supported")
        };
    }

    /// <summary>
    /// Clears indexes in both engines
    /// </summary>
    public async Task ClearIndexAsync(CancellationToken ct = default)
    {
        await Task.WhenAll(
            _lexicalEngine.ClearIndexAsync(ct),
            _vectorEngine.ClearIndexAsync(ct)
        );
    }

    /// <summary>
    /// Performs hybrid search using Reciprocal Rank Fusion (RRF)
    /// </summary>
    private async Task<IReadOnlyList<SearchResult>> HybridSearchAsync(
        string query, 
        int maxResults,
        int contextCount,
        CancellationToken ct)
    {
        // 1. Fetch 3x results from each engine for better fusion
        var fetchCount = maxResults * 3;
        
        // 2. Run both searches in parallel
        var lexicalTask = _lexicalEngine.SearchAsync(query, SearchMode.Lexical, fetchCount, contextCount, ct);
        var semanticTask = _vectorEngine.SearchAsync(query, SearchMode.Semantic, fetchCount, contextCount, ct);
        
        await Task.WhenAll(lexicalTask, semanticTask);
        
        var lexicalResults = await lexicalTask;
        var semanticResults = await semanticTask;

        // 3. Apply RRF scoring
        var fusedScores = new Dictionary<string, (double Score, SearchResult Result)>();
        
        // Score lexical results (1-based ranking)
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
        
        // Score semantic results (1-based ranking)
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
        
        // 4. Sort by fused score and return top results
        return fusedScores.Values
            .OrderByDescending(x => x.Score)
            .Take(maxResults)
            .Select(x => x.Result with { Score = x.Score })
            .ToList();
    }

    /// <summary>
    /// Disposes both search engines
    /// </summary>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _lexicalEngine?.Dispose();
        _vectorEngine?.Dispose();
        _disposed = true;
        GC.SuppressFinalize(this);
    }
}
