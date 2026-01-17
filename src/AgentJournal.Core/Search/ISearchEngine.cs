using AgentJournal.Core.Models;

namespace AgentJournal.Core.Search;

/// <summary>
/// Search mode for querying sessions
/// </summary>
public enum SearchMode
{
    /// <summary>
    /// Lexical/keyword-based search
    /// </summary>
    Lexical,

    /// <summary>
    /// Semantic/vector-based search
    /// </summary>
    Semantic,

    /// <summary>
    /// Hybrid search combining lexical and semantic
    /// </summary>
    Hybrid
}

/// <summary>
/// Search result containing a session and relevance score
/// </summary>
public record SearchResult(
    Session Session,
    double Score,
    IReadOnlyList<Message>? MatchingMessages = null,
    string? Highlight = null
)
{
    /// <summary>
    /// Gets whether there are matching messages
    /// </summary>
    public bool HasMatchingMessages => MatchingMessages != null && MatchingMessages.Count > 0;
}

/// <summary>
/// Interface for searching agent sessions
/// </summary>
public interface ISearchEngine
{
    /// <summary>
    /// Gets the search modes supported by this engine
    /// </summary>
    IReadOnlyList<SearchMode> SupportedModes { get; }

    /// <summary>
    /// Indexes a session for searching
    /// </summary>
    /// <param name="session">The session to index</param>
    /// <param name="ct">Cancellation token</param>
    Task IndexSessionAsync(Session session, CancellationToken ct = default);

    /// <summary>
    /// Indexes multiple sessions for searching
    /// </summary>
    /// <param name="sessions">The sessions to index</param>
    /// <param name="ct">Cancellation token</param>
    Task IndexSessionsAsync(IEnumerable<Session> sessions, CancellationToken ct = default);

    /// <summary>
    /// Searches for sessions matching the query
    /// </summary>
    /// <param name="query">The search query</param>
    /// <param name="mode">The search mode to use</param>
    /// <param name="maxResults">Maximum number of results to return</param>
    /// <param name="contextCount">Number of messages before/after matches to include (0 to disable)</param>
    /// <param name="ct">Cancellation token</param>
    /// <returns>Search results ordered by relevance</returns>
    Task<IReadOnlyList<SearchResult>> SearchAsync(
        string query,
        SearchMode mode = SearchMode.Lexical,
        int maxResults = 10,
        int contextCount = 0,
        CancellationToken ct = default);

    /// <summary>
    /// Initializes the search engine (creates indexes, etc.)
    /// </summary>
    /// <param name="ct">Cancellation token</param>
    Task InitializeAsync(CancellationToken ct = default);

    /// <summary>
    /// Clears all indexed data
    /// </summary>
    /// <param name="ct">Cancellation token</param>
    Task ClearIndexAsync(CancellationToken ct = default);
}
