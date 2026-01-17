using AgentJournal.Core.Models;
using AgentJournal.Core.Search;

namespace AgentJournal.Core.Knowledge;

/// <summary>
/// Interface for persisting and retrieving knowledge entries
/// </summary>
public interface IKnowledgeRepository
{
    /// <summary>
    /// Initializes the repository (creates tables, etc.)
    /// </summary>
    /// <param name="ct">Cancellation token</param>
    Task InitializeAsync(CancellationToken ct = default);

    /// <summary>
    /// Saves a knowledge entry to the repository
    /// </summary>
    /// <param name="entry">The knowledge entry to save</param>
    /// <param name="ct">Cancellation token</param>
    /// <returns>The saved entry</returns>
    Task<KnowledgeEntry> SaveAsync(KnowledgeEntry entry, CancellationToken ct = default);

    /// <summary>
    /// Gets a knowledge entry by its ID
    /// </summary>
    /// <param name="id">The entry ID</param>
    /// <param name="ct">Cancellation token</param>
    /// <returns>The entry or null if not found</returns>
    Task<KnowledgeEntry?> GetAsync(string id, CancellationToken ct = default);

    /// <summary>
    /// Searches knowledge entries
    /// </summary>
    /// <param name="query">Search query text</param>
    /// <param name="tags">Filter by tags</param>
    /// <param name="project">Filter by project</param>
    /// <param name="mode">Search mode (keyword or semantic)</param>
    /// <param name="maxResults">Maximum number of results</param>
    /// <param name="ct">Cancellation token</param>
    /// <returns>Search results with decay applied to scores</returns>
    Task<IReadOnlyList<KnowledgeSearchResult>> SearchAsync(
        string query,
        IEnumerable<string>? tags = null,
        string? project = null,
        SearchMode mode = SearchMode.Hybrid,
        int maxResults = 10,
        CancellationToken ct = default);

    /// <summary>
    /// Deletes a knowledge entry
    /// </summary>
    /// <param name="id">The entry ID to delete</param>
    /// <param name="ct">Cancellation token</param>
    /// <returns>True if deleted, false if not found</returns>
    Task<bool> DeleteAsync(string id, CancellationToken ct = default);

    /// <summary>
    /// Deletes multiple knowledge entries in a single transaction
    /// </summary>
    /// <param name="ids">The entry IDs to delete</param>
    /// <param name="ct">Cancellation token</param>
    /// <returns>Number of entries deleted</returns>
    Task<int> DeleteManyAsync(IEnumerable<string> ids, CancellationToken ct = default);

    /// <summary>
    /// Reinforces a knowledge entry (resets decay timer)
    /// </summary>
    /// <param name="id">The entry ID to reinforce</param>
    /// <param name="ct">Cancellation token</param>
    /// <returns>True if reinforced, false if not found</returns>
    Task<bool> ReinforceAsync(string id, CancellationToken ct = default);

    /// <summary>
    /// Reinforces multiple knowledge entries in a single transaction
    /// </summary>
    /// <param name="ids">The entry IDs to reinforce</param>
    /// <param name="ct">Cancellation token</param>
    /// <returns>Number of entries reinforced</returns>
    Task<int> ReinforceManyAsync(IEnumerable<string> ids, CancellationToken ct = default);

    /// <summary>
    /// Lists knowledge entries with optional filtering
    /// </summary>
    /// <param name="project">Filter by project</param>
    /// <param name="tags">Filter by tags</param>
    /// <param name="includeDecaying">Include entries with decay factor below 0.5</param>
    /// <param name="limit">Maximum number of entries to return</param>
    /// <param name="ct">Cancellation token</param>
    /// <returns>List of knowledge entries</returns>
    Task<IReadOnlyList<KnowledgeEntry>> ListAsync(
        string? project = null,
        IEnumerable<string>? tags = null,
        bool includeDecaying = true,
        int limit = 100,
        CancellationToken ct = default);

    /// <summary>
    /// Gets statistics about the knowledge bank
    /// </summary>
    /// <param name="ct">Cancellation token</param>
    /// <returns>Statistics including count, decay distribution, etc.</returns>
    Task<KnowledgeStats> GetStatsAsync(CancellationToken ct = default);

    /// <summary>
    /// Prunes expired knowledge entries below the threshold
    /// </summary>
    /// <param name="threshold">Decay factor threshold (default: 0.05)</param>
    /// <param name="ct">Cancellation token</param>
    /// <returns>Number of entries pruned</returns>
    Task<int> PruneExpiredAsync(double threshold = 0.05, CancellationToken ct = default);
}

/// <summary>
/// Search result for knowledge entries with decay-adjusted score
/// </summary>
public record KnowledgeSearchResult(
    KnowledgeEntry Entry,
    double Score,
    double DecayFactor,
    string? Highlight
);

/// <summary>
/// Statistics about the knowledge bank
/// </summary>
public record KnowledgeStats(
    int TotalEntries,
    int FreshEntries,       // decay > 0.75
    int GoodEntries,        // decay > 0.50
    int AgingEntries,       // decay > 0.25
    int DecayingEntries,    // decay > 0.10
    int ExpiringEntries,    // decay <= 0.10
    Dictionary<string, int> EntriesByProject,
    Dictionary<string, int> EntriesByTag
);
