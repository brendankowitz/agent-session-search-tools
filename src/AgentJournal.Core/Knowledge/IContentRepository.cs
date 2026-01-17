using AgentJournal.Core.Models;

namespace AgentJournal.Core.Knowledge;

/// <summary>
/// Interface for persisting and retrieving content entries
/// </summary>
public interface IContentRepository
{
    /// <summary>
    /// Initializes the repository (creates tables, etc.)
    /// </summary>
    /// <param name="ct">Cancellation token</param>
    Task InitializeAsync(CancellationToken ct = default);

    /// <summary>
    /// Adds or updates a content entry
    /// </summary>
    /// <param name="entry">The content entry to save</param>
    /// <param name="ct">Cancellation token</param>
    /// <returns>The saved entry</returns>
    Task<ContentEntry> AddAsync(ContentEntry entry, CancellationToken ct = default);

    /// <summary>
    /// Updates an existing content entry
    /// </summary>
    /// <param name="entry">The content entry to update</param>
    /// <param name="ct">Cancellation token</param>
    /// <returns>True if updated, false if not found</returns>
    Task<bool> UpdateAsync(ContentEntry entry, CancellationToken ct = default);

    /// <summary>
    /// Gets a content entry by its ID
    /// </summary>
    /// <param name="id">The entry ID</param>
    /// <param name="ct">Cancellation token</param>
    /// <returns>The entry or null if not found</returns>
    Task<ContentEntry?> GetByIdAsync(string id, CancellationToken ct = default);

    /// <summary>
    /// Gets a content entry by its source
    /// </summary>
    /// <param name="source">The source identifier</param>
    /// <param name="ct">Cancellation token</param>
    /// <returns>The entry or null if not found</returns>
    Task<ContentEntry?> GetBySourceAsync(string source, CancellationToken ct = default);

    /// <summary>
    /// Searches content entries using FTS5
    /// </summary>
    /// <param name="query">Search query text</param>
    /// <param name="project">Filter by project</param>
    /// <param name="sourcePrefix">Filter by source starting with prefix</param>
    /// <param name="tags">Filter by any matching tag</param>
    /// <param name="maxResults">Maximum number of results</param>
    /// <param name="ct">Cancellation token</param>
    /// <returns>Search results with decay applied</returns>
    Task<IReadOnlyList<ContentSearchResult>> SearchAsync(
        string query,
        string? project = null,
        string? sourcePrefix = null,
        string[]? tags = null,
        int maxResults = 10,
        CancellationToken ct = default);

    /// <summary>
    /// Lists content entries with optional filtering
    /// </summary>
    /// <param name="project">Filter by project</param>
    /// <param name="sourcePrefix">Filter by source starting with prefix</param>
    /// <param name="tags">Filter by any matching tag</param>
    /// <param name="limit">Maximum number of entries to return</param>
    /// <param name="ct">Cancellation token</param>
    /// <returns>List of content entries</returns>
    Task<IReadOnlyList<ContentEntry>> ListAsync(
        string? project = null,
        string? sourcePrefix = null,
        string[]? tags = null,
        int limit = 100,
        CancellationToken ct = default);

    /// <summary>
    /// Deletes a content entry by source
    /// </summary>
    /// <param name="source">The source identifier</param>
    /// <param name="ct">Cancellation token</param>
    /// <returns>True if deleted, false if not found</returns>
    Task<bool> DeleteAsync(string source, CancellationToken ct = default);

    /// <summary>
    /// Deletes content entries matching the specified criteria
    /// </summary>
    /// <param name="id">Filter by content ID</param>
    /// <param name="source">Filter by exact source match</param>
    /// <param name="sourcePrefix">Filter by source starting with prefix</param>
    /// <param name="project">Filter by project</param>
    /// <param name="deleteAll">Delete all content if true</param>
    /// <param name="ct">Cancellation token</param>
    /// <returns>Number of entries deleted</returns>
    Task<int> DeleteByCriteriaAsync(
        string? id = null,
        string? source = null,
        string? sourcePrefix = null,
        string? project = null,
        bool deleteAll = false,
        CancellationToken ct = default);

    /// <summary>
    /// Counts content entries matching the specified criteria
    /// </summary>
    /// <param name="id">Filter by content ID</param>
    /// <param name="source">Filter by exact source match</param>
    /// <param name="sourcePrefix">Filter by source starting with prefix</param>
    /// <param name="project">Filter by project</param>
    /// <param name="countAll">Count all content if true</param>
    /// <param name="ct">Cancellation token</param>
    /// <returns>Number of matching entries</returns>
    Task<int> CountByCriteriaAsync(
        string? id = null,
        string? source = null,
        string? sourcePrefix = null,
        string? project = null,
        bool countAll = false,
        CancellationToken ct = default);

    /// <summary>
    /// Reinforces a content entry (resets decay timer)
    /// </summary>
    /// <param name="source">The source identifier</param>
    /// <param name="ct">Cancellation token</param>
    /// <returns>True if reinforced, false if not found</returns>
    Task<bool> ReinforceAsync(string source, CancellationToken ct = default);

    /// <summary>
    /// Gets expired content entries below the threshold
    /// </summary>
    /// <param name="threshold">Decay factor threshold (default: 0.05)</param>
    /// <param name="ct">Cancellation token</param>
    /// <returns>List of expired content entries</returns>
    Task<IReadOnlyList<ContentEntry>> GetExpiredAsync(double threshold = 0.05, CancellationToken ct = default);
}

/// <summary>
/// Search result for content entries with decay-adjusted score
/// </summary>
public record ContentSearchResult(
    ContentEntry Entry,
    double Score,
    double DecayFactor,
    string? Highlight
);
