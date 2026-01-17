using AgentJournal.Core.Models;

namespace AgentJournal.Core.Storage;

/// <summary>
/// Interface for persisting and retrieving agent sessions
/// </summary>
public interface ISessionRepository
{
    /// <summary>
    /// Saves a session to the repository
    /// </summary>
    /// <param name="session">The session to save</param>
    /// <param name="ct">Cancellation token</param>
    Task SaveSessionAsync(Session session, CancellationToken ct = default);

    /// <summary>
    /// Saves multiple sessions to the repository
    /// </summary>
    /// <param name="sessions">The sessions to save</param>
    /// <param name="ct">Cancellation token</param>
    Task SaveSessionsAsync(IEnumerable<Session> sessions, CancellationToken ct = default);

    /// <summary>
    /// Gets a session by its ID
    /// </summary>
    /// <param name="sessionId">The session ID</param>
    /// <param name="ct">Cancellation token</param>
    /// <returns>The session or null if not found</returns>
    Task<Session?> GetSessionAsync(string sessionId, CancellationToken ct = default);

    /// <summary>
    /// Gets all sessions
    /// </summary>
    /// <param name="ct">Cancellation token</param>
    /// <returns>All sessions in the repository</returns>
    IAsyncEnumerable<Session> GetAllSessionsAsync(CancellationToken ct = default);

    /// <summary>
    /// Gets sessions by agent type
    /// </summary>
    /// <param name="agentType">The agent type to filter by</param>
    /// <param name="ct">Cancellation token</param>
    /// <returns>Sessions matching the agent type</returns>
    IAsyncEnumerable<Session> GetSessionsByAgentTypeAsync(string agentType, CancellationToken ct = default);

    /// <summary>
    /// Deletes a session by its ID
    /// </summary>
    /// <param name="sessionId">The session ID to delete</param>
    /// <param name="ct">Cancellation token</param>
    Task DeleteSessionAsync(string sessionId, CancellationToken ct = default);

    /// <summary>
    /// Gets the last modified timestamp for a session
    /// </summary>
    /// <param name="sessionId">The session ID</param>
    /// <param name="ct">Cancellation token</param>
    /// <returns>The last modified timestamp or null if not found</returns>
    Task<DateTime?> GetSessionLastModifiedAsync(string sessionId, CancellationToken ct = default);

    /// <summary>
    /// Initializes the repository (creates tables, etc.)
    /// </summary>
    /// <param name="ct">Cancellation token</param>
    Task InitializeAsync(CancellationToken ct = default);
}
