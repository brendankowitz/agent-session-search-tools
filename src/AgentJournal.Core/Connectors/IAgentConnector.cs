using AgentJournal.Core.Models;

namespace AgentJournal.Core.Connectors;

/// <summary>
/// Interface for connecting to and parsing AI agent session data
/// </summary>
public interface IAgentConnector
{
    /// <summary>
    /// Gets the type of agent this connector supports (e.g., "claude-code", "copilot-cli")
    /// </summary>
    string AgentType { get; }

    /// <summary>
    /// Gets all available session paths for this agent type
    /// </summary>
    /// <returns>Collection of session directory or file paths</returns>
    IEnumerable<string> GetSessionPaths();

    /// <summary>
    /// Parses all sessions asynchronously
    /// </summary>
    /// <param name="ct">Cancellation token</param>
    /// <returns>Async enumerable of parsed sessions</returns>
    IAsyncEnumerable<Session> ParseSessionsAsync(CancellationToken ct = default);

    /// <summary>
    /// Parses a single session from the specified path
    /// </summary>
    /// <param name="sessionPath">Path to the session data</param>
    /// <param name="ct">Cancellation token</param>
    /// <returns>Parsed session or null if parsing failed</returns>
    Task<Session?> ParseSessionAsync(string sessionPath, CancellationToken ct = default);
}
