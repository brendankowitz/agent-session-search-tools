namespace AgentJournal.Core.Models;

/// <summary>
/// Represents an AI agent conversation session
/// </summary>
public record Session(
    string Id,
    string AgentType,
    string? ProjectPath,
    string? GitBranch,
    string? AgentVersion,
    DateTime StartedAt,
    DateTime? EndedAt,
    DateTime? LastModified,
    string? Summary,
    IReadOnlyList<Message> Messages
)
{
    /// <summary>
    /// Gets the duration of the session if it has ended
    /// </summary>
    public TimeSpan? Duration => EndedAt.HasValue ? EndedAt.Value - StartedAt : null;

    /// <summary>
    /// Gets whether the session is currently active
    /// </summary>
    public bool IsActive => !EndedAt.HasValue;

    /// <summary>
    /// Gets the total number of messages in the session
    /// </summary>
    public int MessageCount => Messages.Count;

    /// <summary>
    /// Gets the number of user messages in the session
    /// </summary>
    public int UserMessageCount => Messages.Count(m => m.Role == MessageRole.User);

    /// <summary>
    /// Gets the number of assistant messages in the session
    /// </summary>
    public int AssistantMessageCount => Messages.Count(m => m.Role == MessageRole.Assistant);

    /// <summary>
    /// Gets all tool calls across all messages
    /// </summary>
    public IEnumerable<ToolCall> AllToolCalls => Messages
        .Where(m => m.ToolCalls != null)
        .SelectMany(m => m.ToolCalls!);

    /// <summary>
    /// Gets the total number of tool calls in the session
    /// </summary>
    public int ToolCallCount => AllToolCalls.Count();
}
