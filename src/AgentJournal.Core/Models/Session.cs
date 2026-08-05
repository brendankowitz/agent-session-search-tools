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

    /// <summary>
    /// Returns a copy of this session containing only its last <paramref name="count"/> messages.
    /// </summary>
    /// <param name="count">
    /// Number of trailing messages to keep. Values of zero or less, or values at or above
    /// <see cref="MessageCount"/>, return the session unchanged.
    /// </param>
    /// <remarks>
    /// Session metadata (including <see cref="StartedAt"/>) is deliberately preserved so the
    /// truncated view still identifies the full session it came from. Only <see cref="Messages"/>
    /// is narrowed, which means derived counts such as <see cref="MessageCount"/> describe the
    /// returned slice rather than the original session.
    /// </remarks>
    public Session WithLastMessages(int count)
    {
        if (count <= 0 || count >= Messages.Count)
        {
            return this;
        }

        return this with { Messages = Messages.Skip(Messages.Count - count).ToList() };
    }
}
