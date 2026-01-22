namespace AgentJournal.Core.Models;

/// <summary>
/// Represents a message in an AI agent conversation
/// </summary>
public record Message(
    string Id,
    string SessionId,
    MessageRole Role,
    string Content,
    string? RawContent,
    DateTime Timestamp,
    string? ParentId,
    string? Model,
    IReadOnlyList<ToolCall>? ToolCalls
)
{
    /// <summary>
    /// Gets whether this message has any tool calls
    /// </summary>
    public bool HasToolCalls => ToolCalls != null && ToolCalls.Count > 0;

    /// <summary>
    /// Gets the number of tool calls in this message
    /// </summary>
    public int ToolCallCount => ToolCalls?.Count ?? 0;

    /// <summary>
    /// Gets whether this message is a response to another message
    /// </summary>
    public bool IsResponse => !string.IsNullOrEmpty(ParentId);

    /// <summary>
    /// Gets the content length in characters
    /// </summary>
    public int ContentLength => Content?.Length ?? 0;
}
