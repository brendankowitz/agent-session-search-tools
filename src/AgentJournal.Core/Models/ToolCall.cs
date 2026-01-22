namespace AgentJournal.Core.Models;

/// <summary>
/// Represents a tool call made by the AI assistant
/// </summary>
public record ToolCall(
    string Id,
    string MessageId,
    string Name,
    string? Arguments,
    string? Result,
    bool? Success
)
{
    /// <summary>
    /// Gets whether this tool call has completed
    /// </summary>
    public bool IsCompleted => Result != null;

    /// <summary>
    /// Gets whether this tool call was successful
    /// </summary>
    public bool IsSuccessful => Success == true;

    /// <summary>
    /// Gets whether this tool call has arguments
    /// </summary>
    public bool HasArguments => !string.IsNullOrEmpty(Arguments);

    /// <summary>
    /// Gets the result length in characters
    /// </summary>
    public int ResultLength => Result?.Length ?? 0;
}
