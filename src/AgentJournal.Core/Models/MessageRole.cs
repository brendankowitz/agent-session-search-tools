namespace AgentJournal.Core.Models;

/// <summary>
/// Represents the role of a message in an AI agent conversation
/// </summary>
public enum MessageRole
{
    /// <summary>
    /// Message from the user
    /// </summary>
    User,

    /// <summary>
    /// Message from the AI assistant
    /// </summary>
    Assistant,

    /// <summary>
    /// System message
    /// </summary>
    System,

    /// <summary>
    /// Tool result message
    /// </summary>
    Tool
}
