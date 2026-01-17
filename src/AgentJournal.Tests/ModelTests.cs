using AgentJournal.Core.Models;

namespace AgentJournal.Tests;

public class ModelTests
{
    [Fact]
    public void Session_ShouldCalculateDuration()
    {
        // Arrange
        var startedAt = DateTime.UtcNow;
        var endedAt = startedAt.AddHours(2);
        var session = new Session(
            Id: "test-1",
            AgentType: "claude-code",
            ProjectPath: "/test/path",
            GitBranch: "main",
            AgentVersion: "1.0.0",
            StartedAt: startedAt,
            EndedAt: endedAt,
            LastModified: null,
            Summary: "Test session",
            Messages: Array.Empty<Message>()
        );

        // Act
        var duration = session.Duration;

        // Assert
        Assert.NotNull(duration);
        Assert.Equal(TimeSpan.FromHours(2), duration.Value);
    }

    [Fact]
    public void Session_ShouldBeActive_WhenNotEnded()
    {
        // Arrange
        var session = new Session(
            Id: "test-1",
            AgentType: "claude-code",
            ProjectPath: "/test/path",
            GitBranch: "main",
            AgentVersion: "1.0.0",
            StartedAt: DateTime.UtcNow,
            EndedAt: null,
            LastModified: null,
            Summary: "Test session",
            Messages: Array.Empty<Message>()
        );

        // Act & Assert
        Assert.True(session.IsActive);
        Assert.Null(session.Duration);
    }

    [Fact]
    public void Message_ShouldHaveToolCalls()
    {
        // Arrange
        var toolCalls = new[]
        {
            new ToolCall("tc-1", "msg-1", "read_file", "{\"path\": \"test.cs\"}", "content", true)
        };
        var message = new Message(
            Id: "msg-1",
            SessionId: "session-1",
            Role: MessageRole.Assistant,
            Content: "Let me read that file",
            RawContent: null,
            Timestamp: DateTime.UtcNow,
            ParentId: null,
            Model: "claude-3.5",
            ToolCalls: toolCalls
        );

        // Act & Assert
        Assert.True(message.HasToolCalls);
        Assert.Equal(1, message.ToolCallCount);
    }

    [Fact]
    public void ToolCall_ShouldBeCompleted()
    {
        // Arrange
        var toolCall = new ToolCall(
            Id: "tc-1",
            MessageId: "msg-1",
            Name: "read_file",
            Arguments: "{\"path\": \"test.cs\"}",
            Result: "file contents",
            Success: true
        );

        // Act & Assert
        Assert.True(toolCall.IsCompleted);
        Assert.True(toolCall.IsSuccessful);
        Assert.True(toolCall.HasArguments);
    }
}
