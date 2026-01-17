using AgentJournal.Core.Connectors;
using AgentJournal.Core.Models;
using System.Text.Json;
using Xunit;

namespace AgentJournal.Tests;

public class ClaudeCodeConnectorTests
{
    [Fact]
    public void GetSessionPaths_ReturnsEmptyWhenDirectoryDoesNotExist()
    {
        // Arrange
        var connector = new ClaudeCodeConnector();
        var homeDir = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        var claudeProjectsPath = Path.Combine(homeDir, ".claude", "projects");

        // Act & Assert - should not throw
        var paths = connector.GetSessionPaths().ToList();

        // If directory exists, we should get results, otherwise empty
        if (Directory.Exists(claudeProjectsPath))
        {
            Assert.NotNull(paths);
        }
        else
        {
            Assert.Empty(paths);
        }
    }

    [Fact]
    public async Task ParseSessionAsync_ReturnsNullForNonExistentFile()
    {
        // Arrange
        var connector = new ClaudeCodeConnector();
        var nonExistentPath = Path.Combine(Path.GetTempPath(), "non-existent-session.jsonl");

        // Act
        var session = await connector.ParseSessionAsync(nonExistentPath);

        // Assert
        Assert.Null(session);
    }

    [Fact]
    public async Task ParseSessionAsync_ParsesValidSessionFile()
    {
        // Arrange
        var connector = new ClaudeCodeConnector();
        var tempFile = Path.Combine(Path.GetTempPath(), $"test-session-{Guid.NewGuid()}.jsonl");

        try
        {
            // Create a valid JSONL test file
            var jsonLines = new[]
            {
                JsonSerializer.Serialize(new
                {
                    type = "user",
                    uuid = "msg-001",
                    parentUuid = (string?)null,
                    sessionId = "session-123",
                    timestamp = "2024-01-15T10:30:00Z",
                    cwd = "/test/project",
                    version = "2.1.6",
                    gitBranch = "main",
                    message = new
                    {
                        role = "user",
                        content = "Hello, Claude!",
                        model = (string?)null
                    }
                }),
                JsonSerializer.Serialize(new
                {
                    type = "assistant",
                    uuid = "msg-002",
                    parentUuid = "msg-001",
                    sessionId = "session-123",
                    timestamp = "2024-01-15T10:30:05Z",
                    cwd = "/test/project",
                    version = "2.1.6",
                    gitBranch = "main",
                    message = new
                    {
                        role = "assistant",
                        content = new object[]
                        {
                            new { type = "text", text = "Hello! How can I help you today?" }
                        },
                        model = "claude-sonnet-4-5-20250929"
                    }
                }),
                JsonSerializer.Serialize(new
                {
                    type = "summary",
                    uuid = "summary-001",
                    parentUuid = (string?)null,
                    sessionId = "session-123",
                    timestamp = "2024-01-15T10:35:00Z",
                    cwd = "/test/project",
                    version = "2.1.6",
                    gitBranch = "main",
                    message = new
                    {
                        role = "assistant",
                        content = "Test session summary",
                        model = (string?)null
                    }
                })
            };

            await File.WriteAllLinesAsync(tempFile, jsonLines);

            // Act
            var session = await connector.ParseSessionAsync(tempFile);

            // Assert
            Assert.NotNull(session);
            Assert.Equal("session-123", session.Id);
            Assert.Equal("claude-code", session.AgentType);
            Assert.Equal("/test/project", session.ProjectPath);
            Assert.Equal("main", session.GitBranch);
            Assert.Equal("2.1.6", session.AgentVersion);
            Assert.Equal(2, session.MessageCount);
            Assert.Equal(1, session.UserMessageCount);
            Assert.Equal(1, session.AssistantMessageCount);
            Assert.NotNull(session.Summary);
            Assert.Contains("summary", session.Summary);

            // Check first message
            var userMsg = session.Messages[0];
            Assert.Equal("msg-001", userMsg.Id);
            Assert.Equal(MessageRole.User, userMsg.Role);
            Assert.Equal("Hello, Claude!", userMsg.Content);
            Assert.Null(userMsg.ParentId);

            // Check second message
            var assistantMsg = session.Messages[1];
            Assert.Equal("msg-002", assistantMsg.Id);
            Assert.Equal(MessageRole.Assistant, assistantMsg.Role);
            Assert.Contains("How can I help you", assistantMsg.Content);
            Assert.Equal("msg-001", assistantMsg.ParentId);
            Assert.Equal("claude-sonnet-4-5-20250929", assistantMsg.Model);
        }
        finally
        {
            // Cleanup
            if (File.Exists(tempFile))
            {
                File.Delete(tempFile);
            }
        }
    }

    [Fact]
    public async Task ParseSessionAsync_HandlesToolCalls()
    {
        // Arrange
        var connector = new ClaudeCodeConnector();
        var tempFile = Path.Combine(Path.GetTempPath(), $"test-session-tools-{Guid.NewGuid()}.jsonl");

        try
        {
            // Create a session with tool use
            var jsonLines = new[]
            {
                JsonSerializer.Serialize(new
                {
                    type = "assistant",
                    uuid = "msg-001",
                    parentUuid = (string?)null,
                    sessionId = "session-456",
                    timestamp = "2024-01-15T11:00:00Z",
                    cwd = "/test/project",
                    version = "2.1.6",
                    gitBranch = "main",
                    message = new
                    {
                        role = "assistant",
                        content = new object[]
                        {
                            new { type = "text", text = "I'll view that file for you." },
                            new
                            {
                                type = "tool_use",
                                id = "toolu_123",
                                name = "view",
                                input = new { path = "/test/file.txt" }
                            }
                        },
                        model = "claude-sonnet-4-5-20250929"
                    }
                })
            };

            await File.WriteAllLinesAsync(tempFile, jsonLines);

            // Act
            var session = await connector.ParseSessionAsync(tempFile);

            // Assert
            Assert.NotNull(session);
            Assert.Equal(1, session.MessageCount);
            Assert.Equal(1, session.ToolCallCount);

            var message = session.Messages[0];
            Assert.True(message.HasToolCalls);
            Assert.Single(message.ToolCalls!);

            var toolCall = message.ToolCalls![0];
            Assert.Equal("toolu_123", toolCall.Id);
            Assert.Equal("view", toolCall.Name);
            Assert.NotNull(toolCall.Arguments);
            Assert.Contains("/test/file.txt", toolCall.Arguments);
        }
        finally
        {
            // Cleanup
            if (File.Exists(tempFile))
            {
                File.Delete(tempFile);
            }
        }
    }

    [Fact]
    public async Task ParseSessionAsync_HandlesMalformedLines()
    {
        // Arrange
        var connector = new ClaudeCodeConnector();
        var tempFile = Path.Combine(Path.GetTempPath(), $"test-session-malformed-{Guid.NewGuid()}.jsonl");

        try
        {
            // Create a file with some malformed lines
            var lines = new[]
            {
                "{invalid json",  // Malformed
                "",  // Empty
                JsonSerializer.Serialize(new
                {
                    type = "user",
                    uuid = "msg-001",
                    sessionId = "session-789",
                    timestamp = "2024-01-15T12:00:00Z",
                    cwd = "/test",
                    message = new
                    {
                        role = "user",
                        content = "Valid message"
                    }
                })
            };

            await File.WriteAllLinesAsync(tempFile, lines);

            // Act
            var session = await connector.ParseSessionAsync(tempFile);

            // Assert - Should still parse the valid message
            Assert.NotNull(session);
            Assert.Equal(1, session.MessageCount);
            Assert.Equal("Valid message", session.Messages[0].Content);
        }
        finally
        {
            // Cleanup
            if (File.Exists(tempFile))
            {
                File.Delete(tempFile);
            }
        }
    }
}
