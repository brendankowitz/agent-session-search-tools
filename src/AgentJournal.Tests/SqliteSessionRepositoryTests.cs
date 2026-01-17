using AgentJournal.Core.Models;
using AgentJournal.Core.Storage;
using Microsoft.Data.Sqlite;

namespace AgentJournal.Tests;

public class SqliteSessionRepositoryTests : IDisposable
{
    private readonly string _testDbPath;
    private readonly SqliteSessionRepository _repository;

    public SqliteSessionRepositoryTests()
    {
        _testDbPath = Path.Combine(Path.GetTempPath(), $"test-{Guid.NewGuid()}.db");
        _repository = new SqliteSessionRepository(_testDbPath);
    }

    public void Dispose()
    {
        // Clear any pooled connections
        SqliteConnection.ClearAllPools();
        
        // Note: File cleanup is best effort - SQLite may hold locks for a moment
        // This doesn't affect test validity as each test uses a unique database file
    }

    [Fact]
    public async Task InitializeAsync_ShouldCreateTables()
    {
        // Act
        await _repository.InitializeAsync();

        // Assert - if no exception is thrown, initialization succeeded
        Assert.True(File.Exists(_testDbPath));
    }

    [Fact]
    public async Task SaveSessionAsync_ShouldSaveSession()
    {
        // Arrange
        await _repository.InitializeAsync();
        var session = CreateTestSession("session-1");

        // Act
        await _repository.SaveSessionAsync(session);

        // Assert
        var retrieved = await _repository.GetSessionAsync("session-1");
        Assert.NotNull(retrieved);
        Assert.Equal(session.Id, retrieved.Id);
        Assert.Equal(session.AgentType, retrieved.AgentType);
        Assert.Equal(session.ProjectPath, retrieved.ProjectPath);
        Assert.Equal(session.Messages.Count, retrieved.Messages.Count);
    }

    [Fact]
    public async Task SaveSessionAsync_ShouldSaveMessagesWithToolCalls()
    {
        // Arrange
        await _repository.InitializeAsync();
        var toolCalls = new List<ToolCall>
        {
            new ToolCall("tc-1", "msg-1", "read_file", "{\"path\": \"test.cs\"}", "file contents", true),
            new ToolCall("tc-2", "msg-1", "write_file", "{\"path\": \"output.txt\"}", "written", true)
        };
        var messages = new List<Message>
        {
            new Message(
                "msg-1",
                "session-1",
                MessageRole.Assistant,
                "Let me help you",
                null,
                DateTime.UtcNow,
                null,
                "claude-3.5",
                toolCalls
            )
        };
        var session = new Session(
            "session-1",
            "claude-code",
            "/test/path",
            "main",
            "1.0.0",
            DateTime.UtcNow,
            null,
            null,
            "Test session",
            messages
        );

        // Act
        await _repository.SaveSessionAsync(session);

        // Assert
        var retrieved = await _repository.GetSessionAsync("session-1");
        Assert.NotNull(retrieved);
        Assert.Equal(1, retrieved.Messages.Count);
        Assert.NotNull(retrieved.Messages[0].ToolCalls);
        Assert.Equal(2, retrieved.Messages[0].ToolCalls.Count);
        Assert.Equal("read_file", retrieved.Messages[0].ToolCalls[0].Name);
        Assert.Equal("write_file", retrieved.Messages[0].ToolCalls[1].Name);
    }

    [Fact]
    public async Task GetSessionAsync_ShouldReturnNull_WhenSessionNotFound()
    {
        // Arrange
        await _repository.InitializeAsync();

        // Act
        var retrieved = await _repository.GetSessionAsync("non-existent");

        // Assert
        Assert.Null(retrieved);
    }

    [Fact]
    public async Task GetAllSessionsAsync_ShouldReturnAllSessions()
    {
        // Arrange
        await _repository.InitializeAsync();
        await _repository.SaveSessionAsync(CreateTestSession("session-1"));
        await _repository.SaveSessionAsync(CreateTestSession("session-2"));
        await _repository.SaveSessionAsync(CreateTestSession("session-3"));

        // Act
        var sessions = new List<Session>();
        await foreach (var session in _repository.GetAllSessionsAsync())
        {
            sessions.Add(session);
        }

        // Assert
        Assert.Equal(3, sessions.Count);
    }

    [Fact]
    public async Task GetSessionsByAgentTypeAsync_ShouldFilterByAgentType()
    {
        // Arrange
        await _repository.InitializeAsync();
        await _repository.SaveSessionAsync(CreateTestSession("session-1", "claude-code"));
        await _repository.SaveSessionAsync(CreateTestSession("session-2", "copilot"));
        await _repository.SaveSessionAsync(CreateTestSession("session-3", "claude-code"));

        // Act
        var sessions = new List<Session>();
        await foreach (var session in _repository.GetSessionsByAgentTypeAsync("claude-code"))
        {
            sessions.Add(session);
        }

        // Assert
        Assert.Equal(2, sessions.Count);
        Assert.All(sessions, s => Assert.Equal("claude-code", s.AgentType));
    }

    [Fact]
    public async Task DeleteSessionAsync_ShouldRemoveSession()
    {
        // Arrange
        await _repository.InitializeAsync();
        var session = CreateTestSession("session-1");
        await _repository.SaveSessionAsync(session);

        // Act
        await _repository.DeleteSessionAsync("session-1");

        // Assert
        var retrieved = await _repository.GetSessionAsync("session-1");
        Assert.Null(retrieved);
    }

    [Fact]
    public async Task SaveSessionAsync_ShouldUpdateExistingSession()
    {
        // Arrange
        await _repository.InitializeAsync();
        var session1 = CreateTestSession("session-1");
        await _repository.SaveSessionAsync(session1);

        var session2 = new Session(
            "session-1",
            "updated-agent",
            "/updated/path",
            "develop",
            "2.0.0",
            session1.StartedAt,
            DateTime.UtcNow,
            null,
            "Updated summary",
            Array.Empty<Message>()
        );

        // Act
        await _repository.SaveSessionAsync(session2);

        // Assert
        var retrieved = await _repository.GetSessionAsync("session-1");
        Assert.NotNull(retrieved);
        Assert.Equal("updated-agent", retrieved.AgentType);
        Assert.Equal("/updated/path", retrieved.ProjectPath);
        Assert.Equal("develop", retrieved.GitBranch);
        Assert.NotNull(retrieved.EndedAt);
    }

    [Fact]
    public async Task SaveSessionsAsync_ShouldSaveMultipleSessions()
    {
        // Arrange
        await _repository.InitializeAsync();
        var sessions = new[]
        {
            CreateTestSession("session-1"),
            CreateTestSession("session-2"),
            CreateTestSession("session-3")
        };

        // Act
        await _repository.SaveSessionsAsync(sessions);

        // Assert
        var allSessions = new List<Session>();
        await foreach (var session in _repository.GetAllSessionsAsync())
        {
            allSessions.Add(session);
        }
        Assert.Equal(3, allSessions.Count);
    }

    [Fact]
    public async Task SaveSessionAsync_ShouldHandleNullValues()
    {
        // Arrange
        await _repository.InitializeAsync();
        var session = new Session(
            "session-1",
            "claude-code",
            null, // null project path
            null, // null git branch
            null, // null agent version
            DateTime.UtcNow,
            null, // null ended at
            null, // null last modified
            null, // null summary
            Array.Empty<Message>()
        );

        // Act
        await _repository.SaveSessionAsync(session);

        // Assert
        var retrieved = await _repository.GetSessionAsync("session-1");
        Assert.NotNull(retrieved);
        Assert.Null(retrieved.ProjectPath);
        Assert.Null(retrieved.GitBranch);
        Assert.Null(retrieved.AgentVersion);
        Assert.Null(retrieved.EndedAt);
        Assert.Null(retrieved.LastModified);
        Assert.Null(retrieved.Summary);
    }

    private static Session CreateTestSession(string id, string agentType = "claude-code")
    {
        var messages = new List<Message>
        {
            new Message(
                $"{id}-msg-1",
                id,
                MessageRole.User,
                "Hello",
                null,
                DateTime.UtcNow,
                null,
                null,
                null
            ),
            new Message(
                $"{id}-msg-2",
                id,
                MessageRole.Assistant,
                "Hi there!",
                null,
                DateTime.UtcNow,
                $"{id}-msg-1",
                "claude-3.5",
                null
            )
        };

        return new Session(
            id,
            agentType,
            "/test/path",
            "main",
            "1.0.0",
            DateTime.UtcNow,
            null,
            null,
            "Test session",
            messages
        );
    }
}
