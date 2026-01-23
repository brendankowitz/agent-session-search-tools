using AgentJournal.Core.Embeddings;
using AgentJournal.Core.Models;
using AgentJournal.Core.Search;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Xunit;

namespace AgentJournal.Tests.Search;

/// <summary>
/// Tests for VectorSearchEngine functionality
/// </summary>
public class VectorSearchEngineTests : IDisposable
{
    private readonly string _testIndexPath;
    private readonly VectorSearchEngine _searchEngine;
    private readonly IEmbeddingProvider _embedder;

    public VectorSearchEngineTests()
    {
        _testIndexPath = Path.Combine(Path.GetTempPath(), "vector-test-" + Guid.NewGuid().ToString("N"));
        _embedder = new HashEmbeddingProvider(); // Use hash embedder for testing
        _searchEngine = new VectorSearchEngine(_testIndexPath, _embedder);
    }

    [Fact]
    public async Task InitializeAsync_CreatesIndexDirectory()
    {
        // Act
        await _searchEngine.InitializeAsync();

        // Assert
        Assert.True(Directory.Exists(_testIndexPath));
        Assert.True(File.Exists(Path.Combine(_testIndexPath, "index.ajvi")));
    }

    [Fact]
    public async Task IndexSessionAsync_AddsSessionToIndex()
    {
        // Arrange
        await _searchEngine.InitializeAsync();
        var session = CreateTestSession("session-1", "machine learning algorithms", "neural networks");

        // Act
        await _searchEngine.IndexSessionAsync(session);

        // Assert
        var results = await _searchEngine.SearchAsync("machine learning", SearchMode.Semantic, maxResults: 10);
        Assert.Single(results);
        Assert.Equal("session-1", results[0].Session.Id);
    }

    [Fact]
    public async Task SearchAsync_FindsSemanticallySimilarContent()
    {
        // Arrange
        await _searchEngine.InitializeAsync();
        var session1 = CreateTestSession("session-1", "deep learning neural networks", "backpropagation");
        var session2 = CreateTestSession("session-2", "database query optimization", "indexing strategies");

        await _searchEngine.IndexSessionAsync(session1);
        await _searchEngine.IndexSessionAsync(session2);

        // Act
        var results = await _searchEngine.SearchAsync("artificial intelligence", SearchMode.Semantic, maxResults: 10);

        // Assert - Should find session-1 as more similar to "artificial intelligence"
        Assert.NotEmpty(results);
        Assert.True(results[0].Score > 0);
    }

    [Fact]
    public async Task SearchAsync_DeduplicatesByContentHash()
    {
        // Arrange
        await _searchEngine.InitializeAsync();
        var session1 = CreateTestSession("session-1", "duplicate content", "answer 1");
        var session2 = CreateTestSession("session-2", "duplicate content", "answer 2"); // Same user message

        await _searchEngine.IndexSessionAsync(session1);
        await _searchEngine.IndexSessionAsync(session2);

        // Act
        var results = await _searchEngine.SearchAsync("duplicate", SearchMode.Semantic, maxResults: 10);

        // Assert - Both sessions should be found, but no duplicate messages
        Assert.Equal(2, results.Count);
    }

    [Fact]
    public async Task IndexSessionsAsync_BulkIndexing()
    {
        // Arrange
        await _searchEngine.InitializeAsync();
        var sessions = Enumerable.Range(1, 5)
            .Select(i => CreateTestSession($"session-{i}", $"Query about topic {i}", $"Answer {i}"))
            .ToList();

        // Act
        await _searchEngine.IndexSessionsAsync(sessions);

        // Assert
        var results = await _searchEngine.SearchAsync("topic", SearchMode.Semantic, maxResults: 10);
        Assert.Equal(5, results.Count);
    }

    [Fact]
    public async Task ClearIndexAsync_RemovesAllEntries()
    {
        // Arrange
        await _searchEngine.InitializeAsync();
        var sessions = Enumerable.Range(1, 3)
            .Select(i => CreateTestSession($"session-{i}", $"Query {i}", $"Answer {i}"))
            .ToList();
        await _searchEngine.IndexSessionsAsync(sessions);

        // Act
        await _searchEngine.ClearIndexAsync();

        // Assert
        var results = await _searchEngine.SearchAsync("Query", SearchMode.Semantic, maxResults: 10);
        Assert.Empty(results);
    }

    [Fact]
    public async Task SearchAsync_WithNoResults_ReturnsEmptyList()
    {
        // Arrange
        await _searchEngine.InitializeAsync();

        // Act
        var results = await _searchEngine.SearchAsync("nonexistent", SearchMode.Semantic, maxResults: 10);

        // Assert
        Assert.Empty(results);
    }

    [Fact]
    public async Task SearchAsync_MaxResults_LimitsResultCount()
    {
        // Arrange
        await _searchEngine.InitializeAsync();
        var sessions = Enumerable.Range(1, 10)
            .Select(i => CreateTestSession($"session-{i}", "common semantic query", $"Answer {i}"))
            .ToList();
        await _searchEngine.IndexSessionsAsync(sessions);

        // Act
        var results = await _searchEngine.SearchAsync("common query", SearchMode.Semantic, maxResults: 5);

        // Assert
        Assert.Equal(5, results.Count);
    }

    [Fact]
    public async Task SearchAsync_UnsupportedMode_ThrowsException()
    {
        // Arrange
        await _searchEngine.InitializeAsync();

        // Act & Assert
        await Assert.ThrowsAsync<NotSupportedException>(
            () => _searchEngine.SearchAsync("query", SearchMode.Lexical)
        );
    }

    [Fact]
    public async Task SearchAsync_NotInitialized_ThrowsException()
    {
        // Arrange
        var engine = new VectorSearchEngine(_testIndexPath, _embedder);

        // Act & Assert
        await Assert.ThrowsAsync<InvalidOperationException>(
            () => engine.SearchAsync("query", SearchMode.Semantic)
        );
    }

    [Fact]
    public async Task SearchAsync_EmptyQuery_ReturnsEmptyResults()
    {
        // Arrange
        await _searchEngine.InitializeAsync();
        var session = CreateTestSession("session-1", "Test", "Answer");
        await _searchEngine.IndexSessionAsync(session);

        // Act
        var results = await _searchEngine.SearchAsync("", SearchMode.Semantic, maxResults: 10);

        // Assert
        Assert.Empty(results);
    }

    [Fact]
    public async Task IndexSessionAsync_MapsAgentTypes()
    {
        // Arrange
        await _searchEngine.InitializeAsync();
        var copilotSession = CreateTestSession("session-1", "copilot query", "answer", "copilot-cli");
        var claudeSession = CreateTestSession("session-2", "claude query", "answer", "claude-code");
        var otherSession = CreateTestSession("session-3", "other query", "answer", "other-agent");

        // Act
        await _searchEngine.IndexSessionAsync(copilotSession);
        await _searchEngine.IndexSessionAsync(claudeSession);
        await _searchEngine.IndexSessionAsync(otherSession);

        // Assert - All sessions should be indexed
        var results = await _searchEngine.SearchAsync("query", SearchMode.Semantic, maxResults: 10);
        Assert.Equal(3, results.Count);
    }

    [Fact]
    public async Task SearchAsync_ReturnsMatchingMessages()
    {
        // Arrange
        await _searchEngine.InitializeAsync();
        var session = CreateTestSession("session-1", "test query", "test answer");
        await _searchEngine.IndexSessionAsync(session);

        // Act
        var results = await _searchEngine.SearchAsync("test", SearchMode.Semantic, maxResults: 10);

        // Assert
        Assert.Single(results);
        Assert.NotNull(results[0].MatchingMessages);
        Assert.True(results[0].HasMatchingMessages);
    }

    [Fact]
    public async Task SearchAsync_ProvidesHighlight()
    {
        // Arrange
        await _searchEngine.InitializeAsync();
        var session = CreateTestSession("session-1",
            "This is a long message with important information about machine learning and neural networks",
            "answer");
        await _searchEngine.IndexSessionAsync(session);

        // Act
        var results = await _searchEngine.SearchAsync("machine learning", SearchMode.Semantic, maxResults: 10);

        // Assert
        Assert.Single(results);
        Assert.NotNull(results[0].Highlight);
        Assert.Contains("machine", results[0].Highlight, StringComparison.OrdinalIgnoreCase);
    }

    private Session CreateTestSession(string sessionId, string userMessage, string assistantMessage, string agentType = "test-agent")
    {
        return new Session(
            Id: sessionId,
            AgentType: agentType,
            ProjectPath: "/test/project",
            GitBranch: "main",
            AgentVersion: "1.0.0",
            StartedAt: DateTime.UtcNow,
            EndedAt: null,
            LastModified: null,
            Summary: null,
            Messages: new[]
            {
                new Message(
                    Id: $"{sessionId}-msg-1",
                    SessionId: sessionId,
                    Role: MessageRole.User,
                    Content: userMessage,
                    RawContent: null,
                    Timestamp: DateTime.UtcNow,
                    ParentId: null,
                    Model: null,
                    ToolCalls: null
                ),
                new Message(
                    Id: $"{sessionId}-msg-2",
                    SessionId: sessionId,
                    Role: MessageRole.Assistant,
                    Content: assistantMessage,
                    RawContent: null,
                    Timestamp: DateTime.UtcNow.AddSeconds(1),
                    ParentId: $"{sessionId}-msg-1",
                    Model: "test-model",
                    ToolCalls: null
                )
            }
        );
    }

    public void Dispose()
    {
        _searchEngine.Dispose();

        if (Directory.Exists(_testIndexPath))
        {
            try
            {
                Directory.Delete(_testIndexPath, recursive: true);
            }
            catch
            {
                // Ignore cleanup errors
            }
        }
    }
}
