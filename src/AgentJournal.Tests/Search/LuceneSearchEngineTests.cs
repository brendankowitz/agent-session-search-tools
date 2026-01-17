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
/// Tests for LuceneSearchEngine functionality
/// </summary>
public class LuceneSearchEngineTests : IDisposable
{
    private readonly string _testIndexPath;
    private readonly LuceneSearchEngine _searchEngine;

    public LuceneSearchEngineTests()
    {
        _testIndexPath = Path.Combine(Path.GetTempPath(), "lucene-test-" + Guid.NewGuid().ToString("N"));
        _searchEngine = new LuceneSearchEngine(_testIndexPath);
    }

    [Fact]
    public async Task InitializeAsync_CreatesIndexDirectory()
    {
        // Act
        await _searchEngine.InitializeAsync();

        // Assert
        Assert.True(Directory.Exists(_testIndexPath));
    }

    [Fact]
    public async Task IndexSessionAsync_AddsSessionToIndex()
    {
        // Arrange
        await _searchEngine.InitializeAsync();
        var session = CreateTestSession("session-1", "How to search documents", "Search using full text");

        // Act
        await _searchEngine.IndexSessionAsync(session);

        // Assert
        var results = await _searchEngine.SearchAsync("search", maxResults: 10);
        Assert.Single(results);
        Assert.Equal("session-1", results[0].Session.Id);
    }

    [Fact]
    public async Task SearchAsync_FindsRelevantContent()
    {
        // Arrange
        await _searchEngine.InitializeAsync();
        var session1 = CreateTestSession("session-1", "Lucene full-text search", "Use BM25 for ranking");
        var session2 = CreateTestSession("session-2", "Vector embeddings", "Use cosine similarity");
        
        await _searchEngine.IndexSessionAsync(session1);
        await _searchEngine.IndexSessionAsync(session2);

        // Act
        var results = await _searchEngine.SearchAsync("Lucene search", maxResults: 10);

        // Assert
        Assert.Single(results);
        Assert.Equal("session-1", results[0].Session.Id);
        Assert.True(results[0].Score > 0);
    }

    [Fact]
    public async Task SearchAsync_BooleanQuery_AND()
    {
        // Arrange
        await _searchEngine.InitializeAsync();
        var session1 = CreateTestSession("session-1", "Lucene search engine", "Full text search");
        var session2 = CreateTestSession("session-2", "Lucene library", "Vector search");
        
        await _searchEngine.IndexSessionAsync(session1);
        await _searchEngine.IndexSessionAsync(session2);

        // Act
        var results = await _searchEngine.SearchAsync("Lucene AND text", maxResults: 10);

        // Assert
        Assert.Single(results);
        Assert.Equal("session-1", results[0].Session.Id);
    }

    [Fact]
    public async Task SearchAsync_PhraseQuery()
    {
        // Arrange
        await _searchEngine.InitializeAsync();
        var session1 = CreateTestSession("session-1", "full text search", "Answer");
        var session2 = CreateTestSession("session-2", "search full text", "Answer");
        
        await _searchEngine.IndexSessionAsync(session1);
        await _searchEngine.IndexSessionAsync(session2);

        // Act
        var results = await _searchEngine.SearchAsync("\"full text search\"", maxResults: 10);

        // Assert
        Assert.Single(results);
        Assert.Equal("session-1", results[0].Session.Id);
    }

    [Fact]
    public async Task IndexSessionsAsync_BulkIndexing()
    {
        // Arrange
        await _searchEngine.InitializeAsync();
        var sessions = Enumerable.Range(1, 5)
            .Select(i => CreateTestSession($"session-{i}", $"Query {i}", $"Answer {i}"))
            .ToList();

        // Act
        await _searchEngine.IndexSessionsAsync(sessions);

        // Assert
        var results = await _searchEngine.SearchAsync("Query", maxResults: 10);
        Assert.Equal(5, results.Count);
    }

    [Fact]
    public async Task DeleteSessionAsync_RemovesSession()
    {
        // Arrange
        await _searchEngine.InitializeAsync();
        var session = CreateTestSession("session-1", "Test query", "Test answer");
        await _searchEngine.IndexSessionAsync(session);

        // Act
        await _searchEngine.DeleteSessionAsync("session-1");

        // Assert
        var results = await _searchEngine.SearchAsync("Test", maxResults: 10);
        Assert.Empty(results);
    }

    [Fact]
    public async Task ClearIndexAsync_RemovesAllDocuments()
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
        var results = await _searchEngine.SearchAsync("Query", maxResults: 10);
        Assert.Empty(results);
    }

    [Fact]
    public async Task GetIndexStatsAsync_ReturnsCorrectStats()
    {
        // Arrange
        await _searchEngine.InitializeAsync();
        var session = CreateTestSession("session-1", "Query", "Answer");
        await _searchEngine.IndexSessionAsync(session);

        // Act
        var stats = await _searchEngine.GetIndexStatsAsync();

        // Assert
        Assert.Equal(2, stats.DocumentCount); // 1 user message + 1 assistant message
        Assert.True(stats.SizeBytes > 0);
        Assert.Equal(1, stats.SessionCount);
    }

    [Fact]
    public async Task SearchAsync_WithNoResults_ReturnsEmptyList()
    {
        // Arrange
        await _searchEngine.InitializeAsync();
        var session = CreateTestSession("session-1", "Lucene", "Search");
        await _searchEngine.IndexSessionAsync(session);

        // Act
        var results = await _searchEngine.SearchAsync("NonexistentTerm", maxResults: 10);

        // Assert
        Assert.Empty(results);
    }

    [Fact]
    public async Task SearchAsync_MaxResults_LimitsResultCount()
    {
        // Arrange
        await _searchEngine.InitializeAsync();
        var sessions = Enumerable.Range(1, 10)
            .Select(i => CreateTestSession($"session-{i}", "common query", $"Answer {i}"))
            .ToList();
        await _searchEngine.IndexSessionsAsync(sessions);

        // Act
        var results = await _searchEngine.SearchAsync("common", maxResults: 5);

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
            () => _searchEngine.SearchAsync("query", SearchMode.Semantic)
        );
    }

    [Fact]
    public async Task SearchAsync_NotInitialized_ThrowsException()
    {
        // Arrange
        var engine = new LuceneSearchEngine(_testIndexPath);

        // Act & Assert
        await Assert.ThrowsAsync<InvalidOperationException>(
            () => engine.SearchAsync("query")
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
        var results = await _searchEngine.SearchAsync("", maxResults: 10);

        // Assert
        Assert.Empty(results);
    }

    [Fact]
    public async Task SearchAsync_SpecialCharacters_HandlesGracefully()
    {
        // Arrange
        await _searchEngine.InitializeAsync();
        var session = CreateTestSession("session-1", "C# programming", "Use async/await");
        await _searchEngine.IndexSessionAsync(session);

        // Act
        var results = await _searchEngine.SearchAsync("C#", maxResults: 10);

        // Assert - Should either find results or handle parse error gracefully
        Assert.NotNull(results);
    }

    [Fact]
    public async Task IndexSessionAsync_UpdatesExistingSession()
    {
        // Arrange
        await _searchEngine.InitializeAsync();
        var session1 = CreateTestSession("session-1", "Original query", "Original answer");
        var session2 = CreateTestSession("session-1", "Updated query", "Updated answer");

        // Act
        await _searchEngine.IndexSessionAsync(session1);
        await _searchEngine.IndexSessionAsync(session2);

        // Assert
        var results = await _searchEngine.SearchAsync("Updated", maxResults: 10);
        Assert.Single(results);
        Assert.Equal("session-1", results[0].Session.Id);
    }

    private Session CreateTestSession(string sessionId, string userMessage, string assistantMessage)
    {
        return new Session(
            Id: sessionId,
            AgentType: "test-agent",
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
            Directory.Delete(_testIndexPath, recursive: true);
        }
    }
}
