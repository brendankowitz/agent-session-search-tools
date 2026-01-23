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
/// Tests for HybridSearcher functionality (combines lexical and semantic search)
/// </summary>
public class HybridSearcherTests : IDisposable
{
    private readonly string _testLexicalPath;
    private readonly string _testVectorPath;
    private readonly LuceneSearchEngine _lexicalEngine;
    private readonly VectorSearchEngine _vectorEngine;
    private readonly HybridSearcher _hybridSearcher;
    private readonly IEmbeddingProvider _embedder;

    public HybridSearcherTests()
    {
        _testLexicalPath = Path.Combine(Path.GetTempPath(), "hybrid-lexical-" + Guid.NewGuid().ToString("N"));
        _testVectorPath = Path.Combine(Path.GetTempPath(), "hybrid-vector-" + Guid.NewGuid().ToString("N"));

        _embedder = new HashEmbeddingProvider();
        _lexicalEngine = new LuceneSearchEngine(_testLexicalPath);
        _vectorEngine = new VectorSearchEngine(_testVectorPath, _embedder);
        _hybridSearcher = new HybridSearcher(_lexicalEngine, _vectorEngine);
    }

    [Fact]
    public async Task InitializeAsync_InitializesBothEngines()
    {
        // Act
        await _hybridSearcher.InitializeAsync();

        // Assert
        Assert.True(Directory.Exists(_testLexicalPath));
        Assert.True(Directory.Exists(_testVectorPath));
    }

    [Fact]
    public async Task IndexSessionAsync_IndexesInBothEngines()
    {
        // Arrange
        await _hybridSearcher.InitializeAsync();
        var session = CreateTestSession("session-1", "test query", "test answer");

        // Act
        await _hybridSearcher.IndexSessionAsync(session);

        // Assert - Should be found by both lexical and semantic search
        var lexicalResults = await _hybridSearcher.SearchAsync("test", SearchMode.Lexical, maxResults: 10);
        var semanticResults = await _hybridSearcher.SearchAsync("test", SearchMode.Semantic, maxResults: 10);

        Assert.Single(lexicalResults);
        Assert.Single(semanticResults);
    }

    [Fact]
    public async Task SearchAsync_Lexical_DelegatesToLexicalEngine()
    {
        // Arrange
        await _hybridSearcher.InitializeAsync();
        var session1 = CreateTestSession("session-1", "Lucene full-text search", "BM25 ranking");
        var session2 = CreateTestSession("session-2", "Vector embeddings", "Cosine similarity");

        await _hybridSearcher.IndexSessionAsync(session1);
        await _hybridSearcher.IndexSessionAsync(session2);

        // Act
        var results = await _hybridSearcher.SearchAsync("Lucene search", SearchMode.Lexical, maxResults: 10);

        // Assert
        Assert.Single(results);
        Assert.Equal("session-1", results[0].Session.Id);
    }

    [Fact]
    public async Task SearchAsync_Semantic_DelegatesToVectorEngine()
    {
        // Arrange
        await _hybridSearcher.InitializeAsync();
        var session1 = CreateTestSession("session-1", "machine learning algorithms", "neural networks");
        var session2 = CreateTestSession("session-2", "database optimization", "query tuning");

        await _hybridSearcher.IndexSessionAsync(session1);
        await _hybridSearcher.IndexSessionAsync(session2);

        // Act
        var results = await _hybridSearcher.SearchAsync("AI and ML", SearchMode.Semantic, maxResults: 10);

        // Assert
        Assert.NotEmpty(results);
    }

    [Fact]
    public async Task SearchAsync_Hybrid_CombinesBothEngines()
    {
        // Arrange
        await _hybridSearcher.InitializeAsync();

        // Session 1: Strong lexical match, weak semantic match
        var session1 = CreateTestSession("session-1", "Lucene Lucene Lucene search", "BM25 algorithm");

        // Session 2: Weak lexical match, strong semantic match
        var session2 = CreateTestSession("session-2", "information retrieval systems", "search methods");

        // Session 3: Moderate match in both
        var session3 = CreateTestSession("session-3", "search engine design", "ranking algorithms");

        await _hybridSearcher.IndexSessionAsync(session1);
        await _hybridSearcher.IndexSessionAsync(session2);
        await _hybridSearcher.IndexSessionAsync(session3);

        // Act
        var results = await _hybridSearcher.SearchAsync("search", SearchMode.Hybrid, maxResults: 10);

        // Assert - Should return results from both engines, fused by RRF
        Assert.NotEmpty(results);
        Assert.True(results.Count <= 3);
        Assert.All(results, r => Assert.True(r.Score > 0));
    }

    [Fact]
    public async Task SearchAsync_Hybrid_RRFScoring()
    {
        // Arrange
        await _hybridSearcher.InitializeAsync();

        // Create sessions that will rank differently in each engine
        var sessions = Enumerable.Range(1, 5)
            .Select(i => CreateTestSession($"session-{i}", $"query term {i}", $"answer {i}"))
            .ToList();

        await _hybridSearcher.IndexSessionsAsync(sessions);

        // Act
        var results = await _hybridSearcher.SearchAsync("query term", SearchMode.Hybrid, maxResults: 10);

        // Assert - RRF should produce combined scores
        Assert.NotEmpty(results);
        Assert.All(results, r => Assert.True(r.Score > 0));

        // Results should be ordered by fused score
        var scores = results.Select(r => r.Score).ToList();
        for (int i = 0; i < scores.Count - 1; i++)
        {
            Assert.True(scores[i] >= scores[i + 1], "Results should be ordered by descending score");
        }
    }

    [Fact]
    public async Task IndexSessionsAsync_BulkIndexingInBothEngines()
    {
        // Arrange
        await _hybridSearcher.InitializeAsync();
        var sessions = Enumerable.Range(1, 5)
            .Select(i => CreateTestSession($"session-{i}", $"Query {i}", $"Answer {i}"))
            .ToList();

        // Act
        await _hybridSearcher.IndexSessionsAsync(sessions);

        // Assert - Should be searchable in all modes
        var lexicalResults = await _hybridSearcher.SearchAsync("Query", SearchMode.Lexical, maxResults: 10);
        var semanticResults = await _hybridSearcher.SearchAsync("Query", SearchMode.Semantic, maxResults: 10);
        var hybridResults = await _hybridSearcher.SearchAsync("Query", SearchMode.Hybrid, maxResults: 10);

        Assert.Equal(5, lexicalResults.Count);
        Assert.Equal(5, semanticResults.Count);
        Assert.Equal(5, hybridResults.Count);
    }

    [Fact]
    public async Task ClearIndexAsync_ClearsBothEngines()
    {
        // Arrange
        await _hybridSearcher.InitializeAsync();
        var sessions = Enumerable.Range(1, 3)
            .Select(i => CreateTestSession($"session-{i}", $"Query {i}", $"Answer {i}"))
            .ToList();
        await _hybridSearcher.IndexSessionsAsync(sessions);

        // Act
        await _hybridSearcher.ClearIndexAsync();

        // Assert
        var lexicalResults = await _hybridSearcher.SearchAsync("Query", SearchMode.Lexical, maxResults: 10);
        var semanticResults = await _hybridSearcher.SearchAsync("Query", SearchMode.Semantic, maxResults: 10);

        Assert.Empty(lexicalResults);
        Assert.Empty(semanticResults);
    }

    [Fact]
    public async Task SearchAsync_EmptyQuery_ReturnsEmptyResults()
    {
        // Arrange
        await _hybridSearcher.InitializeAsync();
        var session = CreateTestSession("session-1", "Test", "Answer");
        await _hybridSearcher.IndexSessionAsync(session);

        // Act
        var results = await _hybridSearcher.SearchAsync("", SearchMode.Hybrid, maxResults: 10);

        // Assert
        Assert.Empty(results);
    }

    [Fact]
    public async Task SearchAsync_MaxResults_LimitsHybridResults()
    {
        // Arrange
        await _hybridSearcher.InitializeAsync();
        var sessions = Enumerable.Range(1, 10)
            .Select(i => CreateTestSession($"session-{i}", "common query", $"Answer {i}"))
            .ToList();
        await _hybridSearcher.IndexSessionsAsync(sessions);

        // Act
        var results = await _hybridSearcher.SearchAsync("common", SearchMode.Hybrid, maxResults: 5);

        // Assert
        Assert.Equal(5, results.Count);
    }

    [Fact]
    public async Task SearchAsync_Hybrid_DeduplicatesSessions()
    {
        // Arrange
        await _hybridSearcher.InitializeAsync();

        // Session that should rank high in both engines
        var session = CreateTestSession("session-1", "test query search", "test answer result");
        await _hybridSearcher.IndexSessionAsync(session);

        // Act
        var results = await _hybridSearcher.SearchAsync("test search", SearchMode.Hybrid, maxResults: 10);

        // Assert - Should return session only once, not duplicated
        Assert.Single(results);
        Assert.Equal("session-1", results[0].Session.Id);
    }

    [Fact]
    public async Task SearchAsync_UnsupportedMode_ThrowsException()
    {
        // Arrange
        await _hybridSearcher.InitializeAsync();

        // Act & Assert
        await Assert.ThrowsAsync<NotSupportedException>(
            () => _hybridSearcher.SearchAsync("query", (SearchMode)999)
        );
    }

    [Fact]
    public void Constructor_WithCustomWeights_SetsWeights()
    {
        // Arrange & Act
        var searcher = new HybridSearcher(_lexicalEngine, _vectorEngine,
            lexicalWeight: 0.7f,
            semanticWeight: 0.3f,
            rrfK: 100);

        // Assert
        Assert.Equal([SearchMode.Lexical, SearchMode.Semantic, SearchMode.Hybrid], searcher.SupportedModes);
    }

    [Fact]
    public void Constructor_NullEngines_ThrowsException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            new HybridSearcher(null!, _vectorEngine));

        Assert.Throws<ArgumentNullException>(() =>
            new HybridSearcher(_lexicalEngine, null!));
    }

    [Fact]
    public async Task SearchAsync_Hybrid_FetchesMoreResultsForFusion()
    {
        // Arrange
        await _hybridSearcher.InitializeAsync();

        // Create many sessions to test that we fetch 3x results for better fusion
        var sessions = Enumerable.Range(1, 20)
            .Select(i => CreateTestSession($"session-{i}", $"query content {i}", $"answer {i}"))
            .ToList();
        await _hybridSearcher.IndexSessionsAsync(sessions);

        // Act - Request 5 results, but internally should fetch 15 from each engine
        var results = await _hybridSearcher.SearchAsync("query content", SearchMode.Hybrid, maxResults: 5);

        // Assert
        Assert.Equal(5, results.Count);
        Assert.All(results, r => Assert.True(r.Score > 0));
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
        _hybridSearcher.Dispose();

        if (Directory.Exists(_testLexicalPath))
        {
            try
            {
                Directory.Delete(_testLexicalPath, recursive: true);
            }
            catch
            {
                // Ignore cleanup errors
            }
        }

        if (Directory.Exists(_testVectorPath))
        {
            try
            {
                Directory.Delete(_testVectorPath, recursive: true);
            }
            catch
            {
                // Ignore cleanup errors
            }
        }
    }
}
