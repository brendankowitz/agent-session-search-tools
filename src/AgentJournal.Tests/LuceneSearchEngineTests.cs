using AgentJournal.Core.Models;
using AgentJournal.Core.Search;
using AgentJournal.Core.Storage;
using Xunit;

namespace AgentJournal.Tests;

/// <summary>
/// Covers session-grouping behaviour of the lexical search engine.
/// </summary>
/// <remarks>
/// The grouping loop previously kept only the first hit per session, so <c>MatchingMessages</c>
/// silently collapsed to a single message no matter how many matched. These tests pin the
/// multi-match behaviour and the per-session cap that bounds it.
/// </remarks>
public class LuceneSearchEngineTests : IAsyncLifetime
{
    private readonly string _indexPath;
    private LuceneSearchEngine _engine = null!;

    public LuceneSearchEngineTests()
    {
        _indexPath = Path.Combine(
            Path.GetTempPath(),
            "lucene-tests-" + Guid.NewGuid().ToString("N"));
    }

    public async Task InitializeAsync()
    {
        _engine = new LuceneSearchEngine(_indexPath);
        await _engine.InitializeAsync();
    }

    public Task DisposeAsync()
    {
        _engine.Dispose();

        try
        {
            if (Directory.Exists(_indexPath))
            {
                Directory.Delete(_indexPath, recursive: true);
            }
        }
        catch (IOException)
        {
            // A retained Lucene handle should not fail the test run.
        }

        return Task.CompletedTask;
    }

    [Fact]
    public async Task SearchAsync_ShouldReturnEveryMatchingMessageInASession()
    {
        var session = CreateSession(
            "session-multi",
            "the quick brown widget",
            "an unrelated line",
            "another widget entirely",
            "a third widget here");

        await _engine.IndexSessionAsync(session);

        var results = await _engine.SearchAsync("widget");

        var result = Assert.Single(results);
        Assert.NotNull(result.MatchingMessages);
        Assert.Equal(3, result.MatchingMessages!.Count);
        Assert.All(result.MatchingMessages, m => Assert.Contains("widget", m.Content));
    }

    [Fact]
    public async Task SearchAsync_ShouldCapMatchingMessagesPerSession()
    {
        // 25 matches exceeds the documented cap of 20.
        var contents = Enumerable.Range(0, 25).Select(i => $"widget number {i}").ToArray();
        await _engine.IndexSessionAsync(CreateSession("session-chatty", contents));

        var results = await _engine.SearchAsync("widget");

        var result = Assert.Single(results);
        Assert.NotNull(result.MatchingMessages);
        Assert.Equal(20, result.MatchingMessages!.Count);
    }

    [Fact]
    public async Task SearchAsync_ShouldGroupHitsBySessionAndHonourMaxResults()
    {
        for (var s = 0; s < 3; s++)
        {
            await _engine.IndexSessionAsync(CreateSession(
                $"session-{s}",
                "widget one",
                "widget two"));
        }

        var all = await _engine.SearchAsync("widget", maxResults: 10);
        Assert.Equal(3, all.Count);
        Assert.Equal(3, all.Select(r => r.Session.Id).Distinct().Count());
        Assert.All(all, r => Assert.Equal(2, r.MatchingMessages!.Count));

        // A session must not be truncated just because maxResults limits how many sessions return.
        var limited = await _engine.SearchAsync("widget", maxResults: 2);
        Assert.Equal(2, limited.Count);
        Assert.All(limited, r => Assert.Equal(2, r.MatchingMessages!.Count));
    }

    [Fact]
    public async Task SearchAsync_ShouldNotReportNonMatchingMessagesAsMatches()
    {
        // "widget" appears in one message only; the session-level all_content field contains it for
        // every document, which is what previously caused all four messages to be reported.
        var session = CreateSession(
            "session-precision",
            "alpha only",
            "the widget lives here",
            "beta only",
            "gamma only");

        await _engine.IndexSessionAsync(session);

        var result = Assert.Single(await _engine.SearchAsync("widget"));

        var match = Assert.Single(result.MatchingMessages!);
        Assert.Equal("the widget lives here", match.Content);
    }

    [Fact]
    public async Task SearchAsync_ShouldDistinguishMatchesFromContext()
    {
        // With context expansion the returned list deliberately includes non-matching neighbours.
        // Consumers previously had no way to tell them apart and rendered every one as a match.
        var session = CreateSession(
            "session-context-marking",
            "alpha only",
            "the widget lives here",
            "beta only",
            "gamma only");

        await _engine.IndexSessionAsync(session);

        var result = Assert.Single(await _engine.SearchAsync("widget", contextCount: 1));

        // Context expansion pulled in the neighbours...
        Assert.Equal(3, result.MatchingMessages!.Count);

        // ...but exactly one of them actually matched.
        var actualMatches = result.MatchingMessages!.Where(result.IsMatch).ToList();
        var onlyMatch = Assert.Single(actualMatches);
        Assert.Equal("the widget lives here", onlyMatch.Content);

        Assert.All(
            result.MatchingMessages!.Where(m => m.Content != "the widget lives here"),
            m => Assert.False(result.IsMatch(m)));
    }

    [Fact]
    public async Task SearchAsync_ShouldMatchSessionWhenTermsAreSplitAcrossMessages()
    {
        // Neither message contains both terms, so the session matches only at session level.
        var session = CreateSession(
            "session-split",
            "the widget is broken",
            "we replaced the sprocket");

        await _engine.IndexSessionAsync(session);

        var result = Assert.Single(await _engine.SearchAsync("widget sprocket"));

        // Session-level recall is preserved...
        Assert.Equal("session-split", result.Session.Id);
        // ...but no individual message is claimed as a match, because none is one.
        Assert.Empty(result.MatchingMessages!);
        Assert.False(result.HasMatchingMessages);
    }

    /// <summary>
    /// Every real invocation searches an index built by a *previous* process, so the in-memory
    /// session cache is cold. This test reopens the index the way the CLI does.
    /// </summary>
    [Fact]
    public async Task SearchAsync_ShouldReturnMatchingMessagesWhenSessionCacheIsCold()
    {
        var session = CreateSession(
            "session-cold",
            "the quick brown widget",
            "an unrelated line",
            "another widget entirely");

        var repository = new StubSessionRepository(session);

        await _engine.IndexSessionAsync(session);
        _engine.Dispose();

        // A fresh engine over the same index directory: exactly what `agent-journal search` does.
        using var coldEngine = new LuceneSearchEngine(_indexPath, repository);
        await coldEngine.InitializeAsync();

        var result = Assert.Single(await coldEngine.SearchAsync("widget"));

        Assert.Equal("session-cold", result.Session.Id);
        Assert.True(result.HasMatchingMessages);
        Assert.Equal(2, result.MatchingMessages!.Count);
        Assert.All(result.MatchingMessages, m => Assert.Contains("widget", m.Content));

        // The hydrated session must carry its messages, or --context has nothing to expand over.
        Assert.Equal(3, result.Session.Messages.Count);
    }

    /// <summary>
    /// A session that is not in the repository must still be returned, using Lucene metadata only.
    /// </summary>
    [Fact]
    public async Task SearchAsync_ShouldStillReturnSessionWhenRepositoryHasNoRecord()
    {
        var session = CreateSession("session-missing", "the quick brown widget");

        await _engine.IndexSessionAsync(session);
        _engine.Dispose();

        using var coldEngine = new LuceneSearchEngine(_indexPath, new StubSessionRepository());
        await coldEngine.InitializeAsync();

        var result = Assert.Single(await coldEngine.SearchAsync("widget"));

        Assert.Equal("session-missing", result.Session.Id);
        Assert.Equal("claude-code", result.Session.AgentType);
        Assert.False(result.HasMatchingMessages);
    }

    /// <summary>
    /// A single chatty session must not starve other sessions out of the result set.
    /// </summary>
    [Fact]
    public async Task SearchAsync_ShouldNotLetOneChattySessionStarveOthers()
    {
        // 25 matching messages each - more than MAX_MATCHING_MESSAGES_PER_SESSION.
        foreach (var id in new[] { "session-a", "session-b", "session-c" })
        {
            var contents = Enumerable.Range(0, 25).Select(i => $"widget number {i}").ToArray();
            await _engine.IndexSessionAsync(CreateSession(id, contents));
        }

        var results = await _engine.SearchAsync("widget", maxResults: 3);

        Assert.Equal(3, results.Count);
        Assert.Equal(
            new[] { "session-a", "session-b", "session-c" },
            results.Select(r => r.Session.Id).OrderBy(id => id, StringComparer.Ordinal));

        // Each session reports the full per-session cap, not a truncated remainder.
        Assert.All(results, r => Assert.Equal(20, r.MatchingMessages!.Count));
    }

    [Fact]
    public async Task IndexSessionAsync_ShouldNotGrowQuadraticallyWithSessionLength()
    {
        // Every message document used to carry a copy of the whole session's text in all_content,
        // so index size grew with messages x session length. A few hundred ordinary messages
        // produced hundreds of megabytes of postings and indexing a real corpus never finished.
        const int messageCount = 400;
        var body = string.Join(" ", Enumerable.Range(0, 40).Select(w => $"lorem{w} ipsum{w}"));
        var contents = Enumerable.Range(0, messageCount)
            .Select(i => $"message {i} {body}")
            .ToArray();

        var sessionBytes = contents.Sum(c => (long)c.Length);

        await _engine.IndexSessionAsync(CreateSession("session-large", contents));

        var indexBytes = new DirectoryInfo(_indexPath)
            .EnumerateFiles("*", SearchOption.AllDirectories)
            .Sum(f => f.Length);

        // Linear indexing lands near the source size; the quadratic layout was ~messageCount times
        // larger. A 10x source-size ceiling separates the two without pinning exact codec output.
        Assert.True(
            indexBytes < sessionBytes * 10,
            $"index was {indexBytes:N0} bytes for {sessionBytes:N0} bytes of session text");
    }

    private static Session CreateSession(string sessionId, params string[] contents)    {
        var start = new DateTime(2026, 1, 1, 0, 0, 0, DateTimeKind.Utc);

        var messages = contents.Select((content, i) => new Message(
            Id: $"{sessionId}-msg-{i}",
            SessionId: sessionId,
            Role: i % 2 == 0 ? MessageRole.User : MessageRole.Assistant,
            Content: content,
            RawContent: null,
            Timestamp: start.AddMinutes(i),
            ParentId: null,
            Model: null,
            ToolCalls: null)).ToList();

        return new Session(
            Id: sessionId,
            AgentType: "claude-code",
            ProjectPath: "/tmp/project",
            GitBranch: null,
            AgentVersion: null,
            StartedAt: start,
            EndedAt: start.AddHours(1),
            LastModified: start.AddHours(1),
            Summary: null,
            Messages: messages);
    }

    /// <summary>
    /// Serves only <see cref="ISessionRepository.GetSessionAsync"/>; the search path needs nothing else.
    /// </summary>
    private sealed class StubSessionRepository : ISessionRepository
    {
        private readonly Dictionary<string, Session> _sessions;

        public StubSessionRepository(params Session[] sessions)
            => _sessions = sessions.ToDictionary(s => s.Id, StringComparer.Ordinal);

        public Task<Session?> GetSessionAsync(string sessionId, CancellationToken ct = default)
            => Task.FromResult(_sessions.TryGetValue(sessionId, out var session) ? session : null);

        public Task SaveSessionAsync(Session session, CancellationToken ct = default)
            => throw new NotSupportedException();

        public Task SaveSessionsAsync(IEnumerable<Session> sessions, CancellationToken ct = default)
            => throw new NotSupportedException();

        public IAsyncEnumerable<Session> GetAllSessionsAsync(CancellationToken ct = default)
            => throw new NotSupportedException();

        public IAsyncEnumerable<Session> GetSessionsByAgentTypeAsync(string agentType, CancellationToken ct = default)
            => throw new NotSupportedException();

        public Task DeleteSessionAsync(string sessionId, CancellationToken ct = default)
            => throw new NotSupportedException();

        public Task<DateTime?> GetSessionLastModifiedAsync(string sessionId, CancellationToken ct = default)
            => throw new NotSupportedException();

        public Task InitializeAsync(CancellationToken ct = default) => Task.CompletedTask;
    }
}
