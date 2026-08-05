using AgentJournal.Core.Storage;
using AgentJournal.Core.Tasks;

namespace AgentJournal.Tests;

/// <summary>
/// Tests for searching task journals.
/// </summary>
/// <remarks>
/// Without search, a journal can only be retrieved by an agent that already knows its name - which
/// is exactly what an agent arriving fresh after context loss does not know. These tests cover the
/// index staying truthful as the underlying rows change, because a stale index is worse than no
/// index: it answers confidently with work that no longer exists.
/// </remarks>
public class TaskJournalSearchTests : IDisposable
{
    private readonly string _root;
    private readonly string _planPath;
    private readonly TaskJournalStore _store;

    public TaskJournalSearchTests()
    {
        _root = Path.Combine(Path.GetTempPath(), "agent-journal-search-tests", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_root);

        _planPath = Path.Combine(_root, "plan.md");
        File.WriteAllText(_planPath, "# Plan\n\n## Task 1: One\n\n## Task 2: Two\n\n## Task 3: Three\n");

        _store = new TaskJournalStore(Path.Combine(_root, ".agent-journal", "tasks"));
    }

    public void Dispose()
    {
        try
        {
            if (Directory.Exists(_root))
            {
                Directory.Delete(_root, recursive: true);
            }
        }
        catch (IOException)
        {
            // Temp cleanup is best-effort; a locked file must not fail the test run.
        }

        GC.SuppressFinalize(this);
    }

    [Fact]
    public async Task SearchAsync_FindsProgressNotes()
    {
        await _store.InitAsync(_planPath, name: "plan");
        await _store.AppendAsync("plan", 2, TaskJournalState.InProgress, "raised the kestrel request timeout");

        var hit = Assert.Single(await _store.SearchAsync("kestrel"));

        Assert.Equal("plan", hit.JournalName);
        Assert.Equal(2, hit.TaskNumber);
        Assert.Equal("note", hit.Kind);
        Assert.Contains("kestrel", hit.Excerpt, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public async Task SearchAsync_FindsHandoverArtifacts()
    {
        await _store.InitAsync(_planPath, name: "plan");
        await _store.WriteArtifactAsync(
            "plan",
            1,
            TaskArtifactKind.Report,
            "Replaced the retry policy with exponential backoff.");

        var hit = Assert.Single(await _store.SearchAsync("backoff"));

        Assert.Equal(1, hit.TaskNumber);
        Assert.Equal("report", hit.Kind);
    }

    [Fact]
    public async Task SearchAsync_RanksAndLimitsResults()
    {
        await _store.InitAsync(_planPath, name: "plan");
        await _store.AppendAsync("plan", 1, TaskJournalState.InProgress, "caching layer added");
        await _store.AppendAsync("plan", 2, TaskJournalState.InProgress, "caching caching caching everywhere");
        await _store.AppendAsync("plan", 3, TaskJournalState.InProgress, "unrelated work");

        var results = await _store.SearchAsync("caching");

        Assert.Equal(2, results.Count);
        Assert.True(
            results[0].Score >= results[1].Score,
            "Results must be ordered with the strongest match first.");

        var limited = await _store.SearchAsync("caching", limit: 1);
        Assert.Single(limited);
    }

    [Fact]
    public async Task SearchAsync_TreatsQueryAsLiteralText()
    {
        // FTS5 reads characters such as ':' and '"' as query operators, so an ordinary note phrase
        // would raise a syntax error rather than searching.
        await _store.InitAsync(_planPath, name: "plan");
        await _store.AppendAsync("plan", 1, TaskJournalState.InProgress, "fix: the \"timeout\" bug");

        var results = await _store.SearchAsync("fix: \"timeout\"");

        Assert.NotEmpty(results);
    }

    [Fact]
    public async Task SearchAsync_ReflectsRewrittenArtifacts()
    {
        // Artifacts are upserted, so the index has to follow an UPDATE as well as an INSERT.
        await _store.InitAsync(_planPath, name: "plan");
        await _store.WriteArtifactAsync("plan", 1, TaskArtifactKind.Report, "first attempt used polling");
        await _store.WriteArtifactAsync("plan", 1, TaskArtifactKind.Report, "second attempt used webhooks");

        Assert.Empty(await _store.SearchAsync("polling"));
        Assert.Single(await _store.SearchAsync("webhooks"));
    }

    [Fact]
    public async Task SearchAsync_DoesNotReturnDeletedJournals()
    {
        await _store.InitAsync(_planPath, name: "plan");
        await _store.AppendAsync("plan", 1, TaskJournalState.InProgress, "kestrel timeout raised");
        await _store.WriteArtifactAsync("plan", 1, TaskArtifactKind.Report, "kestrel work finished");

        Assert.NotEmpty(await _store.SearchAsync("kestrel"));

        Assert.True(await _store.DeleteAsync("plan"));

        Assert.Empty(await _store.SearchAsync("kestrel"));
    }

    [Fact]
    public async Task SearchAsync_ScopesResultsToTheirJournal()
    {
        var otherPlan = Path.Combine(_root, "other.md");
        File.WriteAllText(otherPlan, "# Plan\n\n## Task 1: One\n");

        await _store.InitAsync(_planPath, name: "plan");
        await _store.InitAsync(otherPlan, name: "other");

        await _store.AppendAsync("plan", 1, TaskJournalState.InProgress, "shared keyword here");
        await _store.AppendAsync("other", 1, TaskJournalState.InProgress, "shared keyword there");

        var results = await _store.SearchAsync("shared");

        Assert.Equal(2, results.Count);
        Assert.Contains(results, r => r.JournalName == "plan");
        Assert.Contains(results, r => r.JournalName == "other");
    }

    [Fact]
    public async Task SearchAsync_IgnoresEntriesWithoutNotes()
    {
        await _store.InitAsync(_planPath, name: "plan");
        await _store.AppendAsync("plan", 1, TaskJournalState.Complete);

        Assert.Empty(await _store.SearchAsync("plan"));
    }

    [Fact]
    public async Task RebuildSearchIndexAsync_RestoresIndexWithoutDuplicating()
    {
        await _store.InitAsync(_planPath, name: "plan");
        await _store.AppendAsync("plan", 1, TaskJournalState.InProgress, "kestrel timeout raised");
        await _store.WriteArtifactAsync("plan", 1, TaskArtifactKind.Report, "kestrel work finished");

        var before = await _store.SearchAsync("kestrel");

        await _store.RebuildSearchIndexAsync();

        var after = await _store.SearchAsync("kestrel");

        Assert.Equal(before.Count, after.Count);
        Assert.Equal(2, after.Count);
    }

    [Fact]
    public async Task SearchAsync_BackfillsDatabasesCreatedBeforeTheIndexExisted()
    {
        // Journals written by an earlier build have notes and artifacts but no index, and the
        // triggers only fire on future writes - so without a backfill every pre-existing journal
        // would be permanently unfindable.
        await _store.InitAsync(_planPath, name: "plan");
        await _store.AppendAsync("plan", 1, TaskJournalState.InProgress, "kestrel timeout raised");
        await _store.WriteArtifactAsync("plan", 1, TaskArtifactKind.Report, "kestrel work finished");

        await DropSearchIndexAsync(_store.DatabasePath);

        // A new store re-runs schema creation against the now index-less database.
        var reopened = new TaskJournalStore(_store.TasksRoot);

        var results = await reopened.SearchAsync("kestrel");

        Assert.Equal(2, results.Count);
    }

    [Fact]
    public async Task SearchAsync_KeepsIndexingAfterBackfill()
    {
        // The backfill must restore the triggers too, not just the rows.
        await _store.InitAsync(_planPath, name: "plan");
        await _store.AppendAsync("plan", 1, TaskJournalState.InProgress, "kestrel timeout raised");

        await DropSearchIndexAsync(_store.DatabasePath);

        var reopened = new TaskJournalStore(_store.TasksRoot);
        await reopened.AppendAsync("plan", 2, TaskJournalState.InProgress, "sprocket replaced");

        Assert.Single(await reopened.SearchAsync("sprocket"));
    }

    /// <summary>
    /// Removes the search index and its triggers, reproducing a database written before the index
    /// was introduced.
    /// </summary>
    private static async Task DropSearchIndexAsync(string databasePath)
    {
        await using var connection = await SqliteConnectionFactory.OpenAsync(
            SqliteConnectionFactory.BuildConnectionString(databasePath));

        await using var command = connection.CreateCommand();
        command.CommandText = """
            DROP TRIGGER IF EXISTS task_entries_search_ai;
            DROP TRIGGER IF EXISTS task_entries_search_ad;
            DROP TRIGGER IF EXISTS task_artifacts_search_ai;
            DROP TRIGGER IF EXISTS task_artifacts_search_au;
            DROP TRIGGER IF EXISTS task_artifacts_search_ad;
            DROP TABLE IF EXISTS task_search;
            """;
        await command.ExecuteNonQueryAsync();
    }

    [Fact]
    public async Task SearchAsync_RejectsEmptyQuery()
    {
        await _store.InitAsync(_planPath, name: "plan");

        await Assert.ThrowsAsync<ArgumentException>(() => _store.SearchAsync("  "));
    }

    [Fact]
    public async Task SearchAsync_RejectsNonPositiveLimit()
    {
        await _store.InitAsync(_planPath, name: "plan");

        await Assert.ThrowsAsync<ArgumentOutOfRangeException>(() => _store.SearchAsync("kestrel", limit: 0));
    }
}
