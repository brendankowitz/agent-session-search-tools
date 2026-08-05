using AgentJournal.Core.Tasks;

namespace AgentJournal.Tests;

/// <summary>
/// Tests for the task journal. The critical behaviour is resume: after context loss, the journal
/// alone must determine which task to work on next.
/// </summary>
public class TaskJournalStoreTests : IDisposable
{
    private readonly string _root;
    private readonly string _planPath;
    private readonly TaskJournalStore _store;

    public TaskJournalStoreTests()
    {
        _root = Path.Combine(Path.GetTempPath(), "agent-journal-tests", Guid.NewGuid().ToString("N"));
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
    public async Task InitAsync_CountsTasksFromPlanHeadings()
    {
        var snapshot = await _store.InitAsync(_planPath);

        Assert.Equal(3, snapshot.Tasks.Count);
        Assert.All(snapshot.Tasks, t => Assert.Equal(TaskJournalState.Pending, t.State));
        Assert.Equal(0, snapshot.CompletedCount);
        Assert.False(snapshot.IsComplete);
    }

    [Fact]
    public async Task InitAsync_ExplicitTaskCountOverridesPlanHeadings()
    {
        var snapshot = await _store.InitAsync(_planPath, taskCount: 7);

        Assert.Equal(7, snapshot.Tasks.Count);
    }

    /// <summary>
    /// A gap in the numbering used to be filled with a phantom task: it had no plan section, no
    /// brief, and could never be completed, so <c>next</c> parked the agent on it forever.
    /// </summary>
    [Fact]
    public async Task InitAsync_ThrowsWhenPlanHeadingsAreNotContiguous()
    {
        var gapped = Path.Combine(_root, "gapped.md");
        await File.WriteAllTextAsync(gapped, "## Task 1\ntext\n\n## Task 3\ntext\n");

        var ex = await Assert.ThrowsAsync<InvalidOperationException>(
            () => _store.InitAsync(gapped));

        Assert.Contains("Task 2", ex.Message);
    }

    /// <summary>
    /// An explicit count is the documented escape hatch when the plan cannot be renumbered.
    /// </summary>
    [Fact]
    public async Task InitAsync_AllowsNonContiguousPlanWhenTaskCountIsExplicit()
    {
        var gapped = Path.Combine(_root, "gapped-explicit.md");
        await File.WriteAllTextAsync(gapped, "## Task 1\ntext\n\n## Task 3\ntext\n");

        var snapshot = await _store.InitAsync(gapped, taskCount: 3);

        Assert.Equal(3, snapshot.Tasks.Count);
    }

    /// <summary>
    /// A heading inside a fenced block is prose about a task, not a task.
    /// </summary>
    [Fact]
    public async Task InitAsync_IgnoresTaskHeadingsInsideFencedCodeBlocks()
    {
        var fenced = Path.Combine(_root, "fenced.md");
        await File.WriteAllTextAsync(
            fenced,
            "## Task 1\ntext\n\n## Task 2\n\n```markdown\n## Task 99\n```\n");

        var snapshot = await _store.InitAsync(fenced);

        Assert.Equal(2, snapshot.Tasks.Count);
    }

    [Fact]
    public async Task InitAsync_ThrowsWhenPlanMissing()
    {
        var missing = Path.Combine(_root, "nope.md");

        await Assert.ThrowsAsync<FileNotFoundException>(() => _store.InitAsync(missing));
    }

    [Fact]
    public async Task InitAsync_ThrowsWhenPlanMissingEvenWithExplicitCount()
    {
        var missing = Path.Combine(_root, "nope.md");

        // An explicit count skips heading parsing, so without this check the bad path is never noticed.
        await Assert.ThrowsAsync<FileNotFoundException>(() => _store.InitAsync(missing, taskCount: 3));
    }

    [Fact]
    public async Task InitAsync_RaisesTaskCountWhenPlanGrows()
    {
        await _store.InitAsync(_planPath, taskCount: 3, name: "p");

        // Re-initialising after the plan gained tasks must expose them; otherwise the journal
        // reports "all complete" while real work remains invisible.
        var grown = await _store.InitAsync(_planPath, taskCount: 5, name: "p");

        Assert.Equal(5, grown.Tasks.Count);
    }

    [Fact]
    public async Task InitAsync_DoesNotLowerTaskCount()
    {
        await _store.InitAsync(_planPath, taskCount: 5, name: "p");
        await _store.AppendAsync("p", 5, TaskJournalState.Complete);

        // Lowering would orphan entries that already exist for the truncated tasks.
        var shrunk = await _store.InitAsync(_planPath, taskCount: 2, name: "p");

        Assert.Equal(5, shrunk.Tasks.Count);
    }

    [Fact]
    public async Task InitAsync_RejectsReuseWithDifferentPlan()
    {
        await _store.InitAsync(_planPath, name: "shared");

        var otherPlan = Path.Combine(_root, "other.md");
        await File.WriteAllTextAsync(otherPlan, "# Other\n\n## Task 1: X\n");

        // A journal that silently rebinds to a different plan would resume the wrong work.
        var ex = await Assert.ThrowsAsync<InvalidOperationException>(
            () => _store.InitAsync(otherPlan, name: "shared"));

        Assert.Contains("already tracks plan", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public async Task InitAsync_IsIdempotentForSamePlan()
    {
        var first = await _store.InitAsync(_planPath, name: "shared");
        await _store.AppendAsync("shared", 1, TaskJournalState.Complete);

        var second = await _store.InitAsync(_planPath, name: "shared");

        // Re-initialising must not wipe recorded progress.
        Assert.Equal(first.Tasks.Count, second.Tasks.Count);
        Assert.Equal(1, second.CompletedCount);
    }

    [Fact]
    public async Task NextTask_IsLowestIncompleteTask()
    {
        await _store.InitAsync(_planPath, name: "p");
        await _store.AppendAsync("p", 1, TaskJournalState.Complete);

        var snapshot = await _store.LoadAsync("p");

        Assert.Equal(2, snapshot.NextTask?.Number);
    }

    [Fact]
    public async Task NextTask_SkipsCompletedTasksOutOfOrder()
    {
        await _store.InitAsync(_planPath, name: "p");
        await _store.AppendAsync("p", 2, TaskJournalState.Complete);

        var snapshot = await _store.LoadAsync("p");

        Assert.Equal(1, snapshot.NextTask?.Number);
        Assert.Equal(1, snapshot.CompletedCount);
    }

    [Fact]
    public async Task ReopenedTask_BecomesNextAgain()
    {
        await _store.InitAsync(_planPath, name: "p");
        await _store.AppendAsync("p", 1, TaskJournalState.Complete);
        await _store.AppendAsync("p", 2, TaskJournalState.Complete);

        // Review found a problem in task 1 after it was marked complete.
        await _store.AppendAsync("p", 1, TaskJournalState.FixRound, "leak found");

        var snapshot = await _store.LoadAsync("p");

        Assert.Equal(1, snapshot.NextTask?.Number);
        Assert.Equal(TaskJournalState.FixRound, snapshot.NextTask?.State);
        Assert.Equal(1, snapshot.NextTask?.FixRound);
        Assert.Equal("leak found", snapshot.NextTask?.LastNote);
        Assert.Equal(1, snapshot.CompletedCount);
    }

    [Fact]
    public async Task FixRoundsIncrementAcrossReopens()
    {
        await _store.InitAsync(_planPath, name: "p");
        await _store.AppendAsync("p", 1, TaskJournalState.Complete);
        await _store.AppendAsync("p", 1, TaskJournalState.FixRound, "round one");
        await _store.AppendAsync("p", 1, TaskJournalState.Complete);
        await _store.AppendAsync("p", 1, TaskJournalState.FixRound, "round two");

        var snapshot = await _store.LoadAsync("p");

        Assert.Equal(2, snapshot.Tasks[0].FixRound);
    }

    [Fact]
    public async Task IsComplete_TrueOnlyWhenEveryTaskComplete()
    {
        await _store.InitAsync(_planPath, name: "p");

        for (var i = 1; i <= 3; i++)
        {
            await _store.AppendAsync("p", i, TaskJournalState.Complete);
        }

        var snapshot = await _store.LoadAsync("p");

        Assert.True(snapshot.IsComplete);
        Assert.Null(snapshot.NextTask);
        Assert.Equal(3, snapshot.CompletedCount);
    }

    [Fact]
    public async Task LedgerIsAppendOnly()
    {
        await _store.InitAsync(_planPath, name: "p");
        await _store.AppendAsync("p", 1, TaskJournalState.InProgress);
        var afterFirst = await _store.ReadEntriesAsync("p");

        await _store.AppendAsync("p", 1, TaskJournalState.Complete);
        var afterSecond = await _store.ReadEntriesAsync("p");

        // History must only ever grow. Rewriting an earlier entry would let a resumed agent see a
        // different past than the one that actually happened.
        Assert.Equal(afterFirst, afterSecond.Take(afterFirst.Count));
        Assert.Equal(afterFirst.Count + 1, afterSecond.Count);
    }

    [Fact]
    public async Task AppendAsync_PreservesMultilineNotes()
    {
        await _store.InitAsync(_planPath, name: "p");

        // Review feedback is the main source of notes and is naturally multi-line. The old
        // line-based ledger had to flatten it; nothing forces that loss now.
        const string note = "line one\nline two\nline three";
        await _store.AppendAsync("p", 1, TaskJournalState.Complete, note);

        var snapshot = await _store.LoadAsync("p");

        Assert.Equal(note, snapshot.Tasks[0].LastNote);
    }

    [Fact]
    public async Task AppendAsync_RejectsOutOfRangeTaskNumbers()
    {
        await _store.InitAsync(_planPath, name: "p");

        await Assert.ThrowsAsync<ArgumentOutOfRangeException>(
            () => _store.AppendAsync("p", 0, TaskJournalState.Complete));
        await Assert.ThrowsAsync<ArgumentOutOfRangeException>(
            () => _store.AppendAsync("p", 4, TaskJournalState.Complete));
    }

    [Fact]
    public async Task LoadAsync_ThrowsForUnknownJournal()
    {
        await Assert.ThrowsAsync<TaskJournalNotFoundException>(() => _store.LoadAsync("does-not-exist"));
    }

    [Fact]
    public async Task WriteArtifact_RejectsOutOfRangeTaskNumbers()
    {
        await _store.InitAsync(_planPath, name: "p");

        // An orphan artifact would look like a successful handover but never surface in a snapshot.
        await Assert.ThrowsAsync<ArgumentOutOfRangeException>(
            () => _store.WriteArtifactAsync("p", 9, TaskArtifactKind.Brief, "x"));
        await Assert.ThrowsAsync<ArgumentOutOfRangeException>(
            () => _store.WriteArtifactAsync("p", 0, TaskArtifactKind.Report, "x"));
    }

    [Fact]
    public async Task WriteArtifact_RoundTripsAndIsReflectedInSnapshot()
    {
        await _store.InitAsync(_planPath, name: "p");

        await _store.WriteArtifactAsync("p", 1, TaskArtifactKind.Brief, "do the thing");
        await _store.WriteArtifactAsync("p", 1, TaskArtifactKind.Report, "did the thing");

        Assert.Equal("do the thing", await _store.ReadArtifactAsync("p", 1, TaskArtifactKind.Brief));
        Assert.Equal("did the thing", await _store.ReadArtifactAsync("p", 1, TaskArtifactKind.Report));

        var snapshot = await _store.LoadAsync("p");

        Assert.True(snapshot.Tasks[0].HasBrief);
        Assert.True(snapshot.Tasks[0].HasReport);
        Assert.False(snapshot.Tasks[1].HasBrief);
    }

    [Fact]
    public async Task WriteArtifact_OverwritesPreviousContent()
    {
        await _store.InitAsync(_planPath, name: "p");

        // A fix round rewrites the brief; the stale one must not linger.
        await _store.WriteArtifactAsync("p", 1, TaskArtifactKind.Brief, "first attempt");
        await _store.WriteArtifactAsync("p", 1, TaskArtifactKind.Brief, "second attempt");

        Assert.Equal("second attempt", await _store.ReadArtifactAsync("p", 1, TaskArtifactKind.Brief));
    }

    [Fact]
    public async Task ReadArtifact_ReturnsNullWhenNothingStored()
    {
        await _store.InitAsync(_planPath, name: "p");

        Assert.Null(await _store.ReadArtifactAsync("p", 1, TaskArtifactKind.Brief));
    }

    [Fact]
    public async Task ExportArtifact_WritesFileAndThrowsWhenAbsent()
    {
        await _store.InitAsync(_planPath, name: "p");
        await _store.WriteArtifactAsync("p", 1, TaskArtifactKind.Brief, "handover content");

        var target = Path.Combine(_root, "exported", "brief.md");
        var written = await _store.ExportArtifactAsync("p", 1, TaskArtifactKind.Brief, target);

        Assert.Equal("handover content", await File.ReadAllTextAsync(written));

        // Exporting something that was never written must fail loudly rather than emit an empty file.
        await Assert.ThrowsAsync<InvalidOperationException>(
            () => _store.ExportArtifactAsync("p", 2, TaskArtifactKind.Report, target));
    }

    [Fact]
    public async Task List_ReturnsInitialisedJournals()
    {
        await _store.InitAsync(_planPath, name: "alpha");
        await _store.InitAsync(_planPath, name: "beta");

        var journals = await _store.ListAsync();

        Assert.Equal(new[] { "alpha", "beta" }, journals.OrderBy(j => j, StringComparer.Ordinal));
    }

    [Fact]
    public async Task List_IsEmptyWhenNoJournalsExist()
    {
        Assert.Empty(await _store.ListAsync());
    }

    [Fact]
    public async Task ConcurrentFixRounds_EachGetADistinctRoundNumber()
    {
        await _store.InitAsync(_planPath, name: "p");
        await _store.AppendAsync("p", 1, TaskJournalState.Complete);

        // Two reviewers filing fixes at once is the case a read-then-write ledger loses: both read
        // round 0 and both write round 1, so one round vanishes from the history.
        var stores = Enumerable.Range(0, 2)
            .Select(_ => new TaskJournalStore(Path.Combine(_root, ".agent-journal", "tasks")))
            .ToArray();

        await Task.WhenAll(stores.Select(s => s.AppendAsync("p", 1, TaskJournalState.FixRound, "review")));

        var entries = await _store.ReadEntriesAsync("p");
        var rounds = entries
            .Where(e => e.State == TaskJournalState.FixRound)
            .Select(e => e.Round)
            .OrderBy(r => r)
            .ToArray();

        Assert.Equal(new int?[] { 1, 2 }, rounds);
    }

    [Fact]
    public async Task ConcurrentWritersFromSeparateStores_AllEntriesSurvive()
    {
        await _store.InitAsync(_planPath, taskCount: 20, name: "p");

        // A file-backed ledger throws IOException here when two writers collide on the same handle.
        var writes = Enumerable.Range(1, 20).Select(async taskNumber =>
        {
            var store = new TaskJournalStore(Path.Combine(_root, ".agent-journal", "tasks"));
            await store.AppendAsync("p", taskNumber, TaskJournalState.Complete);
        });

        await Task.WhenAll(writes);

        var snapshot = await _store.LoadAsync("p");

        Assert.Equal(20, snapshot.CompletedCount);
        Assert.True(snapshot.IsComplete);
    }

    [Fact]
    public async Task JournalSurvivesProcessRestart()
    {
        await _store.InitAsync(_planPath, name: "p");
        await _store.AppendAsync("p", 1, TaskJournalState.Complete);

        // A fresh store instance stands in for a new process with no in-memory state.
        var reopened = new TaskJournalStore(Path.Combine(_root, ".agent-journal", "tasks"));
        var snapshot = await reopened.LoadAsync("p");

        Assert.Equal(2, snapshot.NextTask?.Number);
        Assert.Equal(1, snapshot.CompletedCount);
    }

    /// <summary>
    /// An empty handover must fail loudly rather than be recorded as a successful one.
    /// </summary>
    [Theory]
    [InlineData("")]
    [InlineData("   \n\t ")]
    public async Task WriteArtifactAsync_RejectsEmptyContent(string content)
    {
        await _store.InitAsync(_planPath, name: "p");

        await Assert.ThrowsAsync<ArgumentException>(
            () => _store.WriteArtifactAsync("p", 1, TaskArtifactKind.Report, content));

        var snapshot = await _store.LoadAsync("p");
        Assert.False(snapshot.Tasks[0].HasReport);
    }
}
