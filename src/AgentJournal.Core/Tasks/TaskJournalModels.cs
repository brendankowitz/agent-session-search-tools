namespace AgentJournal.Core.Tasks;

/// <summary>
/// Lifecycle state of a single task in a task journal.
/// </summary>
public enum TaskJournalState
{
    /// <summary>No events recorded yet.</summary>
    Pending,

    /// <summary>Work has started but the task has not been marked complete.</summary>
    InProgress,

    /// <summary>
    /// Work was reviewed and sent back for corrections. The task is mid-loop: a subagent is
    /// expected to address the review feedback and report again.
    /// </summary>
    FixRound,

    /// <summary>The task finished and passed review.</summary>
    Complete
}

/// <summary>
/// The kind of handover artifact attached to a task.
/// </summary>
public enum TaskArtifactKind
{
    /// <summary>What a subagent was told to do.</summary>
    Brief,

    /// <summary>What the subagent reported back.</summary>
    Report
}

/// <summary>
/// Raised when a named task journal does not exist.
/// </summary>
/// <remarks>
/// A dedicated type so callers can distinguish "no such journal" - recoverable, and the answer is
/// usually <c>task init</c> - from a genuine storage failure. The CLI maps this to its own exit
/// code so an agent can branch on it without parsing stderr.
/// </remarks>
public sealed class TaskJournalNotFoundException : Exception
{
    /// <summary>Creates the exception for a journal name.</summary>
    public TaskJournalNotFoundException(string journalName)
        : base($"No task journal named '{journalName}'. Run 'task init' first.")
    {
        JournalName = journalName;
    }

    /// <summary>The journal name that was not found.</summary>
    public string JournalName { get; }
}

/// <summary>
/// A single append-only entry in a task journal.
/// </summary>
/// <param name="TaskNumber">The 1-based task this entry refers to.</param>
/// <param name="State">The state the task entered.</param>
/// <param name="Round">Fix round number, set only when <paramref name="State"/> is <see cref="TaskJournalState.FixRound"/>.</param>
/// <param name="Timestamp">When the entry was recorded (UTC).</param>
/// <param name="Note">Optional free-text note, typically why a fix round was opened.</param>
public record TaskJournalEntry(
    int TaskNumber,
    TaskJournalState State,
    int? Round,
    DateTimeOffset Timestamp,
    string? Note);

/// <summary>
/// The derived current state of one task, folded from all of its entries.
/// </summary>
/// <param name="Number">The 1-based task number.</param>
/// <param name="State">Current state.</param>
/// <param name="FixRound">Highest fix round opened so far; 0 when none.</param>
/// <param name="StartedAt">When the task first entered <see cref="TaskJournalState.InProgress"/>.</param>
/// <param name="CompletedAt">When the task was last marked complete, if it was.</param>
/// <param name="LastNote">Note attached to the most recent entry, if any.</param>
/// <param name="HasBrief">True when a brief has been stored for this task.</param>
/// <param name="HasReport">True when a report has been stored for this task.</param>
public record TaskJournalTask(
    int Number,
    TaskJournalState State,
    int FixRound,
    DateTimeOffset? StartedAt,
    DateTimeOffset? CompletedAt,
    string? LastNote,
    bool HasBrief,
    bool HasReport);

/// <summary>
/// A point-in-time view of a task journal: its identity, its tasks, and where to resume.
/// </summary>
/// <param name="Name">Journal name.</param>
/// <param name="PlanPath">Absolute path to the plan file this journal tracks.</param>
/// <param name="DatabasePath">Absolute path to the SQLite database backing the journal.</param>
/// <param name="Tasks">All tasks, ordered by task number.</param>
public record TaskJournalSnapshot(
    string Name,
    string PlanPath,
    string DatabasePath,
    IReadOnlyList<TaskJournalTask> Tasks)
{
    /// <summary>
    /// The task to work on next: the lowest-numbered task that is not complete.
    /// Null when every task is complete.
    /// </summary>
    /// <remarks>
    /// This is the whole point of the journal. Conversation memory does not survive context
    /// compaction or an agent restart, so the resume point must be derivable from storage alone.
    /// </remarks>
    public TaskJournalTask? NextTask => Tasks.FirstOrDefault(t => t.State != TaskJournalState.Complete);

    /// <summary>True when the journal has tasks and all of them are complete.</summary>
    public bool IsComplete => Tasks.Count > 0 && Tasks.All(t => t.State == TaskJournalState.Complete);

    /// <summary>Number of completed tasks.</summary>
    public int CompletedCount => Tasks.Count(t => t.State == TaskJournalState.Complete);
}

/// <summary>
/// A single hit from searching task journal notes and artifacts.
/// </summary>
/// <param name="JournalName">Journal the text was recorded in.</param>
/// <param name="TaskNumber">Task the text belongs to.</param>
/// <param name="Kind">
/// What the text is: <c>note</c> for a progress note, otherwise the artifact kind ("brief" or
/// "report").
/// </param>
/// <param name="Excerpt">Extract of the matching text, centred on the match.</param>
/// <param name="Score">Relevance; larger is a better match.</param>
public record TaskSearchResult(
    string JournalName,
    int TaskNumber,
    string Kind,
    string Excerpt,
    double Score);

