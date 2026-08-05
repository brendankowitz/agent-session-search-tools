using System.Globalization;
using System.Text;
using System.Text.RegularExpressions;
using AgentJournal.Core.Storage;
using Microsoft.Data.Sqlite;

namespace AgentJournal.Core.Tasks;

/// <summary>
/// Reads and writes SQLite-backed task journals.
/// </summary>
/// <remarks>
/// <para>
/// A task journal records how far a multi-task plan has progressed. It exists because an agent's
/// conversation memory does not survive context compaction, a crash, or a handover to a different
/// agent - but durable storage does. Every question an agent needs in order to resume ("which task
/// is next?", "was task 3 sent back for fixes?", "what did the subagent report?") is answered by
/// querying the journal rather than by remembering.
/// </para>
/// <para>
/// State lives in SQLite rather than in a parsed markdown ledger. The distinction matters more
/// than it first appears: with a text ledger, a truncated write, a hand-edit, or an unparseable
/// header degrades into <em>silently wrong</em> answers - a damaged task count reads as zero, so
/// the journal reports "all tasks complete" and the agent stops with work outstanding. Columns
/// cannot be misparsed, and a transaction cannot half-apply.
/// </para>
/// <para>
/// Concurrency is the other reason. Two agents running <c>task fix</c> against the same journal
/// previously both read fix round 1 and both wrote round 2, losing a round. Every mutation here
/// runs inside a <c>BEGIN IMMEDIATE</c> transaction, so the read-compute-write sequence is atomic.
/// </para>
/// <para>
/// Entries are append-only. Folding them into current state happens on read, which keeps the
/// history that makes a partially-finished task diagnosable: a task that was completed and later
/// reopened still shows both events.
/// </para>
/// </remarks>
public sealed class TaskJournalStore
{
    /// <summary>File name of the journal database inside the tasks root.</summary>
    private const string DatabaseFileName = "journals.db";

    /// <summary>
    /// Value stored in the search index's <c>kind</c> column for progress notes, distinguishing
    /// them from artifacts, which are indexed under their own kind ("brief", "report").
    /// </summary>
    private const string EntryKind = "note";

    private readonly string _tasksRoot;
    private readonly string _databasePath;
    private readonly string _connectionString;
    private readonly SemaphoreSlim _schemaLock = new(1, 1);
    private bool _schemaReady;

    /// <summary>
    /// Creates a store over an explicit tasks root directory.
    /// </summary>
    /// <param name="tasksRoot">Directory that holds the journal database.</param>
    public TaskJournalStore(string tasksRoot)
    {
        if (string.IsNullOrWhiteSpace(tasksRoot))
        {
            throw new ArgumentException("Tasks root cannot be null or empty", nameof(tasksRoot));
        }

        _tasksRoot = Path.GetFullPath(tasksRoot);
        _databasePath = Path.Combine(_tasksRoot, DatabaseFileName);
        _connectionString = SqliteConnectionFactory.BuildConnectionString(_databasePath);
    }

    /// <summary>Absolute path to the directory holding the journal database.</summary>
    public string TasksRoot => _tasksRoot;

    /// <summary>Absolute path to the SQLite database backing all journals in this store.</summary>
    public string DatabasePath => _databasePath;

    /// <summary>
    /// Creates a store rooted at the enclosing repository's <c>.agent-journal/tasks</c> directory.
    /// </summary>
    /// <param name="startDirectory">Directory to start searching from; defaults to the current directory.</param>
    /// <exception cref="InvalidOperationException">Thrown when no enclosing repository is found.</exception>
    /// <remarks>
    /// The journal is deliberately repo-local rather than user-global: task state belongs to the
    /// checkout being worked on, so a second agent in the same worktree finds the same journal, and
    /// switching branches or repos does not blend unrelated plans together.
    /// <para>
    /// A missing repository is an error rather than a silent fall back to the current directory.
    /// The fallback meant an MCP server started outside a repo would quietly create a journal in an
    /// arbitrary directory and then report "no journals exist" for the repo that actually had one.
    /// </para>
    /// </remarks>
    public static TaskJournalStore ForRepository(string? startDirectory = null)
    {
        var start = string.IsNullOrWhiteSpace(startDirectory)
            ? Directory.GetCurrentDirectory()
            : startDirectory;

        var repoRoot = FindRepositoryRoot(start)
            ?? throw new InvalidOperationException(
                $"No repository found at or above '{Path.GetFullPath(start)}'. " +
                "Task journals are repo-local; run this from inside a repository or pass an explicit path.");

        return new TaskJournalStore(Path.Combine(repoRoot, ".agent-journal", "tasks"));
    }

    /// <summary>
    /// Walks up from <paramref name="startDirectory"/> looking for a directory containing
    /// <c>.git</c>. Returns null when none is found.
    /// </summary>
    private static string? FindRepositoryRoot(string startDirectory)
    {
        var current = new DirectoryInfo(Path.GetFullPath(startDirectory));

        while (current != null)
        {
            // A linked worktree stores .git as a file rather than a directory, so check for both.
            var gitPath = Path.Combine(current.FullName, ".git");
            if (Directory.Exists(gitPath) || File.Exists(gitPath))
            {
                return current.FullName;
            }

            current = current.Parent;
        }

        return null;
    }

    /// <summary>
    /// Creates a new journal for a plan, or returns the existing one when it already exists.
    /// </summary>
    /// <param name="planPath">Path to the plan file being executed.</param>
    /// <param name="taskCount">
    /// Number of tasks in the plan. When null, the plan file is scanned for headings of the form
    /// <c>## Task N</c>. When supplied for an existing journal whose plan has grown, the count is
    /// raised to match; it is never lowered, because that would hide recorded work.
    /// </param>
    /// <param name="name">Journal name; defaults to the plan file name without its extension.</param>
    /// <param name="ct">Cancellation token.</param>
    /// <exception cref="InvalidOperationException">
    /// Thrown when a journal of the same name already tracks a different plan file. Silently
    /// reusing it would append this plan's progress onto an unrelated journal.
    /// </exception>
    public async Task<TaskJournalSnapshot> InitAsync(
        string planPath,
        int? taskCount = null,
        string? name = null,
        CancellationToken ct = default)
    {
        if (string.IsNullOrWhiteSpace(planPath))
        {
            throw new ArgumentException("Plan path cannot be null or empty", nameof(planPath));
        }

        var journalName = string.IsNullOrWhiteSpace(name)
            ? Slugify(Path.GetFileNameWithoutExtension(planPath))
            : Slugify(name);

        if (string.IsNullOrEmpty(journalName))
        {
            throw new ArgumentException("Could not derive a journal name from the plan path", nameof(planPath));
        }

        // Store the plan path absolute. A relative path would be re-resolved against whatever the
        // current directory happens to be when the journal is next read, which is exactly the
        // situation the journal exists to survive.
        var absolutePlanPath = Path.GetFullPath(planPath);

        await using var connection = await OpenAsync(ct);
        await using var transaction = (SqliteTransaction)await connection.BeginTransactionAsync(ct);

        var existing = await TryReadJournalRowAsync(connection, transaction, journalName, ct);

        if (existing is { } row)
        {
            if (!PlanPathsMatch(row.PlanPath, absolutePlanPath))
            {
                throw new InvalidOperationException(
                    $"Journal '{journalName}' already tracks plan '{row.PlanPath}'. " +
                    $"Use a different --name to track '{absolutePlanPath}' separately.");
            }

            // An explicit count for a plan that has since grown must take effect, otherwise the
            // journal reports "complete" while the new tasks are invisible.
            if (taskCount is { } requested && requested > row.TaskCount)
            {
                await using var update = connection.CreateCommand();
                update.Transaction = transaction;
                update.CommandText = "UPDATE task_journals SET task_count = @count WHERE name = @name;";
                update.Parameters.AddWithValue("@count", requested);
                update.Parameters.AddWithValue("@name", journalName);
                await update.ExecuteNonQueryAsync(ct);
            }

            var reloaded = await ReadSnapshotAsync(connection, transaction, journalName, ct);
            await transaction.CommitAsync(ct);
            return reloaded;
        }

        // Validate the plan exists even when an explicit count is supplied. Otherwise a typo'd path
        // creates a journal pointing at nothing, and the mistake only surfaces much later when an
        // agent tries to read the plan it is supposed to be executing.
        if (!File.Exists(absolutePlanPath))
        {
            throw new FileNotFoundException($"Plan file not found: {absolutePlanPath}", absolutePlanPath);
        }

        var resolvedTaskCount = taskCount ?? await CountTasksInPlanAsync(absolutePlanPath, ct);
        if (resolvedTaskCount <= 0)
        {
            throw new InvalidOperationException(
                $"Could not determine the number of tasks in '{absolutePlanPath}'. " +
                "Pass an explicit task count, or add '## Task N' headings to the plan.");
        }

        await using (var insert = connection.CreateCommand())
        {
            insert.Transaction = transaction;
            insert.CommandText = """
                INSERT INTO task_journals (name, plan_path, task_count, created_at)
                VALUES (@name, @plan, @count, @created);
                """;
            insert.Parameters.AddWithValue("@name", journalName);
            insert.Parameters.AddWithValue("@plan", absolutePlanPath);
            insert.Parameters.AddWithValue("@count", resolvedTaskCount);
            insert.Parameters.AddWithValue("@created", FormatTimestamp(DateTimeOffset.UtcNow));
            await insert.ExecuteNonQueryAsync(ct);
        }

        var snapshot = await ReadSnapshotAsync(connection, transaction, journalName, ct);
        await transaction.CommitAsync(ct);
        return snapshot;
    }

    /// <summary>
    /// Loads a journal and folds its entries into current per-task state.
    /// </summary>
    /// <param name="name">Journal name.</param>
    /// <param name="ct">Cancellation token.</param>
    /// <exception cref="TaskJournalNotFoundException">Thrown when the journal does not exist.</exception>
    public async Task<TaskJournalSnapshot> LoadAsync(string name, CancellationToken ct = default)
    {
        var journalName = Slugify(name);

        await using var connection = await OpenAsync(ct);
        return await ReadSnapshotAsync(connection, transaction: null, journalName, ct);
    }

    /// <summary>
    /// Lists the names of all journals in this store, oldest first.
    /// </summary>
    public async Task<IReadOnlyList<string>> ListAsync(CancellationToken ct = default)
    {
        await using var connection = await OpenAsync(ct);

        await using var command = connection.CreateCommand();
        command.CommandText = "SELECT name FROM task_journals ORDER BY created_at, name;";

        var names = new List<string>();
        await using var reader = await command.ExecuteReaderAsync(ct);
        while (await reader.ReadAsync(ct))
        {
            names.Add(reader.GetString(0));
        }

        return names;
    }

    /// <summary>
    /// Records a state change for a task.
    /// </summary>
    /// <param name="name">Journal name.</param>
    /// <param name="taskNumber">1-based task number.</param>
    /// <param name="state">State the task is entering.</param>
    /// <param name="note">Optional note; newlines are collapsed so notes stay single-line in output.</param>
    /// <param name="ct">Cancellation token.</param>
    /// <remarks>
    /// The whole read-compute-write runs in one immediate transaction. Computing the next fix round
    /// outside a transaction let two concurrent callers both observe round 1 and both write round 2.
    /// </remarks>
    public async Task<TaskJournalSnapshot> AppendAsync(
        string name,
        int taskNumber,
        TaskJournalState state,
        string? note = null,
        CancellationToken ct = default)
    {
        if (taskNumber < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(taskNumber), taskNumber, "Task numbers start at 1.");
        }

        if (state == TaskJournalState.Pending)
        {
            throw new ArgumentException("Pending is the absence of entries and cannot be recorded.", nameof(state));
        }

        var journalName = Slugify(name);

        await using var connection = await OpenAsync(ct);
        await using var transaction = (SqliteTransaction)await connection.BeginTransactionAsync(ct);

        var row = await TryReadJournalRowAsync(connection, transaction, journalName, ct)
            ?? throw new TaskJournalNotFoundException(journalName);

        EnsureTaskInRange(journalName, taskNumber, row.TaskCount);

        var fixRound = 0;
        if (state == TaskJournalState.FixRound)
        {
            await using var maxRound = connection.CreateCommand();
            maxRound.Transaction = transaction;
            maxRound.CommandText = """
                SELECT COALESCE(MAX(fix_round), 0) FROM task_entries
                WHERE journal_name = @name AND task_number = @task;
                """;
            maxRound.Parameters.AddWithValue("@name", journalName);
            maxRound.Parameters.AddWithValue("@task", taskNumber);
            fixRound = Convert.ToInt32(await maxRound.ExecuteScalarAsync(ct), CultureInfo.InvariantCulture) + 1;
        }

        await using (var insert = connection.CreateCommand())
        {
            insert.Transaction = transaction;
            insert.CommandText = """
                INSERT INTO task_entries (journal_name, task_number, state, fix_round, note, created_at)
                VALUES (@name, @task, @state, @round, @note, @created);
                """;
            insert.Parameters.AddWithValue("@name", journalName);
            insert.Parameters.AddWithValue("@task", taskNumber);
            insert.Parameters.AddWithValue("@state", StateToText(state));
            insert.Parameters.AddWithValue("@round", fixRound);
            insert.Parameters.AddWithValue("@note",
                string.IsNullOrWhiteSpace(note) ? DBNull.Value : note.Trim());
            insert.Parameters.AddWithValue("@created", FormatTimestamp(DateTimeOffset.UtcNow));
            await insert.ExecuteNonQueryAsync(ct);
        }

        var snapshot = await ReadSnapshotAsync(connection, transaction, journalName, ct);
        await transaction.CommitAsync(ct);
        return snapshot;
    }

    /// <summary>
    /// Stores a task's brief or report, replacing any previous content.
    /// </summary>
    /// <param name="name">Journal name.</param>
    /// <param name="taskNumber">1-based task number.</param>
    /// <param name="kind">Whether this is the brief or the report.</param>
    /// <param name="content">Artifact body.</param>
    /// <param name="ct">Cancellation token.</param>
    /// <remarks>
    /// Content is stored rather than written to a file so that a brief and the task state it
    /// belongs to commit together, and so a second agent sees the artifact the moment it is
    /// written. Callers that need a file - a subagent told to read a path - use
    /// <see cref="ExportArtifactAsync"/>.
    /// </remarks>
    public async Task WriteArtifactAsync(
        string name,
        int taskNumber,
        TaskArtifactKind kind,
        string content,
        CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(content);

        // An empty artifact is worse than no artifact: presence-based reporting would flip
        // hasBrief/hasReport to true, so a subagent whose report generation failed would look
        // like a successful handover and the coordinator would move on.
        if (string.IsNullOrWhiteSpace(content))
        {
            throw new ArgumentException(
                $"Refusing to store an empty {kind.ToString().ToLowerInvariant()} for task {taskNumber}: " +
                "it would be reported as stored while carrying nothing.",
                nameof(content));
        }

        var journalName = Slugify(name);

        await using var connection = await OpenAsync(ct);
        await using var transaction = (SqliteTransaction)await connection.BeginTransactionAsync(ct);

        var row = await TryReadJournalRowAsync(connection, transaction, journalName, ct)
            ?? throw new TaskJournalNotFoundException(journalName);

        EnsureTaskInRange(journalName, taskNumber, row.TaskCount);

        await using (var upsert = connection.CreateCommand())
        {
            upsert.Transaction = transaction;
            upsert.CommandText = """
                INSERT INTO task_artifacts (journal_name, task_number, kind, content, updated_at)
                VALUES (@name, @task, @kind, @content, @updated)
                ON CONFLICT(journal_name, task_number, kind)
                DO UPDATE SET content = excluded.content, updated_at = excluded.updated_at;
                """;
            upsert.Parameters.AddWithValue("@name", journalName);
            upsert.Parameters.AddWithValue("@task", taskNumber);
            upsert.Parameters.AddWithValue("@kind", kind.ToString().ToLowerInvariant());
            upsert.Parameters.AddWithValue("@content", content);
            upsert.Parameters.AddWithValue("@updated", FormatTimestamp(DateTimeOffset.UtcNow));
            await upsert.ExecuteNonQueryAsync(ct);
        }

        await transaction.CommitAsync(ct);
    }

    /// <summary>
    /// Reads a task's brief or report. Returns null when none has been stored.
    /// </summary>
    public async Task<string?> ReadArtifactAsync(
        string name,
        int taskNumber,
        TaskArtifactKind kind,
        CancellationToken ct = default)
    {
        var journalName = Slugify(name);

        await using var connection = await OpenAsync(ct);

        var row = await TryReadJournalRowAsync(connection, transaction: null, journalName, ct)
            ?? throw new TaskJournalNotFoundException(journalName);

        EnsureTaskInRange(journalName, taskNumber, row.TaskCount);

        await using var command = connection.CreateCommand();
        command.CommandText = """
            SELECT content FROM task_artifacts
            WHERE journal_name = @name AND task_number = @task AND kind = @kind;
            """;
        command.Parameters.AddWithValue("@name", journalName);
        command.Parameters.AddWithValue("@task", taskNumber);
        command.Parameters.AddWithValue("@kind", kind.ToString().ToLowerInvariant());

        var result = await command.ExecuteScalarAsync(ct);
        return result as string;
    }

    /// <summary>
    /// Writes a stored artifact out to a file and returns its absolute path.
    /// </summary>
    /// <remarks>
    /// The coordinator/subagent handover works by passing a reference rather than the content
    /// itself, so a large brief never enters the coordinator's context window. A subagent with tool
    /// access can read the artifact directly; one that can only be handed a file path gets a real
    /// file from here.
    /// </remarks>
    /// <exception cref="InvalidOperationException">Thrown when the artifact has not been written.</exception>
    public async Task<string> ExportArtifactAsync(
        string name,
        int taskNumber,
        TaskArtifactKind kind,
        string outputPath,
        CancellationToken ct = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(outputPath);

        var content = await ReadArtifactAsync(name, taskNumber, kind, ct)
            ?? throw new InvalidOperationException(
                $"Task {taskNumber} has no {kind.ToString().ToLowerInvariant()} stored in journal '{Slugify(name)}'.");

        var absolute = Path.GetFullPath(outputPath);
        var directory = Path.GetDirectoryName(absolute);
        if (!string.IsNullOrEmpty(directory))
        {
            Directory.CreateDirectory(directory);
        }

        await File.WriteAllTextAsync(absolute, content, ct);
        return absolute;
    }

    /// <summary>
    /// Returns every recorded entry for a journal, in insertion order.
    /// </summary>
    public async Task<IReadOnlyList<TaskJournalEntry>> ReadEntriesAsync(
        string name,
        CancellationToken ct = default)
    {
        var journalName = Slugify(name);

        await using var connection = await OpenAsync(ct);

        _ = await TryReadJournalRowAsync(connection, transaction: null, journalName, ct)
            ?? throw new TaskJournalNotFoundException(journalName);

        return await ReadEntriesAsync(connection, transaction: null, journalName, ct);
    }

    /// <summary>
    /// Deletes a journal and everything recorded against it.
    /// </summary>
    /// <returns>True when a journal was removed; false when no such journal existed.</returns>
    public async Task<bool> DeleteAsync(string name, CancellationToken ct = default)
    {
        var journalName = Slugify(name);

        await using var connection = await OpenAsync(ct);

        // Entries and artifacts are removed by ON DELETE CASCADE, which the connection factory
        // enables per connection. Those cascaded deletes fire the search index's delete triggers,
        // so the journal's text leaves the index with it - covered by
        // TaskJournalSearchTests.SearchAsync_DoesNotReturnDeletedJournals, since a deleted journal
        // that still answered searches would send agents to work that no longer exists.
        await using var command = connection.CreateCommand();
        command.CommandText = "DELETE FROM task_journals WHERE name = @name;";
        command.Parameters.AddWithValue("@name", journalName);

        return await command.ExecuteNonQueryAsync(ct) > 0;
    }

    /// <summary>
    /// Searches every journal in this repository's store for text recorded in progress notes and
    /// task artifacts, ranked by relevance.
    /// </summary>
    /// <param name="query">Free text to search for.</param>
    /// <param name="limit">Maximum number of results to return.</param>
    /// <param name="ct">Cancellation token.</param>
    /// <remarks>
    /// This is what makes a journal useful beyond the plan that created it: without it, prior work
    /// can only be retrieved by an agent that already knows the journal's name, which is precisely
    /// the knowledge a fresh agent lacks.
    /// </remarks>
    public async Task<IReadOnlyList<TaskSearchResult>> SearchAsync(
        string query,
        int limit = 10,
        CancellationToken ct = default)
    {
        if (string.IsNullOrWhiteSpace(query))
        {
            throw new ArgumentException("Search query cannot be null or empty", nameof(query));
        }

        if (limit <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(limit), limit, "Limit must be greater than zero.");
        }

        await using var connection = await OpenAsync(ct);

        await using var command = connection.CreateCommand();
        command.CommandText = """
            SELECT journal_name,
                   task_number,
                   kind,
                   snippet(task_search, 2, '', '', '...', 24) AS excerpt,
                   bm25(task_search) AS rank
            FROM task_search
            WHERE task_search MATCH @query
            ORDER BY rank
            LIMIT @limit;
            """;
        command.Parameters.AddWithValue("@query", BuildMatchExpression(query));
        command.Parameters.AddWithValue("@limit", limit);

        var results = new List<TaskSearchResult>();

        await using var reader = await command.ExecuteReaderAsync(ct);
        while (await reader.ReadAsync(ct))
        {
            results.Add(new TaskSearchResult(
                JournalName: reader.GetString(0),
                TaskNumber: reader.GetInt32(1),
                Kind: reader.GetString(2),
                Excerpt: reader.IsDBNull(3) ? string.Empty : reader.GetString(3),
                // bm25 returns increasingly negative values as relevance rises; invert so callers
                // can treat a larger score as a better match, matching every other search path.
                Score: -reader.GetDouble(4)));
        }

        return results;
    }

    /// <summary>
    /// Rebuilds the search index from the notes and artifacts it indexes.
    /// </summary>
    /// <remarks>
    /// The triggers keep the index correct in normal operation, so this exists for recovery: a
    /// database restored from a partial copy, or one written by a build that predates the index.
    /// </remarks>
    public async Task RebuildSearchIndexAsync(CancellationToken ct = default)
    {
        await using var connection = await OpenAsync(ct);
        await BackfillSearchIndexAsync(connection, ct);
    }

    /// <summary>
    /// Converts free text into an FTS5 MATCH expression.
    /// </summary>
    /// <remarks>
    /// Raw user input cannot be passed to MATCH: characters such as <c>"</c>, <c>*</c>, <c>:</c>
    /// and <c>-</c> are query operators, so an ordinary phrase like <c>fix: don't</c> raises a
    /// syntax error instead of searching. Each whitespace-separated term is quoted, which makes it
    /// a literal, and the terms are ANDed.
    /// </remarks>
    private static string BuildMatchExpression(string query)
    {
        var terms = query
            .Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Select(term => '"' + term.Replace("\"", "\"\"", StringComparison.Ordinal) + '"');

        return string.Join(' ', terms);
    }

    // ---------------------------------------------------------------------------------------
    // Storage
    // ---------------------------------------------------------------------------------------

    private async Task<SqliteConnection> OpenAsync(CancellationToken ct)
    {
        await EnsureSchemaAsync(ct);
        return await SqliteConnectionFactory.OpenAsync(_connectionString, ct);
    }

    private async Task EnsureSchemaAsync(CancellationToken ct)
    {
        if (_schemaReady)
        {
            return;
        }

        await _schemaLock.WaitAsync(ct);
        try
        {
            if (_schemaReady)
            {
                return;
            }

            Directory.CreateDirectory(_tasksRoot);

            await using var connection = await SqliteConnectionFactory.OpenAsync(_connectionString, ct);

            await using var command = connection.CreateCommand();
            command.CommandText = """
                PRAGMA journal_mode=WAL;

                CREATE TABLE IF NOT EXISTS task_journals (
                    name        TEXT PRIMARY KEY,
                    plan_path   TEXT NOT NULL,
                    task_count  INTEGER NOT NULL,
                    created_at  TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS task_entries (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    journal_name  TEXT NOT NULL REFERENCES task_journals(name) ON DELETE CASCADE,
                    task_number   INTEGER NOT NULL,
                    state         TEXT NOT NULL,
                    fix_round     INTEGER NOT NULL DEFAULT 0,
                    note          TEXT,
                    created_at    TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_task_entries_lookup
                    ON task_entries (journal_name, task_number, id);

                CREATE TABLE IF NOT EXISTS task_artifacts (
                    journal_name  TEXT NOT NULL REFERENCES task_journals(name) ON DELETE CASCADE,
                    task_number   INTEGER NOT NULL,
                    kind          TEXT NOT NULL,
                    content       TEXT NOT NULL,
                    updated_at    TEXT NOT NULL,
                    PRIMARY KEY (journal_name, task_number, kind)
                );
                """;
            await command.ExecuteNonQueryAsync(ct);

            await EnsureSearchIndexAsync(connection, ct);

            _schemaReady = true;
        }
        finally
        {
            _schemaLock.Release();
        }
    }

    /// <summary>
    /// Creates the full-text search index over notes and artifacts, and backfills it from any rows
    /// that already exist.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The index lives inside the journal database rather than in the global Lucene index, because
    /// journals are repo-local while that index is user-global. Indexing repo-local content
    /// globally would mean <c>index --rebuild</c> run in one repository wiped another repository's
    /// task documents with no way to restore them - the source rows live in the other repo's
    /// database, which the rebuilding process cannot see.
    /// </para>
    /// <para>
    /// Synchronisation is done by triggers rather than in application code so that a write and its
    /// index update share one transaction. Index-on-write in C# would let a crash between the two
    /// leave a note that exists but cannot be found.
    /// </para>
    /// </remarks>
    private static async Task EnsureSearchIndexAsync(SqliteConnection connection, CancellationToken ct)
    {
        await using (var probe = connection.CreateCommand())
        {
            probe.CommandText =
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'task_search';";
            if (await probe.ExecuteScalarAsync(ct) != null)
            {
                return;
            }
        }

        await using var create = connection.CreateCommand();
        create.CommandText = $"""
            CREATE VIRTUAL TABLE task_search USING fts5(
                journal_name,
                kind,
                body,
                task_number UNINDEXED,
                ref UNINDEXED,
                tokenize = 'porter unicode61'
            );

            CREATE TRIGGER task_entries_search_ai AFTER INSERT ON task_entries
            WHEN new.note IS NOT NULL AND trim(new.note) <> ''
            BEGIN
                INSERT INTO task_search (journal_name, kind, body, task_number, ref)
                VALUES (new.journal_name, '{EntryKind}', new.note, new.task_number, 'e:' || new.id);
            END;

            CREATE TRIGGER task_entries_search_ad AFTER DELETE ON task_entries
            BEGIN
                DELETE FROM task_search WHERE ref = 'e:' || old.id;
            END;

            CREATE TRIGGER task_artifacts_search_ai AFTER INSERT ON task_artifacts
            BEGIN
                INSERT INTO task_search (journal_name, kind, body, task_number, ref)
                VALUES (new.journal_name, new.kind, new.content, new.task_number,
                        'a:' || new.journal_name || ':' || new.task_number || ':' || new.kind);
            END;

            CREATE TRIGGER task_artifacts_search_au AFTER UPDATE ON task_artifacts
            BEGIN
                DELETE FROM task_search
                    WHERE ref = 'a:' || old.journal_name || ':' || old.task_number || ':' || old.kind;
                INSERT INTO task_search (journal_name, kind, body, task_number, ref)
                VALUES (new.journal_name, new.kind, new.content, new.task_number,
                        'a:' || new.journal_name || ':' || new.task_number || ':' || new.kind);
            END;

            CREATE TRIGGER task_artifacts_search_ad AFTER DELETE ON task_artifacts
            BEGIN
                DELETE FROM task_search
                    WHERE ref = 'a:' || old.journal_name || ':' || old.task_number || ':' || old.kind;
            END;
            """;
        await create.ExecuteNonQueryAsync(ct);

        // Databases created before the index existed already hold notes and artifacts, and the
        // triggers above only fire on future writes.
        await BackfillSearchIndexAsync(connection, ct);
    }

    /// <summary>
    /// Repopulates the search index from the rows it indexes. Safe to run repeatedly.
    /// </summary>
    private static async Task BackfillSearchIndexAsync(SqliteConnection connection, CancellationToken ct)
    {
        await using var backfill = connection.CreateCommand();
        backfill.CommandText = $"""
            DELETE FROM task_search;

            INSERT INTO task_search (journal_name, kind, body, task_number, ref)
            SELECT journal_name, '{EntryKind}', note, task_number, 'e:' || id
            FROM task_entries
            WHERE note IS NOT NULL AND trim(note) <> '';

            INSERT INTO task_search (journal_name, kind, body, task_number, ref)
            SELECT journal_name, kind, content, task_number,
                   'a:' || journal_name || ':' || task_number || ':' || kind
            FROM task_artifacts;
            """;
        await backfill.ExecuteNonQueryAsync(ct);
    }

    private sealed record JournalRow(string Name, string PlanPath, int TaskCount);

    private static async Task<JournalRow?> TryReadJournalRowAsync(
        SqliteConnection connection,
        SqliteTransaction? transaction,
        string journalName,
        CancellationToken ct)
    {
        await using var command = connection.CreateCommand();
        command.Transaction = transaction;
        command.CommandText = "SELECT name, plan_path, task_count FROM task_journals WHERE name = @name;";
        command.Parameters.AddWithValue("@name", journalName);

        await using var reader = await command.ExecuteReaderAsync(ct);
        if (!await reader.ReadAsync(ct))
        {
            return null;
        }

        return new JournalRow(reader.GetString(0), reader.GetString(1), reader.GetInt32(2));
    }

    private static async Task<IReadOnlyList<TaskJournalEntry>> ReadEntriesAsync(
        SqliteConnection connection,
        SqliteTransaction? transaction,
        string journalName,
        CancellationToken ct)
    {
        await using var command = connection.CreateCommand();
        command.Transaction = transaction;
        command.CommandText = """
            SELECT task_number, state, fix_round, note, created_at
            FROM task_entries
            WHERE journal_name = @name
            ORDER BY id;
            """;
        command.Parameters.AddWithValue("@name", journalName);

        var entries = new List<TaskJournalEntry>();
        await using var reader = await command.ExecuteReaderAsync(ct);
        while (await reader.ReadAsync(ct))
        {
            var state = TextToState(reader.GetString(1));
            var round = reader.GetInt32(2);

            entries.Add(new TaskJournalEntry(
                TaskNumber: reader.GetInt32(0),
                State: state,
                Round: state == TaskJournalState.FixRound ? round : null,
                Timestamp: ParseTimestamp(reader.GetString(4)),
                Note: reader.IsDBNull(3) ? null : reader.GetString(3)));
        }

        return entries;
    }

    private async Task<TaskJournalSnapshot> ReadSnapshotAsync(
        SqliteConnection connection,
        SqliteTransaction? transaction,
        string journalName,
        CancellationToken ct)
    {
        var row = await TryReadJournalRowAsync(connection, transaction, journalName, ct)
            ?? throw new TaskJournalNotFoundException(journalName);

        var entries = await ReadEntriesAsync(connection, transaction, journalName, ct);
        var artifacts = await ReadArtifactPresenceAsync(connection, transaction, journalName, ct);

        var tasks = new List<TaskJournalTask>(row.TaskCount);
        for (var number = 1; number <= row.TaskCount; number++)
        {
            tasks.Add(FoldTask(
                number,
                entries,
                hasBrief: artifacts.Contains((number, TaskArtifactKind.Brief)),
                hasReport: artifacts.Contains((number, TaskArtifactKind.Report))));
        }

        return new TaskJournalSnapshot(row.Name, row.PlanPath, _databasePath, tasks);
    }

    private static async Task<HashSet<(int Task, TaskArtifactKind Kind)>> ReadArtifactPresenceAsync(
        SqliteConnection connection,
        SqliteTransaction? transaction,
        string journalName,
        CancellationToken ct)
    {
        await using var command = connection.CreateCommand();
        command.Transaction = transaction;
        command.CommandText = "SELECT task_number, kind FROM task_artifacts WHERE journal_name = @name;";
        command.Parameters.AddWithValue("@name", journalName);

        var present = new HashSet<(int, TaskArtifactKind)>();
        await using var reader = await command.ExecuteReaderAsync(ct);
        while (await reader.ReadAsync(ct))
        {
            if (Enum.TryParse<TaskArtifactKind>(reader.GetString(1), ignoreCase: true, out var kind))
            {
                present.Add((reader.GetInt32(0), kind));
            }
        }

        return present;
    }

    // ---------------------------------------------------------------------------------------
    // Folding and helpers
    // ---------------------------------------------------------------------------------------

    private static TaskJournalTask FoldTask(
        int number,
        IReadOnlyList<TaskJournalEntry> entries,
        bool hasBrief,
        bool hasReport)
    {
        var state = TaskJournalState.Pending;
        var fixRound = 0;
        DateTimeOffset? startedAt = null;
        DateTimeOffset? completedAt = null;
        string? lastNote = null;

        foreach (var entry in entries.Where(e => e.TaskNumber == number))
        {
            // Entries are applied in insertion order, so the last entry wins. This is what makes a
            // reopened task - complete, then a later fix round - resolve to "not done".
            state = entry.State;
            lastNote = entry.Note;

            switch (entry.State)
            {
                case TaskJournalState.InProgress:
                    startedAt ??= entry.Timestamp;
                    break;
                case TaskJournalState.Complete:
                    completedAt = entry.Timestamp;
                    break;
                case TaskJournalState.FixRound:
                    fixRound = Math.Max(fixRound, entry.Round ?? fixRound + 1);
                    break;
            }
        }

        return new TaskJournalTask(
            Number: number,
            State: state,
            FixRound: fixRound,
            StartedAt: startedAt,
            CompletedAt: completedAt,
            LastNote: lastNote,
            HasBrief: hasBrief,
            HasReport: hasReport);
    }

    /// <summary>
    /// Guards against entries and artifacts for tasks the plan does not contain. Such rows would
    /// never appear in a snapshot, so the write would appear to succeed but have no effect.
    /// </summary>
    private static void EnsureTaskInRange(string name, int taskNumber, int taskCount)
    {
        if (taskNumber < 1 || taskNumber > taskCount)
        {
            throw new ArgumentOutOfRangeException(
                nameof(taskNumber),
                taskNumber,
                $"Journal '{name}' tracks {taskCount} tasks.");
        }
    }

    private static string StateToText(TaskJournalState state) => state switch
    {
        TaskJournalState.InProgress => "started",
        TaskJournalState.Complete => "complete",
        TaskJournalState.FixRound => "fix",
        _ => throw new ArgumentOutOfRangeException(nameof(state), state, "Unsupported state.")
    };

    private static TaskJournalState TextToState(string text) => text switch
    {
        "started" => TaskJournalState.InProgress,
        "complete" => TaskJournalState.Complete,
        "fix" => TaskJournalState.FixRound,
        _ => throw new InvalidOperationException(
            $"Unrecognised task state '{text}' in the journal database.")
    };

    private static string FormatTimestamp(DateTimeOffset value) =>
        value.UtcDateTime.ToString("yyyy-MM-ddTHH:mm:ss.fffZ", CultureInfo.InvariantCulture);

    private static DateTimeOffset ParseTimestamp(string value) =>
        DateTimeOffset.TryParse(
            value,
            CultureInfo.InvariantCulture,
            DateTimeStyles.AssumeUniversal | DateTimeStyles.AdjustToUniversal,
            out var parsed)
            ? parsed
            : DateTimeOffset.MinValue;

    /// <summary>
    /// Counts <c>## Task N</c> style headings in a plan file.
    /// </summary>
    /// <remarks>
    /// Returns the number of distinct tasks, not the highest number seen. A gap in the numbering is
    /// rejected rather than filled in: a phantom task has no plan section and no brief, so nothing
    /// can ever complete it and <c>next</c> would hand it to an agent forever.
    /// </remarks>
    private static async Task<int> CountTasksInPlanAsync(string planPath, CancellationToken ct)
    {
        if (!File.Exists(planPath))
        {
            throw new FileNotFoundException($"Plan file not found: {planPath}", planPath);
        }

        var headingPattern = new Regex(@"^#{1,6}\s+Task\s+(\d+)\b", RegexOptions.IgnoreCase);
        var numbers = new SortedSet<int>();
        var inFencedBlock = false;

        foreach (var line in await File.ReadAllLinesAsync(planPath, ct))
        {
            var trimmed = line.Trim();

            // Headings inside a fenced block are prose about a task, not a task.
            if (trimmed.StartsWith("```", StringComparison.Ordinal) ||
                trimmed.StartsWith("~~~", StringComparison.Ordinal))
            {
                inFencedBlock = !inFencedBlock;
                continue;
            }

            if (inFencedBlock)
            {
                continue;
            }

            var match = headingPattern.Match(trimmed);
            if (match.Success && int.TryParse(match.Groups[1].Value, out var n) && n > 0)
            {
                numbers.Add(n);
            }
        }

        if (numbers.Count == 0)
        {
            return 0;
        }

        var missing = Enumerable.Range(1, numbers.Max).Except(numbers).ToList();
        if (missing.Count > 0)
        {
            throw new InvalidOperationException(
                $"Plan '{planPath}' has Task headings numbered up to {numbers.Max} but is missing " +
                $"Task {string.Join(", ", missing)}. Renumber the plan so tasks run 1..N, " +
                "or pass an explicit task count.");
        }

        return numbers.Count;
    }

    private static bool PlanPathsMatch(string a, string b)
    {
        static string Normalize(string p) => Path.GetFullPath(p).TrimEnd(Path.DirectorySeparatorChar);

        try
        {
            return string.Equals(Normalize(a), Normalize(b), StringComparison.OrdinalIgnoreCase);
        }
        catch (Exception ex) when (ex is ArgumentException or NotSupportedException or PathTooLongException)
        {
            // One of the paths is not a valid filesystem path (for example a URL recorded by hand).
            // Fall back to a literal comparison rather than treating them as a match.
            return string.Equals(a, b, StringComparison.OrdinalIgnoreCase);
        }
    }

    /// <summary>
    /// Normalises a journal name so that the same plan resolves to the same journal regardless of
    /// how the caller cased or spaced it.
    /// </summary>
    private static string Slugify(string value)
    {
        if (string.IsNullOrWhiteSpace(value))
        {
            return string.Empty;
        }

        var invalid = Path.GetInvalidFileNameChars();
        var builder = new StringBuilder(value.Length);

        foreach (var ch in value.Trim())
        {
            if (invalid.Contains(ch) || char.IsWhiteSpace(ch))
            {
                builder.Append('-');
            }
            else
            {
                builder.Append(char.ToLowerInvariant(ch));
            }
        }

        return builder.ToString().Trim('-');
    }
}
