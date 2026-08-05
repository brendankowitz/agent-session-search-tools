using System.CommandLine;
using System.Data.Common;
using System.Text.Json;
using AgentJournal.Core.Tasks;

namespace AgentJournal.Commands;

/// <summary>
/// Commands for the task journal, which lets an agent resume a multi-task plan after its
/// conversation context is compacted or lost.
/// </summary>
public class TaskCommand : Command
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = true,
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase
    };

    private TaskCommand() : base("task", "Track progress through a multi-task plan so it survives context loss")
    {
    }

    public static Command Create(IServiceProvider serviceProvider)
    {
        var command = new TaskCommand();

        command.AddCommand(InitSubCommand.Create());
        command.AddCommand(StatusSubCommand.Create());
        command.AddCommand(NextSubCommand.Create());
        command.AddCommand(StartSubCommand.Create());
        command.AddCommand(CompleteSubCommand.Create());
        command.AddCommand(FixSubCommand.Create());
        command.AddCommand(ShowSubCommand.Create());
        command.AddCommand(ListSubCommand.Create());

        return command;
    }

    /// <summary>
    /// Resolves the journal to operate on. When no name is given and exactly one journal exists,
    /// that one is used - an agent driving a single plan should not have to repeat its name.
    /// </summary>
    private static async Task<TaskJournalSnapshot?> ResolveAsync(string? name, CancellationToken ct)
    {
        var store = TaskJournalStore.ForRepository();

        if (!string.IsNullOrWhiteSpace(name))
        {
            return await store.LoadAsync(name, ct);
        }

        var journals = await store.ListAsync(ct);

        if (journals.Count == 0)
        {
            Console.Error.WriteLine("Error: No task journals found. Run 'agent-journal task init <plan-file>' first.");
            CommandOutcome.Fail(CommandOutcome.NotFound);
            return null;
        }

        if (journals.Count > 1)
        {
            Console.Error.WriteLine($"Error: {journals.Count} task journals exist; specify one with --name.");
            foreach (var journal in journals)
            {
                Console.Error.WriteLine($"  {journal}");
            }

            CommandOutcome.Fail();
            return null;
        }

        return await store.LoadAsync(journals[0], ct);
    }

    private static Option<string?> NameOption() => new(
        name: "--name",
        description: "Journal name (defaults to the only journal in this repository)");

    private static Option<bool> RobotOption() => new(
        name: "--robot",
        description: "Emit JSON for machine consumption");

    /// <summary>
    /// Reads text from a file path, or from stdin when the path is "-".
    /// </summary>
    private static async Task<string?> ReadContentAsync(string? source, CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(source))
        {
            return null;
        }

        if (source == "-")
        {
            return await Console.In.ReadToEndAsync(ct);
        }

        if (!File.Exists(source))
        {
            Console.Error.WriteLine($"Error: File not found: {source}");
            CommandOutcome.Fail(CommandOutcome.NotFound);
            return null;
        }

        return await File.ReadAllTextAsync(source, ct);
    }

    /// <summary>
    /// Failures a task command can produce through normal use, as opposed to a defect. Anything
    /// outside this set propagates so it is not mistaken for an expected outcome.
    /// </summary>
    private static bool IsExpectedFailure(Exception ex) =>
        ex is TaskJournalNotFoundException
            or InvalidOperationException
            or FileNotFoundException
            or DirectoryNotFoundException
            or ArgumentException
            or IOException
            or UnauthorizedAccessException
            or DbException;

    /// <summary>
    /// Reports a failure and sets the process exit code.
    /// </summary>
    /// <remarks>
    /// The exit code is the machine-readable half of this contract: an agent branches on 2 to mean
    /// "no such journal, initialise one" without having to parse stderr. Collapsing every failure
    /// into 1 - or into 2 - would make that impossible.
    /// </remarks>
    private static void HandleFailure(Exception ex)
    {
        Console.Error.WriteLine($"Error: {DescribeError(ex)}");

        CommandOutcome.Fail(
            ex is TaskJournalNotFoundException or FileNotFoundException or DirectoryNotFoundException
                ? CommandOutcome.NotFound
                : CommandOutcome.GeneralFailure);
    }

    /// <summary>
    /// Trims the framework's argument-name suffix so users see the message rather than
    /// "(Parameter 'taskNumber') Actual value was 9."
    /// </summary>
    private static string DescribeError(Exception ex)
    {
        var message = ex.Message;

        var parameterMarker = message.IndexOf(" (Parameter", StringComparison.Ordinal);
        if (parameterMarker > 0)
        {
            message = message[..parameterMarker];
        }

        return message.ReplaceLineEndings(" ").Trim();
    }

    private static void PrintSnapshot(TaskJournalSnapshot snapshot, bool robot)
    {
        if (robot)
        {
            Console.WriteLine(JsonSerializer.Serialize(ToDto(snapshot), JsonOptions));
            return;
        }

        Console.WriteLine($"Journal: {snapshot.Name}");
        Console.WriteLine($"Plan:    {snapshot.PlanPath}");
        Console.WriteLine($"Store:   {snapshot.DatabasePath}");
        Console.WriteLine($"Progress: {snapshot.CompletedCount}/{snapshot.Tasks.Count} complete");
        Console.WriteLine();

        foreach (var task in snapshot.Tasks)
        {
            var marker = task.State switch
            {
                TaskJournalState.Complete => "[x]",
                TaskJournalState.InProgress => "[~]",
                TaskJournalState.FixRound => "[!]",
                _ => "[ ]"
            };

            var detail = Describe(task);

            Console.WriteLine($"  {marker} Task {task.Number}: {detail}");

            if (!string.IsNullOrWhiteSpace(task.LastNote))
            {
                // Notes are stored verbatim; collapse them only for this single-line view.
                Console.WriteLine($"        note: {task.LastNote.ReplaceLineEndings(" ")}");
            }
        }

        Console.WriteLine();

        if (snapshot.IsComplete)
        {
            Console.WriteLine("All tasks complete.");
        }
        else if (snapshot.NextTask is { } next)
        {
            Console.WriteLine($"Next: Task {next.Number}");
            Console.WriteLine($"  brief:  {(next.HasBrief ? "stored" : "not written yet")}");
            Console.WriteLine($"  report: {(next.HasReport ? "stored" : "not written yet")}");
        }
    }

    private static string Describe(TaskJournalTask task) => task.State switch
    {
        TaskJournalState.FixRound => $"fix round {task.FixRound}",
        TaskJournalState.InProgress => "in progress",
        TaskJournalState.Complete => "complete",
        _ => "pending"
    };

    private static object ToDto(TaskJournalSnapshot snapshot) => new
    {
        name = snapshot.Name,
        planPath = snapshot.PlanPath,
        databasePath = snapshot.DatabasePath,
        totalTasks = snapshot.Tasks.Count,
        completedTasks = snapshot.CompletedCount,
        isComplete = snapshot.IsComplete,
        nextTask = snapshot.NextTask is null ? null : ToDto(snapshot.NextTask),
        tasks = snapshot.Tasks.Select(ToDto).ToList()
    };

    private static object ToDto(TaskJournalTask task) => new
    {
        number = task.Number,
        state = task.State.ToString(),
        fixRound = task.FixRound,
        startedAt = task.StartedAt,
        completedAt = task.CompletedAt,
        lastNote = task.LastNote,
        hasBrief = task.HasBrief,
        hasReport = task.HasReport
    };

    private sealed class InitSubCommand : Command
    {
        private InitSubCommand() : base("init", "Create a task journal for a plan file")
        {
        }

        public static Command Create()
        {
            var command = new InitSubCommand();

            var planArgument = new Argument<string>(
                name: "plan-file",
                description: "Path to the plan file being executed");

            var tasksOption = new Option<int?>(
                name: "--tasks",
                description: "Number of tasks (default: count '## Task N' headings in the plan)");

            var nameOption = NameOption();
            var robotOption = RobotOption();

            command.AddArgument(planArgument);
            command.AddOption(tasksOption);
            command.AddOption(nameOption);
            command.AddOption(robotOption);

            command.SetHandler(async (plan, tasks, name, robot) =>
            {
                try
                {
                    var store = TaskJournalStore.ForRepository();
                    var snapshot = await store.InitAsync(plan, tasks, name, CancellationToken.None);
                    PrintSnapshot(snapshot, robot);
                }
                catch (Exception ex) when (IsExpectedFailure(ex))
                {
                    HandleFailure(ex);
                }
            }, planArgument, tasksOption, nameOption, robotOption);

            return command;
        }
    }

    private sealed class StatusSubCommand : Command
    {
        private StatusSubCommand() : base("status", "Show progress through a plan")
        {
        }

        public static Command Create()
        {
            var command = new StatusSubCommand();
            var nameOption = NameOption();
            var robotOption = RobotOption();

            command.AddOption(nameOption);
            command.AddOption(robotOption);

            command.SetHandler(async (name, robot) =>
            {
                try
                {
                    var snapshot = await ResolveAsync(name, CancellationToken.None);
                    if (snapshot != null)
                    {
                        PrintSnapshot(snapshot, robot);
                    }
                }
                catch (Exception ex) when (IsExpectedFailure(ex))
                {
                    HandleFailure(ex);
                }
            }, nameOption, robotOption);

            return command;
        }
    }

    private sealed class NextSubCommand : Command
    {
        private NextSubCommand() : base("next", "Show the next task to work on")
        {
        }

        public static Command Create()
        {
            var command = new NextSubCommand();
            var nameOption = NameOption();
            var robotOption = RobotOption();

            command.AddOption(nameOption);
            command.AddOption(robotOption);

            command.SetHandler(async (name, robot) =>
            {
                try
                {
                    var snapshot = await ResolveAsync(name, CancellationToken.None);
                    if (snapshot == null)
                    {
                        return;
                    }

                    if (snapshot.NextTask is not { } next)
                    {
                        if (robot)
                        {
                            Console.WriteLine(JsonSerializer.Serialize(new { isComplete = true, nextTask = (object?)null }, JsonOptions));
                        }
                        else
                        {
                            Console.WriteLine("All tasks complete.");
                        }

                        return;
                    }

                    if (robot)
                    {
                        Console.WriteLine(JsonSerializer.Serialize(new
                        {
                            isComplete = false,
                            planPath = snapshot.PlanPath,
                            nextTask = ToDto(next)
                        }, JsonOptions));
                        return;
                    }

                    Console.WriteLine($"Task {next.Number} ({Describe(next)})");
                    Console.WriteLine($"  plan:   {snapshot.PlanPath}");
                    Console.WriteLine($"  brief:  {(next.HasBrief ? "stored" : "not written yet")}");
                    Console.WriteLine($"  report: {(next.HasReport ? "stored" : "not written yet")}");

                    if (next.State == TaskJournalState.FixRound)
                    {
                        Console.WriteLine($"  status: mid-loop, fix round {next.FixRound}");
                        if (!string.IsNullOrWhiteSpace(next.LastNote))
                        {
                            Console.WriteLine($"  note:   {next.LastNote.ReplaceLineEndings(" ")}");
                        }
                    }
                }
                catch (Exception ex) when (IsExpectedFailure(ex))
                {
                    HandleFailure(ex);
                }
            }, nameOption, robotOption);

            return command;
        }
    }

    private sealed class StartSubCommand : Command
    {
        private StartSubCommand() : base("start", "Record that work on a task has started")
        {
        }

        public static Command Create()
        {
            var command = new StartSubCommand();

            var taskArgument = new Argument<int>(name: "task", description: "Task number");
            var briefOption = new Option<string?>(
                name: "--brief",
                description: "File containing the task brief, or '-' to read from stdin");
            var nameOption = NameOption();
            var robotOption = RobotOption();

            command.AddArgument(taskArgument);
            command.AddOption(briefOption);
            command.AddOption(nameOption);
            command.AddOption(robotOption);

            command.SetHandler(async (task, brief, name, robot) =>
            {
                await RecordAsync(name, task, TaskJournalState.InProgress, note: null, brief, isReport: false, robot);
            }, taskArgument, briefOption, nameOption, robotOption);

            return command;
        }
    }

    private sealed class CompleteSubCommand : Command
    {
        private CompleteSubCommand() : base("complete", "Record that a task is finished")
        {
        }

        public static Command Create()
        {
            var command = new CompleteSubCommand();

            var taskArgument = new Argument<int>(name: "task", description: "Task number");
            var reportOption = new Option<string?>(
                name: "--report",
                description: "File containing the task report, or '-' to read from stdin");
            var noteOption = new Option<string?>(name: "--note", description: "Short note for the ledger");
            var nameOption = NameOption();
            var robotOption = RobotOption();

            command.AddArgument(taskArgument);
            command.AddOption(reportOption);
            command.AddOption(noteOption);
            command.AddOption(nameOption);
            command.AddOption(robotOption);

            command.SetHandler(async (task, report, note, name, robot) =>
            {
                await RecordAsync(name, task, TaskJournalState.Complete, note, report, isReport: true, robot);
            }, taskArgument, reportOption, noteOption, nameOption, robotOption);

            return command;
        }
    }

    private sealed class FixSubCommand : Command
    {
        private FixSubCommand() : base("fix", "Open a fix round on a task after review found problems")
        {
        }

        public static Command Create()
        {
            var command = new FixSubCommand();

            var taskArgument = new Argument<int>(name: "task", description: "Task number");
            var noteOption = new Option<string?>(
                name: "--note",
                description: "Why the task was sent back");
            var nameOption = NameOption();
            var robotOption = RobotOption();

            command.AddArgument(taskArgument);
            command.AddOption(noteOption);
            command.AddOption(nameOption);
            command.AddOption(robotOption);

            command.SetHandler(async (task, note, name, robot) =>
            {
                await RecordAsync(name, task, TaskJournalState.FixRound, note, contentSource: null, isReport: false, robot);
            }, taskArgument, noteOption, nameOption, robotOption);

            return command;
        }
    }

    /// <summary>
    /// Reads a stored brief or report back out. Artifacts live in the database rather than at a
    /// predictable file path, so this - or <c>--out</c> - is the only way to retrieve one.
    /// </summary>
    private sealed class ShowSubCommand : Command
    {
        private ShowSubCommand() : base("show", "Print a task's stored brief or report")
        {
        }

        public static Command Create()
        {
            var command = new ShowSubCommand();

            var kindArgument = new Argument<string>(
                name: "kind",
                description: "Which artifact to read: brief or report");

            var taskArgument = new Argument<int>(
                name: "task-number",
                description: "Task number");

            var outOption = new Option<string?>(
                name: "--out",
                description: "Write the artifact to this file instead of stdout");

            var nameOption = NameOption();

            command.AddArgument(kindArgument);
            command.AddArgument(taskArgument);
            command.AddOption(outOption);
            command.AddOption(nameOption);

            command.SetHandler(async (kindText, taskNumber, outPath, name) =>
            {
                var ct = CancellationToken.None;

                try
                {
                    if (!Enum.TryParse<TaskArtifactKind>(kindText, ignoreCase: true, out var kind))
                    {
                        Console.Error.WriteLine($"Error: Unknown artifact kind '{kindText}'. Expected 'brief' or 'report'.");
                        CommandOutcome.Fail();
                        return;
                    }

                    var snapshot = await ResolveAsync(name, ct);
                    if (snapshot == null)
                    {
                        return;
                    }

                    var store = TaskJournalStore.ForRepository();

                    if (!string.IsNullOrWhiteSpace(outPath))
                    {
                        var written = await store.ExportArtifactAsync(snapshot.Name, taskNumber, kind, outPath, ct);
                        Console.WriteLine(written);
                        return;
                    }

                    var content = await store.ReadArtifactAsync(snapshot.Name, taskNumber, kind, ct);

                    if (content == null)
                    {
                        Console.Error.WriteLine(
                            $"Error: Task {taskNumber} has no {kind.ToString().ToLowerInvariant()} stored in journal '{snapshot.Name}'.");
                        CommandOutcome.Fail(CommandOutcome.NotFound);
                        return;
                    }

                    Console.Out.Write(content);
                }
                catch (Exception ex) when (IsExpectedFailure(ex))
                {
                    HandleFailure(ex);
                }
            }, kindArgument, taskArgument, outOption, nameOption);

            return command;
        }
    }

    private sealed class ListSubCommand : Command
    {        private ListSubCommand() : base("list", "List task journals in this repository")
        {
        }

        public static Command Create()
        {
            var command = new ListSubCommand();
            var robotOption = RobotOption();
            command.AddOption(robotOption);

            command.SetHandler(async (bool robot) =>
            {
                try
                {
                    var store = TaskJournalStore.ForRepository();
                    var journals = await store.ListAsync(CancellationToken.None);

                    if (robot)
                    {
                        Console.WriteLine(JsonSerializer.Serialize(new { root = store.TasksRoot, journals }, JsonOptions));
                        return;
                    }

                    if (journals.Count == 0)
                    {
                        Console.WriteLine("No task journals found.");
                        return;
                    }

                    Console.WriteLine($"Task journals in {store.TasksRoot}:");
                    foreach (var journal in journals)
                    {
                        Console.WriteLine($"  {journal}");
                    }
                }
                catch (Exception ex) when (IsExpectedFailure(ex))
                {
                    HandleFailure(ex);
                }
            }, robotOption);

            return command;
        }
    }

    /// <summary>
    /// Shared implementation for the state-recording subcommands: optionally persist a handover
    /// artifact, then append the ledger entry.
    /// </summary>
    private static async Task RecordAsync(
        string? name,
        int taskNumber,
        TaskJournalState state,
        string? note,
        string? contentSource,
        bool isReport,
        bool robot)
    {
        var ct = CancellationToken.None;

        try
        {
            var snapshot = await ResolveAsync(name, ct);
            if (snapshot == null)
            {
                return;
            }

            var store = TaskJournalStore.ForRepository();

            if (!string.IsNullOrWhiteSpace(contentSource))
            {
                var content = await ReadContentAsync(contentSource, ct);
                if (content == null)
                {
                    return;
                }

                var kind = isReport ? TaskArtifactKind.Report : TaskArtifactKind.Brief;
                await store.WriteArtifactAsync(snapshot.Name, taskNumber, kind, content, ct);

                if (!robot)
                {
                    Console.WriteLine($"Stored {kind.ToString().ToLowerInvariant()} for task {taskNumber}.");
                }
            }

            var updated = await store.AppendAsync(snapshot.Name, taskNumber, state, note, ct);
            PrintSnapshot(updated, robot);
        }
        catch (Exception ex) when (IsExpectedFailure(ex))
        {
            HandleFailure(ex);
        }
    }
}
