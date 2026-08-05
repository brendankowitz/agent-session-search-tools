using System.CommandLine;
using System.Text.Json;
using Microsoft.Extensions.DependencyInjection;
using AgentJournal.Configuration;
using AgentJournal.Core.Search;
using AgentJournal.Core.Storage;
using AgentJournal.Core.Models;
using AgentJournal.Core.Knowledge;
using AgentJournal.Core.Tasks;

namespace AgentJournal.Commands;

/// <summary>
/// Command to search indexed sessions
/// </summary>
public class SearchCommand : Command
{
    /// <summary>
    /// Caps how many messages are printed per result so a long context expansion cannot bury the
    /// rest of the result list.
    /// </summary>
    private const int MaxRenderedMessages = 8;
    private readonly Argument<string> _queryArgument;
    private readonly Option<string> _modeOption;
    private readonly Option<int> _contextOption;
    private readonly Option<int> _maxResultsOption;
    private readonly Option<string?> _agentOption;
    private readonly Option<string?> _projectOption;
    private readonly Option<bool> _robotOption;
    private readonly Option<bool> _includeKnowledgeOption;
    private readonly Option<bool> _includeTasksOption;

    private SearchCommand() : base("search", "Search indexed agent sessions")
    {
        _queryArgument = new Argument<string>(
            name: "query",
            description: "Search query");

        _modeOption = new Option<string>(
            name: "--mode",
            getDefaultValue: () => "lexical",
            description: "Search mode: lexical, semantic, or hybrid");
        _modeOption.AddAlias("-m");

        _contextOption = new Option<int>(
            name: "--context",
            getDefaultValue: () => 3,
            description: "Number of surrounding messages to include in results");
        _contextOption.AddAlias("-c");

        _maxResultsOption = new Option<int>(
            name: "--max",
            getDefaultValue: () => 20,
            description: "Maximum number of results to return");
        _maxResultsOption.AddAlias("-n");

        _agentOption = new Option<string?>(
            name: "--agent",
            description: "Filter by agent type (claude-code, copilot-cli)");
        _agentOption.AddAlias("-a");

        _projectOption = new Option<string?>(
            name: "--project",
            description: "Filter by project path");
        _projectOption.AddAlias("-p");

        _robotOption = new Option<bool>(
            name: "--robot",
            description: "Output results as JSON for scripting");
        _robotOption.AddAlias("-r");

        _includeKnowledgeOption = new Option<bool>(
            name: "--include-knowledge",
            getDefaultValue: () => false,
            description: "Include knowledge entries in search results");
        _includeKnowledgeOption.AddAlias("-k");

        _includeTasksOption = new Option<bool>(
            name: "--include-tasks",
            getDefaultValue: () => false,
            description: "Include task journal notes and artifacts from the current repository");
        _includeTasksOption.AddAlias("-t");

        this.AddArgument(_queryArgument);
        this.AddOption(_modeOption);
        this.AddOption(_contextOption);
        this.AddOption(_maxResultsOption);
        this.AddOption(_agentOption);
        this.AddOption(_projectOption);
        this.AddOption(_robotOption);
        this.AddOption(_includeKnowledgeOption);
        this.AddOption(_includeTasksOption);
    }

    public static Command Create(IServiceProvider serviceProvider)
    {
        var command = new SearchCommand();

        // Bound by name through the parse result rather than positionally: this command has more
        // options than the positional SetHandler overloads accept, and positional binding silently
        // mis-maps arguments when the list is reordered.
        command.SetHandler(async context =>
        {
            var parsed = context.ParseResult;
            var configService = serviceProvider.GetRequiredService<ConfigurationService>();
            var searchEngine = serviceProvider.GetRequiredService<ISearchEngine>();
            var repository = serviceProvider.GetRequiredService<ISessionRepository>();
            var knowledgeRepo = serviceProvider.GetService<IKnowledgeRepository>();

            await ExecuteAsync(
                parsed.GetValueForArgument(command._queryArgument),
                parsed.GetValueForOption(command._modeOption),
                parsed.GetValueForOption(command._contextOption),
                parsed.GetValueForOption(command._maxResultsOption),
                parsed.GetValueForOption(command._agentOption),
                parsed.GetValueForOption(command._projectOption),
                parsed.GetValueForOption(command._robotOption),
                parsed.GetValueForOption(command._includeKnowledgeOption),
                parsed.GetValueForOption(command._includeTasksOption),
                configService,
                searchEngine,
                repository,
                knowledgeRepo,
                context.GetCancellationToken());
        });

        return command;
    }

    private static async Task ExecuteAsync(
        string query,
        string? mode,
        int contextCount,
        int maxResults,
        string? agentType,
        string? project,
        bool robot,
        bool includeKnowledge,
        bool includeTasks,
        ConfigurationService configService,
        ISearchEngine searchEngine,
        ISessionRepository repository,
        IKnowledgeRepository? knowledgeRepository,
        CancellationToken ct)
    {
        var config = await configService.LoadConfigAsync(ct);

        // Validate and clamp parameters
        maxResults = Math.Clamp(maxResults, 1, 1000);
        contextCount = Math.Clamp(contextCount, 0, 50);

        // Parse search mode
        var searchMode = mode?.ToLowerInvariant() switch
        {
            "semantic" => SearchMode.Semantic,
            "hybrid" => SearchMode.Hybrid,
            _ => SearchMode.Lexical
        };

        if (includeKnowledge && searchMode != SearchMode.Lexical)
        {
            // The knowledge bank is FTS5-only, so the knowledge half of these results is lexical
            // regardless of the requested mode. Say so rather than letting "Mode: Semantic" imply
            // otherwise.
            Console.Error.WriteLine(
                $"Note: knowledge entries are matched lexically (FTS5); '{searchMode}' mode applies to sessions only.");
        }

        if (includeTasks && searchMode != SearchMode.Lexical)
        {
            // Same caveat as the knowledge bank: task journals are FTS5-only.
            Console.Error.WriteLine(
                $"Note: task journals are matched lexically (FTS5); '{searchMode}' mode applies to sessions only.");
        }

        if (!robot)
        {
            Console.WriteLine($"Searching for: \"{query}\"");
            Console.WriteLine($"Mode: {searchMode}");
            if (includeKnowledge)
            {
                Console.WriteLine("Including: Knowledge entries");
            }
            if (includeTasks)
            {
                Console.WriteLine("Including: Task journals (current repository)");
            }
            Console.WriteLine();
        }

        // Collect results per source. Sources are merged by rank, not by raw score, so each list
        // is kept separate until fusion.
        var resultsBySource = new List<List<UnifiedSearchResult>>();

        // Execute session search
        var sessionResults = await searchEngine.SearchAsync(query, searchMode, maxResults, contextCount, ct);

        // Filter session results by agent type and project if specified
        var filteredSessionResults = sessionResults.AsEnumerable();

        if (!string.IsNullOrWhiteSpace(agentType))
        {
            filteredSessionResults = filteredSessionResults.Where(r =>
                r.Session.AgentType.Equals(agentType, StringComparison.OrdinalIgnoreCase));
        }

        if (!string.IsNullOrWhiteSpace(project))
        {
            filteredSessionResults = filteredSessionResults.Where(r =>
                r.Session.ProjectPath?.Contains(project, StringComparison.OrdinalIgnoreCase) == true);
        }

        // Add session results to unified collection
        resultsBySource.Add(filteredSessionResults.Select(UnifiedSearchResult.FromSession).ToList());

        // Execute knowledge search if requested
        if (includeKnowledge)
        {
            if (knowledgeRepository == null)
            {
                // stderr, not stdout: --robot callers parse stdout as JSON.
                Console.Error.WriteLine("Warning: knowledge repository not available; results cover sessions only.");
                CommandOutcome.Fail(CommandOutcome.PartialFailure);
            }
            else
            {
                try
                {
                    var knowledgeResults = await knowledgeRepository.SearchAsync(
                        query,
                        tags: null,
                        project: project,
                        mode: searchMode,
                        maxResults: maxResults,
                        ct: ct);

                    // Add knowledge results to unified collection
                    resultsBySource.Add(knowledgeResults.Select(UnifiedSearchResult.FromKnowledge).ToList());
                }
                catch (Exception ex) when (ex is not OperationCanceledException)
                {
                    // Always report, and set a non-zero exit code: a caller that only checks the
                    // exit code would otherwise treat a half-sourced answer as complete.
                    Console.Error.WriteLine($"Warning: knowledge search failed: {ex.Message}");
                    Console.Error.WriteLine("Results below cover sessions only.");
                    CommandOutcome.Fail(CommandOutcome.PartialFailure);
                }
            }
        }

        // Execute task journal search if requested. Unlike sessions and knowledge, task journals
        // live in the repository being worked on, so this half of the search is repo-scoped and is
        // simply unavailable outside a repository.
        if (includeTasks)
        {
            try
            {
                var taskStore = TaskJournalStore.ForRepository(Directory.GetCurrentDirectory());
                var taskResults = await taskStore.SearchAsync(query, maxResults, ct);
                resultsBySource.Add(taskResults.Select(UnifiedSearchResult.FromTask).ToList());
            }
            catch (Exception ex) when (ex is not OperationCanceledException)
            {
                // Non-zero exit code for the same reason as knowledge: the caller asked for task
                // coverage and did not get it.
                Console.Error.WriteLine($"Warning: task journal search failed: {ex.Message}");
                Console.Error.WriteLine("Results below do not include task journals.");
                CommandOutcome.Fail(CommandOutcome.PartialFailure);
            }
        }

        // Merge the sources by rank rather than by raw score. Lucene relevance and FTS5 bm25 are
        // on different scales - in a small per-repository corpus bm25 collapses towards zero - so
        // sorting the combined list by Score buries exact task and knowledge matches underneath
        // weak session matches. Reciprocal Rank Fusion compares positions instead of magnitudes.
        // With a single source it reproduces that source's order exactly, so an ordinary session
        // search is unaffected.
        var finalResults = RankFusion.Fuse(resultsBySource, maxResults);

        if (robot)
        {
            // Output as JSON for scripting
            var jsonResults = finalResults.Select<UnifiedSearchResult, object?>(r =>
            {
                if (r.Type == SearchResultType.Session && r.TryGetSession(out var session))
                {
                    return new
                    {
                        type = "session",
                        sessionId = session!.Id,
                        agentType = session.AgentType,
                        projectPath = session.ProjectPath,
                        startedAt = session.StartedAt,
                        messageCount = session.MessageCount,
                        score = r.Score,
                        highlight = r.Highlight,
                        matchingMessages = r.MatchingMessages?.Select(m => new
                        {
                            role = m.Role.ToString(),
                            content = m.Content,
                            timestamp = m.Timestamp
                        }).ToList()
                    };
                }
                else if (r.Type == SearchResultType.Knowledge && r.TryGetKnowledge(out var entry))
                {
                    return new
                    {
                        type = "knowledge",
                        id = entry!.Id,
                        content = entry.Content,
                        tags = entry.Tags,
                        project = entry.Project,
                        source = entry.Source,
                        createdAt = entry.CreatedAt,
                        lastReinforcedAt = entry.LastReinforcedAt,
                        reinforcementCount = entry.ReinforcementCount,
                        score = r.Score,
                        decayFactor = r.DecayFactor,
                        highlight = r.Highlight
                    };
                }
                else if (r.Type == SearchResultType.Task && r.TryGetTask(out var task))
                {
                    return new
                    {
                        type = "task",
                        journalName = task!.JournalName,
                        taskNumber = task.TaskNumber,
                        kind = task.Kind,
                        excerpt = task.Excerpt,
                        score = r.Score
                    };
                }
                return null;
            }).Where(r => r != null);

            var json = JsonSerializer.Serialize(jsonResults, new JsonSerializerOptions
            {
                WriteIndented = true
            });
            Console.WriteLine(json);
        }
        else
        {
            // Human-readable output
            if (finalResults.Count == 0)
            {
                Console.WriteLine("No results found.");
                return;
            }

            Console.WriteLine($"Found {finalResults.Count} result(s):\n");

            for (int i = 0; i < finalResults.Count; i++)
            {
                var result = finalResults[i];

                if (result.Type == SearchResultType.Session && result.TryGetSession(out var session))
                {
                    DisplaySessionResult(i + 1, session!, result, contextCount);
                }
                else if (result.Type == SearchResultType.Knowledge && result.TryGetKnowledge(out var entry))
                {
                    DisplayKnowledgeResult(i + 1, entry!, result);
                }
                else if (result.Type == SearchResultType.Task && result.TryGetTask(out var task))
                {
                    DisplayTaskResult(i + 1, task!);
                }

                Console.WriteLine();
            }
        }
    }

    /// <summary>
    /// Displays a session search result
    /// </summary>
    private static void DisplaySessionResult(int index, Session session, UnifiedSearchResult result, int contextCount)
    {
        Console.WriteLine($"[{index}] Session: {session.Id}");
        Console.WriteLine($"    Agent: {session.AgentType}");
        Console.WriteLine($"    Score: {result.Score:F2}");

        if (!string.IsNullOrWhiteSpace(session.ProjectPath))
        {
            Console.WriteLine($"    Project: {session.ProjectPath}");
        }

        Console.WriteLine($"    Started: {session.StartedAt:yyyy-MM-dd HH:mm:ss}");

        if (result.MatchingMessages != null && result.MatchingMessages.Count > 0)
        {
            // MatchingMessages is already context-expanded, so label it accordingly and mark which
            // entries actually matched. Rendering it as "Matching messages" reported surrounding
            // context as matches, leaving no way to tell what the query hit.
            var header = contextCount > 0 ? "Matching messages (with context):" : "Matching messages:";
            Console.WriteLine($"    {header}");

            foreach (var message in result.MatchingMessages.Take(MaxRenderedMessages))
            {
                var preview = message.Content.Length > 150
                    ? message.Content[..150] + "..."
                    : message.Content;
                var marker = result.IsMatch(message) ? "→" : " ";
                Console.WriteLine($"      {marker} [{message.Role}] {preview}");
            }

            var hidden = result.MatchingMessages.Count - MaxRenderedMessages;
            if (hidden > 0)
            {
                Console.WriteLine($"      ... {hidden} more");
            }
        }
        else if (!string.IsNullOrWhiteSpace(result.Highlight))
        {
            var preview = result.Highlight.Length > 200
                ? result.Highlight[..200] + "..."
                : result.Highlight;
            Console.WriteLine($"    Preview: {preview}");
        }
    }

    /// <summary>
    /// Displays a knowledge search result
    /// </summary>
    /// <summary>
    /// Displays a task journal search result. Prints the journal name and task number so the
    /// caller can follow up with `task show`.
    /// </summary>
    private static void DisplayTaskResult(int index, TaskSearchResult task)
    {
        // No score is printed here. Task results are ranked by FTS5 bm25 over a single
        // repository's journal, where the corpus is small enough that IDF collapses and an exact
        // match still scores near zero. Displayed beside Lucene session scores it reads as
        // "irrelevant" for a result that RRF has just ranked highly. The ordinal conveys the
        // ranking; --robot still carries the native score for callers that want it.
        Console.WriteLine($"[{index}] Task: {task.JournalName} #{task.TaskNumber} ({task.Kind})");
        Console.WriteLine($"    {task.Excerpt}");
        Console.WriteLine($"    Show: agent-journal task show {task.JournalName} --task {task.TaskNumber}");
    }

    private static void DisplayKnowledgeResult(int index, KnowledgeEntry entry, UnifiedSearchResult result)
    {
        Console.WriteLine($"[{index}] Knowledge: {entry.Id}");
        Console.WriteLine($"    Score: {result.Score:F2} {RenderDecayBar(result.DecayFactor ?? 1.0)}");

        if (entry.Tags.Length > 0)
        {
            Console.WriteLine($"    Tags: {string.Join(", ", entry.Tags)}");
        }

        if (!string.IsNullOrWhiteSpace(entry.Project))
        {
            Console.WriteLine($"    Project: {entry.Project}");
        }

        if (!string.IsNullOrWhiteSpace(entry.Source))
        {
            Console.WriteLine($"    Source: {entry.Source}");
        }

        Console.WriteLine($"    Created: {entry.CreatedAt:yyyy-MM-dd HH:mm:ss}");
        Console.WriteLine($"    Last reinforced: {entry.LastReinforcedAt:yyyy-MM-dd HH:mm:ss} ({entry.ReinforcementCount}x)");

        // Show content preview
        var contentPreview = !string.IsNullOrWhiteSpace(result.Highlight)
            ? result.Highlight
            : entry.Content;

        if (contentPreview.Length > 200)
        {
            contentPreview = contentPreview[..200] + "...";
        }

        Console.WriteLine($"    Content: {contentPreview}");
    }

    /// <summary>
    /// Renders a visual decay bar with factor
    /// </summary>
    private static string RenderDecayBar(double decayFactor)
    {
        int filled = (int)(decayFactor * 10);
        string bar = new string('█', filled) + new string('░', 10 - filled);
        string status = decayFactor < 0.25 ? " ⚠️ decaying" : "";
        return $"(decay: {decayFactor:F2} {bar}){status}";
    }
}
