using System.CommandLine;
using System.Text.Json;
using Microsoft.Extensions.DependencyInjection;
using AgentJournal.Configuration;
using AgentJournal.Core.Search;
using AgentJournal.Core.Storage;
using AgentJournal.Core.Models;
using AgentJournal.Core.Knowledge;

namespace AgentJournal.Commands;

/// <summary>
/// Command to search indexed sessions
/// </summary>
public class SearchCommand : Command
{
    private SearchCommand() : base("search", "Search indexed agent sessions")
    {
        var queryArgument = new Argument<string>(
            name: "query",
            description: "Search query");

        var modeOption = new Option<string>(
            name: "--mode",
            getDefaultValue: () => "lexical",
            description: "Search mode: lexical, semantic, or hybrid");
        modeOption.AddAlias("-m");

        var contextOption = new Option<int>(
            name: "--context",
            getDefaultValue: () => 3,
            description: "Number of surrounding messages to include in results");
        contextOption.AddAlias("-c");

        var maxResultsOption = new Option<int>(
            name: "--max",
            getDefaultValue: () => 20,
            description: "Maximum number of results to return");
        maxResultsOption.AddAlias("-n");

        var agentOption = new Option<string?>(
            name: "--agent",
            description: "Filter by agent type (claude-code, copilot-cli)");
        agentOption.AddAlias("-a");

        var projectOption = new Option<string?>(
            name: "--project",
            description: "Filter by project path");
        projectOption.AddAlias("-p");

        var robotOption = new Option<bool>(
            name: "--robot",
            description: "Output results as JSON for scripting");
        robotOption.AddAlias("-r");

        var includeKnowledgeOption = new Option<bool>(
            name: "--include-knowledge",
            getDefaultValue: () => false,
            description: "Include knowledge entries in search results");
        includeKnowledgeOption.AddAlias("-k");

        this.AddArgument(queryArgument);
        this.AddOption(modeOption);
        this.AddOption(contextOption);
        this.AddOption(maxResultsOption);
        this.AddOption(agentOption);
        this.AddOption(projectOption);
        this.AddOption(robotOption);
        this.AddOption(includeKnowledgeOption);
    }

    public static Command Create(IServiceProvider serviceProvider)
    {
        var command = new SearchCommand();

        command.SetHandler(async (query, mode, contextCount, maxResults, agentType, project, robot, includeKnowledge) =>
        {
            var configService = serviceProvider.GetRequiredService<ConfigurationService>();
            var searchEngine = serviceProvider.GetRequiredService<ISearchEngine>();
            var repository = serviceProvider.GetRequiredService<ISessionRepository>();
            var knowledgeRepo = serviceProvider.GetService<IKnowledgeRepository>();

            await ExecuteAsync(
                query,
                mode,
                contextCount,
                maxResults,
                agentType,
                project,
                robot,
                includeKnowledge,
                configService,
                searchEngine,
                repository,
                knowledgeRepo,
                CancellationToken.None);
        },
        command.Arguments[0] as Argument<string> ?? throw new InvalidOperationException("Missing query argument"),
        command.Options[0] as Option<string> ?? throw new InvalidOperationException("Missing mode option"),
        command.Options[1] as Option<int> ?? throw new InvalidOperationException("Missing context option"),
        command.Options[2] as Option<int> ?? throw new InvalidOperationException("Missing max option"),
        command.Options[3] as Option<string?> ?? throw new InvalidOperationException("Missing agent option"),
        command.Options[4] as Option<string?> ?? throw new InvalidOperationException("Missing project option"),
        command.Options[5] as Option<bool> ?? throw new InvalidOperationException("Missing robot option"),
        command.Options[6] as Option<bool> ?? throw new InvalidOperationException("Missing include-knowledge option"));

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

        if (!robot)
        {
            Console.WriteLine($"Searching for: \"{query}\"");
            Console.WriteLine($"Mode: {searchMode}");
            if (includeKnowledge)
            {
                Console.WriteLine("Including: Knowledge entries");
            }
            Console.WriteLine();
        }

        // Collect unified results from both sessions and knowledge
        var unifiedResults = new List<UnifiedSearchResult>();

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
        unifiedResults.AddRange(filteredSessionResults.Select(UnifiedSearchResult.FromSession));

        // Execute knowledge search if requested
        if (includeKnowledge)
        {
            if (knowledgeRepository == null)
            {
                if (!robot)
                {
                    Console.WriteLine("Warning: Knowledge repository not available");
                }
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
                    unifiedResults.AddRange(knowledgeResults.Select(UnifiedSearchResult.FromKnowledge));
                }
                catch (Exception ex)
                {
                    if (!robot)
                    {
                        Console.WriteLine($"Warning: Knowledge search failed: {ex.Message}");
                    }
                }
            }
        }

        // Sort all results by score and take top maxResults
        var finalResults = unifiedResults
            .OrderByDescending(r => r.Score)
            .Take(maxResults)
            .ToList();

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
            Console.WriteLine($"    Matching messages:");

            foreach (var message in result.MatchingMessages.Take(3))
            {
                var preview = message.Content.Length > 150
                    ? message.Content[..150] + "..."
                    : message.Content;
                Console.WriteLine($"      [{message.Role}] {preview}");
            }
        }
        else if (!string.IsNullOrWhiteSpace(result.Highlight))
        {
            var preview = result.Highlight.Length > 200
                ? result.Highlight[..200] + "..."
                : result.Highlight;
            Console.WriteLine($"    Preview: {preview}");
        }

        if (contextCount > 0 && result.MatchingMessages != null && result.MatchingMessages.Count > 0)
        {
            Console.WriteLine($"    Context messages:");

            // Get context messages around matches
            var matchedIndices = result.MatchingMessages
                .Select(m => session.Messages.ToList().IndexOf(m))
                .Where(idx => idx >= 0)
                .ToHashSet();

            var contextMessages = session.Messages
                .Select((msg, idx) => (msg, idx))
                .Where(x => matchedIndices.Any(matchIdx =>
                    Math.Abs(x.idx - matchIdx) <= contextCount))
                .Take(5)
                .ToList();

            foreach (var (msg, idx) in contextMessages)
            {
                var isMatch = matchedIndices.Contains(idx);
                var marker = isMatch ? "→" : " ";
                var preview = msg.Content.Length > 100
                    ? msg.Content[..100] + "..."
                    : msg.Content;
                Console.WriteLine($"      {marker} [{msg.Role}] {preview}");
            }
        }
    }

    /// <summary>
    /// Displays a knowledge search result
    /// </summary>
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
