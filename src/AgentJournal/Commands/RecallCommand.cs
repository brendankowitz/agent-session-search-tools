using System.CommandLine;
using Microsoft.Extensions.DependencyInjection;
using AgentJournal.Configuration;
using AgentJournal.Core.Knowledge;
using AgentJournal.Core.Search;

namespace AgentJournal.Commands;

/// <summary>
/// Command to search and recall knowledge from the knowledge bank
/// </summary>
public class RecallCommand : Command
{
    private readonly Argument<string> _queryArgument;
    private readonly Option<string?> _tagsOption;
    private readonly Option<string?> _projectOption;
    private readonly Option<string?> _modeOption;
    private readonly Option<int> _limitOption;
    private readonly Option<bool> _jsonOption;

    private RecallCommand() : base("recall", "Search and recall knowledge from the knowledge bank")
    {
        _queryArgument = new Argument<string>(
            name: "query",
            description: "Search query for knowledge recall");

        _tagsOption = new Option<string?>(
            name: "--tags",
            description: "Filter by comma-separated tags");
        _tagsOption.AddAlias("-t");

        _projectOption = new Option<string?>(
            name: "--project",
            description: "Filter by project name or path");
        _projectOption.AddAlias("-p");

        // No default value: an unset --mode must stay distinguishable from an explicit one so the
        // "mode not supported" note below only fires when the user actually asked for a mode.
        _modeOption = new Option<string?>(
            name: "--mode",
            description: "Search mode: keyword, semantic, or hybrid (knowledge search is always lexical)");
        _modeOption.AddAlias("-m");

        _limitOption = new Option<int>(
            name: "--limit",
            getDefaultValue: () => 10,
            description: "Maximum number of results to return");
        _limitOption.AddAlias("-n");

        _jsonOption = new Option<bool>(
            name: "--json",
            description: "Output results as JSON");

        this.AddArgument(_queryArgument);
        this.AddOption(_tagsOption);
        this.AddOption(_projectOption);
        this.AddOption(_modeOption);
        this.AddOption(_limitOption);
        this.AddOption(_jsonOption);
    }

    public static Command Create(IServiceProvider serviceProvider)
    {
        var command = new RecallCommand();

        command.SetHandler(async (query, tags, project, mode, limit, json) =>
        {
            var repository = serviceProvider.GetRequiredService<IKnowledgeRepository>();
            var configService = serviceProvider.GetRequiredService<ConfigurationService>();

            await ExecuteAsync(
                query,
                tags,
                project,
                mode,
                limit,
                json,
                repository,
                configService,
                CancellationToken.None);
        },
        command._queryArgument,
        command._tagsOption,
        command._projectOption,
        command._modeOption,
        command._limitOption,
        command._jsonOption);

        return command;
    }

    private static async Task ExecuteAsync(
        string query,
        string? tags,
        string? project,
        string? mode,
        int limit,
        bool json,
        IKnowledgeRepository repository,
        ConfigurationService configService,
        CancellationToken ct)
    {
        var config = await configService.LoadConfigAsync(ct);

        try
        {
            // Parse tags
            var tagList = string.IsNullOrWhiteSpace(tags)
                ? null
                : tags.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).AsEnumerable();

            // Parse search mode
            var searchMode = mode?.ToLowerInvariant() switch
            {
                "keyword" or "lexical" => SearchMode.Lexical,
                "semantic" => SearchMode.Semantic,
                _ => SearchMode.Hybrid
            };

            // Clamp limit
            limit = Math.Clamp(limit, 1, 100);

            // The knowledge bank has no vector index, so every mode is served by FTS5 lexical
            // search. Only say so when a mode was explicitly requested - otherwise every default
            // invocation prints a note about a mode the user never asked for.
            if (mode != null && searchMode != SearchMode.Lexical)
            {
                Console.Error.WriteLine(
                    $"Note: knowledge search does not support '{searchMode}' mode; using lexical (FTS5) search.");
            }

            if (!json)
            {
                Console.WriteLine($"Searching knowledge bank for: \"{query}\"");
                Console.WriteLine($"Mode: Lexical (FTS5)");
                Console.WriteLine();
            }

            // Execute search
            var results = await repository.SearchAsync(query, tagList, project, searchMode, limit, ct);

            if (json)
            {
                OutputJson(results);
            }
            else
            {
                OutputHuman(results);
            }
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Error recalling knowledge: {ex.Message}");
            CommandOutcome.Fail();
            if (config.VerboseLogging)
            {
                Console.Error.WriteLine(ex.StackTrace);
            }
        }
    }

    private static void OutputJson(IReadOnlyList<KnowledgeSearchResult> results)
    {
        var jsonResults = results.Select(r => new
        {
            id = r.Entry.Id,
            score = r.Score,
            decayFactor = r.DecayFactor,
            content = r.Entry.Content,
            tags = r.Entry.Tags,
            project = r.Entry.Project,
            source = r.Entry.Source,
            createdAt = r.Entry.CreatedAt,
            lastReinforcedAt = r.Entry.LastReinforcedAt,
            reinforcementCount = r.Entry.ReinforcementCount,
            daysSinceReinforcement = r.Entry.DaysSinceReinforcement
        });

        var json = System.Text.Json.JsonSerializer.Serialize(jsonResults, new System.Text.Json.JsonSerializerOptions
        {
            WriteIndented = true
        });
        Console.WriteLine(json);
    }

    private static void OutputHuman(IReadOnlyList<KnowledgeSearchResult> results)
    {
        if (results.Count == 0)
        {
            Console.WriteLine("No knowledge found.");
            return;
        }

        Console.WriteLine($"Found {results.Count} result(s):\n");

        for (int i = 0; i < results.Count; i++)
        {
            var result = results[i];
            var entry = result.Entry;
            var decayStatus = DecayCalculator.GetDecayStatus(result.DecayFactor);
            var isDecaying = result.DecayFactor <= 0.5;
            var isExpiring = result.DecayFactor <= 0.1;

            Console.WriteLine($"[{i + 1}] ID: {entry.Id}");
            Console.Write($"    Score: {result.Score:F2} (decay: {result.DecayFactor:F2} {RenderDecayBar(result.DecayFactor)})");

            if (isExpiring)
            {
                Console.Write(" ⚠️ expiring");
            }
            else if (isDecaying)
            {
                Console.Write(" ⚠️ decaying");
            }

            Console.WriteLine();

            if (entry.Tags.Length > 0)
            {
                Console.WriteLine($"    Tags: {string.Join(", ", entry.Tags)}");
            }

            if (!string.IsNullOrWhiteSpace(entry.Project))
            {
                Console.WriteLine($"    Project: {entry.Project}");
            }

            Console.WriteLine($"    Reinforced: {FormatTimeSince(entry.LastReinforcedAt)}");
            Console.WriteLine($"    Content: {TruncateContent(entry.Content, 200)}");

            if (!string.IsNullOrWhiteSpace(result.Highlight))
            {
                Console.WriteLine($"    Highlight: {result.Highlight}");
            }

            Console.WriteLine();
        }
    }

    private static string RenderDecayBar(double decayFactor)
    {
        const int barLength = 10;
        var filled = (int)Math.Round(decayFactor * barLength);
        var empty = barLength - filled;

        return new string('█', filled) + new string('░', empty);
    }

    private static string FormatTimeSince(DateTime timestamp)
    {
        var span = DateTime.UtcNow - timestamp;

        if (span.TotalDays < 1)
        {
            return span.TotalHours < 1
                ? $"{(int)span.TotalMinutes} minutes ago"
                : $"{(int)span.TotalHours} hours ago";
        }

        if (span.TotalDays < 7)
        {
            return $"{(int)span.TotalDays} days ago";
        }

        if (span.TotalDays < 30)
        {
            return $"{(int)(span.TotalDays / 7)} weeks ago";
        }

        if (span.TotalDays < 365)
        {
            return $"{(int)(span.TotalDays / 30)} months ago";
        }

        return $"{(int)(span.TotalDays / 365)} years ago";
    }

    private static string TruncateContent(string content, int maxLength)
    {
        if (content.Length <= maxLength)
        {
            return content;
        }

        return content[..maxLength] + "...";
    }
}
