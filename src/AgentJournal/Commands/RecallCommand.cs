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
    private RecallCommand() : base("recall", "Search and recall knowledge from the knowledge bank")
    {
        var queryArgument = new Argument<string>(
            name: "query",
            description: "Search query for knowledge recall");

        var tagsOption = new Option<string?>(
            name: "--tags",
            description: "Filter by comma-separated tags");
        tagsOption.AddAlias("-t");

        var projectOption = new Option<string?>(
            name: "--project",
            description: "Filter by project name or path");
        projectOption.AddAlias("-p");

        var modeOption = new Option<string>(
            name: "--mode",
            getDefaultValue: () => "hybrid",
            description: "Search mode: keyword, semantic, or hybrid");
        modeOption.AddAlias("-m");

        var limitOption = new Option<int>(
            name: "--limit",
            getDefaultValue: () => 10,
            description: "Maximum number of results to return");
        limitOption.AddAlias("-n");

        var jsonOption = new Option<bool>(
            name: "--json",
            description: "Output results as JSON");

        this.AddArgument(queryArgument);
        this.AddOption(tagsOption);
        this.AddOption(projectOption);
        this.AddOption(modeOption);
        this.AddOption(limitOption);
        this.AddOption(jsonOption);
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
        command.Arguments[0] as Argument<string> ?? throw new InvalidOperationException("Missing query argument"),
        command.Options[0] as Option<string?> ?? throw new InvalidOperationException("Missing tags option"),
        command.Options[1] as Option<string?> ?? throw new InvalidOperationException("Missing project option"),
        command.Options[2] as Option<string> ?? throw new InvalidOperationException("Missing mode option"),
        command.Options[3] as Option<int> ?? throw new InvalidOperationException("Missing limit option"),
        command.Options[4] as Option<bool> ?? throw new InvalidOperationException("Missing json option"));

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

            if (!json)
            {
                Console.WriteLine($"Searching knowledge bank for: \"{query}\"");
                Console.WriteLine($"Mode: {searchMode}");
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
