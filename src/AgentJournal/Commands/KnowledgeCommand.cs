using System.CommandLine;
using System.Text.Json;
using Microsoft.Extensions.DependencyInjection;
using AgentJournal.Configuration;
using AgentJournal.Core.Knowledge;
using AgentJournal.Core.Models;

namespace AgentJournal.Commands;

/// <summary>
/// Command for managing the knowledge bank with subcommands
/// </summary>
public class KnowledgeCommand : Command
{
    private KnowledgeCommand() : base("knowledge", "Manage the knowledge bank")
    {
    }

    public static Command Create(IServiceProvider serviceProvider)
    {
        var command = new KnowledgeCommand();

        // Add subcommands
        command.AddCommand(CreateListCommand(serviceProvider));
        command.AddCommand(CreateStatsCommand(serviceProvider));
        command.AddCommand(CreateExportCommand(serviceProvider));
        command.AddCommand(CreateImportCommand(serviceProvider));
        command.AddCommand(CreatePruneCommand(serviceProvider));
        command.AddCommand(CreateClearCommand(serviceProvider));

        return command;
    }

    private static Command CreateListCommand(IServiceProvider serviceProvider)
    {
        var listCommand = new Command("list", "List knowledge entries");

        var projectOption = new Option<string?>(
            name: "--project",
            description: "Filter by project");
        projectOption.AddAlias("-p");

        var tagsOption = new Option<string?>(
            name: "--tags",
            description: "Filter by comma-separated tags");
        tagsOption.AddAlias("-t");

        var decayingOption = new Option<bool>(
            name: "--decaying",
            description: "Show only decaying entries (decay < 0.5)");

        var expiringOption = new Option<bool>(
            name: "--expiring",
            description: "Show only expiring entries (decay < 0.1)");

        var limitOption = new Option<int>(
            name: "--limit",
            getDefaultValue: () => 50,
            description: "Maximum number of entries to list");
        limitOption.AddAlias("-n");

        listCommand.AddOption(projectOption);
        listCommand.AddOption(tagsOption);
        listCommand.AddOption(decayingOption);
        listCommand.AddOption(expiringOption);
        listCommand.AddOption(limitOption);

        listCommand.SetHandler(async (project, tags, decaying, expiring, limit) =>
        {
            var repository = serviceProvider.GetRequiredService<IKnowledgeRepository>();
            var configService = serviceProvider.GetRequiredService<ConfigurationService>();

            await ExecuteListAsync(project, tags, decaying, expiring, limit, repository, configService, CancellationToken.None);
        },
        projectOption, tagsOption, decayingOption, expiringOption, limitOption);

        return listCommand;
    }

    private static Command CreateStatsCommand(IServiceProvider serviceProvider)
    {
        var statsCommand = new Command("stats", "Show knowledge bank statistics");

        statsCommand.SetHandler(async () =>
        {
            var repository = serviceProvider.GetRequiredService<IKnowledgeRepository>();
            var configService = serviceProvider.GetRequiredService<ConfigurationService>();

            await ExecuteStatsAsync(repository, configService, CancellationToken.None);
        });

        return statsCommand;
    }

    private static Command CreateExportCommand(IServiceProvider serviceProvider)
    {
        var exportCommand = new Command("export", "Export knowledge bank to file");

        var formatOption = new Option<string>(
            name: "--format",
            getDefaultValue: () => "json",
            description: "Export format (json)");
        formatOption.AddAlias("-f");

        var outputOption = new Option<string?>(
            name: "--output",
            description: "Output file path");
        outputOption.AddAlias("-o");

        exportCommand.AddOption(formatOption);
        exportCommand.AddOption(outputOption);

        exportCommand.SetHandler(async (format, output) =>
        {
            var repository = serviceProvider.GetRequiredService<IKnowledgeRepository>();
            var configService = serviceProvider.GetRequiredService<ConfigurationService>();

            await ExecuteExportAsync(format, output, repository, configService, CancellationToken.None);
        },
        formatOption, outputOption);

        return exportCommand;
    }

    private static Command CreateImportCommand(IServiceProvider serviceProvider)
    {
        var importCommand = new Command("import", "Import knowledge bank from file");

        var fileArgument = new Argument<string>(
            name: "file",
            description: "JSON file to import");

        importCommand.AddArgument(fileArgument);

        importCommand.SetHandler(async (file) =>
        {
            var repository = serviceProvider.GetRequiredService<IKnowledgeRepository>();
            var configService = serviceProvider.GetRequiredService<ConfigurationService>();

            await ExecuteImportAsync(file, repository, configService, CancellationToken.None);
        },
        fileArgument);

        return importCommand;
    }

    private static Command CreatePruneCommand(IServiceProvider serviceProvider)
    {
        var pruneCommand = new Command("prune", "Remove expired knowledge entries");

        var thresholdOption = new Option<double>(
            name: "--threshold",
            getDefaultValue: () => 0.05,
            description: "Decay factor threshold for pruning");
        thresholdOption.AddAlias("-t");

        pruneCommand.AddOption(thresholdOption);

        pruneCommand.SetHandler(async (threshold) =>
        {
            var repository = serviceProvider.GetRequiredService<IKnowledgeRepository>();
            var configService = serviceProvider.GetRequiredService<ConfigurationService>();

            await ExecutePruneAsync(threshold, repository, configService, CancellationToken.None);
        },
        thresholdOption);

        return pruneCommand;
    }

    private static Command CreateClearCommand(IServiceProvider serviceProvider)
    {
        var clearCommand = new Command("clear", "Clear all knowledge entries");

        var confirmOption = new Option<bool>(
            name: "--confirm",
            description: "Confirm clearing all knowledge");
        confirmOption.AddAlias("-y");

        clearCommand.AddOption(confirmOption);

        clearCommand.SetHandler(async (confirm) =>
        {
            var repository = serviceProvider.GetRequiredService<IKnowledgeRepository>();
            var configService = serviceProvider.GetRequiredService<ConfigurationService>();

            await ExecuteClearAsync(confirm, repository, configService, CancellationToken.None);
        },
        confirmOption);

        return clearCommand;
    }

    // Command implementations

    private static async Task ExecuteListAsync(
        string? project,
        string? tags,
        bool decaying,
        bool expiring,
        int limit,
        IKnowledgeRepository repository,
        ConfigurationService configService,
        CancellationToken ct)
    {
        var config = await configService.LoadConfigAsync(ct);

        try
        {
            var tagList = string.IsNullOrWhiteSpace(tags)
                ? null
                : tags.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).AsEnumerable();

            limit = Math.Clamp(limit, 1, 1000);

            var entries = await repository.ListAsync(project, tagList, includeDecaying: true, limit, ct);

            // Apply filters
            if (expiring)
            {
                entries = entries.Where(e => DecayCalculator.CalculateDecayFactor(e.LastReinforcedAt) < 0.1).ToList();
            }
            else if (decaying)
            {
                entries = entries.Where(e => DecayCalculator.CalculateDecayFactor(e.LastReinforcedAt) < 0.5).ToList();
            }

            if (entries.Count == 0)
            {
                Console.WriteLine("No knowledge entries found.");
                return;
            }

            Console.WriteLine($"Knowledge Bank ({entries.Count} entries):\n");

            for (int i = 0; i < entries.Count; i++)
            {
                var entry = entries[i];
                var decayFactor = DecayCalculator.CalculateDecayFactor(entry.LastReinforcedAt);
                var decayStatus = DecayCalculator.GetDecayStatus(decayFactor);
                var isDecaying = decayFactor <= 0.5;
                var isExpiring = decayFactor <= 0.1;

                Console.WriteLine($"[{i + 1}] ID: {entry.Id}");
                Console.Write($"    Decay: {decayFactor:F2} {RenderDecayBar(decayFactor)} ({decayStatus})");

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

                Console.WriteLine($"    Reinforced: {FormatTimeSince(entry.LastReinforcedAt)} ({entry.ReinforcementCount} times)");
                Console.WriteLine($"    Content: {TruncateContent(entry.Content, 200)}");
                Console.WriteLine();
            }
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Error listing knowledge: {ex.Message}");
            CommandOutcome.Fail();
            if (config.VerboseLogging)
            {
                Console.Error.WriteLine(ex.StackTrace);
            }
        }
    }

    private static async Task ExecuteStatsAsync(
        IKnowledgeRepository repository,
        ConfigurationService configService,
        CancellationToken ct)
    {
        var config = await configService.LoadConfigAsync(ct);

        try
        {
            var stats = await repository.GetStatsAsync(ct);

            Console.WriteLine("Knowledge Bank Statistics:\n");
            Console.WriteLine($"Total Entries: {stats.TotalEntries}");
            Console.WriteLine();
            Console.WriteLine("Decay Distribution:");
            Console.WriteLine($"  Fresh (>75%):    {stats.FreshEntries,6} {RenderPercentageBar(stats.FreshEntries, stats.TotalEntries)}");
            Console.WriteLine($"  Good (>50%):     {stats.GoodEntries,6} {RenderPercentageBar(stats.GoodEntries, stats.TotalEntries)}");
            Console.WriteLine($"  Aging (>25%):    {stats.AgingEntries,6} {RenderPercentageBar(stats.AgingEntries, stats.TotalEntries)}");
            Console.WriteLine($"  Decaying (>10%): {stats.DecayingEntries,6} {RenderPercentageBar(stats.DecayingEntries, stats.TotalEntries)}");
            Console.WriteLine($"  Expiring (≤10%): {stats.ExpiringEntries,6} {RenderPercentageBar(stats.ExpiringEntries, stats.TotalEntries)}");

            if (stats.EntriesByProject.Count > 0)
            {
                Console.WriteLine();
                Console.WriteLine("By Project:");
                foreach (var (project, count) in stats.EntriesByProject.OrderByDescending(kv => kv.Value).Take(10))
                {
                    Console.WriteLine($"  {project,-30} {count,6}");
                }
            }

            if (stats.EntriesByTag.Count > 0)
            {
                Console.WriteLine();
                Console.WriteLine("By Tag:");
                foreach (var (tag, count) in stats.EntriesByTag.OrderByDescending(kv => kv.Value).Take(10))
                {
                    Console.WriteLine($"  {tag,-30} {count,6}");
                }
            }
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Error getting statistics: {ex.Message}");
            CommandOutcome.Fail();
            if (config.VerboseLogging)
            {
                Console.Error.WriteLine(ex.StackTrace);
            }
        }
    }

    private static async Task ExecuteExportAsync(
        string? format,
        string? output,
        IKnowledgeRepository repository,
        ConfigurationService configService,
        CancellationToken ct)
    {
        var config = await configService.LoadConfigAsync(ct);

        try
        {
            var entries = await repository.ListAsync(limit: int.MaxValue, ct: ct);

            if (entries.Count == 0)
            {
                Console.WriteLine("No knowledge entries to export.");
                return;
            }

            var json = JsonSerializer.Serialize(entries, new JsonSerializerOptions
            {
                WriteIndented = true
            });

            if (string.IsNullOrWhiteSpace(output))
            {
                Console.WriteLine(json);
            }
            else
            {
                if (!Path.IsPathRooted(output))
                {
                    output = Path.Combine(Environment.CurrentDirectory, output);
                }

                await File.WriteAllTextAsync(output, json, ct);
                Console.WriteLine($"✓ Exported {entries.Count} knowledge entries to {output}");
            }
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Error exporting knowledge: {ex.Message}");
            CommandOutcome.Fail();
            if (config.VerboseLogging)
            {
                Console.Error.WriteLine(ex.StackTrace);
            }
        }
    }

    private static async Task ExecuteImportAsync(
        string file,
        IKnowledgeRepository repository,
        ConfigurationService configService,
        CancellationToken ct)
    {
        var config = await configService.LoadConfigAsync(ct);

        try
        {
            if (!File.Exists(file))
            {
                Console.Error.WriteLine($"Error: File not found: {file}");
                CommandOutcome.Fail(CommandOutcome.NotFound);
                return;
            }

            var json = await File.ReadAllTextAsync(file, ct);
            var entries = JsonSerializer.Deserialize<KnowledgeEntry[]>(json);

            if (entries == null || entries.Length == 0)
            {
                Console.WriteLine("No entries found in file.");
                return;
            }

            Console.WriteLine($"Importing {entries.Length} knowledge entries...");

            int imported = 0;
            foreach (var entry in entries)
            {
                await repository.SaveAsync(entry, ct);
                imported++;
            }

            Console.WriteLine($"✓ Imported {imported} knowledge entries");
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Error importing knowledge: {ex.Message}");
            CommandOutcome.Fail();
            if (config.VerboseLogging)
            {
                Console.Error.WriteLine(ex.StackTrace);
            }
        }
    }

    private static async Task ExecutePruneAsync(
        double threshold,
        IKnowledgeRepository repository,
        ConfigurationService configService,
        CancellationToken ct)
    {
        var config = await configService.LoadConfigAsync(ct);

        try
        {
            threshold = Math.Clamp(threshold, 0.0, 1.0);

            Console.WriteLine($"Pruning entries with decay factor below {threshold:F2}...");

            var pruned = await repository.PruneExpiredAsync(threshold, ct);

            Console.WriteLine($"✓ Pruned {pruned} expired knowledge entries");
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Error pruning knowledge: {ex.Message}");
            CommandOutcome.Fail();
            if (config.VerboseLogging)
            {
                Console.Error.WriteLine(ex.StackTrace);
            }
        }
    }

    private static async Task ExecuteClearAsync(
        bool confirm,
        IKnowledgeRepository repository,
        ConfigurationService configService,
        CancellationToken ct)
    {
        var config = await configService.LoadConfigAsync(ct);

        if (!confirm)
        {
            Console.Error.WriteLine("Error: Clearing all knowledge requires --confirm flag");
            Console.Error.WriteLine("Use --confirm to proceed with clearing");
            CommandOutcome.Fail();
            return;
        }

        try
        {
            var entries = await repository.ListAsync(limit: int.MaxValue, ct: ct);

            Console.WriteLine($"Clearing {entries.Count} knowledge entries...");

            var ids = entries.Select(e => e.Id).ToList();
            var deleted = await repository.DeleteManyAsync(ids, ct);

            Console.WriteLine($"✓ Cleared {deleted} knowledge entries");
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Error clearing knowledge: {ex.Message}");
            CommandOutcome.Fail();
            if (config.VerboseLogging)
            {
                Console.Error.WriteLine(ex.StackTrace);
            }
        }
    }

    // Helper methods

    private static string RenderDecayBar(double decayFactor)
    {
        const int barLength = 10;
        var filled = (int)Math.Round(decayFactor * barLength);
        var empty = barLength - filled;

        return new string('█', filled) + new string('░', empty);
    }

    private static string RenderPercentageBar(int count, int total)
    {
        if (total == 0) return new string('░', 20);

        const int barLength = 20;
        var percentage = (double)count / total;
        var filled = (int)Math.Round(percentage * barLength);
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
