using System.CommandLine;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Microsoft.Extensions.DependencyInjection;
using AgentJournal.Configuration;
using AgentJournal.Core.Knowledge;
using AgentJournal.Core.Models;
using AgentJournal.Core.Utilities;
using Microsoft.Extensions.FileSystemGlobbing;

namespace AgentJournal.Commands;

/// <summary>
/// Command for managing content indexing with subcommands
/// </summary>
public class ContentCommand : Command
{
    private ContentCommand() : base("content", "Manage content indexing")
    {
    }

    public static Command Create(IServiceProvider serviceProvider)
    {
        var command = new ContentCommand();

        // Add subcommands
        command.AddCommand(CreateIndexCommand(serviceProvider));
        command.AddCommand(CreateAddCommand(serviceProvider));
        command.AddCommand(CreateSearchCommand(serviceProvider));
        command.AddCommand(CreateListCommand(serviceProvider));
        command.AddCommand(CreateRemoveCommand(serviceProvider));
        command.AddCommand(CreateReinforceCommand(serviceProvider));

        return command;
    }

    private static Command CreateIndexCommand(IServiceProvider serviceProvider)
    {
        var indexCommand = new Command("index", "Index markdown files from a directory");

        var pathArgument = new Argument<string>(
            name: "path",
            description: "Directory path to scan for markdown files");

        var filterOption = new Option<string>(
            name: "--filter",
            getDefaultValue: () => "*.md",
            description: "Glob pattern for file matching");
        filterOption.AddAlias("-f");

        var projectOption = new Option<string?>(
            name: "--project",
            description: "Project name to associate with indexed content");
        projectOption.AddAlias("-p");

        var recursiveOption = new Option<bool>(
            name: "--recursive",
            getDefaultValue: () => true,
            description: "Recursively scan subdirectories");
        recursiveOption.AddAlias("-r");

        var rebuildOption = new Option<bool>(
            name: "--rebuild",
            description: "Rebuild index (re-index unchanged files)");

        indexCommand.AddArgument(pathArgument);
        indexCommand.AddOption(filterOption);
        indexCommand.AddOption(projectOption);
        indexCommand.AddOption(recursiveOption);
        indexCommand.AddOption(rebuildOption);

        indexCommand.SetHandler(async (path, filter, project, recursive, rebuild) =>
        {
            var repository = serviceProvider.GetRequiredService<IContentRepository>();
            var configService = serviceProvider.GetRequiredService<ConfigurationService>();

            await ExecuteIndexAsync(path, filter, project, recursive, rebuild, repository, configService, CancellationToken.None);
        },
        pathArgument, filterOption, projectOption, recursiveOption, rebuildOption);

        return indexCommand;
    }

    private static Command CreateAddCommand(IServiceProvider serviceProvider)
    {
        var addCommand = new Command("add", "Add content directly");

        var sourceOption = new Option<string>(
            name: "--source",
            description: "Source identifier for this content");
        sourceOption.AddAlias("-s");
        sourceOption.IsRequired = true;

        var titleOption = new Option<string>(
            name: "--title",
            description: "Content title");
        titleOption.AddAlias("-t");
        titleOption.IsRequired = true;

        var contentOption = new Option<string?>(
            name: "--content",
            description: "Content text (if not provided, reads from stdin)");
        contentOption.AddAlias("-c");

        var projectOption = new Option<string?>(
            name: "--project",
            description: "Project name");
        projectOption.AddAlias("-p");

        var tagsOption = new Option<string?>(
            name: "--tags",
            description: "Comma-separated tags");

        addCommand.AddOption(sourceOption);
        addCommand.AddOption(titleOption);
        addCommand.AddOption(contentOption);
        addCommand.AddOption(projectOption);
        addCommand.AddOption(tagsOption);

        addCommand.SetHandler(async (source, title, content, project, tags) =>
        {
            var repository = serviceProvider.GetRequiredService<IContentRepository>();
            var configService = serviceProvider.GetRequiredService<ConfigurationService>();

            await ExecuteAddAsync(source, title, content, project, tags, repository, configService, CancellationToken.None);
        },
        sourceOption, titleOption, contentOption, projectOption, tagsOption);

        return addCommand;
    }

    private static Command CreateSearchCommand(IServiceProvider serviceProvider)
    {
        var searchCommand = new Command("search", "Search indexed content");

        var queryArgument = new Argument<string>(
            name: "query",
            description: "Search query");

        var maxOption = new Option<int>(
            name: "--max",
            getDefaultValue: () => 10,
            description: "Maximum number of results");
        maxOption.AddAlias("-n");

        var projectOption = new Option<string?>(
            name: "--project",
            description: "Filter by project");
        projectOption.AddAlias("-p");

        var sourcePrefixOption = new Option<string?>(
            name: "--source-prefix",
            description: "Filter by source path prefix");
        sourcePrefixOption.AddAlias("-s");

        var tagsOption = new Option<string?>(
            name: "--tags",
            description: "Filter by tags (comma-separated)");
        tagsOption.AddAlias("-t");

        var robotOption = new Option<bool>(
            name: "--robot",
            description: "Output JSON for automation");

        searchCommand.AddArgument(queryArgument);
        searchCommand.AddOption(maxOption);
        searchCommand.AddOption(projectOption);
        searchCommand.AddOption(sourcePrefixOption);
        searchCommand.AddOption(tagsOption);
        searchCommand.AddOption(robotOption);

        searchCommand.SetHandler(async (query, max, project, sourcePrefix, tags, robot) =>
        {
            var repository = serviceProvider.GetRequiredService<IContentRepository>();
            var configService = serviceProvider.GetRequiredService<ConfigurationService>();

            await ExecuteSearchAsync(query, max, project, sourcePrefix, tags, robot, repository, configService, CancellationToken.None);
        },
        queryArgument, maxOption, projectOption, sourcePrefixOption, tagsOption, robotOption);

        return searchCommand;
    }

    private static Command CreateListCommand(IServiceProvider serviceProvider)
    {
        var listCommand = new Command("list", "List indexed content");

        var projectOption = new Option<string?>(
            name: "--project",
            description: "Filter by project");
        projectOption.AddAlias("-p");

        var sourcePrefixOption = new Option<string?>(
            name: "--source-prefix",
            description: "Filter by source path prefix");
        sourcePrefixOption.AddAlias("-s");

        var tagsOption = new Option<string?>(
            name: "--tags",
            description: "Filter by tags (comma-separated)");
        tagsOption.AddAlias("-t");

        var robotOption = new Option<bool>(
            name: "--robot",
            description: "Output JSON for automation");

        var expiredOption = new Option<bool>(
            name: "--expired",
            description: "Show only expired content");

        var limitOption = new Option<int>(
            name: "--limit",
            getDefaultValue: () => 50,
            description: "Maximum number of entries to list");
        limitOption.AddAlias("-n");

        listCommand.AddOption(projectOption);
        listCommand.AddOption(sourcePrefixOption);
        listCommand.AddOption(tagsOption);
        listCommand.AddOption(robotOption);
        listCommand.AddOption(expiredOption);
        listCommand.AddOption(limitOption);

        listCommand.SetHandler(async (project, sourcePrefix, tags, robot, expired, limit) =>
        {
            var repository = serviceProvider.GetRequiredService<IContentRepository>();
            var configService = serviceProvider.GetRequiredService<ConfigurationService>();

            await ExecuteListAsync(project, sourcePrefix, tags, robot, expired, limit, repository, configService, CancellationToken.None);
        },
        projectOption, sourcePrefixOption, tagsOption, robotOption, expiredOption, limitOption);

        return listCommand;
    }

    private static Command CreateRemoveCommand(IServiceProvider serviceProvider)
    {
        var removeCommand = new Command("remove", "Remove content by various criteria");

        var idOption = new Option<string?>(
            name: "--id",
            description: "Remove by content ID");

        var sourceOption = new Option<string?>(
            name: "--source",
            description: "Remove by exact source match");
        sourceOption.AddAlias("-s");

        var sourcePrefixOption = new Option<string?>(
            name: "--source-prefix",
            description: "Remove all content where source starts with prefix");

        var projectOption = new Option<string?>(
            name: "--project",
            description: "Remove all content for a project");
        projectOption.AddAlias("-p");

        var allOption = new Option<bool>(
            name: "--all",
            description: "Remove all content (requires confirmation)");

        var forceOption = new Option<bool>(
            name: "--force",
            description: "Skip confirmation prompts");
        forceOption.AddAlias("-f");

        removeCommand.AddOption(idOption);
        removeCommand.AddOption(sourceOption);
        removeCommand.AddOption(sourcePrefixOption);
        removeCommand.AddOption(projectOption);
        removeCommand.AddOption(allOption);
        removeCommand.AddOption(forceOption);

        removeCommand.SetHandler(async (id, source, sourcePrefix, project, all, force) =>
        {
            var repository = serviceProvider.GetRequiredService<IContentRepository>();
            var configService = serviceProvider.GetRequiredService<ConfigurationService>();

            await ExecuteRemoveAsync(id, source, sourcePrefix, project, all, force, repository, configService, CancellationToken.None);
        },
        idOption, sourceOption, sourcePrefixOption, projectOption, allOption, forceOption);

        return removeCommand;
    }

    private static Command CreateReinforceCommand(IServiceProvider serviceProvider)
    {
        var reinforceCommand = new Command("reinforce", "Reset decay timer for content");

        var sourceOption = new Option<string>(
            name: "--source",
            description: "Source identifier of content to reinforce");
        sourceOption.AddAlias("-s");
        sourceOption.IsRequired = true;

        reinforceCommand.AddOption(sourceOption);

        reinforceCommand.SetHandler(async (source) =>
        {
            var repository = serviceProvider.GetRequiredService<IContentRepository>();
            var configService = serviceProvider.GetRequiredService<ConfigurationService>();

            await ExecuteReinforceAsync(source, repository, configService, CancellationToken.None);
        },
        sourceOption);

        return reinforceCommand;
    }

    // Command implementations

    private static async Task ExecuteIndexAsync(
        string path,
        string filter,
        string? project,
        bool recursive,
        bool rebuild,
        IContentRepository repository,
        ConfigurationService configService,
        CancellationToken ct)
    {
        var config = await configService.LoadConfigAsync(ct);

        try
        {
            // Validate path to prevent directory traversal
            var validatedPath = ContentUtils.ValidatePath(path);

            if (!Directory.Exists(validatedPath))
            {
                Console.Error.WriteLine($"Error: Directory not found: {validatedPath}");
                return;
            }

            Console.WriteLine($"Indexing markdown files from: {validatedPath}");
            Console.WriteLine($"Filter: {filter}");
            Console.WriteLine($"Recursive: {recursive}");
            if (!string.IsNullOrWhiteSpace(project))
            {
                Console.WriteLine($"Project: {project}");
            }
            Console.WriteLine();

            var matcher = new Matcher();
            matcher.AddInclude(filter);

            var searchOption = recursive ? SearchOption.AllDirectories : SearchOption.TopDirectoryOnly;
            var files = Directory.GetFiles(validatedPath, "*", searchOption);

            var matchedFiles = matcher.Match(Path.GetFullPath(validatedPath), files.Select(f => Path.GetRelativePath(validatedPath, f)))
                .Files
                .Select(f => Path.Combine(validatedPath, f.Path))
                .ToList();

            Console.WriteLine($"Found {matchedFiles.Count} files matching pattern");
            Console.WriteLine();

            int indexed = 0;
            int skipped = 0;
            int errors = 0;

            foreach (var file in matchedFiles)
            {
                try
                {
                    // Validate file size before reading
                    try
                    {
                        ContentUtils.ValidateFileSize(file);
                    }
                    catch (InvalidOperationException ex)
                    {
                        errors++;
                        Console.Error.WriteLine($"  ⊙ Skipped (too large): {Path.GetFileName(file)} - {ex.Message}");
                        continue;
                    }

                    var fileContent = await File.ReadAllTextAsync(file, ct);
                    var contentHash = ContentUtils.ComputeHash(fileContent);

                    // Check if file already indexed and unchanged
                    var existing = await repository.GetBySourceAsync(file, ct);
                    if (!rebuild && existing != null && existing.ContentHash == contentHash)
                    {
                        skipped++;
                        if (config.VerboseLogging)
                        {
                            Console.WriteLine($"  ⊙ Skipped (unchanged): {Path.GetFileName(file)}");
                        }
                        continue;
                    }

                    // Extract title from first line or filename
                    var title = ContentUtils.ExtractTitle(fileContent, file);

                    var entry = new ContentEntry(
                        Id: Guid.NewGuid().ToString("N")[..12],
                        Title: title,
                        Content: fileContent,
                        Source: file,
                        Project: project,
                        Tags: null,
                        CreatedAt: existing?.CreatedAt ?? DateTimeOffset.UtcNow,
                        LastReinforcedAt: DateTimeOffset.UtcNow,
                        ContentHash: contentHash
                    );

                    await repository.AddAsync(entry, ct);
                    indexed++;

                    if (config.VerboseLogging)
                    {
                        Console.WriteLine($"  ✓ Indexed: {Path.GetFileName(file)}");
                    }
                    else if (indexed % 10 == 0)
                    {
                        Console.Write($"\r  Indexed: {indexed}/{matchedFiles.Count} files...");
                    }
                }
                catch (Exception ex)
                {
                    errors++;
                    Console.Error.WriteLine($"  ✗ Error indexing {Path.GetFileName(file)}: {ex.Message}");
                }
            }

            if (!config.VerboseLogging && indexed > 0)
            {
                Console.WriteLine($"\r  Indexed: {indexed} files    ");
            }

            Console.WriteLine();
            Console.WriteLine($"✓ Indexing complete!");
            Console.WriteLine($"  Indexed: {indexed}");
            Console.WriteLine($"  Skipped: {skipped}");
            if (errors > 0)
            {
                Console.WriteLine($"  Errors: {errors}");
            }
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Error indexing content: {ex.Message}");
            if (config.VerboseLogging)
            {
                Console.Error.WriteLine(ex.StackTrace);
            }
        }
    }

    private static async Task ExecuteAddAsync(
        string source,
        string title,
        string? contentText,
        string? project,
        string? tags,
        IContentRepository repository,
        ConfigurationService configService,
        CancellationToken ct)
    {
        var config = await configService.LoadConfigAsync(ct);

        try
        {
            // Read from stdin if content not provided
            if (string.IsNullOrWhiteSpace(contentText))
            {
                Console.WriteLine("Enter content (Ctrl+Z or Ctrl+D to finish):");
                contentText = await Console.In.ReadToEndAsync();

                if (string.IsNullOrWhiteSpace(contentText))
                {
                    Console.Error.WriteLine("Error: No content provided");
                    return;
                }
            }

            var contentHash = ContentUtils.ComputeHash(contentText);
            var tagArray = string.IsNullOrWhiteSpace(tags)
                ? null
                : tags.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);

            // Check if source already exists
            var existing = await repository.GetBySourceAsync(source, ct);

            var entry = new ContentEntry(
                Id: existing?.Id ?? Guid.NewGuid().ToString("N")[..12],
                Title: title,
                Content: contentText,
                Source: source,
                Project: project,
                Tags: tagArray,
                CreatedAt: existing?.CreatedAt ?? DateTimeOffset.UtcNow,
                LastReinforcedAt: DateTimeOffset.UtcNow,
                ContentHash: contentHash
            );

            await repository.AddAsync(entry, ct);

            if (existing != null)
            {
                Console.WriteLine($"✓ Updated content: {source}");
            }
            else
            {
                Console.WriteLine($"✓ Added content: {source}");
            }
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Error adding content: {ex.Message}");
            if (config.VerboseLogging)
            {
                Console.Error.WriteLine(ex.StackTrace);
            }
        }
    }

    private static async Task ExecuteSearchAsync(
        string query,
        int max,
        string? project,
        string? sourcePrefix,
        string? tags,
        bool robot,
        IContentRepository repository,
        ConfigurationService configService,
        CancellationToken ct)
    {
        var config = await configService.LoadConfigAsync(ct);

        try
        {
            max = Math.Clamp(max, 1, 100);

            var tagArray = string.IsNullOrWhiteSpace(tags)
                ? null
                : tags.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);

            var results = await repository.SearchAsync(query, project, sourcePrefix, tagArray, max, ct);

            if (robot)
            {
                var json = JsonSerializer.Serialize(results, new JsonSerializerOptions
                {
                    WriteIndented = true
                });
                Console.WriteLine(json);
                return;
            }

            if (results.Count == 0)
            {
                Console.WriteLine("No results found.");
                return;
            }

            Console.WriteLine($"Content Search Results ({results.Count} results):\n");

            for (int i = 0; i < results.Count; i++)
            {
                var result = results[i];
                var entry = result.Entry;
                var decayStatus = DecayCalculator.GetDecayStatus(result.DecayFactor);

                Console.WriteLine($"[{i + 1}] {entry.Title}");
                Console.WriteLine($"    Source: {entry.Source}");
                Console.WriteLine($"    Score: {result.Score:F2} | Decay: {result.DecayFactor:F2} ({decayStatus})");

                if (!string.IsNullOrWhiteSpace(entry.Project))
                {
                    Console.WriteLine($"    Project: {entry.Project}");
                }

                if (entry.Tags != null && entry.Tags.Length > 0)
                {
                    Console.WriteLine($"    Tags: {string.Join(", ", entry.Tags)}");
                }

                if (!string.IsNullOrWhiteSpace(result.Highlight))
                {
                    Console.WriteLine($"    {result.Highlight}");
                }

                Console.WriteLine();
            }
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Error searching content: {ex.Message}");
            if (config.VerboseLogging)
            {
                Console.Error.WriteLine(ex.StackTrace);
            }
        }
    }

    private static async Task ExecuteListAsync(
        string? project,
        string? sourcePrefix,
        string? tags,
        bool robot,
        bool expired,
        int limit,
        IContentRepository repository,
        ConfigurationService configService,
        CancellationToken ct)
    {
        var config = await configService.LoadConfigAsync(ct);

        try
        {
            limit = Math.Clamp(limit, 1, 1000);

            IReadOnlyList<ContentEntry> entries;

            if (expired)
            {
                entries = await repository.GetExpiredAsync(0.05, ct);
            }
            else
            {
                var tagArray = string.IsNullOrWhiteSpace(tags)
                    ? null
                    : tags.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);

                entries = await repository.ListAsync(project, sourcePrefix, tagArray, limit, ct);
            }

            if (robot)
            {
                var json = JsonSerializer.Serialize(entries, new JsonSerializerOptions
                {
                    WriteIndented = true
                });
                Console.WriteLine(json);
                return;
            }

            if (entries.Count == 0)
            {
                Console.WriteLine(expired ? "No expired content found." : "No content found.");
                return;
            }

            Console.WriteLine($"Indexed Content ({entries.Count} entries):\n");

            for (int i = 0; i < entries.Count; i++)
            {
                var entry = entries[i];
                var decayFactor = DecayCalculator.CalculateDecayFactor(entry.LastReinforcedAt.DateTime);
                var decayStatus = DecayCalculator.GetDecayStatus(decayFactor);

                Console.WriteLine($"[{i + 1}] {entry.Title}");
                Console.WriteLine($"    Source: {entry.Source}");
                Console.WriteLine($"    Decay: {decayFactor:F2} {RenderDecayBar(decayFactor)} ({decayStatus})");

                if (!string.IsNullOrWhiteSpace(entry.Project))
                {
                    Console.WriteLine($"    Project: {entry.Project}");
                }

                if (entry.Tags != null && entry.Tags.Length > 0)
                {
                    Console.WriteLine($"    Tags: {string.Join(", ", entry.Tags)}");
                }

                Console.WriteLine($"    Created: {FormatTimeSince(entry.CreatedAt)}");
                Console.WriteLine($"    Last Reinforced: {FormatTimeSince(entry.LastReinforcedAt)}");
                Console.WriteLine();
            }
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Error listing content: {ex.Message}");
            if (config.VerboseLogging)
            {
                Console.Error.WriteLine(ex.StackTrace);
            }
        }
    }

    private static async Task ExecuteRemoveAsync(
        string? id,
        string? source,
        string? sourcePrefix,
        string? project,
        bool all,
        bool force,
        IContentRepository repository,
        ConfigurationService configService,
        CancellationToken ct)
    {
        var config = await configService.LoadConfigAsync(ct);

        try
        {
            // Validate that at least one criteria is specified
            if (!all && string.IsNullOrWhiteSpace(id) && string.IsNullOrWhiteSpace(source) &&
                string.IsNullOrWhiteSpace(sourcePrefix) && string.IsNullOrWhiteSpace(project))
            {
                Console.Error.WriteLine("Error: At least one removal criteria must be specified.");
                Console.Error.WriteLine("Use --id, --source, --source-prefix, --project, or --all");
                return;
            }

            // Count entries to be deleted
            var count = await repository.CountByCriteriaAsync(id, source, sourcePrefix, project, all, ct);

            if (count == 0)
            {
                Console.WriteLine("No content found matching the criteria.");
                return;
            }

            // Display what will be deleted
            Console.WriteLine($"Found {count} content {(count == 1 ? "entry" : "entries")} matching:");
            if (!string.IsNullOrWhiteSpace(id))
            {
                Console.WriteLine($"  - ID: {id}");
            }
            if (!string.IsNullOrWhiteSpace(source))
            {
                Console.WriteLine($"  - Source: {source}");
            }
            if (!string.IsNullOrWhiteSpace(sourcePrefix))
            {
                Console.WriteLine($"  - Source prefix: {sourcePrefix}");
            }
            if (!string.IsNullOrWhiteSpace(project))
            {
                Console.WriteLine($"  - Project: {project}");
            }
            if (all)
            {
                Console.WriteLine("  - All content");
            }
            Console.WriteLine();

            // Confirm deletion unless --force is specified
            if (!force)
            {
                Console.Write($"Are you sure you want to delete {count} {(count == 1 ? "entry" : "entries")}? (y/N): ");
                var response = Console.ReadLine()?.Trim().ToLowerInvariant();

                if (response != "y" && response != "yes")
                {
                    Console.WriteLine("Deletion cancelled.");
                    return;
                }
            }

            // Perform deletion
            var deleted = await repository.DeleteByCriteriaAsync(id, source, sourcePrefix, project, all, ct);

            Console.WriteLine($"✓ Removed {deleted} content {(deleted == 1 ? "entry" : "entries")}");
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Error removing content: {ex.Message}");
            if (config.VerboseLogging)
            {
                Console.Error.WriteLine(ex.StackTrace);
            }
        }
    }

    private static async Task ExecuteReinforceAsync(
        string source,
        IContentRepository repository,
        ConfigurationService configService,
        CancellationToken ct)
    {
        var config = await configService.LoadConfigAsync(ct);

        try
        {
            var reinforced = await repository.ReinforceAsync(source, ct);

            if (reinforced)
            {
                Console.WriteLine($"✓ Reinforced content: {source}");
            }
            else
            {
                Console.WriteLine($"Content not found: {source}");
            }
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Error reinforcing content: {ex.Message}");
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

    private static string FormatTimeSince(DateTimeOffset timestamp)
    {
        var span = DateTimeOffset.UtcNow - timestamp;

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
}
