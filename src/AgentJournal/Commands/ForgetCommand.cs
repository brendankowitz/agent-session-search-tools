using System.CommandLine;
using Microsoft.Extensions.DependencyInjection;
using AgentJournal.Configuration;
using AgentJournal.Core.Knowledge;
using AgentJournal.Core.Search;

namespace AgentJournal.Commands;

/// <summary>
/// Command to delete knowledge from the knowledge bank
/// </summary>
public class ForgetCommand : Command
{
    private readonly Argument<string?> _idArgument;
    private readonly Option<string?> _matchOption;
    private readonly Option<string?> _projectOption;
    private readonly Option<bool> _allOption;
    private readonly Option<bool> _confirmOption;

    private ForgetCommand() : base("forget", "Delete knowledge from the knowledge bank")
    {
        _idArgument = new Argument<string?>(
            name: "id",
            description: "ID of the knowledge entry to delete",
            getDefaultValue: () => null);

        _matchOption = new Option<string?>(
            name: "--match",
            description: "Delete entries matching this query");
        _matchOption.AddAlias("-m");

        _projectOption = new Option<string?>(
            name: "--project",
            description: "Delete entries from this project");
        _projectOption.AddAlias("-p");

        _allOption = new Option<bool>(
            name: "--all",
            description: "Delete all entries (requires --confirm)");

        _confirmOption = new Option<bool>(
            name: "--confirm",
            description: "Confirm deletion (required for batch operations)");
        _confirmOption.AddAlias("-y");

        this.AddArgument(_idArgument);
        this.AddOption(_matchOption);
        this.AddOption(_projectOption);
        this.AddOption(_allOption);
        this.AddOption(_confirmOption);
    }

    public static Command Create(IServiceProvider serviceProvider)
    {
        var command = new ForgetCommand();

        command.SetHandler(async (id, match, project, all, confirm) =>
        {
            var repository = serviceProvider.GetRequiredService<IKnowledgeRepository>();
            var configService = serviceProvider.GetRequiredService<ConfigurationService>();

            await ExecuteAsync(
                id,
                match,
                project,
                all,
                confirm,
                repository,
                configService,
                CancellationToken.None);
        },
        command._idArgument,
        command._matchOption,
        command._projectOption,
        command._allOption,
        command._confirmOption);

        return command;
    }

    private static async Task ExecuteAsync(
        string? id,
        string? match,
        string? project,
        bool all,
        bool confirm,
        IKnowledgeRepository repository,
        ConfigurationService configService,
        CancellationToken ct)
    {
        var config = await configService.LoadConfigAsync(ct);

        try
        {
            // Single ID deletion
            if (!string.IsNullOrWhiteSpace(id))
            {
                await DeleteSingleAsync(id, repository, ct);
                return;
            }

            // Batch deletion requires confirmation
            if (!confirm)
            {
                Console.Error.WriteLine("Error: Batch deletion requires --confirm flag");
                Console.Error.WriteLine("Use --confirm to proceed with deletion");
                CommandOutcome.Fail();
                return;
            }

            // Delete all entries
            if (all)
            {
                await DeleteAllAsync(repository, ct);
                return;
            }

            // Delete by match query
            if (!string.IsNullOrWhiteSpace(match))
            {
                await DeleteByMatchAsync(match, project, repository, ct);
                return;
            }

            // Delete by project
            if (!string.IsNullOrWhiteSpace(project))
            {
                await DeleteByProjectAsync(project, repository, ct);
                return;
            }

            Console.Error.WriteLine("Error: Must specify an ID, --match, --project, or --all");
            Console.Error.WriteLine("Examples:");
            Console.Error.WriteLine("  agent-journal forget abc123");
            Console.Error.WriteLine("  agent-journal forget --match \"old convention\" --confirm");
            Console.Error.WriteLine("  agent-journal forget --project my-app --confirm");
            CommandOutcome.Fail();
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Error deleting knowledge: {ex.Message}");
            CommandOutcome.Fail();
            if (config.VerboseLogging)
            {
                Console.Error.WriteLine(ex.StackTrace);
            }
        }
    }

    private static async Task DeleteSingleAsync(string id, IKnowledgeRepository repository, CancellationToken ct)
    {
        var entry = await repository.GetAsync(id, ct);
        if (entry == null)
        {
            Console.Error.WriteLine($"Error: Knowledge entry '{id}' not found");
            CommandOutcome.Fail(CommandOutcome.NotFound);
            return;
        }

        var deleted = await repository.DeleteAsync(id, ct);
        if (deleted)
        {
            Console.WriteLine($"✓ Knowledge entry deleted");
            Console.WriteLine($"  ID: {id}");
            Console.WriteLine($"  Content: {TruncateContent(entry.Content, 80)}");
        }
        else
        {
            Console.Error.WriteLine($"Error: Failed to delete entry '{id}'");
            CommandOutcome.Fail();
        }
    }

    private static async Task DeleteAllAsync(IKnowledgeRepository repository, CancellationToken ct)
    {
        var entries = await repository.ListAsync(limit: int.MaxValue, ct: ct);

        Console.WriteLine($"Deleting {entries.Count} knowledge entries...");

        var ids = entries.Select(e => e.Id).ToList();
        var deleted = await repository.DeleteManyAsync(ids, ct);

        Console.WriteLine($"✓ Deleted {deleted} knowledge entries");
    }

    private static async Task DeleteByMatchAsync(
        string match,
        string? project,
        IKnowledgeRepository repository,
        CancellationToken ct)
    {
        // Search for matching entries
        var results = await repository.SearchAsync(match, project: project, maxResults: 100, ct: ct);

        if (results.Count == 0)
        {
            Console.WriteLine("No matching knowledge entries found.");
            return;
        }

        Console.WriteLine($"Found {results.Count} matching entries:");
        foreach (var result in results.Take(10))
        {
            Console.WriteLine($"  - {result.Entry.Id}: {TruncateContent(result.Entry.Content, 60)}");
        }

        if (results.Count > 10)
        {
            Console.WriteLine($"  ... and {results.Count - 10} more");
        }

        Console.WriteLine();
        Console.WriteLine($"Deleting {results.Count} entries...");

        var ids = results.Select(r => r.Entry.Id).ToList();
        var deleted = await repository.DeleteManyAsync(ids, ct);

        Console.WriteLine($"✓ Deleted {deleted} knowledge entries");
    }

    private static async Task DeleteByProjectAsync(
        string project,
        IKnowledgeRepository repository,
        CancellationToken ct)
    {
        var entries = await repository.ListAsync(project: project, limit: int.MaxValue, ct: ct);

        if (entries.Count == 0)
        {
            Console.WriteLine($"No knowledge entries found for project '{project}'.");
            return;
        }

        Console.WriteLine($"Deleting {entries.Count} entries from project '{project}'...");

        var ids = entries.Select(e => e.Id).ToList();
        var deleted = await repository.DeleteManyAsync(ids, ct);

        Console.WriteLine($"✓ Deleted {deleted} knowledge entries");
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
