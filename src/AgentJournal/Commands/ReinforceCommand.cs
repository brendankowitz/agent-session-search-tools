using System.CommandLine;
using Microsoft.Extensions.DependencyInjection;
using AgentJournal.Configuration;
using AgentJournal.Core.Knowledge;
using AgentJournal.Core.Search;

namespace AgentJournal.Commands;

/// <summary>
/// Command to reinforce knowledge entries (reset decay timer)
/// </summary>
public class ReinforceCommand : Command
{
    private readonly Argument<string[]> _idsArgument;
    private readonly Option<string?> _matchOption;
    private readonly Option<string?> _projectOption;
    private readonly Option<bool> _decayingOption;
    private readonly Option<bool> _expiringOption;

    private ReinforceCommand() : base("reinforce", "Reinforce knowledge entries to reset decay timer")
    {
        _idsArgument = new Argument<string[]>(
            name: "ids",
            description: "IDs of knowledge entries to reinforce",
            getDefaultValue: () => Array.Empty<string>());

        _matchOption = new Option<string?>(
            name: "--match",
            description: "Reinforce entries matching this query");
        _matchOption.AddAlias("-m");

        _projectOption = new Option<string?>(
            name: "--project",
            description: "Reinforce entries from this project");
        _projectOption.AddAlias("-p");

        _decayingOption = new Option<bool>(
            name: "--decaying",
            description: "Reinforce all decaying entries (decay < 0.5)");

        _expiringOption = new Option<bool>(
            name: "--expiring",
            description: "Reinforce all expiring entries (decay < 0.1)");

        this.AddArgument(_idsArgument);
        this.AddOption(_matchOption);
        this.AddOption(_projectOption);
        this.AddOption(_decayingOption);
        this.AddOption(_expiringOption);
    }

    public static Command Create(IServiceProvider serviceProvider)
    {
        var command = new ReinforceCommand();

        command.SetHandler(async (ids, match, project, decaying, expiring) =>
        {
            var repository = serviceProvider.GetRequiredService<IKnowledgeRepository>();
            var configService = serviceProvider.GetRequiredService<ConfigurationService>();

            await ExecuteAsync(
                ids,
                match,
                project,
                decaying,
                expiring,
                repository,
                configService,
                CancellationToken.None);
        },
        command._idsArgument,
        command._matchOption,
        command._projectOption,
        command._decayingOption,
        command._expiringOption);

        return command;
    }

    private static async Task ExecuteAsync(
        string[] ids,
        string? match,
        string? project,
        bool decaying,
        bool expiring,
        IKnowledgeRepository repository,
        ConfigurationService configService,
        CancellationToken ct)
    {
        var config = await configService.LoadConfigAsync(ct);

        try
        {
            // Reinforce specific IDs
            if (ids.Length > 0)
            {
                await ReinforceByIdsAsync(ids, repository, ct);
                return;
            }

            // Reinforce by match query
            if (!string.IsNullOrWhiteSpace(match))
            {
                await ReinforceByMatchAsync(match, project, repository, ct);
                return;
            }

            // Reinforce expiring entries
            if (expiring)
            {
                await ReinforceExpiringAsync(project, repository, ct);
                return;
            }

            // Reinforce decaying entries
            if (decaying)
            {
                await ReinforceDecayingAsync(project, repository, ct);
                return;
            }

            Console.Error.WriteLine("Error: Must specify IDs, --match, --decaying, or --expiring");
            Console.Error.WriteLine("Examples:");
            Console.Error.WriteLine("  agent-journal reinforce abc123 def456");
            Console.Error.WriteLine("  agent-journal reinforce --match \"important\"");
            Console.Error.WriteLine("  agent-journal reinforce --decaying");
            CommandOutcome.Fail();
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Error reinforcing knowledge: {ex.Message}");
            CommandOutcome.Fail();
            if (config.VerboseLogging)
            {
                Console.Error.WriteLine(ex.StackTrace);
            }
        }
    }

    private static async Task ReinforceByIdsAsync(
        string[] ids,
        IKnowledgeRepository repository,
        CancellationToken ct)
    {
        Console.WriteLine($"Reinforcing {ids.Length} knowledge entries...");

        // Use batch operation for efficiency
        var reinforced = await repository.ReinforceManyAsync(ids, ct);

        Console.WriteLine();
        Console.WriteLine($"✓ Reinforced {reinforced} of {ids.Length} knowledge entries");
    }

    private static async Task ReinforceByMatchAsync(
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
            var decayFactor = DecayCalculator.CalculateDecayFactor(result.Entry.LastReinforcedAt);
            Console.WriteLine($"  - {result.Entry.Id} (decay: {decayFactor:F2}): {TruncateContent(result.Entry.Content, 50)}");
        }

        if (results.Count > 10)
        {
            Console.WriteLine($"  ... and {results.Count - 10} more");
        }

        Console.WriteLine();
        Console.WriteLine($"Reinforcing {results.Count} entries...");

        var ids = results.Select(r => r.Entry.Id).ToList();
        var reinforced = await repository.ReinforceManyAsync(ids, ct);

        Console.WriteLine($"✓ Reinforced {reinforced} knowledge entries");
    }

    private static async Task ReinforceDecayingAsync(
        string? project,
        IKnowledgeRepository repository,
        CancellationToken ct)
    {
        var entries = await repository.ListAsync(project: project, includeDecaying: true, limit: int.MaxValue, ct: ct);

        // Filter to decaying entries (decay factor < 0.5)
        var decayingEntries = entries
            .Where(e => DecayCalculator.CalculateDecayFactor(e.LastReinforcedAt) < 0.5)
            .ToList();

        if (decayingEntries.Count == 0)
        {
            Console.WriteLine("No decaying knowledge entries found.");
            return;
        }

        Console.WriteLine($"Found {decayingEntries.Count} decaying entries:");
        foreach (var entry in decayingEntries.Take(10))
        {
            var decayFactor = DecayCalculator.CalculateDecayFactor(entry.LastReinforcedAt);
            Console.WriteLine($"  - {entry.Id} (decay: {decayFactor:F2}): {TruncateContent(entry.Content, 50)}");
        }

        if (decayingEntries.Count > 10)
        {
            Console.WriteLine($"  ... and {decayingEntries.Count - 10} more");
        }

        Console.WriteLine();
        Console.WriteLine($"Reinforcing {decayingEntries.Count} entries...");

        var ids = decayingEntries.Select(e => e.Id).ToList();
        var reinforced = await repository.ReinforceManyAsync(ids, ct);

        Console.WriteLine($"✓ Reinforced {reinforced} knowledge entries");
    }

    private static async Task ReinforceExpiringAsync(
        string? project,
        IKnowledgeRepository repository,
        CancellationToken ct)
    {
        var entries = await repository.ListAsync(project: project, includeDecaying: true, limit: int.MaxValue, ct: ct);

        // Filter to expiring entries (decay factor < 0.1)
        var expiringEntries = entries
            .Where(e => DecayCalculator.CalculateDecayFactor(e.LastReinforcedAt) < 0.1)
            .ToList();

        if (expiringEntries.Count == 0)
        {
            Console.WriteLine("No expiring knowledge entries found.");
            return;
        }

        Console.WriteLine($"Found {expiringEntries.Count} expiring entries:");
        foreach (var entry in expiringEntries.Take(10))
        {
            var decayFactor = DecayCalculator.CalculateDecayFactor(entry.LastReinforcedAt);
            Console.WriteLine($"  - {entry.Id} (decay: {decayFactor:F2}): {TruncateContent(entry.Content, 50)}");
        }

        if (expiringEntries.Count > 10)
        {
            Console.WriteLine($"  ... and {expiringEntries.Count - 10} more");
        }

        Console.WriteLine();
        Console.WriteLine($"Reinforcing {expiringEntries.Count} entries...");

        var ids = expiringEntries.Select(e => e.Id).ToList();
        var reinforced = await repository.ReinforceManyAsync(ids, ct);

        Console.WriteLine($"✓ Reinforced {reinforced} knowledge entries");
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
