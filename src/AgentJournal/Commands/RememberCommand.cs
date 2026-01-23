using System.CommandLine;
using Microsoft.Extensions.DependencyInjection;
using AgentJournal.Configuration;
using AgentJournal.Core.Knowledge;
using AgentJournal.Core.Models;

namespace AgentJournal.Commands;

/// <summary>
/// Command to add knowledge to the knowledge bank
/// </summary>
public class RememberCommand : Command
{
    private RememberCommand() : base("remember", "Store knowledge in the knowledge bank")
    {
        var contentArgument = new Argument<string>(
            name: "content",
            description: "The knowledge content to remember");

        var tagsOption = new Option<string?>(
            name: "--tags",
            description: "Comma-separated tags (e.g., code-style,linting)");
        tagsOption.AddAlias("-t");

        var projectOption = new Option<string?>(
            name: "--project",
            description: "Project name or path associated with this knowledge");
        projectOption.AddAlias("-p");

        var sourceOption = new Option<string?>(
            name: "--source",
            description: "Source of the knowledge (e.g., URL, document)");
        sourceOption.AddAlias("-s");

        this.AddArgument(contentArgument);
        this.AddOption(tagsOption);
        this.AddOption(projectOption);
        this.AddOption(sourceOption);
    }

    public static Command Create(IServiceProvider serviceProvider)
    {
        var command = new RememberCommand();

        command.SetHandler(async (content, tags, project, source) =>
        {
            var repository = serviceProvider.GetRequiredService<IKnowledgeRepository>();
            var configService = serviceProvider.GetRequiredService<ConfigurationService>();

            await ExecuteAsync(
                content,
                tags,
                project,
                source,
                repository,
                configService,
                CancellationToken.None);
        },
        command.Arguments[0] as Argument<string> ?? throw new InvalidOperationException("Missing content argument"),
        command.Options[0] as Option<string?> ?? throw new InvalidOperationException("Missing tags option"),
        command.Options[1] as Option<string?> ?? throw new InvalidOperationException("Missing project option"),
        command.Options[2] as Option<string?> ?? throw new InvalidOperationException("Missing source option"));

        return command;
    }

    private static async Task ExecuteAsync(
        string content,
        string? tags,
        string? project,
        string? source,
        IKnowledgeRepository repository,
        ConfigurationService configService,
        CancellationToken ct)
    {
        var config = await configService.LoadConfigAsync(ct);

        try
        {
            // Validate input
            if (string.IsNullOrWhiteSpace(content))
            {
                Console.Error.WriteLine("Error: Content cannot be empty");
                return;
            }

            if (content.Length > 10_000)
            {
                Console.Error.WriteLine("Error: Content exceeds maximum length of 10,000 characters");
                return;
            }

            // Parse tags
            var tagArray = string.IsNullOrWhiteSpace(tags)
                ? Array.Empty<string>()
                : tags.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);

            // Create knowledge entry
            var id = GenerateId();
            var now = DateTime.UtcNow;
            var entry = new KnowledgeEntry(
                Id: id,
                Content: content,
                Tags: tagArray,
                Project: project,
                Source: source,
                CreatedAt: now,
                LastReinforcedAt: now,
                ReinforcementCount: 0
            );

            // Save to repository
            var saved = await repository.SaveAsync(entry, ct);

            Console.WriteLine($"✓ Knowledge stored successfully");
            Console.WriteLine($"  ID: {saved.Id}");
            Console.WriteLine($"  Content: {TruncateContent(saved.Content, 80)}");

            if (saved.Tags.Length > 0)
            {
                Console.WriteLine($"  Tags: {string.Join(", ", saved.Tags)}");
            }

            if (!string.IsNullOrWhiteSpace(saved.Project))
            {
                Console.WriteLine($"  Project: {saved.Project}");
            }

            if (!string.IsNullOrWhiteSpace(saved.Source))
            {
                Console.WriteLine($"  Source: {saved.Source}");
            }
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Error storing knowledge: {ex.Message}");
            if (config.VerboseLogging)
            {
                Console.Error.WriteLine(ex.StackTrace);
            }
        }
    }

    private static string GenerateId()
    {
        return Guid.NewGuid().ToString("N")[..12];
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
