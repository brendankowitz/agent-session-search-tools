using System.CommandLine;
using Microsoft.Extensions.DependencyInjection;
using AgentJournal.Configuration;
using AgentJournal.Core.Export;
using AgentJournal.Core.Storage;

namespace AgentJournal.Commands;

/// <summary>
/// Command to export sessions to various formats
/// </summary>
public class ExportCommand : Command
{
    private ExportCommand() : base("export", "Export a session to a file")
    {
        var sessionIdArgument = new Argument<string>(
            name: "session-id",
            description: "ID of the session to export");

        var formatOption = new Option<string>(
            name: "--format",
            getDefaultValue: () => "html",
            description: "Export format: html, md, or json");
        formatOption.AddAlias("-f");

        var outputOption = new Option<string?>(
            name: "--output",
            description: "Output file path (default: session-{id}.{ext})");
        outputOption.AddAlias("-o");

        var stdoutOption = new Option<bool>(
            name: "--stdout",
            description: "Write output to stdout instead of a file");

        this.AddArgument(sessionIdArgument);
        this.AddOption(formatOption);
        this.AddOption(outputOption);
        this.AddOption(stdoutOption);
    }

    public static Command Create(IServiceProvider serviceProvider)
    {
        var command = new ExportCommand();
        
        command.SetHandler(async (sessionId, format, output, stdout) =>
        {
            var configService = serviceProvider.GetRequiredService<ConfigurationService>();
            var repository = serviceProvider.GetRequiredService<ISessionRepository>();
            var exporters = serviceProvider.GetRequiredService<IEnumerable<IExporter>>();

            await ExecuteAsync(
                sessionId,
                format,
                output,
                stdout,
                configService,
                repository,
                exporters,
                CancellationToken.None);
        }, 
        command.Arguments[0] as Argument<string> ?? throw new InvalidOperationException("Missing session-id argument"),
        command.Options[0] as Option<string> ?? throw new InvalidOperationException("Missing format option"),
        command.Options[1] as Option<string?> ?? throw new InvalidOperationException("Missing output option"),
        command.Options[2] as Option<bool> ?? throw new InvalidOperationException("Missing stdout option"));

        return command;
    }

    private static async Task ExecuteAsync(
        string sessionId,
        string? format,
        string? output,
        bool stdout,
        ConfigurationService configService,
        ISessionRepository repository,
        IEnumerable<IExporter> exporters,
        CancellationToken ct)
    {
        var config = await configService.LoadConfigAsync(ct);

        // Parse export format
        var exportFormat = format?.ToLowerInvariant() switch
        {
            "md" or "markdown" => ExportFormat.Markdown,
            "json" => ExportFormat.Json,
            _ => ExportFormat.Html
        };

        // Get the appropriate exporter
        var exporter = exporters.FirstOrDefault(e => e.Format == exportFormat);
        if (exporter == null)
        {
            Console.Error.WriteLine($"Error: No exporter found for format '{format}'");
            return;
        }

        if (!stdout)
        {
            Console.WriteLine($"Exporting session: {sessionId}");
            Console.WriteLine($"Format: {exportFormat}");
        }

        // Load session from repository
        var session = await repository.GetSessionAsync(sessionId, ct);
        if (session == null)
        {
            Console.Error.WriteLine($"Error: Session '{sessionId}' not found");
            Console.Error.WriteLine("Use 'aj search' to find available sessions");
            return;
        }

        // Export the session
        try
        {
            if (stdout)
            {
                // Export to stdout
                var content = await exporter.ExportAsync(session, ct);
                Console.WriteLine(content);
            }
            else
            {
                // Determine output path
                var outputPath = output;
                if (string.IsNullOrWhiteSpace(outputPath))
                {
                    outputPath = $"session-{sessionId}.{exporter.FileExtension}";
                }

                // Expand relative paths
                if (!Path.IsPathRooted(outputPath))
                {
                    outputPath = Path.Combine(Environment.CurrentDirectory, outputPath);
                }

                Console.WriteLine($"Output: {outputPath}");

                // Export to file
                await exporter.ExportToFileAsync(session, outputPath, ct);

                Console.WriteLine($"Export complete!");
                Console.WriteLine($"Session exported: {session.MessageCount} messages");
                
                if (session.ToolCallCount > 0)
                {
                    Console.WriteLine($"Tool calls included: {session.ToolCallCount}");
                }
            }
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Error exporting session: {ex.Message}");
            if (config.VerboseLogging)
            {
                Console.Error.WriteLine(ex.StackTrace);
            }
        }
    }
}
