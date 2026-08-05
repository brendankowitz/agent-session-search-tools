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
    // Options are held as fields so handler binding is by reference. The previous code bound via
    // positional option-index casts, where inserting an option silently rebound every handler
    // argument after it.
    private readonly Argument<string> _sessionIdArgument;
    private readonly Option<string> _formatOption;
    private readonly Option<string?> _outputOption;
    private readonly Option<bool> _stdoutOption;
    private readonly Option<int?> _lastOption;

    private ExportCommand() : base("export", "Export a session to a file")
    {
        _sessionIdArgument = new Argument<string>(
            name: "session-id",
            description: "ID of the session to export");

        _formatOption = new Option<string>(
            name: "--format",
            getDefaultValue: () => "html",
            description: "Export format: html, md, or json");
        _formatOption.AddAlias("-f");

        _outputOption = new Option<string?>(
            name: "--output",
            description: "Output file path (default: session-{id}.{ext})");
        _outputOption.AddAlias("-o");

        _stdoutOption = new Option<bool>(
            name: "--stdout",
            description: "Write output to stdout instead of a file");

        _lastOption = new Option<int?>(
            name: "--last",
            description: "Export only the last N messages of the session");
        _lastOption.AddAlias("-n");

        this.AddArgument(_sessionIdArgument);
        this.AddOption(_formatOption);
        this.AddOption(_outputOption);
        this.AddOption(_stdoutOption);
        this.AddOption(_lastOption);
    }

    public static Command Create(IServiceProvider serviceProvider)
    {
        var command = new ExportCommand();

        command.SetHandler(async (sessionId, format, output, stdout, last) =>
        {
            var configService = serviceProvider.GetRequiredService<ConfigurationService>();
            var repository = serviceProvider.GetRequiredService<ISessionRepository>();
            var exporters = serviceProvider.GetRequiredService<IEnumerable<IExporter>>();

            await ExecuteAsync(
                sessionId,
                format,
                output,
                stdout,
                last,
                configService,
                repository,
                exporters,
                CancellationToken.None);
        },
        command._sessionIdArgument,
        command._formatOption,
        command._outputOption,
        command._stdoutOption,
        command._lastOption);

        return command;
    }

    private static async Task ExecuteAsync(
        string sessionId,
        string? format,
        string? output,
        bool stdout,
        int? last,
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
            CommandOutcome.Fail();
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
            CommandOutcome.Fail(CommandOutcome.NotFound);
            return;
        }

        if (last.HasValue)
        {
            if (last.Value <= 0)
            {
                Console.Error.WriteLine("Error: --last must be greater than zero");
                CommandOutcome.Fail();
                return;
            }

            var totalMessages = session.MessageCount;
            session = session.WithLastMessages(last.Value);

            if (!stdout)
            {
                Console.WriteLine($"Messages: last {session.MessageCount} of {totalMessages}");
            }
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
            CommandOutcome.Fail();
            if (config.VerboseLogging)
            {
                Console.Error.WriteLine(ex.StackTrace);
            }
        }
    }
}
