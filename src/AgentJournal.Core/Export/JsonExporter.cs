using System.Text.Json;
using AgentJournal.Core.Models;

namespace AgentJournal.Core.Export;

/// <summary>
/// Exports sessions to JSON format
/// </summary>
public class JsonExporter : IExporter
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = true,
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase
    };

    public ExportFormat Format => ExportFormat.Json;
    public string FileExtension => ".json";

    public Task<string> ExportAsync(Session session, CancellationToken ct = default)
    {
        var json = JsonSerializer.Serialize(session, JsonOptions);
        return Task.FromResult(json);
    }

    public async Task ExportToFileAsync(Session session, string outputPath, CancellationToken ct = default)
    {
        var content = await ExportAsync(session, ct);
        await File.WriteAllTextAsync(outputPath, content, ct);
    }

    public Task<string> ExportMultipleAsync(IEnumerable<Session> sessions, CancellationToken ct = default)
    {
        var json = JsonSerializer.Serialize(sessions, JsonOptions);
        return Task.FromResult(json);
    }
}
