using AgentJournal.Core.Models;

namespace AgentJournal.Core.Export;

/// <summary>
/// Export format
/// </summary>
public enum ExportFormat
{
    /// <summary>
    /// HTML format
    /// </summary>
    Html,

    /// <summary>
    /// Markdown format
    /// </summary>
    Markdown,

    /// <summary>
    /// JSON format
    /// </summary>
    Json
}

/// <summary>
/// Interface for exporting sessions to various formats
/// </summary>
public interface IExporter
{
    /// <summary>
    /// Gets the export format this exporter supports
    /// </summary>
    ExportFormat Format { get; }

    /// <summary>
    /// Gets the file extension for exported files
    /// </summary>
    string FileExtension { get; }

    /// <summary>
    /// Exports a session to a string
    /// </summary>
    /// <param name="session">The session to export</param>
    /// <param name="ct">Cancellation token</param>
    /// <returns>The exported content as a string</returns>
    Task<string> ExportAsync(Session session, CancellationToken ct = default);

    /// <summary>
    /// Exports a session to a file
    /// </summary>
    /// <param name="session">The session to export</param>
    /// <param name="outputPath">The output file path</param>
    /// <param name="ct">Cancellation token</param>
    Task ExportToFileAsync(Session session, string outputPath, CancellationToken ct = default);

    /// <summary>
    /// Exports multiple sessions to a single output
    /// </summary>
    /// <param name="sessions">The sessions to export</param>
    /// <param name="ct">Cancellation token</param>
    /// <returns>The exported content as a string</returns>
    Task<string> ExportMultipleAsync(IEnumerable<Session> sessions, CancellationToken ct = default);
}
