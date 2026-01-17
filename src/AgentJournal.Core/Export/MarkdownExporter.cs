using System.Text;
using AgentJournal.Core.Models;

namespace AgentJournal.Core.Export;

/// <summary>
/// Exports sessions to Markdown format
/// </summary>
public class MarkdownExporter : IExporter
{
    private readonly ExportOptions _options;

    public ExportFormat Format => ExportFormat.Markdown;
    public string FileExtension => ".md";

    public MarkdownExporter(ExportOptions? options = null)
    {
        _options = options ?? ExportOptions.Default;
    }

    public Task<string> ExportAsync(Session session, CancellationToken ct = default)
    {
        var sb = new StringBuilder();
        
        // Header
        sb.AppendLine($"# {session.AgentType} Session");
        sb.AppendLine();
        sb.AppendLine("## Session Information");
        sb.AppendLine();
        sb.AppendLine($"- **Project**: {session.ProjectPath ?? "N/A"}");
        sb.AppendLine($"- **Branch**: {session.GitBranch ?? "N/A"}");
        sb.AppendLine($"- **Started**: {session.StartedAt:yyyy-MM-dd HH:mm:ss}");
        
        if (session.EndedAt.HasValue)
        {
            sb.AppendLine($"- **Ended**: {session.EndedAt.Value:yyyy-MM-dd HH:mm:ss}");
            sb.AppendLine($"- **Duration**: {FormatDuration(session.Duration)}");
        }
        
        sb.AppendLine($"- **Messages**: {session.MessageCount} ({session.UserMessageCount} user, {session.AssistantMessageCount} assistant)");
        sb.AppendLine($"- **Tool Calls**: {session.ToolCallCount}");
        
        if (!string.IsNullOrEmpty(session.AgentVersion))
        {
            sb.AppendLine($"- **Agent Version**: {session.AgentVersion}");
        }

        if (!string.IsNullOrEmpty(session.Summary))
        {
            sb.AppendLine();
            sb.AppendLine("## Summary");
            sb.AppendLine();
            sb.AppendLine(session.Summary);
        }

        sb.AppendLine();
        sb.AppendLine("---");
        sb.AppendLine();
        sb.AppendLine("## Conversation");
        sb.AppendLine();

        // Messages
        foreach (var message in session.Messages)
        {
            AppendMessage(sb, message);
        }

        // Footer
        sb.AppendLine();
        sb.AppendLine("---");
        sb.AppendLine();
        sb.AppendLine($"*Exported on {DateTime.Now:yyyy-MM-dd HH:mm:ss} by AgentJournal*");

        return Task.FromResult(sb.ToString());
    }

    public async Task ExportToFileAsync(Session session, string outputPath, CancellationToken ct = default)
    {
        var content = await ExportAsync(session, ct);
        await File.WriteAllTextAsync(outputPath, content, ct);
    }

    public Task<string> ExportMultipleAsync(IEnumerable<Session> sessions, CancellationToken ct = default)
    {
        var sessionList = sessions.ToList();
        var sb = new StringBuilder();

        // Header
        sb.AppendLine("# Agent Sessions Export");
        sb.AppendLine();
        sb.AppendLine($"**Total Sessions**: {sessionList.Count}");
        sb.AppendLine($"**Exported**: {DateTime.Now:yyyy-MM-dd HH:mm:ss}");
        sb.AppendLine();
        sb.AppendLine("---");
        sb.AppendLine();

        // Table of contents
        sb.AppendLine("## Table of Contents");
        sb.AppendLine();
        for (int i = 0; i < sessionList.Count; i++)
        {
            var session = sessionList[i];
            sb.AppendLine($"{i + 1}. [{session.AgentType} - {session.StartedAt:yyyy-MM-dd HH:mm}](#{MakeLinkId(session.Id)})");
        }
        sb.AppendLine();
        sb.AppendLine("---");
        sb.AppendLine();

        // Sessions
        foreach (var session in sessionList)
        {
            AppendSessionSummary(sb, session);
            sb.AppendLine();
        }

        return Task.FromResult(sb.ToString());
    }

    private void AppendMessage(StringBuilder sb, Message message)
    {
        // Role header
        var roleIcon = message.Role switch
        {
            MessageRole.User => "👤",
            MessageRole.Assistant => "🤖",
            MessageRole.System => "⚙️",
            MessageRole.Tool => "🔧",
            _ => "❓"
        };

        sb.AppendLine($"### {roleIcon} {message.Role}");
        sb.AppendLine();

        // Timestamp
        if (_options.IncludeTimestamps)
        {
            sb.AppendLine($"*{message.Timestamp:yyyy-MM-dd HH:mm:ss}*");
            sb.AppendLine();
        }

        // Content
        if (!string.IsNullOrEmpty(message.Content))
        {
            sb.AppendLine(message.Content);
            sb.AppendLine();
        }

        // Model info
        if (!string.IsNullOrEmpty(message.Model))
        {
            sb.AppendLine($"<sub>Model: {message.Model}</sub>");
            sb.AppendLine();
        }

        // Tool calls
        if (_options.IncludeToolCalls && message.HasToolCalls)
        {
            sb.AppendLine("#### 🔧 Tool Calls");
            sb.AppendLine();

            foreach (var tool in message.ToolCalls!)
            {
                sb.AppendLine($"##### {tool.Name}");
                sb.AppendLine();

                if (!string.IsNullOrEmpty(tool.Arguments))
                {
                    sb.AppendLine("**Arguments:**");
                    sb.AppendLine("```json");
                    sb.AppendLine(tool.Arguments);
                    sb.AppendLine("```");
                    sb.AppendLine();
                }

                if (!string.IsNullOrEmpty(tool.Result))
                {
                    sb.AppendLine("**Result:**");
                    sb.AppendLine("```");
                    
                    var result = tool.Result;
                    if (_options.MaxToolResultLength.HasValue && result.Length > _options.MaxToolResultLength.Value)
                    {
                        result = result[.._options.MaxToolResultLength.Value] + "\n\n... (truncated)";
                    }
                    
                    sb.AppendLine(result);
                    sb.AppendLine("```");
                    sb.AppendLine();
                }

                if (tool.Success.HasValue)
                {
                    var status = tool.Success.Value ? "✅ Success" : "❌ Failed";
                    sb.AppendLine($"**Status:** {status}");
                    sb.AppendLine();
                }
            }
        }

        sb.AppendLine("---");
        sb.AppendLine();
    }

    private void AppendSessionSummary(StringBuilder sb, Session session)
    {
        sb.AppendLine($"## {session.AgentType} Session {{#{MakeLinkId(session.Id)}}}");
        sb.AppendLine();
        
        sb.AppendLine("### Session Information");
        sb.AppendLine();
        sb.AppendLine($"- **ID**: `{session.Id}`");
        sb.AppendLine($"- **Project**: {session.ProjectPath ?? "N/A"}");
        sb.AppendLine($"- **Branch**: {session.GitBranch ?? "N/A"}");
        sb.AppendLine($"- **Started**: {session.StartedAt:yyyy-MM-dd HH:mm:ss}");
        
        if (session.EndedAt.HasValue)
        {
            sb.AppendLine($"- **Ended**: {session.EndedAt.Value:yyyy-MM-dd HH:mm:ss}");
            sb.AppendLine($"- **Duration**: {FormatDuration(session.Duration)}");
        }
        
        sb.AppendLine($"- **Messages**: {session.MessageCount}");
        sb.AppendLine($"- **Tool Calls**: {session.ToolCallCount}");

        if (!string.IsNullOrEmpty(session.Summary))
        {
            sb.AppendLine();
            sb.AppendLine("### Summary");
            sb.AppendLine();
            sb.AppendLine(session.Summary);
        }

        sb.AppendLine();
        sb.AppendLine("### Messages");
        sb.AppendLine();

        foreach (var message in session.Messages)
        {
            AppendMessage(sb, message);
        }

        sb.AppendLine("---");
        sb.AppendLine();
    }

    private static string FormatDuration(TimeSpan? duration)
    {
        if (!duration.HasValue)
            return "N/A";
        
        var d = duration.Value;
        if (d.TotalMinutes < 1)
            return $"{d.Seconds}s";
        if (d.TotalHours < 1)
            return $"{d.Minutes}m {d.Seconds}s";
        if (d.TotalDays < 1)
            return $"{(int)d.TotalHours}h {d.Minutes}m";
        
        return $"{(int)d.TotalDays}d {d.Hours}h {d.Minutes}m";
    }

    private static string MakeLinkId(string id)
    {
        // Create a valid markdown anchor link ID
        return id.ToLowerInvariant().Replace(" ", "-");
    }
}
