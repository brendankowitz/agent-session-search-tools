using System.Text;
using System.Web;
using AgentJournal.Core.Models;
using Scriban;
using Scriban.Runtime;

namespace AgentJournal.Core.Export;

/// <summary>
/// Exports sessions to HTML format using Scriban templates
/// </summary>
public class HtmlExporter : IExporter
{
    private readonly ExportOptions _options;

    public ExportFormat Format => ExportFormat.Html;
    public string FileExtension => ".html";

    public HtmlExporter(ExportOptions? options = null)
    {
        _options = options ?? ExportOptions.Default;
    }

    public Task<string> ExportAsync(Session session, CancellationToken ct = default)
    {
        var template = Template.Parse(GetSingleSessionTemplate());
        var context = CreateScribanContext(session);
        var html = template.Render(context);
        return Task.FromResult(html);
    }

    public async Task ExportToFileAsync(Session session, string outputPath, CancellationToken ct = default)
    {
        var content = await ExportAsync(session, ct);
        await File.WriteAllTextAsync(outputPath, content, ct);
    }

    public Task<string> ExportMultipleAsync(IEnumerable<Session> sessions, CancellationToken ct = default)
    {
        var sessionList = sessions.ToList();
        var template = Template.Parse(GetMultipleSessionsTemplate());

        var scriptObject = new ScriptObject
        {
            { "sessions", sessionList },
            { "options", _options },
            { "export_date", DateTime.Now },
            { "total_sessions", sessionList.Count }
        };

        var context = new TemplateContext();
        context.PushGlobal(scriptObject);

        var html = template.Render(context);
        return Task.FromResult(html);
    }

    private ScriptObject CreateScribanContext(Session session)
    {
        var scriptObject = new ScriptObject
        {
            { "session", session },
            { "options", _options },
            { "export_date", DateTime.Now }
        };

        // Add custom functions
        scriptObject.Import("escape_html", new Func<string, string>(EscapeHtml));
        scriptObject.Import("truncate_text", new Func<string?, int, string>(TruncateText));
        scriptObject.Import("format_date", new Func<DateTime, string>(d => d.ToString("yyyy-MM-dd HH:mm:ss")));
        scriptObject.Import("format_duration", new Func<TimeSpan?, string>(FormatDuration));

        var context = new TemplateContext();
        context.PushGlobal(scriptObject);

        return scriptObject;
    }

    private static string EscapeHtml(string text)
    {
        if (string.IsNullOrEmpty(text))
            return string.Empty;

        return HttpUtility.HtmlEncode(text);
    }

    private static string TruncateText(string? text, int maxLength)
    {
        if (string.IsNullOrEmpty(text) || text.Length <= maxLength)
            return text ?? string.Empty;

        return text[..maxLength] + "...";
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

        return $"{(int)d.TotalHours}h {d.Minutes}m";
    }

    private string GetSingleSessionTemplate()
    {
        var isDark = _options.Theme.Equals("dark", StringComparison.OrdinalIgnoreCase);
        var cssVars = isDark
            ? "--bg: #1a1a2e; --text: #eee; --user-bg: #0f3460; --assistant-bg: #16213e; --header-border: #333; --tool-bg: #1e1e1e; --code-bg: #0d0d0d;"
            : "--bg: #ffffff; --text: #333; --user-bg: #e3f2fd; --assistant-bg: #f5f5f5; --header-border: #ddd; --tool-bg: #f9f9f9; --code-bg: #f5f5f5;";

        return @"<!DOCTYPE html>
<html lang=""en"">
<head>
    <meta charset=""utf-8"">
    <meta name=""viewport"" content=""width=device-width, initial-scale=1.0"">
    <title>{{ session.agent_type }} - {{ format_date session.started_at }}</title>
    <style>
        :root { " + cssVars + @" }
        * { box-sizing: border-box; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica', 'Arial', sans-serif;
            background: var(--bg); 
            color: var(--text); 
            max-width: 1000px; 
            margin: 0 auto; 
            padding: 2rem;
            line-height: 1.6;
        }
        .header { 
            border-bottom: 2px solid var(--header-border); 
            padding-bottom: 1.5rem; 
            margin-bottom: 2rem; 
        }
        .header h1 { 
            margin: 0 0 1rem 0; 
            font-size: 2rem;
        }
        .header .meta { 
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 0.5rem;
            margin-top: 1rem;
        }
        .header .meta-item { 
            font-size: 0.9rem; 
        }
        .header .meta-item strong { 
            font-weight: 600; 
        }
        .message { 
            padding: 1.25rem; 
            margin: 1rem 0; 
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }
        .message:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        .message.user { 
            background: var(--user-bg); 
            margin-left: 2rem;
        }
        .message.assistant { 
            background: var(--assistant-bg);
            margin-right: 2rem;
        }
        .message.system { 
            background: var(--tool-bg);
            font-style: italic;
            opacity: 0.8;
        }
        .message-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.75rem;
        }
        .role-badge {
            font-weight: 600;
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .timestamp { 
            font-size: 0.8em; 
            opacity: 0.6;
        }
        .content { 
            white-space: pre-wrap;
            word-wrap: break-word;
            margin: 0.75rem 0;
        }
        .tool-calls {
            margin-top: 1rem;
            border-top: 1px solid var(--header-border);
            padding-top: 1rem;
        }
        .tool-call-header {
            cursor: pointer;
            padding: 0.5rem;
            background: var(--tool-bg);
            border-radius: 6px;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            user-select: none;
            transition: opacity 0.2s;
        }
        .tool-call-header:hover {
            opacity: 0.8;
        }
        .tool-call-header::before {
            content: '▶';
            display: inline-block;
            transition: transform 0.2s;
        }
        .tool-call-header.expanded::before {
            transform: rotate(90deg);
        }
        .tool-call-content {
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.3s ease-out;
        }
        .tool-call-content.expanded {
            max-height: 2000px;
        }
        .tool-call { 
            background: var(--tool-bg); 
            padding: 1rem; 
            margin: 0.5rem 0;
            border-radius: 6px;
            border-left: 3px solid var(--user-bg);
        }
        .tool-call strong {
            display: block;
            margin-bottom: 0.5rem;
            color: var(--user-bg);
        }
        .tool-result {
            background: var(--code-bg);
            padding: 0.75rem;
            border-radius: 4px;
            font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
            font-size: 0.85em;
            overflow-x: auto;
            white-space: pre-wrap;
            word-break: break-all;
        }
        pre { 
            background: var(--code-bg); 
            padding: 1rem; 
            overflow-x: auto; 
            border-radius: 6px;
            margin: 0.5rem 0;
        }
        code {
            font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
            font-size: 0.9em;
        }
        .footer {
            margin-top: 3rem;
            padding-top: 1rem;
            border-top: 1px solid var(--header-border);
            text-align: center;
            font-size: 0.85rem;
            opacity: 0.6;
        }
    </style>
    <script>
        function toggleToolCalls(element) {
            element.classList.toggle('expanded');
            const content = element.nextElementSibling;
            content.classList.toggle('expanded');
        }
    </script>
</head>
<body>
    <div class=""header"">
        <h1>{{ session.agent_type }} Session</h1>
        <div class=""meta"">
            <div class=""meta-item"">
                <strong>Project:</strong> {{ session.project_path ?? ""N/A"" }}
            </div>
            <div class=""meta-item"">
                <strong>Branch:</strong> {{ session.git_branch ?? ""N/A"" }}
            </div>
            <div class=""meta-item"">
                <strong>Started:</strong> {{ format_date session.started_at }}
            </div>
            {{~ if session.ended_at ~}}
            <div class=""meta-item"">
                <strong>Duration:</strong> {{ format_duration session.duration }}
            </div>
            {{~ end ~}}
            <div class=""meta-item"">
                <strong>Messages:</strong> {{ session.message_count }}
            </div>
            <div class=""meta-item"">
                <strong>Tool Calls:</strong> {{ session.tool_call_count }}
            </div>
        </div>
        {{~ if session.summary ~}}
        <div style=""margin-top: 1rem;"">
            <strong>Summary:</strong> {{ session.summary }}
        </div>
        {{~ end ~}}
    </div>

    {{~ for message in session.messages ~}}
    <div class=""message {{ message.role | string.downcase }}"">
        <div class=""message-header"">
            <span class=""role-badge"">{{ message.role }}</span>
            {{~ if options.include_timestamps ~}}
            <span class=""timestamp"">{{ format_date message.timestamp }}</span>
            {{~ end ~}}
        </div>
        <div class=""content"">{{ escape_html message.content }}</div>
        
        {{~ if options.include_tool_calls && message.has_tool_calls ~}}
        <div class=""tool-calls"">
            <div class=""tool-call-header{{~ if !options.collapse_tool_calls_by_default ~}} expanded{{~ end ~}}"" 
                 onclick=""toggleToolCalls(this)"">
                🔧 {{ message.tool_call_count }} tool call(s)
            </div>
            <div class=""tool-call-content{{~ if !options.collapse_tool_calls_by_default ~}} expanded{{~ end ~}}"">
                {{~ for tool in message.tool_calls ~}}
                <div class=""tool-call"">
                    <strong>{{ tool.name }}</strong>
                    {{~ if tool.arguments ~}}
                    <div style=""margin-bottom: 0.5rem; font-size: 0.9em;"">
                        <em>Arguments:</em> {{ escape_html (truncate_text tool.arguments 200) }}
                    </div>
                    {{~ end ~}}
                    {{~ if tool.result ~}}
                    <div class=""tool-result"">{{ escape_html (truncate_text tool.result (options.max_tool_result_length ?? 500)) }}</div>
                    {{~ end ~}}
                </div>
                {{~ end ~}}
            </div>
        </div>
        {{~ end ~}}
    </div>
    {{~ end ~}}

    <div class=""footer"">
        Exported on {{ format_date export_date }} | AgentJournal
    </div>
</body>
</html>";
    }

    private string GetMultipleSessionsTemplate()
    {
        var isDark = _options.Theme.Equals("dark", StringComparison.OrdinalIgnoreCase);
        var cssVars = isDark
            ? "--bg: #1a1a2e; --text: #eee; --card-bg: #16213e; --header-border: #333; --hover-bg: #0f3460;"
            : "--bg: #ffffff; --text: #333; --card-bg: #f5f5f5; --header-border: #ddd; --hover-bg: #e3f2fd;";

        return @"<!DOCTYPE html>
<html lang=""en"">
<head>
    <meta charset=""utf-8"">
    <meta name=""viewport"" content=""width=device-width, initial-scale=1.0"">
    <title>Agent Sessions Export</title>
    <style>
        :root { " + cssVars + @" }
        * { box-sizing: border-box; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica', 'Arial', sans-serif;
            background: var(--bg); 
            color: var(--text); 
            max-width: 1200px; 
            margin: 0 auto; 
            padding: 2rem;
            line-height: 1.6;
        }
        .header { 
            border-bottom: 2px solid var(--header-border); 
            padding-bottom: 1.5rem; 
            margin-bottom: 2rem; 
        }
        .session-card {
            background: var(--card-bg);
            padding: 1.5rem;
            margin: 1rem 0;
            border-radius: 12px;
            border: 1px solid var(--header-border);
            cursor: pointer;
            transition: all 0.2s;
        }
        .session-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            background: var(--hover-bg);
        }
        .session-title {
            font-size: 1.25rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }
        .session-meta {
            display: flex;
            gap: 1.5rem;
            flex-wrap: wrap;
            font-size: 0.9rem;
            opacity: 0.8;
        }
        .footer {
            margin-top: 3rem;
            padding-top: 1rem;
            border-top: 1px solid var(--header-border);
            text-align: center;
            font-size: 0.85rem;
            opacity: 0.6;
        }
    </style>
</head>
<body>
    <div class=""header"">
        <h1>Agent Sessions Export</h1>
        <p>Total Sessions: {{ total_sessions }}</p>
    </div>

    {{~ for session in sessions ~}}
    <div class=""session-card"">
        <div class=""session-title"">{{ session.agent_type }}</div>
        <div class=""session-meta"">
            <span>📅 {{ session.started_at | date.to_string ""%Y-%m-%d %H:%M"" }}</span>
            <span>💬 {{ session.message_count }} messages</span>
            <span>🔧 {{ session.tool_call_count }} tool calls</span>
            {{~ if session.project_path ~}}
            <span>📁 {{ session.project_path }}</span>
            {{~ end ~}}
        </div>
        {{~ if session.summary ~}}
        <div style=""margin-top: 0.75rem; font-size: 0.95rem;"">
            {{ session.summary }}
        </div>
        {{~ end ~}}
    </div>
    {{~ end ~}}

    <div class=""footer"">
        Exported on {{ export_date | date.to_string ""%Y-%m-%d %H:%M:%S"" }} | AgentJournal
    </div>
</body>
</html>";
    }
}
