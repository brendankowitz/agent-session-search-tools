using AgentJournal.Core.Models;
using AgentJournal.Core.Export;

namespace AgentJournal.Demo;

public class Program
{
    public static async Task Main(string[] args)
    {
        // Create a sample session
        var messages = new List<Message>
        {
            new Message(
                Id: "msg1",
                SessionId: "demo-session",
                Role: MessageRole.User,
                Content: "I need to implement HTML and Markdown exporters for an agent journal system.",
                RawContent: null,
                Timestamp: DateTime.Now.AddMinutes(-20),
                ParentId: null,
                Model: null,
                ToolCalls: null
            ),
            new Message(
                Id: "msg2",
                SessionId: "demo-session",
                Role: MessageRole.Assistant,
                Content: "I'll help you implement both exporters. Let me start by creating the HTML exporter with Scriban templates.\n\nKey features:\n- Dark/Light theme support\n- Collapsible tool calls\n- Beautiful, responsive design\n- Self-contained HTML files",
                RawContent: null,
                Timestamp: DateTime.Now.AddMinutes(-19),
                ParentId: "msg1",
                Model: "claude-3-opus",
                ToolCalls: new List<ToolCall>
                {
                    new ToolCall(
                        Id: "tool1",
                        MessageId: "msg2",
                        Name: "create",
                        Arguments: "{\"path\": \"Export/HtmlExporter.cs\"}",
                        Result: "Created HtmlExporter.cs with Scriban templates for beautiful, self-contained HTML output",
                        Success: true
                    ),
                    new ToolCall(
                        Id: "tool2",
                        MessageId: "msg2",
                        Name: "create",
                        Arguments: "{\"path\": \"Export/MarkdownExporter.cs\"}",
                        Result: "Created MarkdownExporter.cs with clean markdown formatting",
                        Success: true
                    )
                }
            ),
            new Message(
                Id: "msg3",
                SessionId: "demo-session",
                Role: MessageRole.User,
                Content: "Can you add support for themes in the HTML export?",
                RawContent: null,
                Timestamp: DateTime.Now.AddMinutes(-15),
                ParentId: "msg2",
                Model: null,
                ToolCalls: null
            ),
            new Message(
                Id: "msg4",
                SessionId: "demo-session",
                Role: MessageRole.Assistant,
                Content: "I've added theme support with dark and light modes. The CSS variables make it easy to customize.\n\nThemes available:\n- Dark mode (default): Professional dark theme\n- Light mode: Clean light theme\n\nYou can configure via ExportOptions.",
                RawContent: null,
                Timestamp: DateTime.Now.AddMinutes(-14),
                ParentId: "msg3",
                Model: "claude-3-opus",
                ToolCalls: new List<ToolCall>
                {
                    new ToolCall(
                        Id: "tool3",
                        MessageId: "msg4",
                        Name: "edit",
                        Arguments: "{\"path\": \"Export/HtmlExporter.cs\", \"changes\": \"Added theme support\"}",
                        Result: "Updated HtmlExporter.cs with theme support (dark/light)",
                        Success: true
                    )
                }
            ),
            new Message(
                Id: "msg5",
                SessionId: "demo-session",
                Role: MessageRole.User,
                Content: "Perfect! Can you also add support for exporting multiple sessions at once?",
                RawContent: null,
                Timestamp: DateTime.Now.AddMinutes(-10),
                ParentId: "msg4",
                Model: null,
                ToolCalls: null
            ),
            new Message(
                Id: "msg6",
                SessionId: "demo-session",
                Role: MessageRole.Assistant,
                Content: "Done! Both exporters now support:\n1. Single session export\n2. Multiple sessions export (creates an index/TOC)\n3. Export to file\n\nAll tests passing! ✅",
                RawContent: null,
                Timestamp: DateTime.Now.AddMinutes(-5),
                ParentId: "msg5",
                Model: "claude-3-opus",
                ToolCalls: new List<ToolCall>
                {
                    new ToolCall(
                        Id: "tool4",
                        MessageId: "msg6",
                        Name: "powershell",
                        Arguments: "{\"command\": \"dotnet test\"}",
                        Result: "Test Run Successful.\nTotal tests: 11\n     Passed: 11\n Total time: 0.8638 Seconds",
                        Success: true
                    )
                }
            )
        };

        var session = new Session(
            Id: "demo-session",
            AgentType: "Complex Coding Agent",
            ProjectPath: "E:\\data\\src\\agent-session-search-tools",
            GitBranch: "main",
            AgentVersion: "1.0.0",
            StartedAt: DateTime.Now.AddMinutes(-25),
            EndedAt: DateTime.Now,
            Summary: "Successfully implemented HTML and Markdown exporters with Scriban templates, theme support, and comprehensive test coverage",
            Messages: messages
        );

        Console.WriteLine("===========================================");
        Console.WriteLine("     AgentJournal Export Demo");
        Console.WriteLine("===========================================");
        Console.WriteLine();
        Console.WriteLine($"Session ID: {session.Id}");
        Console.WriteLine($"Agent Type: {session.AgentType}");
        Console.WriteLine($"Messages: {session.MessageCount}");
        Console.WriteLine($"Tool Calls: {session.ToolCallCount}");
        Console.WriteLine($"Duration: {session.Duration}");
        Console.WriteLine();
        Console.WriteLine($"Summary: {session.Summary}");
        Console.WriteLine();

        // Export to HTML (Dark Theme)
        Console.WriteLine("Exporting to HTML (Dark Theme)...");
        var htmlExporterDark = new HtmlExporter(ExportOptions.Default);
        var htmlDark = await htmlExporterDark.ExportAsync(session);
        await File.WriteAllTextAsync("demo-session-dark.html", htmlDark);
        Console.WriteLine($"✅ Created: demo-session-dark.html ({htmlDark.Length:N0} bytes)");

        // Export to HTML (Light Theme)
        Console.WriteLine("Exporting to HTML (Light Theme)...");
        var htmlExporterLight = new HtmlExporter(ExportOptions.Light);
        var htmlLight = await htmlExporterLight.ExportAsync(session);
        await File.WriteAllTextAsync("demo-session-light.html", htmlLight);
        Console.WriteLine($"✅ Created: demo-session-light.html ({htmlLight.Length:N0} bytes)");

        // Export to Markdown
        Console.WriteLine("Exporting to Markdown...");
        var mdExporter = new MarkdownExporter();
        var markdown = await mdExporter.ExportAsync(session);
        await File.WriteAllTextAsync("demo-session.md", markdown);
        Console.WriteLine($"✅ Created: demo-session.md ({markdown.Length:N0} bytes)");

        // Export to JSON
        Console.WriteLine("Exporting to JSON...");
        var jsonExporter = new JsonExporter();
        var json = await jsonExporter.ExportAsync(session);
        await File.WriteAllTextAsync("demo-session.json", json);
        Console.WriteLine($"✅ Created: demo-session.json ({json.Length:N0} bytes)");

        // Export multiple sessions
        Console.WriteLine();
        Console.WriteLine("Creating multiple sessions export...");
        var sessions = new[] { session, session with { Id = "session2", AgentType = "Fast Coding Agent" } };

        var htmlMultiple = await htmlExporterDark.ExportMultipleAsync(sessions);
        await File.WriteAllTextAsync("demo-sessions-index.html", htmlMultiple);
        Console.WriteLine($"✅ Created: demo-sessions-index.html ({htmlMultiple.Length:N0} bytes)");

        var mdMultiple = await mdExporter.ExportMultipleAsync(sessions);
        await File.WriteAllTextAsync("demo-sessions.md", mdMultiple);
        Console.WriteLine($"✅ Created: demo-sessions.md ({mdMultiple.Length:N0} bytes)");

        Console.WriteLine();
        Console.WriteLine("===========================================");
        Console.WriteLine("     Export Complete!");
        Console.WriteLine("===========================================");
        Console.WriteLine();
        Console.WriteLine("Open the HTML files in your browser to see:");
        Console.WriteLine("  • Beautiful dark/light themes");
        Console.WriteLine("  • Collapsible tool calls");
        Console.WriteLine("  • Responsive design");
        Console.WriteLine("  • Session metadata");
        Console.WriteLine();
        Console.WriteLine("The Markdown files are perfect for:");
        Console.WriteLine("  • Documentation");
        Console.WriteLine("  • Version control");
        Console.WriteLine("  • Easy reading in any text editor");
    }
}
