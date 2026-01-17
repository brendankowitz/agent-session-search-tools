using AgentJournal.Core.Export;
using AgentJournal.Core.Models;

namespace AgentJournal.Tests;

public class ExporterTests
{
    private static Session CreateTestSession()
    {
        var messages = new List<Message>
        {
            new Message(
                Id: "msg1",
                SessionId: "session1",
                Role: MessageRole.User,
                Content: "Hello, can you help me with my code?",
                RawContent: null,
                Timestamp: DateTime.Now.AddMinutes(-10),
                ParentId: null,
                Model: null,
                ToolCalls: null
            ),
            new Message(
                Id: "msg2",
                SessionId: "session1",
                Role: MessageRole.Assistant,
                Content: "Sure! I'd be happy to help. What issue are you facing?",
                RawContent: null,
                Timestamp: DateTime.Now.AddMinutes(-9),
                ParentId: "msg1",
                Model: "claude-3-opus",
                ToolCalls: null
            ),
            new Message(
                Id: "msg3",
                SessionId: "session1",
                Role: MessageRole.User,
                Content: "I need to implement an export feature.",
                RawContent: null,
                Timestamp: DateTime.Now.AddMinutes(-8),
                ParentId: "msg2",
                Model: null,
                ToolCalls: null
            ),
            new Message(
                Id: "msg4",
                SessionId: "session1",
                Role: MessageRole.Assistant,
                Content: "Let me help you create that export feature.",
                RawContent: null,
                Timestamp: DateTime.Now.AddMinutes(-7),
                ParentId: "msg3",
                Model: "claude-3-opus",
                ToolCalls: new List<ToolCall>
                {
                    new ToolCall(
                        Id: "tool1",
                        MessageId: "msg4",
                        Name: "create_file",
                        Arguments: "{\"path\": \"Export/HtmlExporter.cs\"}",
                        Result: "File created successfully",
                        Success: true
                    ),
                    new ToolCall(
                        Id: "tool2",
                        MessageId: "msg4",
                        Name: "create_file",
                        Arguments: "{\"path\": \"Export/MarkdownExporter.cs\"}",
                        Result: "File created successfully",
                        Success: true
                    )
                }
            )
        };

        return new Session(
            Id: "session1",
            AgentType: "Complex Coding Agent",
            ProjectPath: "E:\\data\\src\\agent-session-search-tools",
            GitBranch: "main",
            AgentVersion: "1.0.0",
            StartedAt: DateTime.Now.AddMinutes(-15),
            EndedAt: DateTime.Now,
            LastModified: null,
            Summary: "Implemented HTML and Markdown exporters with Scriban templates",
            Messages: messages
        );
    }

    [Fact]
    public async Task HtmlExporter_ExportsSession_Successfully()
    {
        // Arrange
        var session = CreateTestSession();
        var exporter = new HtmlExporter();

        // Act
        var html = await exporter.ExportAsync(session);

        // Assert
        Assert.NotNull(html);
        Assert.NotEmpty(html);
        Assert.Contains("<!DOCTYPE html>", html);
        Assert.Contains(session.AgentType, html);
        Assert.Contains("Complex Coding Agent", html);
    }

    [Fact]
    public async Task HtmlExporter_WithLightTheme_UsesCorrectColors()
    {
        // Arrange
        var session = CreateTestSession();
        var options = ExportOptions.Light;
        var exporter = new HtmlExporter(options);

        // Act
        var html = await exporter.ExportAsync(session);

        // Assert
        Assert.Contains("--bg: #ffffff", html);
        Assert.Contains("--text: #333", html);
    }

    [Fact]
    public async Task HtmlExporter_WithDarkTheme_UsesCorrectColors()
    {
        // Arrange
        var session = CreateTestSession();
        var options = ExportOptions.Default; // Dark by default
        var exporter = new HtmlExporter(options);

        // Act
        var html = await exporter.ExportAsync(session);

        // Assert
        Assert.Contains("--bg: #1a1a2e", html);
        Assert.Contains("--text: #eee", html);
    }

    [Fact]
    public async Task MarkdownExporter_ExportsSession_Successfully()
    {
        // Arrange
        var session = CreateTestSession();
        var exporter = new MarkdownExporter();

        // Act
        var markdown = await exporter.ExportAsync(session);

        // Assert
        Assert.NotNull(markdown);
        Assert.NotEmpty(markdown);
        Assert.Contains("# Complex Coding Agent Session", markdown);
        Assert.Contains("## Session Information", markdown);
        Assert.Contains("## Conversation", markdown);
    }

    [Fact]
    public async Task MarkdownExporter_IncludesToolCalls_WhenEnabled()
    {
        // Arrange
        var session = CreateTestSession();
        var options = new ExportOptions(IncludeToolCalls: true);
        var exporter = new MarkdownExporter(options);

        // Act
        var markdown = await exporter.ExportAsync(session);

        // Assert
        Assert.Contains("#### 🔧 Tool Calls", markdown);
        Assert.Contains("create_file", markdown);
    }

    [Fact]
    public async Task MarkdownExporter_ExcludesToolCalls_WhenDisabled()
    {
        // Arrange
        var session = CreateTestSession();
        var options = ExportOptions.NoToolCalls;
        var exporter = new MarkdownExporter(options);

        // Act
        var markdown = await exporter.ExportAsync(session);

        // Assert
        Assert.DoesNotContain("#### 🔧 Tool Calls", markdown);
    }

    [Fact]
    public async Task JsonExporter_ExportsSession_Successfully()
    {
        // Arrange
        var session = CreateTestSession();
        var exporter = new JsonExporter();

        // Act
        var json = await exporter.ExportAsync(session);

        // Assert
        Assert.NotNull(json);
        Assert.NotEmpty(json);
        Assert.Contains("\"agentType\":", json);
        Assert.Contains("Complex Coding Agent", json);
    }

    [Fact]
    public async Task HtmlExporter_ExportMultipleSessions_CreatesIndex()
    {
        // Arrange
        var sessions = new[]
        {
            CreateTestSession(),
            CreateTestSession() with { Id = "session2", AgentType = "Fast Coding Agent" }
        };
        var exporter = new HtmlExporter();

        // Act
        var html = await exporter.ExportMultipleAsync(sessions);

        // Assert
        Assert.Contains("Agent Sessions Export", html);
        Assert.Contains("Total Sessions: 2", html);
        Assert.Contains("Complex Coding Agent", html);
        Assert.Contains("Fast Coding Agent", html);
    }

    [Fact]
    public async Task MarkdownExporter_ExportMultipleSessions_CreatesTableOfContents()
    {
        // Arrange
        var sessions = new[]
        {
            CreateTestSession(),
            CreateTestSession() with { Id = "session2", AgentType = "Fast Coding Agent" }
        };
        var exporter = new MarkdownExporter();

        // Act
        var markdown = await exporter.ExportMultipleAsync(sessions);

        // Assert
        Assert.Contains("# Agent Sessions Export", markdown);
        Assert.Contains("## Table of Contents", markdown);
        Assert.Contains("Complex Coding Agent", markdown);
        Assert.Contains("Fast Coding Agent", markdown);
    }

    [Fact]
    public async Task HtmlExporter_ExportToFile_CreatesFile()
    {
        // Arrange
        var session = CreateTestSession();
        var exporter = new HtmlExporter();
        var tempFile = Path.GetTempFileName() + ".html";

        try
        {
            // Act
            await exporter.ExportToFileAsync(session, tempFile);

            // Assert
            Assert.True(File.Exists(tempFile));
            var content = await File.ReadAllTextAsync(tempFile);
            Assert.Contains("<!DOCTYPE html>", content);
        }
        finally
        {
            // Cleanup
            if (File.Exists(tempFile))
                File.Delete(tempFile);
        }
    }

    [Fact]
    public async Task MarkdownExporter_ExportToFile_CreatesFile()
    {
        // Arrange
        var session = CreateTestSession();
        var exporter = new MarkdownExporter();
        var tempFile = Path.GetTempFileName() + ".md";

        try
        {
            // Act
            await exporter.ExportToFileAsync(session, tempFile);

            // Assert
            Assert.True(File.Exists(tempFile));
            var content = await File.ReadAllTextAsync(tempFile);
            Assert.Contains("# Complex Coding Agent Session", content);
        }
        finally
        {
            // Cleanup
            if (File.Exists(tempFile))
                File.Delete(tempFile);
        }
    }
}
