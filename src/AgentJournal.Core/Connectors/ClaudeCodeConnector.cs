using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Text.Json.Serialization;
using AgentJournal.Core.Models;

namespace AgentJournal.Core.Connectors;

/// <summary>
/// Connector for Claude Code agent sessions
/// </summary>
public class ClaudeCodeConnector : IAgentConnector
{
    public string AgentType => "claude-code";

    private readonly string _projectsPath;

    /// <summary>
    /// Creates a connector rooted at the supplied Claude projects directory.
    /// </summary>
    /// <param name="projectsPath">
    /// Directory to scan for session files. When null or blank the default
    /// <c>~/.claude/projects</c> location is used. This is what the configured
    /// <c>ClaudeProjectsPath</c> setting flows into - previously the default was hardcoded here,
    /// so the setting had no effect at all.
    /// </param>
    public ClaudeCodeConnector(string? projectsPath = null)
    {
        _projectsPath = string.IsNullOrWhiteSpace(projectsPath)
            ? Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
                ".claude",
                "projects")
            : projectsPath;
    }

    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNameCaseInsensitive = true,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        AllowTrailingCommas = true,
        ReadCommentHandling = JsonCommentHandling.Skip
    };

    public IEnumerable<string> GetSessionPaths()
    {
        if (!Directory.Exists(_projectsPath))
        {
            yield break;
        }

        // Find all .jsonl files recursively under the configured projects directory
        foreach (var file in Directory.EnumerateFiles(_projectsPath, "*.jsonl", SearchOption.AllDirectories))
        {
            // Check if the file name looks like a session UUID (contains hyphens, typical of GUIDs)
            var fileName = Path.GetFileNameWithoutExtension(file);
            if (fileName.Contains('-') && fileName.Length >= 32)
            {
                yield return file;
            }
        }
    }

    public async IAsyncEnumerable<Session> ParseSessionsAsync([EnumeratorCancellation] CancellationToken ct = default)
    {
        foreach (var sessionPath in GetSessionPaths())
        {
            ct.ThrowIfCancellationRequested();

            var session = await ParseSessionAsync(sessionPath, ct);
            if (session != null)
            {
                yield return session;
            }
        }
    }

    public async Task<Session?> ParseSessionAsync(string sessionPath, CancellationToken ct = default)
    {
        if (!File.Exists(sessionPath))
        {
            return null;
        }

        try
        {
            var lastModified = File.GetLastWriteTimeUtc(sessionPath);
            var records = new List<ClaudeCodeRecord>();

            // Read JSONL file line by line
            await foreach (var line in File.ReadLinesAsync(sessionPath, ct))
            {
                if (string.IsNullOrWhiteSpace(line))
                    continue;

                try
                {
                    var record = JsonSerializer.Deserialize<ClaudeCodeRecord>(line, JsonOptions);
                    if (record != null)
                    {
                        records.Add(record);
                    }
                }
                catch (JsonException ex)
                {
                    // Skip malformed lines. Diagnostics go to stderr: this code also runs inside the
                    // MCP stdio server and behind `--robot`, where anything written to stdout would
                    // corrupt the JSON-RPC stream / machine-readable output.
                    Console.Error.WriteLine($"Warning: Failed to parse line in {sessionPath}: {ex.Message}");
                    continue;
                }
            }

            if (records.Count == 0)
            {
                return null;
            }

            // Extract session metadata
            var summaryRecord = records.FirstOrDefault(r => r.Type == "summary");
            var firstRecord = records.FirstOrDefault();
            if (firstRecord == null)
            {
                return null;
            }

            var sessionId = firstRecord.SessionId ?? Path.GetFileNameWithoutExtension(sessionPath);
            var messages = new List<Message>();
            var toolCallsMap = new Dictionary<string, List<ToolCall>>();

            // Process message records
            foreach (var record in records.Where(r => r.Type is "user" or "assistant"))
            {
                ct.ThrowIfCancellationRequested();

                var message = ParseMessage(record, sessionId, toolCallsMap);
                if (message != null)
                {
                    messages.Add(message);
                }
            }

            // Sort messages by timestamp
            messages = messages.OrderBy(m => m.Timestamp).ToList();

            // Build session
            var startedAt = messages.FirstOrDefault()?.Timestamp ?? DateTime.UtcNow;
            var endedAt = summaryRecord != null ? messages.LastOrDefault()?.Timestamp : null;

            return new Session(
                Id: sessionId,
                AgentType: AgentType,
                ProjectPath: firstRecord.Cwd,
                GitBranch: firstRecord.GitBranch,
                AgentVersion: firstRecord.Version,
                StartedAt: startedAt,
                EndedAt: endedAt,
                LastModified: lastModified,
                Summary: summaryRecord?.Message?.Content?.ToString(),
                Messages: messages
            );
        }
        catch (Exception ex)
        {
            // Report on stderr, not stdout - see note above about MCP stdio / --robot output.
            Console.Error.WriteLine($"Error parsing session {sessionPath}: {ex.Message}");
            return null;
        }
    }

    private static Message? ParseMessage(
        ClaudeCodeRecord record,
        string sessionId,
        Dictionary<string, List<ToolCall>> toolCallsMap)
    {
        if (record.Message == null || record.Uuid == null)
        {
            return null;
        }

        var messageId = record.Uuid;
        var role = record.Message.Role switch
        {
            "user" => MessageRole.User,
            "assistant" => MessageRole.Assistant,
            _ => MessageRole.User
        };

        // Extract content and tool calls
        var (content, toolCalls) = ExtractContentAndToolCalls(record.Message, messageId);

        // Store tool calls for later matching with results
        if (toolCalls.Count > 0)
        {
            toolCallsMap[messageId] = toolCalls;
        }

        var timestamp = DateTime.TryParse(record.Timestamp, out var parsedTime)
            ? parsedTime
            : DateTime.UtcNow;

        return new Message(
            Id: messageId,
            SessionId: sessionId,
            Role: role,
            Content: content,
            RawContent: JsonSerializer.Serialize(record.Message.Content, JsonOptions),
            Timestamp: timestamp,
            ParentId: record.ParentUuid,
            Model: record.Message.Model,
            ToolCalls: toolCalls.Count > 0 ? toolCalls : null
        );
    }

    private static (string Content, List<ToolCall> ToolCalls) ExtractContentAndToolCalls(
        ClaudeMessage message,
        string messageId)
    {
        var contentParts = new List<string>();
        var toolCalls = new List<ToolCall>();

        if (message.Content == null)
        {
            return (string.Empty, toolCalls);
        }

        // Handle string content
        if (message.Content is JsonElement jsonElement)
        {
            if (jsonElement.ValueKind == JsonValueKind.String)
            {
                return (jsonElement.GetString() ?? string.Empty, toolCalls);
            }

            if (jsonElement.ValueKind == JsonValueKind.Array)
            {
                foreach (var block in jsonElement.EnumerateArray())
                {
                    ProcessContentBlock(block, contentParts, toolCalls, messageId);
                }
            }
        }

        return (string.Join("\n\n", contentParts), toolCalls);
    }

    private static void ProcessContentBlock(
        JsonElement block,
        List<string> contentParts,
        List<ToolCall> toolCalls,
        string messageId)
    {
        if (!block.TryGetProperty("type", out var typeElement))
        {
            return;
        }

        var blockType = typeElement.GetString();

        switch (blockType)
        {
            case "text":
                if (block.TryGetProperty("text", out var textElement))
                {
                    var text = textElement.GetString();
                    if (!string.IsNullOrWhiteSpace(text))
                    {
                        contentParts.Add(text);
                    }
                }
                break;

            case "thinking":
                // Skip thinking blocks for main content, but could be logged separately
                break;

            case "tool_use":
                var toolCall = ParseToolUse(block, messageId);
                if (toolCall != null)
                {
                    toolCalls.Add(toolCall);
                }
                break;

            case "tool_result":
                // Tool results are handled separately by matching tool_use_id
                break;
        }
    }

    private static ToolCall? ParseToolUse(JsonElement block, string messageId)
    {
        if (!block.TryGetProperty("id", out var idElement) ||
            !block.TryGetProperty("name", out var nameElement))
        {
            return null;
        }

        var toolId = idElement.GetString();
        var toolName = nameElement.GetString();

        if (string.IsNullOrEmpty(toolId) || string.IsNullOrEmpty(toolName))
        {
            return null;
        }

        string? arguments = null;
        if (block.TryGetProperty("input", out var inputElement))
        {
            arguments = inputElement.GetRawText();
        }

        // Look for corresponding tool_result in subsequent blocks
        // For now, we'll leave Result as null - it could be matched in a second pass
        return new ToolCall(
            Id: toolId,
            MessageId: messageId,
            Name: toolName,
            Arguments: arguments,
            Result: null,
            Success: null
        );
    }

    #region JSON Models

    private class ClaudeCodeRecord
    {
        [JsonPropertyName("type")]
        public string? Type { get; set; }

        [JsonPropertyName("uuid")]
        public string? Uuid { get; set; }

        [JsonPropertyName("parentUuid")]
        public string? ParentUuid { get; set; }

        [JsonPropertyName("sessionId")]
        public string? SessionId { get; set; }

        [JsonPropertyName("timestamp")]
        public string? Timestamp { get; set; }

        [JsonPropertyName("cwd")]
        public string? Cwd { get; set; }

        [JsonPropertyName("version")]
        public string? Version { get; set; }

        [JsonPropertyName("gitBranch")]
        public string? GitBranch { get; set; }

        [JsonPropertyName("message")]
        public ClaudeMessage? Message { get; set; }
    }

    private class ClaudeMessage
    {
        [JsonPropertyName("role")]
        public string? Role { get; set; }

        [JsonPropertyName("content")]
        public object? Content { get; set; }

        [JsonPropertyName("model")]
        public string? Model { get; set; }
    }

    #endregion
}
