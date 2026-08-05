using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Text.Json.Serialization;
using AgentJournal.Core.Models;

namespace AgentJournal.Core.Connectors;

/// <summary>
/// Connector for GitHub Copilot CLI agent sessions
/// </summary>
public class CopilotCliConnector : IAgentConnector
{
    public string AgentType => "copilot-cli";

    private readonly string[] _sessionDirectories;

    /// <summary>
    /// Creates a connector rooted at the supplied Copilot sessions directory.
    /// </summary>
    /// <param name="sessionsPath">
    /// Directory to scan for session folders. When null or blank the default
    /// <c>~/.copilot/session-state</c> and <c>~/.copilot-cli/sessions</c> locations are probed.
    /// This is what the configured <c>CopilotSessionsPath</c> setting flows into - previously the
    /// defaults were hardcoded here, so the setting had no effect at all.
    /// </param>
    public CopilotCliConnector(string? sessionsPath = null)
    {
        if (!string.IsNullOrWhiteSpace(sessionsPath))
        {
            _sessionDirectories = [sessionsPath];
        }
        else
        {
            var homeDirectory = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
            _sessionDirectories =
            [
                Path.Combine(homeDirectory, ".copilot", "session-state"),
                Path.Combine(homeDirectory, ".copilot-cli", "sessions")
            ];
        }
    }

    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNameCaseInsensitive = true,
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase
    };

    public IEnumerable<string> GetSessionPaths()
    {
        foreach (var sessionsDirectory in _sessionDirectories)
        {
            if (!Directory.Exists(sessionsDirectory))
            {
                continue;
            }

            foreach (var sessionDirectory in Directory.EnumerateDirectories(sessionsDirectory))
            {
                var eventsFile = Path.Combine(sessionDirectory, "events.jsonl");
                if (File.Exists(eventsFile))
                {
                    yield return sessionDirectory;
                }
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
        try
        {
            var eventsFile = Path.Combine(sessionPath, "events.jsonl");
            if (!File.Exists(eventsFile))
            {
                return null;
            }

            var lastModified = File.GetLastWriteTimeUtc(eventsFile);

            var events = new List<CopilotEvent>();

            await foreach (var line in File.ReadLinesAsync(eventsFile, ct))
            {
                if (string.IsNullOrWhiteSpace(line))
                {
                    continue;
                }

                var evt = JsonSerializer.Deserialize<CopilotEvent>(line, JsonOptions);
                if (evt != null)
                {
                    events.Add(evt);
                }
            }

            // Sort events by timestamp to ensure proper order
            events = events.OrderBy(e => e.Timestamp).ToList();

            return BuildSessionFromEvents(events, sessionPath, lastModified);
        }
        catch (Exception ex)
        {
            // A session that cannot be parsed is not the same as "no session here". Report it on
            // stderr so a failed index run is diagnosable, then skip the file.
            Console.Error.WriteLine($"Error parsing session {sessionPath}: {ex.Message}");
            return null;
        }
    }

    private Session? BuildSessionFromEvents(List<CopilotEvent> events, string sessionPath, DateTime lastModified)
    {
        if (events.Count == 0)
        {
            return null;
        }

        // Find session.start event to get session metadata
        var sessionStartEvent = events.FirstOrDefault(e => e.Type == "session.start");
        if (sessionStartEvent?.Data == null)
        {
            return null;
        }

        string sessionId;
        string? copilotVersion = null;
        DateTime startTime;

        try
        {
            if (sessionStartEvent.Data is not JsonElement jsonElement)
            {
                ReportDropped("session", sessionPath, "session.start event carried no JSON payload");
                return null;
            }

            var startData = JsonSerializer.Deserialize<SessionStartData>(
                jsonElement.GetRawText(), JsonOptions);

            if (startData == null || string.IsNullOrEmpty(startData.SessionId))
            {
                ReportDropped("session", sessionPath, "session.start event has no session id");
                return null;
            }

            sessionId = startData.SessionId;
            copilotVersion = startData.CopilotVersion;
            startTime = startData.StartTime ?? sessionStartEvent.Timestamp;
        }
        catch (JsonException ex)
        {
            ReportDropped("session", sessionPath, $"malformed session.start payload: {ex.Message}");
            return null;
        }

        // Find last event timestamp for EndedAt
        var lastEvent = events.LastOrDefault();
        var endTime = lastEvent?.Timestamp;

        // Build messages from events
        var messages = BuildMessagesFromEvents(events, sessionId);

        return new Session(
            Id: sessionId,
            AgentType: AgentType,
            ProjectPath: sessionPath,
            GitBranch: null,
            AgentVersion: copilotVersion,
            StartedAt: startTime,
            EndedAt: endTime,
            LastModified: lastModified,
            Summary: null,
            Messages: messages
        );
    }

    private IReadOnlyList<Message> BuildMessagesFromEvents(List<CopilotEvent> events, string sessionId)
    {
        var messages = new List<Message>();
        var toolCallsMap = new Dictionary<string, ToolCallInfo>();

        foreach (var evt in events)
        {
            if (evt.Data == null)
            {
                continue;
            }

            switch (evt.Type)
            {
                case "user.message":
                    var userMessage = ParseUserMessage(evt, sessionId);
                    if (userMessage != null)
                    {
                        messages.Add(userMessage);
                    }
                    break;

                case "assistant.message":
                    var (assistantMessage, toolCalls) = ParseAssistantMessage(evt, sessionId);
                    if (assistantMessage != null)
                    {
                        messages.Add(assistantMessage);

                        // Track tool calls for later result matching
                        foreach (var toolCall in toolCalls)
                        {
                            toolCallsMap[toolCall.Id] = new ToolCallInfo
                            {
                                ToolCall = toolCall,
                                MessageId = assistantMessage.Id
                            };
                        }
                    }
                    break;

                case "tool.execution_complete":
                    UpdateToolCallResult(evt, toolCallsMap);
                    break;
            }
        }

        // Rebuild messages with updated tool calls
        return RebuildMessagesWithToolCalls(messages, toolCallsMap);
    }

    private Message? ParseUserMessage(CopilotEvent evt, string sessionId)
    {
        try
        {
            if (evt.Data is not JsonElement jsonElement)
            {
                ReportDropped("user message", evt.Id, "event carried no JSON payload");
                return null;
            }

            var data = JsonSerializer.Deserialize<UserMessageData>(
                jsonElement.GetRawText(), JsonOptions);

            if (data == null)
            {
                ReportDropped("user message", evt.Id, "payload deserialized to null");
                return null;
            }

            var content = data.TransformedContent ?? data.Content ?? string.Empty;

            return new Message(
                Id: evt.Id,
                SessionId: sessionId,
                Role: MessageRole.User,
                Content: content,
                RawContent: data.Content,
                Timestamp: evt.Timestamp,
                ParentId: evt.ParentId,
                Model: null,
                ToolCalls: null
            );
        }
        catch (JsonException ex)
        {
            ReportDropped("user message", evt.Id, $"malformed payload: {ex.Message}");
            return null;
        }
    }

    private (Message?, List<ToolCall>) ParseAssistantMessage(CopilotEvent evt, string sessionId)
    {
        try
        {
            if (evt.Data is not JsonElement jsonElement)
            {
                ReportDropped("assistant message", evt.Id, "event carried no JSON payload");
                return (null, new List<ToolCall>());
            }

            var data = JsonSerializer.Deserialize<AssistantMessageData>(
                jsonElement.GetRawText(), JsonOptions);

            if (data == null)
            {
                ReportDropped("assistant message", evt.Id, "payload deserialized to null");
                return (null, new List<ToolCall>());
            }

            var content = data.Content ?? string.Empty;
            var toolCalls = new List<ToolCall>();

            // Parse tool requests if present
            if (data.ToolRequests != null && data.ToolRequests.Count > 0)
            {
                foreach (var toolRequest in data.ToolRequests)
                {
                    if (string.IsNullOrEmpty(toolRequest.ToolCallId))
                    {
                        continue;
                    }

                    var toolCall = new ToolCall(
                        Id: toolRequest.ToolCallId,
                        MessageId: evt.Id,
                        Name: toolRequest.Name ?? "unknown",
                        Arguments: toolRequest.Arguments != null
                            ? JsonSerializer.Serialize(toolRequest.Arguments, JsonOptions)
                            : null,
                        Result: null,
                        Success: null
                    );
                    toolCalls.Add(toolCall);
                }
            }

            var message = new Message(
                Id: evt.Id,
                SessionId: sessionId,
                Role: MessageRole.Assistant,
                Content: content,
                RawContent: null,
                Timestamp: evt.Timestamp,
                ParentId: evt.ParentId,
                Model: null,
                ToolCalls: toolCalls.Count > 0 ? toolCalls : null
            );

            return (message, toolCalls);
        }
        catch (JsonException ex)
        {
            ReportDropped("assistant message", evt.Id, $"malformed payload: {ex.Message}");
            return (null, new List<ToolCall>());
        }
    }

    private void UpdateToolCallResult(CopilotEvent evt, Dictionary<string, ToolCallInfo> toolCallsMap)
    {
        try
        {
            if (evt.Data is not JsonElement jsonElement)
            {
                return;
            }

            var data = JsonSerializer.Deserialize<ToolExecutionCompleteData>(
                jsonElement.GetRawText(), JsonOptions);

            if (data == null || string.IsNullOrEmpty(data.ToolCallId))
            {
                return;
            }

            if (toolCallsMap.TryGetValue(data.ToolCallId, out var toolCallInfo))
            {
                var resultContent = data.Result?.Content ?? string.Empty;

                toolCallInfo.ToolCall = toolCallInfo.ToolCall with
                {
                    Result = resultContent,
                    Success = data.Success
                };
            }
        }
        catch (JsonException ex)
        {
            ReportDropped("tool result", evt.Id, $"malformed payload: {ex.Message}");
        }
    }

    /// <summary>
    /// Reports content the connector could not parse. Indexing continues without it, so this is
    /// the only signal that a session is stored with gaps.
    /// </summary>
    private static void ReportDropped(string what, string? id, string reason)
    {
        Console.Error.WriteLine(
            $"agent-journal: skipped {what} {id ?? "<no id>"} - {reason}");
    }

    private IReadOnlyList<Message> RebuildMessagesWithToolCalls(
        List<Message> messages,
        Dictionary<string, ToolCallInfo> toolCallsMap)
    {
        var rebuiltMessages = new List<Message>();

        foreach (var message in messages)
        {
            if (message.ToolCalls == null || message.ToolCalls.Count == 0)
            {
                rebuiltMessages.Add(message);
                continue;
            }

            // Update tool calls with results
            var updatedToolCalls = message.ToolCalls
                .Select(tc =>
                {
                    if (toolCallsMap.TryGetValue(tc.Id, out var info))
                    {
                        return info.ToolCall;
                    }
                    return tc;
                })
                .ToList();

            var updatedMessage = message with { ToolCalls = updatedToolCalls };
            rebuiltMessages.Add(updatedMessage);
        }

        return rebuiltMessages;
    }

    // Internal data structures for JSON parsing
    private class CopilotEvent
    {
        public string Type { get; set; } = string.Empty;
        public string Id { get; set; } = string.Empty;
        public string? ParentId { get; set; }
        public DateTime Timestamp { get; set; }
        public object? Data { get; set; }
    }

    private class SessionStartData
    {
        public string SessionId { get; set; } = string.Empty;
        public string? CopilotVersion { get; set; }
        public DateTime? StartTime { get; set; }
    }

    private class UserMessageData
    {
        public string? Content { get; set; }
        public string? TransformedContent { get; set; }
        public List<object>? Attachments { get; set; }
    }

    private class AssistantMessageData
    {
        public string? MessageId { get; set; }
        public string? Content { get; set; }
        public List<ToolRequest>? ToolRequests { get; set; }
    }

    private class ToolRequest
    {
        public string? ToolCallId { get; set; }
        public string? Name { get; set; }
        public object? Arguments { get; set; }
    }

    private class ToolExecutionCompleteData
    {
        public string? ToolCallId { get; set; }
        public bool? Success { get; set; }
        public ToolExecutionResult? Result { get; set; }
    }

    private class ToolExecutionResult
    {
        public string? Content { get; set; }
    }

    private class ToolCallInfo
    {
        public ToolCall ToolCall { get; set; } = null!;
        public string MessageId { get; set; } = string.Empty;
    }
}
