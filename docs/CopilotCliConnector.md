# CopilotCliConnector Implementation

## Overview

The `CopilotCliConnector` is a connector implementation for parsing GitHub Copilot CLI agent sessions stored in JSONL (JSON Lines) format.

## Features

### 1. Session Discovery (`GetSessionPaths`)
- Scans `~/.copilot-cli/sessions/` directory
- Returns paths to session directories containing `events.jsonl` files
- Cross-platform compatible using `Environment.SpecialFolder.UserProfile`

### 2. Session Parsing (`ParseSessionAsync`)
Parses JSONL event logs and reconstructs conversation sessions:

#### Supported Event Types
- `session.start` - Extracts session metadata (ID, version, start time)
- `user.message` - User input messages with optional transformedContent
- `assistant.message` - Assistant responses with optional tool requests
- `tool.execution_complete` - Tool results with success status

#### Event Chain Reconstruction
- Events ordered by timestamp
- Parent-child relationships via `parentId`
- Tool calls matched to results via `toolCallId`

### 3. Batch Processing (`ParseSessionsAsync`)
- Iterates through all discovered session paths
- Yields sessions asynchronously for streaming processing
- Gracefully handles parsing failures

## Data Mapping

### Session Fields
```csharp
Session(
    Id: from session.start.data.sessionId,
    AgentType: "copilot-cli",
    ProjectPath: session directory path,
    AgentVersion: from session.start.data.copilotVersion,
    StartedAt: from session.start.data.startTime,
    EndedAt: last event timestamp,
    Messages: reconstructed from events
)
```

### Message Fields
```csharp
Message(
    Id: event.id,
    SessionId: session.id,
    Role: User|Assistant,
    Content: message content (transformedContent preferred for user messages),
    RawContent: original user.message.data.content,
    Timestamp: event.timestamp,
    ParentId: event.parentId,
    ToolCalls: extracted from assistant.message.data.toolRequests
)
```

### ToolCall Fields
```csharp
ToolCall(
    Id: toolRequest.toolCallId,
    MessageId: assistant message id,
    Name: toolRequest.name,
    Arguments: JSON serialized toolRequest.arguments,
    Result: from tool.execution_complete.data.result.content,
    Success: from tool.execution_complete.data.success
)
```

## Implementation Details

### JSON Parsing
- Uses `System.Text.Json` with camelCase naming policy
- Type-safe deserialization with internal DTOs
- Graceful error handling (returns null on parsing failures)

### Tool Call Tracking
- Tool requests stored in map by `toolCallId`
- Results matched and merged when `tool.execution_complete` events arrive
- Messages rebuilt with complete tool call information

### Event Handling
- Reads JSONL line-by-line asynchronously
- Skips empty lines and malformed JSON
- Sorts events by timestamp to ensure correct ordering

## Testing

Test data created in `test-data/copilot-cli-test/` and `test-data/copilot-cli-complex/`:

### Test Case 1: Simple Session
- 1 user message
- 1 assistant message with 1 tool call
- Tool call with successful result

### Test Case 2: Complex Session
- 2 user messages
- 2 assistant messages with multiple tool calls
- Mix of successful and failed tool executions

### Test Results
✅ All messages parsed correctly
✅ Tool calls matched with results
✅ Session metadata extracted
✅ Timestamps and durations calculated
✅ Solution builds without errors or warnings

## Usage Example

```csharp
var connector = new CopilotCliConnector();

// Parse single session
var session = await connector.ParseSessionAsync(sessionPath);

// Parse all sessions
await foreach (var session in connector.ParseSessionsAsync())
{
    Console.WriteLine($"{session.Id}: {session.MessageCount} messages");
    foreach (var msg in session.Messages)
    {
        Console.WriteLine($"[{msg.Role}] {msg.Content}");
    }
}

// Discover session paths
foreach (var path in connector.GetSessionPaths())
{
    Console.WriteLine(path);
}
```

## File Structure

```
CopilotCliConnector.cs
├── Public Interface (IAgentConnector)
│   ├── GetSessionPaths()
│   ├── ParseSessionsAsync()
│   └── ParseSessionAsync()
│
├── Session Building
│   ├── BuildSessionFromEvents()
│   └── BuildMessagesFromEvents()
│
├── Event Parsing
│   ├── ParseUserMessage()
│   ├── ParseAssistantMessage()
│   └── UpdateToolCallResult()
│
├── Utility
│   └── RebuildMessagesWithToolCalls()
│
└── Internal DTOs
    ├── CopilotEvent
    ├── SessionStartData
    ├── UserMessageData
    ├── AssistantMessageData
    ├── ToolRequest
    ├── ToolExecutionCompleteData
    └── ToolCallInfo
```

## Error Handling

- Returns `null` for invalid/unparseable sessions
- Catches and swallows JSON deserialization exceptions
- Validates required fields (sessionId, event types)
- Handles missing Data properties gracefully

## Performance Characteristics

- **Streaming**: Uses `IAsyncEnumerable` for memory-efficient processing
- **Lazy Evaluation**: Sessions parsed on-demand
- **File I/O**: Asynchronous reading with `File.ReadLinesAsync`
- **Memory**: O(n) where n = events per session (small sessions fit in memory)

## Future Enhancements

Potential improvements:
- [ ] Support for session.truncation events
- [ ] Extract MCP info from session.info events
- [ ] Parse attachments in user.message events
- [ ] Model detection from assistant responses
- [ ] Git branch extraction if available in events
- [ ] Session summary generation from content
