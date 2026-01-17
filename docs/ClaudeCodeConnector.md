# Claude Code Connector Implementation

## Overview

The `ClaudeCodeConnector` class implements the `IAgentConnector` interface to parse Claude Code session data from JSONL files stored in the `~/.claude/projects/` directory.

## File Location

**Path**: `src/AgentJournal.Core/Connectors/ClaudeCodeConnector.cs`

## Implementation Details

### 1. GetSessionPaths()

Discovers all Claude Code session files:

- **Location**: `~/.claude/projects/<project-hash>/<session-uuid>.jsonl`
- **Filter**: Only includes `.jsonl` files with UUID-like names (contains hyphens, min 32 chars)
- **Recursion**: Searches all subdirectories in the projects folder

```csharp
public IEnumerable<string> GetSessionPaths()
{
    var homeDir = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
    var claudeProjectsPath = Path.Combine(homeDir, ".claude", "projects");
    
    // Finds all .jsonl files that look like session UUIDs
    foreach (var file in Directory.EnumerateFiles(claudeProjectsPath, "*.jsonl", SearchOption.AllDirectories))
    {
        var fileName = Path.GetFileNameWithoutExtension(file);
        if (fileName.Contains('-') && fileName.Length >= 32)
        {
            yield return file;
        }
    }
}
```

### 2. ParseSessionAsync()

Parses a single JSONL session file:

**Input Format**:
```json
{
  "type": "user|assistant|summary|file-history-snapshot",
  "uuid": "unique-message-id",
  "parentUuid": "parent-message-id|null",
  "sessionId": "session-guid",
  "timestamp": "ISO-8601-timestamp",
  "cwd": "working-directory",
  "version": "2.1.6",
  "gitBranch": "branch-name",
  "message": {
    "role": "user|assistant",
    "content": "string or ContentBlock[]",
    "model": "claude-sonnet-4-5-20250929"
  }
}
```

**Process**:
1. Read file line by line (JSONL format)
2. Deserialize each line as a `ClaudeCodeRecord`
3. Skip malformed lines with warning
4. Extract session metadata from first record and summary record
5. Parse user/assistant messages
6. Extract content from ContentBlocks
7. Build Session object with all messages

### 3. Content Block Handling

Supports multiple ContentBlock types:

#### Text Block
```json
{ "type": "text", "text": "content here" }
```
Extracted to main content.

#### Thinking Block
```json
{ "type": "thinking", "thinking": "...", "signature": "..." }
```
Skipped for main content (could be logged separately).

#### Tool Use Block
```json
{
  "type": "tool_use",
  "id": "toolu_123",
  "name": "view",
  "input": { "path": "/some/file" }
}
```
Converted to `ToolCall` with serialized input as Arguments.

#### Tool Result Block
```json
{
  "type": "tool_result",
  "tool_use_id": "toolu_123",
  "content": "result data"
}
```
Currently skipped (could be matched to tool_use in future enhancement).

### 4. Message Parsing

Each user/assistant record is converted to a `Message`:

- **Id**: UUID from record
- **SessionId**: Extracted from record or filename
- **Role**: User or Assistant
- **Content**: Combined text from text blocks
- **RawContent**: Full JSON serialization of content
- **Timestamp**: Parsed from ISO-8601 string
- **ParentId**: Parent message UUID
- **Model**: AI model identifier (e.g., "claude-sonnet-4-5-20250929")
- **ToolCalls**: List of tool calls from tool_use blocks

### 5. Session Building

The final `Session` object includes:

- **Id**: Session UUID
- **AgentType**: "claude-code"
- **ProjectPath**: Working directory from first record
- **GitBranch**: Git branch name
- **AgentVersion**: Claude Code version (e.g., "2.1.6")
- **StartedAt**: Timestamp of first message
- **EndedAt**: Timestamp of last message (only if summary record exists)
- **Summary**: Content from summary record
- **Messages**: All parsed messages in chronological order

## Error Handling

1. **Missing directories**: Returns empty enumerable
2. **Malformed JSON lines**: Logs warning and skips line
3. **Empty files**: Returns null session
4. **Missing required fields**: Uses null coalescing and sensible defaults
5. **Parse exceptions**: Catches and logs, returns null session

## JSON Serialization

Uses `System.Text.Json` with options:
- Case-insensitive property matching
- Camel case naming policy
- Allows trailing commas
- Skips comments
- Ignores null values

## Usage Example

```csharp
var connector = new ClaudeCodeConnector();

// Get all sessions
await foreach (var session in connector.ParseSessionsAsync())
{
    Console.WriteLine($"Session: {session.Id}");
    Console.WriteLine($"Messages: {session.MessageCount}");
    Console.WriteLine($"Tool Calls: {session.ToolCallCount}");
}

// Parse single session
var sessionPath = "~/.claude/projects/abc123/def456.jsonl";
var session = await connector.ParseSessionAsync(sessionPath);
```

## Future Enhancements

1. **Tool Result Matching**: Link tool_result blocks to corresponding tool_use blocks
2. **Thinking Block Capture**: Optionally include thinking content in RawContent
3. **File History Processing**: Parse file-history-snapshot records for file state tracking
4. **Performance**: Add streaming for large JSONL files
5. **Validation**: Add schema validation for record types

## Testing

Build verification:
```bash
cd E:\data\src\agent-session-search-tools
dotnet build
```

Result: ✅ Build succeeded with 0 errors (3 unrelated warnings in other files)

## Related Files

- **Interface**: `src/AgentJournal.Core/Connectors/IAgentConnector.cs`
- **Models**: 
  - `src/AgentJournal.Core/Models/Session.cs`
  - `src/AgentJournal.Core/Models/Message.cs`
  - `src/AgentJournal.Core/Models/ToolCall.cs`
  - `src/AgentJournal.Core/Models/MessageRole.cs`
