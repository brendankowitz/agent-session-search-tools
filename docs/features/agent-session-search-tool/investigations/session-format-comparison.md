# Investigation: Session Format Comparison

**Feature**: agent-session-search-tool  
**Status**: ✅ Complete  
**Created**: 2026-01-16

## Purpose

Document the verified JSONL formats from Copilot CLI and Claude Code to enable accurate connector implementation and unified data model mapping.

---

## Copilot CLI Format

**Location**: `~/.copilot-cli/sessions/<session-id>/events.jsonl`  
**Structure**: Event-sourced JSONL with parent-child event chain

### Event Types

| Type | Purpose |
|------|---------|
| `session.start` | Session initialization with metadata |
| `session.info` | Authentication, MCP connection info |
| `session.truncation` | Context window management events |
| `user.message` | User input with optional attachments |
| `assistant.turn_start` | Marks beginning of assistant response |
| `assistant.message` | Assistant response with optional tool requests |
| `assistant.turn_end` | Marks end of assistant response |
| `tool.execution_start` | Tool invocation begins |
| `tool.execution_complete` | Tool result with success/failure |

### Schema

```typescript
// Base event structure
interface CopilotEvent {
  id: string;                    // UUID for this event
  type: CopilotEventType;        // Event discriminator
  timestamp: string;             // ISO 8601
  parentId: string | null;       // Links to parent event (chain)
  data: EventData;               // Type-specific payload
}

// session.start
interface SessionStartData {
  sessionId: string;
  version: number;
  producer: string;              // "copilot-agent"
  copilotVersion: string;        // e.g., "0.0.382"
  startTime: string;
}

// user.message
interface UserMessageData {
  content: string;               // Raw user input
  transformedContent: string;    // With injected context (datetime, etc.)
  attachments: Attachment[];
}

// assistant.message
interface AssistantMessageData {
  messageId: string;
  content: string;               // Text response (may be empty if only tools)
  toolRequests: ToolRequest[];
}

interface ToolRequest {
  toolCallId: string;
  name: string;
  arguments: Record<string, unknown>;
}

// tool.execution_complete
interface ToolCompleteData {
  toolCallId: string;
  success: boolean;
  result: { content: string };
  toolTelemetry: Record<string, unknown>;
}
```

### Key Observations

1. **Event chain via parentId**: Events form a linked list; reconstruct conversation order by following chain
2. **Turns are explicit**: `turn_start` / `turn_end` bracket each assistant response
3. **Tool calls inline**: Tools are in `assistant.message.data.toolRequests`, results in separate `tool.execution_complete` events
4. **Rich metadata**: Includes copilot version, truncation stats, telemetry

---

## Claude Code Format

**Location**: `~/.claude/projects/<project-hash>/<session-uuid>.jsonl`  
**Structure**: Flat JSONL with parent-child UUID linking

### Event Types

| Type | Purpose |
|------|---------|
| `summary` | Session summary (appears at file start) |
| `file-history-snapshot` | Git/file state snapshot |
| `user` | User message |
| `assistant` | Assistant response (may contain multiple content blocks) |
| `system` | System prompts/context |
| `queue-operation` | Internal queue state |

### Schema

```typescript
// Base record structure
interface ClaudeRecord {
  type: ClaudeRecordType;
  uuid: string;                  // Unique ID for this record
  parentUuid: string | null;     // Links to parent (conversation chain)
  sessionId: string;             // Groups records into session
  timestamp: string;             // ISO 8601
  cwd: string;                   // Working directory
  version: string;               // Claude Code version
  gitBranch?: string;            // Current git branch
  isSidechain: boolean;          // Branch in conversation tree
}

// user record
interface UserRecord extends ClaudeRecord {
  type: "user";
  userType: "external" | "internal";
  message: {
    role: "user";
    content: string | ContentBlock[];  // String or tool_result blocks
  };
  thinkingMetadata?: ThinkingMetadata;
  todos: Todo[];
}

// assistant record
interface AssistantRecord extends ClaudeRecord {
  type: "assistant";
  requestId: string;
  message: {
    model: string;               // e.g., "claude-sonnet-4-5-20250929"
    id: string;                  // Anthropic message ID
    role: "assistant";
    content: ContentBlock[];     // text, thinking, tool_use blocks
    usage: UsageStats;
    stop_reason: string | null;
  };
}

// Content blocks (Anthropic API format)
type ContentBlock = 
  | { type: "text"; text: string }
  | { type: "thinking"; thinking: string; signature: string }
  | { type: "tool_use"; id: string; name: string; input: unknown }
  | { type: "tool_result"; tool_use_id: string; content: string };
```

### Key Observations

1. **Anthropic API passthrough**: `message` field mirrors Anthropic's API response
2. **Extended thinking visible**: `thinking` blocks contain chain-of-thought with signature
3. **Content blocks array**: Mixed text and tool_use in single message
4. **Tool results as user**: `tool_result` appears as `user` type record
5. **Rich context**: Includes cwd, gitBranch, model name per record
6. **Summary records**: Searchable summaries at file start

---

## Unified Data Model Mapping

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         UNIFIED SESSION MODEL                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Session                                                                    │
│  ├── Id: string (UUID)                                                     │
│  ├── AgentType: "copilot-cli" | "claude-code" | ...                        │
│  ├── ProjectPath: string (cwd)                                             │
│  ├── GitBranch: string?                                                    │
│  ├── AgentVersion: string                                                  │
│  ├── StartedAt: DateTime                                                   │
│  ├── EndedAt: DateTime?                                                    │
│  ├── Summary: string? (from Claude summary records or generated)           │
│  └── Messages: Message[]                                                   │
│                                                                             │
│  Message                                                                    │
│  ├── Id: string (UUID)                                                     │
│  ├── SessionId: string                                                     │
│  ├── Role: User | Assistant | System | Tool                                │
│  ├── Content: string (normalized text)                                     │
│  ├── RawContent: string (original with thinking, etc.)                     │
│  ├── Timestamp: DateTime                                                   │
│  ├── ParentId: string? (for conversation tree)                             │
│  ├── Model: string? (for assistant messages)                               │
│  └── ToolCalls: ToolCall[]                                                 │
│                                                                             │
│  ToolCall                                                                   │
│  ├── Id: string                                                            │
│  ├── MessageId: string                                                     │
│  ├── Name: string                                                          │
│  ├── Arguments: JSON                                                       │
│  ├── Result: string?                                                       │
│  └── Success: bool?                                                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

MAPPING RULES:

Copilot CLI → Unified:
├── session.start.data.sessionId → Session.Id
├── session.start.data.copilotVersion → Session.AgentVersion
├── user.message.data.content → Message(Role=User).Content
├── assistant.message.data.content → Message(Role=Assistant).Content
├── assistant.message.data.toolRequests[*] → ToolCall[]
└── tool.execution_complete.data.result → ToolCall.Result

Claude Code → Unified:
├── sessionId (from any record) → Session.Id
├── version → Session.AgentVersion
├── cwd → Session.ProjectPath
├── gitBranch → Session.GitBranch
├── summary record → Session.Summary
├── user.message.content → Message(Role=User).Content
├── assistant.message.content[type=text].text → Message(Role=Assistant).Content
├── assistant.message.content[type=tool_use] → ToolCall
└── user.message.content[type=tool_result] → ToolCall.Result (linked by tool_use_id)
```

---

## Global Storage Configuration

To enable cross-agent search, indexes must be globally accessible:

```toml
# ~/.agent-journal/config.toml

[storage]
# Global index location (shared across all agents)
index_path = "~/.agent-journal/index"

# Per-agent source locations (auto-discovered if not specified)
[storage.sources]
copilot_cli = "~/.copilot-cli/sessions"
claude_code = "~/.claude/projects"
cursor = "~/.cursor/conversations"      # Platform-specific
aider = "**/.aider.chat.history.md"     # Glob pattern

[index]
# Search engine settings
lucene_ram_buffer_mb = 64
vector_precision = "f16"                 # f16 (compact) or f32 (precise)

[index.vector]
# Vector search settings
enabled = true
model_path = "~/.chats/models"          # ONNX model location
fallback = "hash"                        # "hash" or "none" when model absent

[search]
# Default search behavior
default_mode = "hybrid"                  # lexical, semantic, hybrid
default_context = 3                      # Messages before/after match
max_results = 20

[export]
# Export defaults
default_format = "html"
html_theme = "dark"                      # dark, light, auto
include_tool_calls = true
```

### Environment Variable Overrides

```bash
AJ_INDEX_PATH=~/.agent-journal/index
AJ_COPILOT_PATH=~/.copilot-cli/sessions
AJ_CLAUDE_PATH=~/.claude/projects
```

---

## Implementation Priority

Based on format analysis:

1. **Claude Code first** - Richer metadata, summaries pre-computed, simpler flat structure
2. **Copilot CLI second** - Event chain reconstruction needed, but well-structured
3. **Other agents** - After core proven

## Open Questions

1. **Sidechain handling**: Claude Code has `isSidechain` for conversation branches - how to index?
2. **Thinking content**: Should extended thinking be searchable? (Privacy vs utility)
3. **Tool call verbosity**: Full tool args/results or summarized?
