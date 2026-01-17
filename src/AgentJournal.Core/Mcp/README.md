# MCP (Model Context Protocol) Interface

The Agent Journal MCP interface provides a standardized protocol for AI agents (Claude, Copilot, etc.) to directly query session history and knowledge bank.

## Quick Start

### Start MCP Server

```bash
agent-journal mcp
```

The MCP server runs on stdio and communicates using JSON-RPC 2.0 protocol.

## Client Configuration

### Claude Desktop

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "agent-journal": {
      "command": "agent-journal",
      "args": ["mcp"]
    }
  }
}
```

**Location**:
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`
- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Linux: `~/.config/Claude/claude_desktop_config.json`

### VS Code Copilot

Add to `.vscode/mcp.json` in your workspace:

```json
{
  "servers": {
    "agent-journal": {
      "command": "agent-journal",
      "args": ["mcp"],
      "type": "stdio"
    }
  }
}
```

## Available Tools

### Session Tools

#### search_sessions
Search past AI agent sessions for relevant conversations.

**Parameters**:
- `query` (string): Search query to find relevant sessions
- `mode` (string, optional): Search mode - "lexical", "semantic", or "hybrid" (default: "hybrid")
- `project` (string, optional): Filter by project path or name
- `limit` (int, optional): Maximum number of results (default: 10)

**Example**:
```json
{
  "query": "authentication implementation",
  "mode": "hybrid",
  "project": "myapp",
  "limit": 5
}
```

#### get_session
Get complete session content including all messages and metadata.

**Parameters**:
- `id` (string): Session ID to retrieve

#### list_recent_sessions
List recent agent sessions with optional project filtering.

**Parameters**:
- `limit` (int, optional): Maximum number of sessions (default: 10)
- `project` (string, optional): Filter by project path or name

### Knowledge Tools

#### remember
Store important information in the knowledge bank.

**Parameters**:
- `content` (string): Content to remember
- `tags` (string, optional): Tags for categorization (comma-separated)
- `project` (string, optional): Associated project path
- `source` (string, optional): Source of this knowledge

**Example**:
```json
{
  "content": "The API uses JWT tokens for authentication",
  "tags": "auth,api,security",
  "project": "myapp"
}
```

#### recall
Search the knowledge bank for relevant information.

**Parameters**:
- `query` (string): Search query
- `tags` (string, optional): Filter by tags (comma-separated)
- `project` (string, optional): Filter by project path
- `mode` (string, optional): Search mode (default: "hybrid")
- `limit` (int, optional): Maximum results (default: 10)

#### reinforce
Reinforce knowledge entries by resetting their decay timer.

**Parameters**:
- `ids` (string): Knowledge entry IDs to reinforce (comma-separated)

#### forget
Remove knowledge from the knowledge bank.

**Parameters**:
- `id` (string): Knowledge entry ID to remove

### Unified Search

#### search
Search both sessions and knowledge bank with unified results.

**Parameters**:
- `query` (string): Search query
- `mode` (string, optional): Search mode (default: "hybrid")
- `includeKnowledge` (bool, optional): Include knowledge entries (default: true)
- `project` (string, optional): Filter by project path
- `limit` (int, optional): Maximum total results (default: 20)

## Search Modes

- **lexical**: Keyword-based search using Lucene full-text search
- **semantic**: Meaning-based search using vector embeddings
- **hybrid**: Combines lexical and semantic search (recommended)

## Use Cases

### 1. Context Injection
AI agents can automatically pull relevant past sessions when working on similar tasks.

```
Claude: "Let me search for past work on authentication..."
Tool: search_sessions(query="authentication", limit=3)
```

### 2. Knowledge Persistence
Store important learnings that should be remembered across sessions.

```
Agent: "I'll remember that the API key is stored in environment variables"
Tool: remember(content="API key stored in env vars", tags="config,api")
```

### 3. Project History
Query all sessions related to a specific project.

```
Tool: search_sessions(query="*", project="myapp", limit=20)
```

### 4. Multi-Agent Workflows
Different agents can query each other's session history.

## Architecture

```
┌─────────────────────────────────────┐
│     AI Agent (Claude, Copilot)     │
├─────────────────────────────────────┤
│                                     │
│  ┌──────────────────────────────┐  │
│  │   MCP Client (stdio)         │  │
│  └──────────┬───────────────────┘  │
└─────────────┼──────────────────────┘
              │ JSON-RPC 2.0
              ▼
┌─────────────────────────────────────┐
│      agent-journal mcp              │
├─────────────────────────────────────┤
│  ┌──────────────────────────────┐  │
│  │   AgentJournalMcpServer     │  │
│  │   (stdio transport)         │  │
│  └──────────┬───────────────────┘  │
│             │                       │
│             ▼                       │
│  ┌──────────────────────────────┐  │
│  │   AgentJournalTools          │  │
│  │   - SearchSessions           │  │
│  │   - GetSession               │  │
│  │   - Remember/Recall          │  │
│  │   - Forget/Reinforce         │  │
│  └──────────┬───────────────────┘  │
│             │                       │
│             ▼                       │
│  ┌──────────────────────────────┐  │
│  │   Core Services              │  │
│  │   - ISearchEngine            │  │
│  │   - ISessionRepository       │  │
│  │   - IKnowledgeRepository     │  │
│  └──────────────────────────────┘  │
└─────────────────────────────────────┘
```

## Protocol

Agent Journal implements the [Model Context Protocol](https://modelcontextprotocol.io/) specification.

- **Transport**: stdio (standard input/output)
- **Protocol**: JSON-RPC 2.0
- **Message Format**: MCP messages (initialize, tools/list, tools/call, etc.)

## Troubleshooting

### Server Not Starting

Check that agent-journal is installed globally:
```bash
dotnet tool list -g
```

### Tools Not Appearing

1. Restart your AI client (Claude Desktop, VS Code)
2. Check the configuration file path and format
3. Verify agent-journal is accessible from command line

### Connection Issues

The MCP server logs to stderr (not stdout, which is reserved for protocol messages).
Check your AI client's logs for error messages.

## Development

To test the MCP interface manually:

```bash
# Start the MCP server
agent-journal mcp

# Send a test request (JSON-RPC 2.0 format)
echo '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' | agent-journal mcp
```

## See Also

- [Agent Journal Documentation](../README.md)
- [Model Context Protocol Specification](https://modelcontextprotocol.io/)
- [Search Modes](../KNOWLEDGE_SEARCH_ARCHITECTURE.md)
