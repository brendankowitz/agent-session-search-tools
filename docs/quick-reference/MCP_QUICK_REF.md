# MCP Interface Quick Reference

## Command
```bash
agent-journal mcp
```

## Tools

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `search_sessions` | Search session history | query, mode, project, limit |
| `get_session` | Get full session | id |
| `list_recent_sessions` | List recent sessions | limit, project |
| `remember` | Store knowledge | content, tags, project, source |
| `recall` | Search knowledge | query, tags, project, mode, limit |
| `reinforce` | Prevent decay | ids (comma-separated) |
| `forget` | Remove knowledge | id |
| `search` | Unified search | query, mode, includeKnowledge, project, limit |

## Configuration

### Claude Desktop
`claude_desktop_config.json`:
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

### VS Code
`.vscode/mcp.json`:
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

## Search Modes
- `lexical` - Keyword search (Lucene)
- `semantic` - Meaning search (vectors)
- `hybrid` - Both (default, recommended)

## Example Usage

### From Claude
```
"Search past work on authentication"
→ Tool: search_sessions(query="authentication", mode="hybrid")

"Remember that we use JWT for auth"
→ Tool: remember(content="Using JWT for authentication", tags="auth,security")

"What did we decide about the database schema?"
→ Tool: recall(query="database schema", limit=5)
```

## Files

- **Tools**: `src/AgentJournal.Core/Mcp/AgentJournalTools.cs`
- **Server**: `src/AgentJournal.Core/Mcp/AgentJournalMcpServer.cs`
- **Command**: `src/AgentJournal/Commands/McpCommand.cs`
- **Docs**: `src/AgentJournal.Core/Mcp/README.md`
