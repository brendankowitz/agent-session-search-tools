# Investigation: MCP (Model Context Protocol) Interface

**Feature**: agent-session-search-tool  
**Status**: Investigation  
**Created**: 2026-01-17

## Summary

Investigate adding an MCP stdio server interface to agent-journal, enabling AI agents (Claude, Copilot, etc.) to directly query session history through the standardized Model Context Protocol.

## What is MCP?

Model Context Protocol (MCP) is an open standard for connecting LLM applications with external tools, data sources, and context. Key features:

- **JSON-RPC 2.0** over stdio (or HTTP/SSE)
- **Tools**: Schema-defined functions the model can call
- **Resources**: Data/files the model can read
- **Prompts**: Reusable instruction templates
- **Cross-platform**: Works with Claude Desktop, VS Code Copilot, and other MCP clients

## Why Add MCP to Agent Journal?

| Current State | With MCP |
|---------------|----------|
| CLI only - agents must shell out | Native protocol integration |
| Parse text output | Structured JSON responses |
| No streaming | Real-time streaming support |
| Manual invocation | Automatic tool discovery |

### Use Cases Enabled

1. **Copilot/Claude can directly search past sessions** without shell commands
2. **Context injection** - Agent can pull relevant past sessions automatically
3. **IDE integration** - VS Code MCP panel can show session search
4. **Multi-agent workflows** - Agents can query each other's history

---

## Technical Approach

### Architecture: Dual Interface

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Agent Journal                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────┐              ┌─────────────────┐               │
│  │   CLI Host      │              │   MCP Host      │               │
│  │ (System.CommandLine)           │ (McpServer)     │               │
│  └────────┬────────┘              └────────┬────────┘               │
│           │                                │                         │
│           └──────────────┬─────────────────┘                        │
│                          │                                           │
│                          ▼                                           │
│           ┌─────────────────────────────┐                           │
│           │     Shared Core Services     │                           │
│           │  - ISearchEngine             │                           │
│           │  - ISessionRepository        │                           │
│           │  - IEmbeddingProvider        │                           │
│           └─────────────────────────────┘                           │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### MCP Server Implementation

Using the official C# SDK: `ModelContextProtocol` NuGet package.

```csharp
// Program.cs for MCP mode
var builder = Host.CreateApplicationBuilder(args);
builder.Services
    .AddMcpServer()
    .WithStdioServerTransport()
    .WithToolsFromAssembly();

// Register our services
builder.Services.AddSingleton<ISearchEngine, HybridSearcher>();
// ... other services

await builder.Build().RunAsync();
```

### Proposed MCP Tools

| Tool | Description | Parameters |
|------|-------------|------------|
| `search_sessions` | Search past sessions | `query`, `mode`, `maxResults`, `agent`, `project` |
| `get_session` | Get full session by ID | `sessionId` |
| `list_recent_sessions` | List recent sessions | `count`, `agent` |
| `export_session` | Export session to format | `sessionId`, `format` |
| `index_sessions` | Trigger reindex | `rebuild` |

### Tool Definitions

```csharp
[McpServerToolType]
public class AgentJournalTools
{
    private readonly ISearchEngine _searchEngine;
    private readonly ISessionRepository _repository;

    [McpServerTool(Description = "Search past AI agent sessions for relevant context")]
    public async Task<SearchToolResult> SearchSessions(
        [Description("Search query")] string query,
        [Description("Search mode: lexical, semantic, or hybrid")] string mode = "hybrid",
        [Description("Maximum results")] int maxResults = 10,
        [Description("Filter by agent type")] string? agent = null,
        [Description("Filter by project path")] string? project = null)
    {
        var searchMode = mode switch
        {
            "semantic" => SearchMode.Semantic,
            "lexical" => SearchMode.Lexical,
            _ => SearchMode.Hybrid
        };
        
        var results = await _searchEngine.SearchAsync(query, searchMode, maxResults);
        
        // Filter and return structured results
        return new SearchToolResult { Sessions = results.Select(r => new SessionSummary { ... }) };
    }

    [McpServerTool(Description = "Get full session content by ID")]
    public async Task<SessionContent> GetSession(
        [Description("Session ID")] string sessionId)
    {
        var session = await _repository.GetSessionAsync(sessionId);
        return new SessionContent { ... };
    }
}
```

### Proposed MCP Resources

| Resource URI | Description |
|--------------|-------------|
| `session://{id}` | Full session content |
| `session://{id}/messages` | Session messages only |
| `config://settings` | Current configuration |

### Proposed MCP Prompts

| Prompt | Description | Parameters |
|--------|-------------|------------|
| `find-similar-work` | Search for similar past work | `description` |
| `summarize-project-sessions` | Summarize sessions for a project | `projectPath` |

---

## Implementation Plan

### Phase 1: Core MCP Server (2-3 hours)
1. Add `ModelContextProtocol` NuGet package
2. Create `AgentJournal.Mcp` project or add MCP host to existing
3. Implement basic `search_sessions` tool
4. Add stdio transport

### Phase 2: Full Tool Suite (2-3 hours)
1. Add `get_session`, `list_recent_sessions` tools
2. Add `export_session` tool
3. Add `index_sessions` tool (with progress)

### Phase 3: Resources & Prompts (1-2 hours)
1. Implement session resources
2. Add workflow prompts
3. Add configuration resource

### Phase 4: Integration & Testing (1-2 hours)
1. Test with Claude Desktop
2. Test with VS Code Copilot
3. Add `.vscode/mcp.json` configuration
4. Documentation

---

## Entry Point Options

### Option A: Separate Executable
```
agent-journal          # CLI
agent-journal-mcp      # MCP server
```
**Pros**: Clean separation, simple
**Cons**: Two binaries to distribute

### Option B: Subcommand
```
agent-journal mcp      # Start MCP server mode
agent-journal search   # CLI mode (existing)
```
**Pros**: Single binary, consistent
**Cons**: Slightly more complex startup

### Option C: Auto-detect
Detect if stdin is a pipe/MCP client and switch modes.
**Pros**: Seamless
**Cons**: Complex, potential edge cases

**Recommendation**: Option B (subcommand) - balances simplicity with single distribution.

---

## Client Configuration

### Claude Desktop (`claude_desktop_config.json`)
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

### VS Code (`.vscode/mcp.json`)
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

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `ModelContextProtocol` | 0.1.x (prerelease) | MCP SDK |
| `Microsoft.Extensions.Hosting` | 10.x | Host for MCP server |

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| MCP SDK is prerelease | Pin version, test thoroughly |
| Breaking protocol changes | Follow spec, version our tools |
| Performance with large results | Limit response sizes, streaming |
| Concurrent access to indexes | Already have locking in place |

---

## Alignment Checklist

- [x] Reuses existing search/repository infrastructure
- [x] Doesn't require architectural changes to core
- [x] Follows MCP specification
- [x] Maintains CLI backward compatibility
- [x] Supports all three search modes

---

## Verdict

**Viable**: ✅ YES

MCP integration is straightforward with the official C# SDK. The existing service architecture (ISearchEngine, ISessionRepository) maps directly to MCP tools. Estimated effort: **6-10 hours** for full implementation.

**Recommendation**: Proceed with implementation using Option B (subcommand approach).

---

## Next Steps

1. Add `ModelContextProtocol` package
2. Create `McpCommand` subcommand
3. Implement `AgentJournalTools` class with search/get/list tools
4. Test with Claude Desktop and VS Code
5. Document configuration for users
