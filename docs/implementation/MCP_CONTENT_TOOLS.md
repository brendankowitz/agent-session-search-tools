# MCP Content Tools Implementation

## Overview

MCP (Model Context Protocol) tools for content operations, matching the CLI functionality. These tools enable programmatic content management through MCP-compatible clients like Claude Desktop.

## Implementation Summary

### Files Modified

1. **`src/AgentJournal.Core/Mcp/AgentJournalTools.cs`**
   - Added `IContentRepository` dependency to constructor
   - Added 6 new MCP tool methods for content operations
   - Added helper methods now in `ContentUtils`
   - Added 7 new result types for content operations

2. **`src/AgentJournal.Core/Mcp/AgentJournalMcpServer.cs`**
   - Updated `CreateHostAsync()` to accept `IContentRepository` parameter
   - Updated `RunAsync()` to accept `IContentRepository` parameter
   - Registered `IContentRepository` in DI container

3. **`src/AgentJournal/Commands/McpCommand.cs`**
   - Updated to retrieve `IContentRepository` from service provider
   - Updated `ExecuteAsync()` to pass content repository to MCP server

## MCP Tools

See the main [Content Indexing User Guide](../CONTENT_INDEXING.md) for complete documentation on all content operations.

### Tool Overview

| Tool | Purpose | Key Parameters |
|------|---------|---------------|
| IndexContent | Index markdown files | path, filter, project, recursive |
| AddContent | Add content directly | source, title, content, project, tags |
| SearchContent | Full-text search | query, maxResults, project, sourcePrefix, tags |
| ListContent | List content | project, sourcePrefix, tags, limit, expiredOnly |
| RemoveContent | Remove by criteria | id, source, sourcePrefix, project |
| ReinforceContent | Reset decay timer | source |

## Integration with Claude Desktop

### Configuration

Add to your MCP client configuration:

**Mac/Linux:** `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "agentjournal": {
      "command": "dotnet",
      "args": ["run", "--project", "/path/to/src/AgentJournal", "--", "mcp"]
    }
  }
}
```

## Dependencies

The content tools use these existing components:
- `IContentRepository` - Content storage and retrieval
- `DecayCalculator` - Time-based decay calculations  
- `Microsoft.Extensions.FileSystemGlobbing` - File pattern matching
- `ContentUtils` - Shared security and utility methods

## Build and Verification

```bash
# Build the project
dotnet build src/AgentJournal

# Run with MCP mode (stdio protocol)
dotnet run --project src/AgentJournal -- mcp

# Check help
dotnet run --project src/AgentJournal -- --help
dotnet run --project src/AgentJournal -- content --help
```

All builds completed successfully with no errors.

## Related Documentation

- [Content Indexing User Guide](../CONTENT_INDEXING.md) - Complete user documentation
- [Content Implementation](CONTENT_IMPLEMENTATION.md) - Technical implementation details
- [Content Quick Reference](../quick-reference/CONTENT_QUICK_REF.md) - Command cheat sheet
- [Content Security Review](../reviews/CONTENT_SECURITY_REVIEW.md) - Security analysis
