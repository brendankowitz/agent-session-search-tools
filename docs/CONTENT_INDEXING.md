# Content Indexing - User Guide

## Overview

AgentJournal provides powerful content indexing capabilities that allow you to index markdown files from directories and directly add content for easy retrieval. The content indexing system uses SQLite FTS5 for full-text search and implements a temporal decay system to prioritize recent content.

## Features

- **Full-text search** with FTS5 for fast, accurate results
- **Temporal decay** to prioritize recent content (90-day half-life)
- **Hash-based deduplication** to skip unchanged files during re-indexing
- **Flexible filtering** by project, source prefix, and tags
- **Multiple interfaces**: CLI commands and MCP tools
- **JSON output** for automation and scripting

## Quick Start

### Index Documentation

```bash
# Index markdown files in a directory
agent-journal content index ./docs --project myproject

# Index with custom glob pattern
agent-journal content index ./docs --filter "*.markdown" --project myproject

# Force rebuild of all content
agent-journal content index ./docs --rebuild
```

### Add Content Directly

```bash
# Add content with metadata
agent-journal content add \
  --source "note-1" \
  --title "Important Note" \
  --content "Remember this information" \
  --tags "urgent,todo"

# Add from stdin
echo "Content from pipe" | agent-journal content add \
  --source "pipe-1" \
  --title "Piped Content"
```

### Search Content

```bash
# Basic search
agent-journal content search "database"

# Search with filters
agent-journal content search "API" \
  --project myproject \
  --max 5 \
  --source-prefix "docs/api"

# Search with tag filtering
agent-journal content search "authentication" --tags "api,security"

# Get JSON for automation
agent-journal content search "guide" --robot
```

### List Content

```bash
# List all content
agent-journal content list

# List by project
agent-journal content list --project myproject

# List with filters
agent-journal content list \
  --source-prefix "docs/" \
  --tags "tutorial,guide" \
  --limit 20

# Show only expired content
agent-journal content list --expired

# Get JSON output
agent-journal content list --robot
```

### Remove Content

```bash
# Remove by source
agent-journal content remove --source "docs/obsolete.md"

# Remove by ID
agent-journal content remove --id "abc123def456"

# Remove all under a directory
agent-journal content remove --source-prefix "docs/deprecated/"

# Remove all for a project
agent-journal content remove --project old-project

# Remove with multiple filters (AND logic)
agent-journal content remove --project myproject --source-prefix "specs/"

# Force remove without confirmation
agent-journal content remove --project old-project --force

# Remove all content (with confirmation)
agent-journal content remove --all
```

### Reinforce Content

Keep important content fresh by resetting its decay timer:

```bash
# Reinforce a specific document
agent-journal content reinforce --source "docs/important.md"
```

## Command Reference

### `content index <path>`

Index markdown files from a directory.

**Options:**
- `--filter` / `-f` - Glob pattern for file matching (default: `*.md`)
- `--project` / `-p` - Project name to associate with content
- `--recursive` / `-r` - Recursively scan subdirectories (default: `true`)
- `--rebuild` - Force re-indexing of all files, even if unchanged

**Behavior:**
- Skips files with unchanged content (hash-based detection)
- Extracts title from markdown headers or uses filename
- Reports progress and statistics

### `content add`

Add content directly without reading from a file.

**Options:**
- `--source` / `-s` - Source identifier (required)
- `--title` / `-t` - Content title (required)
- `--content` / `-c` - Content text (or use stdin)
- `--project` / `-p` - Project name
- `--tags` - Comma-separated tags

**Behavior:**
- Updates existing content if source already exists
- Supports stdin for piping content
- Preserves creation timestamp on updates

### `content search <query>`

Search indexed content using full-text search.

**Options:**
- `--max` / `-n` - Maximum number of results (default: 10)
- `--project` / `-p` - Filter by project
- `--source-prefix` / `-s` - Filter by source path prefix
- `--tags` / `-t` - Filter by tags (comma-separated)
- `--robot` - Output as JSON

**Features:**
- FTS5 full-text search with highlighting
- Decay-adjusted relevance scoring
- Multiple filter options can be combined

### `content list`

List indexed content entries.

**Options:**
- `--project` / `-p` - Filter by project
- `--source-prefix` / `-s` - Filter by source path prefix
- `--tags` / `-t` - Filter by tags (comma-separated)
- `--limit` / `-n` - Maximum entries to return (default: 50)
- `--expired` - Show only expired content
- `--robot` - Output as JSON

**Features:**
- Shows decay status and time since last reinforcement
- Supports multiple filter criteria
- Configurable result limit

### `content remove`

Remove content by various criteria.

**Options:**
- `--id` - Remove by content ID
- `--source` / `-s` - Remove by exact source match
- `--source-prefix` - Remove all with source prefix
- `--project` / `-p` - Remove all for a project
- `--all` - Remove all content
- `--force` / `-f` - Skip confirmation prompt

**Behavior:**
- Requires at least one removal criterion
- Shows confirmation prompt (unless `--force` used)
- Supports combining filters with AND logic
- Reports count of removed entries

### `content reinforce --source <source>`

Reset the decay timer for content to extend its lifetime.

**Options:**
- `--source` / `-s` - Source identifier (required)

## Temporal Decay System

Content entries use a temporal decay system to prioritize recently reinforced content:

| Status | Decay Factor | Description |
|--------|-------------|-------------|
| **Fresh** | >75% | Recently reinforced |
| **Good** | >50% | Still relevant |
| **Aging** | >25% | Getting old |
| **Decaying** | >10% | Needs attention |
| **Expiring** | ≤10% | Very old |

- Default half-life: 90 days
- Decay is exponential based on time since last reinforcement
- Use `reinforce` command to reset decay timer
- Search results are scored with decay adjustment

## MCP Tools

Content operations are also available as MCP tools for integration with MCP clients like Claude Desktop.

See [MCP Content Tools Implementation](implementation/MCP_CONTENT_TOOLS.md) for details.

### Available MCP Tools

1. **IndexContent** - Index markdown files from a directory
2. **AddContent** - Add content directly
3. **SearchContent** - Search with full-text search
4. **ListContent** - List content with filters
5. **RemoveContent** - Remove by criteria
6. **ReinforceContent** - Reset decay timer

### MCP Configuration

Add to your MCP client configuration (e.g., Claude Desktop):

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

## Advanced Usage

### Scripting with JSON Output

Use `--robot` flag for JSON output in scripts:

```bash
# Get all titles
agent-journal content list --robot | jq '.[] | .Title'

# Search and extract sources
agent-journal content search "api" --robot | jq '.[] | .Entry.Source'

# Count entries by project
agent-journal content list --robot | \
  jq 'group_by(.Project) | map({project: .[0].Project, count: length})'

# Find expired content
agent-journal content list --expired --robot | jq '.[] | .Source'
```

### Filtering Strategies

**By Location:**
```bash
# Documentation only
agent-journal content search "guide" --source-prefix "docs/"

# Specific subdirectory
agent-journal content list --source-prefix "docs/api/"

# External sources
agent-journal content list --source-prefix "https://example.com"
```

**By Tags:**
```bash
# Single tag
agent-journal content list --tags "tutorial"

# Multiple tags (OR logic - matches any)
agent-journal content search "database" --tags "guide,tutorial,reference"
```

**Combined Filters:**
```bash
# Project + source prefix
agent-journal content list --project "backend" --source-prefix "specs/"

# All filters
agent-journal content search "API" \
  --project "backend" \
  --source-prefix "docs/api/" \
  --tags "reference,guide"
```

### Content Maintenance

**Regular Reinforcement:**
```bash
# Reinforce important documents
agent-journal content reinforce --source "docs/architecture.md"
agent-journal content reinforce --source "docs/security.md"
```

**Cleanup Old Content:**
```bash
# Find expired content
agent-journal content list --expired

# Remove specific expired content
agent-journal content remove --source-prefix "old-docs/"

# Remove entire obsolete project
agent-journal content remove --project "archived-project" --force
```

## Database Location

Content is stored in: `~/.agent-journal/content.db`

The database uses:
- SQLite for storage
- FTS5 for full-text indexing
- Automatic triggers to keep FTS index synchronized
- Indexes on source, project, and last_reinforced_at for fast queries

## Architecture

Content indexing follows the repository pattern:

- **ContentEntry Model** - Represents indexed content with metadata
- **IContentRepository** - Repository interface for content operations
- **SqliteContentRepository** - SQLite implementation with FTS5
- **ContentCommand** - CLI command implementation
- **AgentJournalTools** - MCP tool implementations

For technical details, see [Content Implementation](implementation/CONTENT_IMPLEMENTATION.md).

## Best Practices

1. **Use Projects** - Organize content by project for easier filtering
2. **Tag Consistently** - Use consistent tags for better categorization
3. **Reinforce Important Content** - Keep critical documents fresh
4. **Regular Cleanup** - Remove obsolete content to improve search quality
5. **Use Glob Patterns** - Index only relevant files with `--filter`
6. **Leverage JSON Output** - Automate workflows with `--robot` flag

## Related Documentation

- [Content Quick Reference](quick-reference/CONTENT_QUICK_REF.md) - Command cheat sheet
- [Content Implementation](implementation/CONTENT_IMPLEMENTATION.md) - Technical details
- [MCP Content Tools](implementation/MCP_CONTENT_TOOLS.md) - MCP integration
- [Content Security Review](reviews/CONTENT_SECURITY_REVIEW.md) - Security considerations
- [CLI Usage Guide](CLI_USAGE.md) - General CLI documentation

## Troubleshooting

**Files not being indexed:**
- Check file path and glob pattern
- Ensure files are accessible
- Use `--rebuild` to force re-indexing

**Search not finding expected results:**
- Check query syntax
- Verify project/source-prefix filters
- Consider content may be expired (check with `--expired`)

**Memory issues during indexing:**
- Large files (>10MB) are automatically skipped
- Index directories in smaller batches
- Use more specific glob patterns

**Remove command not working:**
- Ensure at least one criterion is specified
- Check that content exists with `list` command
- Use `--force` to skip confirmation

## See Also

- [Content Filtering Guide](CONTENT_FILTERING_GUIDE.md) - Existing filtering documentation
- [Content Remove Reference](content-remove-reference.md) - Remove command reference
