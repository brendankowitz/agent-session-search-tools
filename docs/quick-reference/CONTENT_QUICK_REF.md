# Content Indexing - Quick Reference

## CLI Commands

### Index Directory
```bash
agent-journal content index <path> [options]

Options:
  --filter <pattern>    Glob pattern (default: *.md)
  --project <name>      Project name
  --recursive           Scan subdirectories (default: true)
  --rebuild             Force re-index unchanged files
```

### Add Content
```bash
agent-journal content add --source <id> --title <title> [options]

Options:
  --source <id>         Source identifier (required)
  --title <title>       Content title (required)
  --content <text>      Content (or use stdin)
  --project <name>      Project name
  --tags <tag1,tag2>    Comma-separated tags
```

### Search
```bash
agent-journal content search <query> [options]

Options:
  --max <n>             Max results (default: 10)
  --project <name>      Filter by project
  --source-prefix <prefix>  Filter by source path
  --tags <tags>         Filter by tags (comma-separated)
  --robot               JSON output
```

### List
```bash
agent-journal content list [options]

Options:
  --project <name>      Filter by project
  --source-prefix <prefix>  Filter by source path
  --tags <tags>         Filter by tags (comma-separated)
  --robot               JSON output
  --expired             Show only expired
  --limit <n>           Max entries (default: 50)
```

### Remove
```bash
agent-journal content remove [options]

Options:
  --id <id>             Remove by content ID
  --source <source>     Remove by exact source
  --source-prefix <prefix>  Remove by source prefix
  --project <name>      Remove by project
  --all                 Remove all content
  --force               Skip confirmation
```

### Reinforce
```bash
agent-journal content reinforce --source <id>
```

## Usage Examples

### Basic Operations
```bash
# Index documentation
agent-journal content index ./docs --project "my-docs"

# Add note with tags
agent-journal content add \
  --source "note-1" \
  --title "Important Note" \
  --content "Remember this" \
  --tags "urgent,todo"

# Search and get top 5 results
agent-journal content search "database" --max 5

# List by project
agent-journal content list --project "my-docs"
```

### Advanced Filtering
```bash
# Search in specific directory
agent-journal content search "API" --source-prefix "docs/api/"

# List with multiple filters
agent-journal content list \
  --project "backend" \
  --source-prefix "docs/" \
  --tags "api,reference"

# Find expired content
agent-journal content list --expired --robot
```

### Bulk Operations
```bash
# Remove by directory
agent-journal content remove --source-prefix "docs/deprecated/"

# Remove by project (with confirmation)
agent-journal content remove --project "old-project"

# Force remove without confirmation
agent-journal content remove --project "old-project" --force
```

### JSON Output for Scripting
```bash
# Get all titles
agent-journal content list --robot | jq '.[] | .Title'

# Search and extract sources
agent-journal content search "api" --robot | jq '.[] | .Entry.Source'

# Count entries by project
agent-journal content list --robot | \
  jq 'group_by(.Project) | map({project: .[0].Project, count: length})'
```

## Decay System

Content relevance decays over time (90-day half-life):

| Status | Factor | Age |
|--------|--------|-----|
| **Fresh** | >75% | < 38 days |
| **Good** | >50% | < 90 days |
| **Aging** | >25% | < 180 days |
| **Decaying** | >10% | < 299 days |
| **Expiring** | ≤10% | > 299 days |

Use `reinforce` to reset the decay timer.

## MCP Tools

All CLI commands are available as MCP tools:
- `IndexContent`
- `AddContent`
- `SearchContent`
- `ListContent`
- `RemoveContent`
- `ReinforceContent`

See [MCP Content Tools](../implementation/MCP_CONTENT_TOOLS.md) for details.

## Database Location

Content stored in: `~/.agent-journal/content.db`

## Related Documentation

- [Content Indexing User Guide](../CONTENT_INDEXING.md) - Full documentation
- [Content Implementation](../implementation/CONTENT_IMPLEMENTATION.md) - Technical details
- [Content Security Review](../reviews/CONTENT_SECURITY_REVIEW.md) - Security considerations
