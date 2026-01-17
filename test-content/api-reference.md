# API Reference

## Content Commands

### content index

Index markdown files from a directory.

**Syntax:**
```
agent-journal content index <path> [options]
```

**Options:**
- `--filter`: Glob pattern for file matching (default: *.md)
- `--project`: Project name to associate with content
- `--recursive`: Scan subdirectories recursively (default: true)
- `--rebuild`: Force re-indexing of unchanged files

### content add

Add content directly via command line or stdin.

**Syntax:**
```
agent-journal content add --source <id> --title <title> [options]
```

**Options:**
- `--source`: Source identifier (required)
- `--title`: Content title (required)
- `--content`: Content text (reads from stdin if not provided)
- `--project`: Project name
- `--tags`: Comma-separated tags

### content search

Search indexed content with decay-adjusted scores.

**Syntax:**
```
agent-journal content search <query> [options]
```

**Options:**
- `--max`: Maximum results (default: 10)
- `--project`: Filter by project
- `--robot`: JSON output for automation
