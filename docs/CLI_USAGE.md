# Agent Journal CLI Usage Guide

## Overview

Agent Journal is a CLI tool that indexes, searches, and exports AI agent conversation sessions from Claude Code and Copilot CLI. It enables agents to learn from past sessions and find relevant context.

## Installation

```bash
# Install as global tool
dotnet tool install -g agent-journal

# Or run from source
cd src/AgentJournal && dotnet run -- <command>
```

## Core Commands

### 1. Search (`agent-journal search`)

Find relevant sessions using different search strategies.

**Syntax:**
```bash
agent-journal search "<query>" [options]
```

**Options:**
| Option | Alias | Default | Description |
|--------|-------|---------|-------------|
| `--mode` | `-m` | lexical | Search mode: `lexical`, `semantic`, `hybrid` |
| `--max` | `-n` | 20 | Maximum results to return |
| `--agent` | `-a` | all | Filter: `claude-code`, `copilot-cli` |
| `--project` | `-p` | - | Filter by project path substring |
| `--context` | `-c` | 3 | Context messages around matches |
| `--robot` | `-r` | false | Output JSON for scripting |

**Examples:**
```bash
# Semantic search for concepts
agent-journal search "implement retry logic" --mode semantic

# Hybrid search (best quality)
agent-journal search "database connection pooling" --mode hybrid --max 10

# Filter by project
agent-journal search "authentication" --project my-api

# JSON output
agent-journal search "error handling" --robot | jq '.[0].sessionId'
```

### 2. Index (`agent-journal index`)

Build or update the search index from session files.

**Syntax:**
```bash
agent-journal index [options]
```

**Options:**
| Option | Alias | Description |
|--------|-------|-------------|
| `--rebuild` | `-r` | Clear and rebuild entire index |
| `--agent` | `-a` | Index specific agent type only |
| `--watch` | `-w` | Continuously watch for new sessions |

**Examples:**
```bash
# Full rebuild
agent-journal index --rebuild

# Index only Claude sessions
agent-journal index --agent claude-code

# Watch mode for continuous indexing
agent-journal index --watch
```

### 3. Export (`agent-journal export`)

Export a session to file.

**Syntax:**
```bash
agent-journal export <session-id> [options]
```

**Options:**
| Option | Alias | Default | Description |
|--------|-------|---------|-------------|
| `--format` | `-f` | html | Format: `html`, `md`, `json` |
| `--output` | `-o` | auto | Output file path |
| `--stdout` | - | false | Write to stdout |

**Examples:**
```bash
# Export to markdown
agent-journal export abc123 --format md

# Export to specific path
agent-journal export abc123 --output ./docs/session.html

# Pipe to another command
agent-journal export abc123 --format json --stdout | jq '.messages'
```

### 4. Config (`agent-journal config`)

View and modify configuration.

```bash
# Show all settings
agent-journal config show

# Set a value
agent-journal config set DefaultSearchMode hybrid
agent-journal config set ClaudeProjectsPath /path/to/projects
```

### 5. Models (`agent-journal models`)

Manage embedding models for semantic search.

```bash
# List installed models
agent-journal models list

# Shows: model name, status, size, execution provider (CPU/GPU)
```

### 6. Content (`agent-journal content`)

Index and search arbitrary markdown content (documentation, notes, etc.).

**Subcommands:**
```bash
# Index markdown files
agent-journal content index <path> --project myproject

# Add content directly
agent-journal content add --source "note-1" --title "Title" --content "Text"

# Search content
agent-journal content search "query" --project myproject

# List content
agent-journal content list --project myproject

# Remove content
agent-journal content remove --source "file.md"

# Reinforce content (reset decay timer)
agent-journal content reinforce --source "file.md"
```

**See:** [Content Indexing User Guide](CONTENT_INDEXING.md) for complete documentation.

## Search Mode Comparison

| Mode | Use Case | Speed | Notes |
|------|----------|-------|-------|
| `lexical` | Exact terms, file names, error codes | ⚡ Fast | BM25 ranking |
| `semantic` | Concepts, "how to" queries | 🔄 Medium | Requires model |
| `hybrid` | General queries, best recall | 🔄 Medium | RRF fusion |

## Data Locations

```
~/.agent-journal/
├── config.json          # Configuration
├── agent-journal.db     # SQLite metadata
├── lucene-index/        # Full-text search index
├── vector-index/        # Semantic embeddings
│   ├── index.ajvi       # Vector data
│   ├── mappings.json    # Message→Session map
│   └── sessions.json    # Session metadata
└── models/
    └── minilm/          # Embedding model (23MB)
```

## Typical Workflows

### Before Starting New Work
```bash
# Check for similar past work
agent-journal search "feature you're about to implement" --mode semantic --max 5
```

### After Completing Work
```bash
# Re-index to include current session
agent-journal index
```

### Creating Documentation
```bash
# Find and export relevant session
agent-journal search "implemented feature X" --mode hybrid --max 1
agent-journal export <session-id> --format md --output docs/feature-x.md
```

### Scripting / Automation
```bash
# Get session IDs matching criteria
SESSIONS=$(agent-journal search "topic" --robot | jq -r '.[].sessionId')
for id in $SESSIONS; do
  agent-journal export $id --format json --stdout >> all-sessions.jsonl
done
```
