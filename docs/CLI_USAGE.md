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
| `--last` | `-n` | all | Export only the last N messages |

**Examples:**
```bash
# Export to markdown
agent-journal export abc123 --format md

# Export to specific path
agent-journal export abc123 --output ./docs/session.html

# Only the tail of a long session, for feeding back into a context window
agent-journal export abc123 --format md --last 20 --stdout

# Pipe to another command
agent-journal export abc123 --format json --stdout | jq '.messages'
```

Session metadata (including the start time) always describes the whole session; `--last` narrows
only the message list, so the export still identifies the session it came from.

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

### 7. Knowledge (`remember` / `recall` / `forget` / `reinforce`)

A persistent knowledge bank for conventions and learnings, separate from session history.

```bash
# Store a learning
agent-journal remember "Use async/await for DB operations" --tags "best-practice,async"

# Recall it later
agent-journal recall "database patterns"

# Search sessions and knowledge together
agent-journal search "retry policy" --include-knowledge

# Knowledge decays on a 90-day half-life; reinforce what stays useful
agent-journal reinforce <id>

# Remove an entry
agent-journal forget <id>
```

`--include-knowledge` is served by an FTS5 index, so it only applies to lexical matching. Combining
it with `--mode semantic` or `--mode hybrid` still searches sessions in that mode, but the knowledge
portion remains lexical.

### 8. Task Journal (`agent-journal task`)

Durable progress tracking for a multi-task plan, so an agent can resume after its context is
compacted or lost. State lives in a SQLite database under the repository, not in the conversation.

```bash
# Bind a journal to a plan file (task count read from '## Task N' headings)
agent-journal task init docs/plans/refactor.md --name refactor

# After context loss: what should I do next?
agent-journal task next --robot

# Record progress as it happens
agent-journal task start 1
agent-journal task complete 1 --note "extracted the parser"

# Reopen a completed task when review finds a problem
agent-journal task fix 1 --note "leak in the disposal path"

# Briefs and reports are stored in the journal rather than pasted through context
agent-journal task show brief 1
agent-journal task show report 1 --out report.md

agent-journal task status
agent-journal task list

# Search the journal's notes, briefs, and reports
agent-journal search "disposal path" --include-tasks
```

**See:** [Task Journal](TASK_JOURNAL.md) for the full loop and schema.

### 9. Searching Across Sources

`search` queries session history by default. Two flags widen it:

```bash
agent-journal search "retry policy" --include-knowledge   # + knowledge bank
agent-journal search "retry policy" --include-tasks       # + task journals (this repo)
```

Scope differs by source, and this matters when interpreting results:

| Source | Scope | Index |
|--------|-------|-------|
| Sessions | All projects (user-global) | Lucene |
| Knowledge | All projects (user-global) | SQLite FTS5 |
| Task journals | **Current repository only** | SQLite FTS5, inside the repo |

Task journals stay repo-local on purpose: they live in `<repo>/.agent-journal/tasks/journals.db`
alongside the work they describe, so they travel with the checkout and never leak one repository's
plan into another's results. `--include-tasks` therefore requires you to be inside a repository.

Because knowledge and task journals are served by FTS5, those portions are always lexical, even
under `--mode semantic` or `--mode hybrid`. The session portion still honours the requested mode.

Results from different sources are merged with Reciprocal Rank Fusion rather than by raw score.
Lucene and FTS5 produce scores on incomparable scales, so sorting on the raw number buries exact
matches from the smaller corpus. Each result still reports its own native `score` in `--robot`
output; fusion governs ordering only.

## Exit Codes

Every command reports failure through its exit code, so scripts and agents can branch on it.

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Failure (bad arguments, unusable input, unexpected error) |
| 2 | Requested item not found |
| 3 | Completed, but some items failed to process |

Diagnostics and errors go to stderr, so `--robot` output on stdout stays valid JSON.

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

Task journals are stored per repository rather than per user, so switching branches or repos does
not blend unrelated plans together:

```
<repo>/.agent-journal/tasks/
└── journals.db          # All task journals for this checkout
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
