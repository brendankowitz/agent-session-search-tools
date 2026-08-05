# Agent Journal - AI Agent Session Search Tool

Agent Journal (`agent-journal`) is a CLI tool for indexing, searching, and exporting AI agent conversation sessions from Claude Code and Copilot CLI.

## Installation

The tool is installed as a .NET global tool:
```bash
dotnet tool install -g agent-journal
```

## Commands

### Search Sessions

Search indexed sessions using lexical, semantic, or hybrid search:

```bash
# Lexical search (keyword-based, fast)
agent-journal search "error handling" --mode lexical

# Semantic search (meaning-based, uses embeddings)
agent-journal search "how to fix bugs" --mode semantic

# Hybrid search (combines both, best quality)
agent-journal search "performance optimization" --mode hybrid

# Limit results
agent-journal search "testing" --max 5

# Filter by agent type
agent-journal search "debugging" --agent claude-code
agent-journal search "git commands" --agent copilot-cli

# Filter by project
agent-journal search "api design" --project my-project

# JSON output for scripting
agent-journal search "database" --robot

# Widen the search beyond sessions
agent-journal search "retry policy" --include-knowledge   # + knowledge bank (all projects)
agent-journal search "retry policy" --include-tasks       # + task journals (this repo only)
```

### Index Sessions

Index sessions from configured agent paths:

```bash
# Index all sessions
agent-journal index

# Rebuild index from scratch
agent-journal index --rebuild

# Index specific agent type only
agent-journal index --agent claude-code
agent-journal index --agent copilot-cli

# Watch for new sessions continuously
agent-journal index --watch
```

### Export Sessions

Export a session to various formats:

```bash
# Export to HTML (default)
agent-journal export <session-id>

# Export to Markdown
agent-journal export <session-id> --format md

# Export to JSON
agent-journal export <session-id> --format json

# Specify output file
agent-journal export <session-id> --output ./exports/session.html

# Output to stdout
agent-journal export <session-id> --stdout

# Export only the last N messages (useful for long sessions)
agent-journal export <session-id> --last 20
```

### Task Journal

Track progress through a multi-task plan in a SQLite journal, so an agent can resume after its
conversation context is compacted or lost:

```bash
# Bind a journal to a plan file (task count from '## Task N' headings)
agent-journal task init docs/plans/refactor.md --name refactor

# After context loss: which task is next?
agent-journal task next --robot

# Record progress as it happens
agent-journal task start 1
agent-journal task complete 1 --note "extracted the parser"

# Review found a problem in an already-completed task
agent-journal task fix 1 --note "leak in the disposal path"

# Briefs and reports are stored in the journal, not pasted through context
agent-journal task show brief 1
agent-journal task show report 1 --out report.md

agent-journal task status
agent-journal task list

# Search the notes, briefs, and reports left behind
agent-journal search "disposal path" --include-tasks
```

Record each task with `task complete` as it finishes rather than holding progress in conversation.
Every subcommand supports `--robot` for JSON. Exit codes: 0 success, 1 failure, 2 not found.

The journal lives in a SQLite database keyed by repository, so several agents working the same
plan share one view of progress without clobbering each other.

Task notes, briefs, and reports are indexed for search inside that same database. Unlike session and
knowledge search, which span every project, `--include-tasks` is scoped to the current repository —
it requires being inside one, and matches lexically regardless of `--mode`.

See [docs/TASK_JOURNAL.md](../docs/TASK_JOURNAL.md) for the full loop.

### Configuration

View and modify configuration:

```bash
# Show current configuration
agent-journal config show

# Set a configuration value
agent-journal config set ClaudeProjectsPath /path/to/claude/projects
agent-journal config set CopilotSessionsPath /path/to/copilot/sessions
agent-journal config set DefaultSearchMode hybrid
```

### Model Management

Manage embedding models for semantic search:

```bash
# List installed models
agent-journal models list

# Download a model
agent-journal models download minilm

# Remove a model
agent-journal models remove minilm
```

## Search Modes

| Mode | Description | Speed | Quality |
|------|-------------|-------|---------|
| `lexical` | Keyword/BM25 search | Fast | Good for exact terms |
| `semantic` | Vector embedding search | Medium | Good for meaning/concepts |
| `hybrid` | Combines both with RRF | Medium | Best overall quality |

## Use Cases

### Find how a problem was solved before
```bash
agent-journal search "fix typescript compilation error" --mode semantic
```

### Find sessions about a specific topic
```bash
agent-journal search "database migration" --mode hybrid --max 10
```

### Export a useful session for documentation
```bash
agent-journal search "implemented authentication" --mode semantic --max 1
# Then export the session ID from results
agent-journal export <session-id> --format md --output docs/auth-implementation.md
```

### Find all sessions for a project
```bash
agent-journal search "*" --project my-app --max 50
```

## Data Locations

- **Config**: `~/.agent-journal/config.json`
- **Database**: `~/.agent-journal/agent-journal.db`
- **Lucene Index**: `~/.agent-journal/lucene-index/`
- **Vector Index**: `~/.agent-journal/vector-index/`
- **Models**: `~/.agent-journal/models/`

## Tips

1. **First run**: Execute `agent-journal index` to build the initial index
2. **Semantic search**: Requires the MiniLM model (auto-downloaded on first use)
3. **GPU acceleration**: DirectML is auto-detected on Windows for faster embeddings
4. **Rebuild index**: Use `--rebuild` if sessions seem missing or corrupted
