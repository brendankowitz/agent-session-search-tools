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
```

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
