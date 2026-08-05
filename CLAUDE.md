# CLAUDE.md - Agent Journal Instructions

This repository contains `agent-journal`, a .NET CLI tool for searching AI agent conversation history and managing a persistent knowledge bank.

## Quick Reference

```bash
# Search past sessions
agent-journal search "your query" --mode hybrid --max 10

# Search sessions + knowledge together
agent-journal search "query" --include-knowledge

# Search sessions + this repo's task journals
agent-journal search "query" --include-tasks

# Remember a convention or learning
agent-journal remember "Use async/await for DB operations" --tags "best-practice,async"

# Recall stored knowledge
agent-journal recall "database patterns" --mode hybrid

# Reinforce useful knowledge (reset decay)
agent-journal reinforce <id>

# Rebuild index after new sessions
agent-journal index --rebuild

# Export a session
agent-journal export <session-id> --format md

# Export only the last N messages of a session
agent-journal export <session-id> --last 20 --format md
```

## When to Use Agent Journal

Use this tool to:
- **Find previous solutions**: Search how similar problems were solved before
- **Recall context**: Find sessions related to a specific project or topic
- **Remember learnings**: Store conventions, patterns, and best practices
- **Reinforce knowledge**: Keep important facts from decaying
- **Learn patterns**: Discover common approaches used across sessions
- **Export documentation**: Convert useful sessions to markdown/HTML

## Search Examples

```bash
# Find error fixes
agent-journal search "fix NullReferenceException" --mode semantic

# Find implementation patterns
agent-journal search "implement caching" --mode hybrid

# Find project-specific sessions
agent-journal search "api endpoint" --project my-api

# Find Claude Code sessions only
agent-journal search "refactoring" --agent claude-code

# JSON output for parsing
agent-journal search "testing" --robot | jq '.[] | .sessionId'
```

## Search Modes

- **lexical**: Fast keyword search (BM25). Best for exact terms, file names, error messages.
- **semantic**: Embedding-based search. Best for concepts, "how to" queries, finding similar approaches.
- **hybrid**: Combines both using Reciprocal Rank Fusion. Best overall quality.

## Workflow Integration

Before starting work on a new feature or bug fix:
```bash
# Check if similar work was done before
agent-journal search "feature description" --mode semantic --max 5
```

After completing significant work:
```bash
# Re-index to include the current session
agent-journal index
```

## Configuration

Default paths are auto-detected. To customize:
```bash
agent-journal config show
agent-journal config set ClaudeProjectsPath /custom/path
```

## Knowledge Bank

Store and recall learnings with automatic decay:

```bash
# Store project-specific knowledge
agent-journal remember "API uses JWT with 24h expiry" --project my-api --tags "auth,security"

# Search knowledge
agent-journal recall "authentication" --tags "security"

# View knowledge bank
agent-journal knowledge list
agent-journal knowledge stats

# Knowledge decays over 90-day half-life - reinforce when useful
agent-journal reinforce <id>
```

## Task Journal

Track progress through a multi-task plan in a SQLite journal, so work survives context loss:

```bash
# Bind a journal to a plan file
agent-journal task init docs/plans/refactor.md

# After context loss: where was I?
agent-journal task next --robot

# Record progress as it happens
agent-journal task start 1
agent-journal task complete 1 --note "extracted the parser"
agent-journal task fix 1 --note "review found a leak"   # reopens task 1

# Hand a brief to a subagent, read its report back
agent-journal task show brief 1
agent-journal task show report 1 --out report.md

agent-journal task status

# Search the notes, briefs, and reports left behind
agent-journal search "leak in disposal" --include-tasks
```

Never hold plan progress only in conversation — record it with `task complete` as each task
finishes. See [docs/TASK_JOURNAL.md](docs/TASK_JOURNAL.md).

### Search scope differs by source

| Source | Flag | Scope |
|---|---|---|
| Sessions | (default) | All projects |
| Knowledge | `--include-knowledge` | All projects |
| Task journals | `--include-tasks` | **Current repository only** |

Task journals are indexed inside the repo (`.agent-journal/tasks/journals.db`), not in the global
index, so one repository's plans never surface in another's search results. `--include-tasks`
requires being inside a repository, and matches lexically regardless of `--mode`.

## MCP Server

For integration with Claude Desktop or other MCP clients:
```bash
agent-journal mcp
```

Add to `claude_desktop_config.json`:
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

## Build & Test

```bash
cd src/AgentJournal
dotnet build
dotnet run -- search "test query"
```
