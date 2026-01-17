# Agent Journal - Session Search, Knowledge Bank & Content Indexing

Search and retrieve past AI agent sessions, stored knowledge, and indexed content.

## When to Use

- Finding previous work: "have I done this before?"
- Context from past sessions about a topic
- How a problem was solved previously
- Recalling project-specific knowledge
- Storing learnings and conventions
- Indexing and searching documentation

## Quick Reference

### Search Sessions
```bash
agent-journal search "<query>" --mode hybrid --context 5
agent-journal search "<query>" --project <name> --agent claude-code
```

### Knowledge Bank
```bash
agent-journal remember "fact" --project <name> --tags "tag1,tag2"
agent-journal recall "query" --project <name>
agent-journal reinforce <id>
```

### Content Indexing
```bash
agent-journal content index ./docs --project <name>
agent-journal content search "query" --project <name>
agent-journal content add --source "id" --title "Title" --content "..."
```

### Export
```bash
agent-journal export <session-id> --format md
```

## MCP Integration

When running as MCP server, these tools are available:
- `SearchSessions` - Search with context window
- `Remember/Recall/Reinforce/Forget` - Knowledge management
- `IndexContent/AddContent/SearchContent` - Content indexing
- `Search` - Unified search across all sources

## Decay System

Knowledge decays over 90-day half-life. Reinforce important entries to keep them fresh.
