# Knowledge Search - Quick Reference

## Command Syntax

```bash
agent-journal search <query> [options]
```

## New Option

```bash
--include-knowledge, -k    Include knowledge entries in results (default: false)
```

## Usage Examples

### Basic Search (Sessions Only)
```bash
agent-journal search "authentication"
```

### Search Sessions + Knowledge
```bash
agent-journal search "authentication" --include-knowledge
agent-journal search "authentication" -k
```

### Semantic Search with Knowledge
```bash
agent-journal search "JWT tokens" -k -m semantic
```

### Hybrid Search
```bash
agent-journal search "database migration" -k -m hybrid
```

### Filter by Project
```bash
agent-journal search "API design" -k -p "api-service"
```

### Limit Results
```bash
agent-journal search "testing" -k --max 10
```

### JSON Output (for scripting)
```bash
agent-journal search "react hooks" -k --robot
```

### Extract Only Knowledge Results with jq
```bash
agent-journal search "security" -k --robot | jq '.[] | select(.type == "knowledge")'
```

### Extract Only Session Results with jq
```bash
agent-journal search "bugs" -k --robot | jq '.[] | select(.type == "session")'
```

## Output Format

### Session Result
```
[1] Session: def456
    Agent: claude-code
    Score: 0.90
    Started: 2026-01-15 10:30:00
    Preview: Implemented authentication...
```

### Knowledge Result
```
[2] Knowledge: abc123
    Score: 0.85 (decay: 0.92 ██████████░)
    Tags: auth, security
    Created: 2026-01-10 14:20:00
    Last reinforced: 2026-01-14 09:15:00 (3x)
    Content: Use JWT with 24h expiry...
```

## Decay Indicator

| Decay Factor | Visual | Status | Meaning |
|--------------|--------|--------|---------|
| > 0.75 | `██████████░` | Fresh | Recently used |
| > 0.50 | `███████░░░` | Good | Still relevant |
| > 0.25 | `████░░░░░░` | Aging | Consider review |
| < 0.25 | `██░░░░░░░░ ⚠️` | Decaying | Needs reinforcement |

## JSON Output Schema

### Session
```json
{
  "type": "session",
  "sessionId": "...",
  "agentType": "...",
  "projectPath": "...",
  "startedAt": "...",
  "messageCount": 0,
  "score": 0.90,
  "highlight": "...",
  "matchingMessages": [...]
}
```

### Knowledge
```json
{
  "type": "knowledge",
  "id": "...",
  "content": "...",
  "tags": ["..."],
  "project": "...",
  "source": "...",
  "createdAt": "...",
  "lastReinforcedAt": "...",
  "reinforcementCount": 0,
  "score": 0.85,
  "decayFactor": 0.92,
  "highlight": "..."
}
```

## Combined with Other Commands

### Search and Add to Knowledge
```bash
# Find relevant sessions
agent-journal search "error handling pattern" -m semantic

# Extract knowledge from a session
agent-journal kb add-from-session <session-id>

# Search again with knowledge
agent-journal search "error handling pattern" -k -m semantic
```

### Search and Reinforce
```bash
# Search knowledge
agent-journal search "deployment steps" -k --robot

# Reinforce relevant entries
agent-journal kb reinforce <knowledge-id>
```

## Tips

1. **Use semantic mode** (`-m semantic`) for concept-based searches
2. **Filter by project** (`-p`) to scope results
3. **Enable knowledge** (`-k`) when looking for facts/patterns
4. **Watch decay indicators** to identify knowledge needing reinforcement
5. **Use JSON mode** (`--robot`) for automation and scripting

## See Also

- `agent-journal kb --help` - Knowledge bank commands
- `agent-journal search --help` - Full search options
- `KNOWLEDGE_SEARCH_INTEGRATION.md` - Detailed documentation
