# Knowledge Bank CLI - Quick Reference

## Quick Command Reference

### Basic Operations

```bash
# Store knowledge
agent-journal remember "Your knowledge here" --tags tag1,tag2 --project my-app

# Search knowledge
agent-journal recall "search query"

# Delete knowledge
agent-journal forget <entry-id>

# Reinforce knowledge (reset decay)
agent-journal reinforce <entry-id>
```

### List and Stats

```bash
# List all knowledge
agent-journal knowledge list

# List with filters
agent-journal knowledge list --project my-app --tags security
agent-journal knowledge list --decaying
agent-journal knowledge list --expiring

# Show statistics
agent-journal knowledge stats
```

### Backup and Restore

```bash
# Export to file
agent-journal knowledge export --output backup.json

# Import from file
agent-journal knowledge import backup.json
```

### Maintenance

```bash
# Remove expired entries
agent-journal knowledge prune

# Clear all (requires confirmation)
agent-journal knowledge clear --confirm
```

## Command Details

### remember
Store knowledge in the knowledge bank.

**Options:**
- `content` (required): Knowledge to remember
- `--tags, -t`: Tags (comma-separated)
- `--project, -p`: Project name
- `--source, -s`: Source reference

**Example:**
```bash
agent-journal remember "Use async/await for I/O" \
  --tags best-practice,async \
  --project my-api \
  --source "Microsoft Docs"
```

### recall
Search the knowledge bank.

**Options:**
- `query` (required): Search text
- `--tags, -t`: Filter by tags
- `--project, -p`: Filter by project
- `--mode, -m`: keyword|semantic|hybrid [default: hybrid]
- `--limit, -n`: Max results [default: 10]
- `--json`: JSON output

**Example:**
```bash
agent-journal recall "authentication" \
  --tags security,auth \
  --mode semantic \
  --limit 20
```

### forget
Delete knowledge entries.

**Options:**
- `id`: Entry ID (no confirmation needed)
- `--match, -m`: Delete by search query (needs --confirm)
- `--project, -p`: Delete by project (needs --confirm)
- `--all`: Delete all (needs --confirm)
- `--confirm, -y`: Confirmation flag

**Example:**
```bash
# Single entry
agent-journal forget abc123

# Batch deletion
agent-journal forget --match "old pattern" --confirm
agent-journal forget --project legacy-app --confirm
```

### reinforce
Reset decay timer to keep knowledge fresh.

**Options:**
- `ids`: Entry IDs (multiple)
- `--match, -m`: Reinforce by search query
- `--project, -p`: Filter by project
- `--decaying`: Reinforce all with decay < 0.5
- `--expiring`: Reinforce all with decay < 0.1

**Example:**
```bash
# Specific entries
agent-journal reinforce abc123 def456

# By query
agent-journal reinforce --match "important"

# All decaying
agent-journal reinforce --decaying
```

### knowledge list
List knowledge entries.

**Options:**
- `--project, -p`: Filter by project
- `--tags, -t`: Filter by tags
- `--decaying`: Show only decaying (decay < 0.5)
- `--expiring`: Show only expiring (decay < 0.1)
- `--limit, -n`: Max entries [default: 50]

**Example:**
```bash
agent-journal knowledge list --project my-app --limit 100
agent-journal knowledge list --decaying
```

### knowledge stats
Show knowledge bank statistics.

**Example:**
```bash
agent-journal knowledge stats
```

### knowledge export
Export to file.

**Options:**
- `--format, -f`: Format [default: json]
- `--output, -o`: Output path

**Example:**
```bash
agent-journal knowledge export --output backup.json
agent-journal knowledge export > backup.json
```

### knowledge import
Import from file.

**Example:**
```bash
agent-journal knowledge import backup.json
```

### knowledge prune
Remove expired entries.

**Options:**
- `--threshold, -t`: Decay threshold [default: 0.05]

**Example:**
```bash
agent-journal knowledge prune
agent-journal knowledge prune --threshold 0.1
```

### knowledge clear
Clear all entries (requires confirmation).

**Example:**
```bash
agent-journal knowledge clear --confirm
```

## Decay System

### Decay Formula
```
decay_factor = 0.5^(days_since_reinforced / 90)
```

### Decay Levels
- **Fresh** (>0.75): 0-26 days
- **Good** (>0.50): 27-90 days
- **Aging** (>0.25): 91-180 days
- **Decaying** (>0.10): 181-299 days
- **Expiring** (≤0.10): 300+ days

### Visual Indicators
```
██████████  = 1.00 (Fresh)
█████░░░░░  = 0.50 (Decaying) ⚠️
█░░░░░░░░░  = 0.10 (Expiring) ⚠️
```

## Workflow Examples

### Daily Knowledge Capture
```bash
# Capture learnings
agent-journal remember "Team prefers React Query for data fetching" \
  --tags convention,react --project webapp

agent-journal remember "API uses JWT with 24h expiry" \
  --tags auth,api --project backend
```

### Weekly Maintenance
```bash
# Review decaying knowledge
agent-journal knowledge list --decaying

# Reinforce important items
agent-journal reinforce --match "critical"

# Review stats
agent-journal knowledge stats
```

### Monthly Cleanup
```bash
# Export backup
agent-journal knowledge export --output monthly-backup-$(date +%Y-%m).json

# Review expiring entries
agent-journal knowledge list --expiring

# Prune very old entries
agent-journal knowledge prune --threshold 0.05

# Check results
agent-journal knowledge stats
```

### Project Handoff
```bash
# Export project knowledge
agent-journal recall "" --project old-project --json > project-knowledge.json

# Review and clean
agent-journal knowledge list --project old-project
agent-journal forget --project old-project --confirm
```

## Search Modes

### Keyword (Lexical)
Fast, exact matching, good for code terms.
```bash
agent-journal recall "useState" --mode keyword
```

### Semantic
Meaning-based, finds related concepts.
```bash
agent-journal recall "state management" --mode semantic
```

### Hybrid (Default)
Best of both, balanced precision/recall.
```bash
agent-journal recall "authentication patterns" --mode hybrid
```

## Tips

1. **Tag Consistently**: Use standard tags across projects
2. **Reinforce Often**: Reset decay on actively used knowledge
3. **Regular Pruning**: Remove expired knowledge monthly
4. **Export Regularly**: Backup before major cleanup
5. **Use Projects**: Organize by codebase/team
6. **Source Attribution**: Track where knowledge came from
7. **Batch Operations**: Use --match for bulk updates

## Common Patterns

### Capture Meeting Notes
```bash
agent-journal remember "Team decided on microservices architecture" \
  --tags architecture,decision \
  --source "Team Meeting 2024-01-15"
```

### Document Conventions
```bash
agent-journal remember "Use kebab-case for file names" \
  --tags convention,naming \
  --project frontend
```

### Save Troubleshooting Solutions
```bash
agent-journal remember "Fixed CORS by adding credentials: true" \
  --tags troubleshooting,cors,api \
  --source "Stack Overflow #12345"
```

### Track Dependencies
```bash
agent-journal remember "Using React 18.2, React Query 4.x" \
  --tags dependencies,versions \
  --project webapp
```

## Automation Examples

### Daily Reminder Script
```bash
#!/bin/bash
# Show expiring knowledge daily
agent-journal knowledge list --expiring | mail -s "Expiring Knowledge" you@example.com
```

### Pre-commit Hook
```bash
#!/bin/bash
# Reinforce project knowledge on commit
agent-journal reinforce --project $(basename $(git rev-parse --show-toplevel))
```

### CI/CD Integration
```bash
#!/bin/bash
# Export knowledge for team documentation
agent-journal knowledge export --output docs/team-knowledge.json
```

## JSON Output Format

```json
[
  {
    "id": "abc123",
    "score": 0.85,
    "decayFactor": 0.95,
    "content": "Knowledge content here",
    "tags": ["tag1", "tag2"],
    "project": "my-app",
    "source": "https://example.com",
    "createdAt": "2024-01-15T10:30:00Z",
    "lastReinforcedAt": "2024-01-20T14:00:00Z",
    "reinforcementCount": 3,
    "daysSinceReinforcement": 5.2
  }
]
```

## Getting Help

```bash
# General help
agent-journal --help

# Command-specific help
agent-journal remember --help
agent-journal recall --help
agent-journal knowledge --help
agent-journal knowledge list --help
```

## Exit Codes
- `0`: Success
- `1`: General error (check error message)

## Data Location
- Database: `{DataPath}/knowledge.db`
- Default: `~/.agent-journal/knowledge.db`

## See Also
- `KNOWLEDGE_BANK_CLI_IMPLEMENTATION.md` - Full implementation details
- `KNOWLEDGE_BANK_CORE_COMPLETE.md` - Core architecture
- `KNOWLEDGE_BANK_QUICK_REFERENCE.md` - Core API reference
