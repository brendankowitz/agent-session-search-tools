# Investigation: Agent Knowledge Bank

**Feature**: agent-session-search-tool  
**Status**: Investigation  
**Created**: 2026-01-17

## Summary

Investigate adding a persistent knowledge bank to agent-journal, enabling agents to store facts, learnings, and notes that persist across sessions and are searchable alongside session history.

## Problem Statement

Currently, agents can search past sessions but cannot:
- Store discrete facts or learnings for later retrieval
- Remember decisions or preferences across sessions
- Build a persistent knowledge base over time
- Share knowledge between different agents/projects

## Vision

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Agent Journal Search                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   Sessions (conversations)     +     Knowledge Bank (facts)         │
│   ────────────────────────          ─────────────────────           │
│   • Full conversation history       • Discrete facts/learnings      │
│   • Indexed automatically           • Explicitly stored by agent    │
│   • Immutable                        • Mutable (update/delete)       │
│   • Large, contextual               • Small, atomic                  │
│                                                                      │
│                    ↓ Unified Search ↓                               │
│                                                                      │
│   "How do I handle auth?"  →  Sessions + Knowledge entries          │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Use Cases

### 1. Store Project Conventions
```bash
agent-journal remember "This project uses tabs, not spaces"
agent-journal remember "API endpoints follow REST conventions with /api/v1 prefix"
agent-journal remember "Use pytest for testing, not unittest"
```

### 2. Store Decisions
```bash
agent-journal remember "Chose PostgreSQL over MySQL for JSON support" --tag decision
agent-journal remember "Authentication uses JWT with 24h expiry" --tag architecture
```

### 3. Store Learnings
```bash
agent-journal remember "User prefers concise responses" --tag preference
agent-journal remember "Build command is 'npm run build:prod'" --tag build
```

### 4. Query Knowledge
```bash
# Search knowledge only
agent-journal recall "authentication"

# Search everything (sessions + knowledge)
agent-journal search "authentication" --include-knowledge
```

### 5. Agent Auto-Memory (MCP)
```
Agent: "I'll remember that you prefer TypeScript over JavaScript"
→ MCP tool call: remember("User prefers TypeScript over JavaScript", tags=["preference"])
```

### 6. Reinforce Important Knowledge
```bash
# Agent finds knowledge useful and reinforces it
agent-journal reinforce abc123

# Or via MCP when agent uses the knowledge
→ MCP tool call: reinforce("abc123")  # Resets decay timer
```

---

## Temporal Decay Model

Knowledge entries decay over time to automatically expire stale information. This mimics how human memory works - unused knowledge fades, while frequently accessed knowledge stays fresh.

### Decay Formula

```
effective_score = base_score × decay_factor
decay_factor = 0.5 ^ (days_since_reinforced / half_life)
```

Where:
- **half_life** = 90 days (configurable)
- **days_since_reinforced** = now - last_reinforced_at

### Example Decay Over Time

| Days Since Reinforced | Decay Factor | Effective Score (if base=1.0) |
|-----------------------|--------------|-------------------------------|
| 0 | 1.00 | 1.00 |
| 30 | 0.79 | 0.79 |
| 60 | 0.63 | 0.63 |
| 90 | 0.50 | 0.50 |
| 180 | 0.25 | 0.25 |
| 270 | 0.13 | 0.13 |
| 365 | 0.06 | 0.06 |

### Reinforcement

Knowledge is reinforced (decay timer reset) when:
1. **Explicit**: Agent calls `reinforce` tool/command
2. **Implicit**: Knowledge appears in search results and agent uses it
3. **Manual**: User runs `agent-journal reinforce <id>`

### Expiration Threshold

Entries with `decay_factor < 0.05` (roughly 1 year without reinforcement) can be:
- Automatically archived (moved to cold storage)
- Flagged for review
- Auto-deleted (configurable)

### Implementation

```csharp
public record KnowledgeEntry(
    ...
    DateTime LastReinforcedAt,      // When decay timer was last reset
    int ReinforcementCount          // How many times reinforced (for analytics)
);

public static class DecayCalculator
{
    private const double HalfLifeDays = 90.0;
    
    public static double CalculateDecayFactor(DateTime lastReinforced)
    {
        var daysSinceReinforced = (DateTime.UtcNow - lastReinforced).TotalDays;
        return Math.Pow(0.5, daysSinceReinforced / HalfLifeDays);
    }
    
    public static double ApplyDecay(double baseScore, DateTime lastReinforced)
    {
        return baseScore * CalculateDecayFactor(lastReinforced);
    }
}
```

---

## Data Model

### Knowledge Entry

```csharp
public record KnowledgeEntry(
    string Id,                      // GUID
    string Content,                 // The fact/knowledge text
    string? Source,                 // Where it came from (session ID, user input, etc.)
    string? Project,                // Project scope (null = global)
    IReadOnlyList<string> Tags,     // Categorization tags
    DateTime CreatedAt,
    DateTime? UpdatedAt,
    DateTime LastReinforcedAt,      // For decay calculation
    int ReinforcementCount,         // Times reinforced
    string? AgentType               // Which agent stored it
);
```

### Storage Schema (SQLite)

```sql
CREATE TABLE knowledge (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    source TEXT,
    project TEXT,
    tags TEXT,                      -- JSON array
    agent_type TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT,
    last_reinforced_at TEXT NOT NULL,  -- For decay
    reinforcement_count INTEGER DEFAULT 0,
    embedding BLOB                  -- Optional: pre-computed vector
);

CREATE INDEX idx_knowledge_project ON knowledge(project);
CREATE INDEX idx_knowledge_tags ON knowledge(tags);
CREATE INDEX idx_knowledge_decay ON knowledge(last_reinforced_at);
CREATE VIRTUAL TABLE knowledge_fts USING fts5(content, tags);
```

---

## CLI Commands

### `remember` - Store Knowledge

```bash
# Basic usage
agent-journal remember "fact to remember"

# With tags
agent-journal remember "use camelCase for variables" --tag convention --tag style

# With project scope
agent-journal remember "API key is in .env" --project my-api

# From stdin (for longer content)
echo "detailed explanation" | agent-journal remember --stdin

# With source attribution
agent-journal remember "from code review" --source "session:abc123"
```

### `recall` - Query Knowledge

```bash
# Search knowledge bank
agent-journal recall "authentication"

# Filter by tag
agent-journal recall "conventions" --tag style

# Filter by project
agent-journal recall "build" --project my-api

# Semantic search
agent-journal recall "how to deploy" --mode semantic

# Show all knowledge
agent-journal recall --all
```

### `forget` - Remove Knowledge

```bash
# Remove by ID
agent-journal forget abc123

# Remove by content match (with confirmation)
agent-journal forget --match "old convention"

# Remove all for a project
agent-journal forget --project old-project --all
```

### `reinforce` - Reset Decay Timer

```bash
# Reinforce a specific entry (reset decay)
agent-journal reinforce abc123

# Reinforce multiple entries
agent-journal reinforce abc123 def456 ghi789

# Reinforce all entries matching a query
agent-journal reinforce --match "important convention"
```

### `knowledge` - Manage Knowledge Bank

```bash
# List all entries (shows decay status)
agent-journal knowledge list

# List entries by decay status
agent-journal knowledge list --decaying      # decay_factor < 0.5
agent-journal knowledge list --expiring      # decay_factor < 0.1

# Show stats (including decay distribution)
agent-journal knowledge stats

# Export knowledge
agent-journal knowledge export --format json > knowledge.json

# Import knowledge
agent-journal knowledge import knowledge.json

# Prune expired entries (decay_factor < threshold)
agent-journal knowledge prune --threshold 0.05

# Clear all (with confirmation)
agent-journal knowledge clear
```

---

## Integration with Search

### Unified Search

The existing `search` command gains knowledge integration:

```bash
# Default: sessions only (backward compatible)
agent-journal search "auth"

# Include knowledge entries
agent-journal search "auth" --include-knowledge

# Knowledge entries appear with [K] prefix in results
```

### Search Result Display

```
Found 5 result(s):

[1] Session: abc123
    Agent: claude-code
    Score: 0.85
    Started: 2026-01-15 10:30:00
    Preview: Implemented JWT authentication...

[2] Knowledge: def456
    Score: 0.82 (decay: 0.95 ████████░░)
    Tags: architecture, auth
    Reinforced: 5 days ago
    Content: Authentication uses JWT with 24h expiry, refresh tokens...

[3] Knowledge: xyz789
    Score: 0.45 (decay: 0.31 ███░░░░░░░)
    Tags: convention
    Reinforced: 156 days ago ⚠️ decaying
    Content: Old formatting convention that may be outdated...

[4] Session: ghi789
    ...
```

### Decay Indicators

| Decay Factor | Status | Visual |
|--------------|--------|--------|
| > 0.75 | Fresh | ████████░░ |
| 0.50 - 0.75 | Good | ██████░░░░ |
| 0.25 - 0.50 | Aging | ████░░░░░░ |
| 0.10 - 0.25 | Decaying ⚠️ | ██░░░░░░░░ |
| < 0.10 | Expiring 🕐 | █░░░░░░░░░ |

---

## MCP Tools

### Additional MCP Tools for Knowledge

| Tool | Description | Parameters |
|------|-------------|------------|
| `remember` | Store a fact/learning | `content`, `tags[]`, `project` |
| `recall` | Search knowledge bank | `query`, `tags[]`, `project`, `mode` |
| `reinforce` | Reset decay timer on knowledge | `id` or `ids[]` |
| `forget` | Remove knowledge entry | `id` |
| `list_knowledge` | List knowledge entries | `project`, `tags[]`, `limit`, `includeDecaying` |

### MCP Tool Definitions

```csharp
[McpServerTool(Description = "Remember a fact or learning for future reference")]
public async Task<RememberResult> Remember(
    [Description("The fact or knowledge to remember")] string content,
    [Description("Categorization tags")] string[]? tags = null,
    [Description("Project scope (null for global)")] string? project = null)
{
    var entry = new KnowledgeEntry(
        Id: Guid.NewGuid().ToString(),
        Content: content,
        Tags: tags ?? [],
        Project: project,
        CreatedAt: DateTime.UtcNow,
        LastReinforcedAt: DateTime.UtcNow,  // Fresh entry
        ReinforcementCount: 0,
        ...
    );
    
    await _knowledgeRepository.SaveAsync(entry);
    await _searchEngine.IndexKnowledgeAsync(entry);
    
    return new RememberResult { Id = entry.Id, Success = true };
}

[McpServerTool(Description = "Search stored knowledge and facts")]
public async Task<RecallResult> Recall(
    [Description("Search query")] string query,
    [Description("Filter by tags")] string[]? tags = null,
    [Description("Filter by project")] string? project = null,
    [Description("Search mode")] string mode = "hybrid")
{
    var results = await _knowledgeRepository.SearchAsync(query, tags, project, mode);
    // Results are scored with decay applied
    return new RecallResult { Entries = results };
}

[McpServerTool(Description = "Reinforce knowledge to prevent decay - call when knowledge was useful")]
public async Task<ReinforceResult> Reinforce(
    [Description("Knowledge entry ID(s) to reinforce")] string[] ids)
{
    var reinforced = 0;
    foreach (var id in ids)
    {
        var entry = await _knowledgeRepository.GetAsync(id);
        if (entry != null)
        {
            await _knowledgeRepository.ReinforceAsync(id);
            reinforced++;
        }
    }
    return new ReinforceResult { ReinforcedCount = reinforced };
}
```

### Automatic Reinforcement

The agent should call `reinforce` when:
1. Knowledge was retrieved and found useful
2. Knowledge influenced a decision
3. User confirms knowledge is still accurate

Example agent behavior:
```
Agent thinking: "Let me check project conventions..."
→ recall("conventions", project="my-app")
→ Returns: "Use ESLint Airbnb config" (id: abc123)

Agent: "Based on your convention of using ESLint with Airbnb config, I'll..."

Agent thinking: "That knowledge was useful, I should reinforce it"
→ reinforce(["abc123"])
```

---

## Architecture

### New Components

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Core Services                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────┐    ┌─────────────────────┐                │
│  │ ISessionRepository  │    │ IKnowledgeRepository │  ← NEW        │
│  │ (sessions)          │    │ (knowledge entries)  │                │
│  └──────────┬──────────┘    └──────────┬──────────┘                │
│             │                          │                            │
│             └────────────┬─────────────┘                           │
│                          ▼                                          │
│           ┌─────────────────────────────┐                          │
│           │    ISearchEngine            │                          │
│           │  - SearchSessionsAsync()    │                          │
│           │  - SearchKnowledgeAsync()   │  ← Extended              │
│           │  - SearchAllAsync()         │  ← NEW                   │
│           └─────────────────────────────┘                          │
│                                                                      │
├─────────────────────────────────────────────────────────────────────┤
│                         Storage Layer                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐ │
│  │ SQLite      │  │ Lucene      │  │ AJVI        │  │ Knowledge  │ │
│  │ (metadata)  │  │ (sessions)  │  │ (vectors)   │  │ (FTS+vec)  │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └────────────┘ │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### IKnowledgeRepository Interface

```csharp
public interface IKnowledgeRepository
{
    Task<KnowledgeEntry> SaveAsync(KnowledgeEntry entry, CancellationToken ct = default);
    Task<KnowledgeEntry?> GetAsync(string id, CancellationToken ct = default);
    Task<bool> DeleteAsync(string id, CancellationToken ct = default);
    Task<IReadOnlyList<KnowledgeEntry>> ListAsync(
        string? project = null, 
        IEnumerable<string>? tags = null,
        int limit = 100,
        CancellationToken ct = default);
    Task<IReadOnlyList<KnowledgeEntry>> SearchAsync(
        string query,
        IEnumerable<string>? tags = null,
        string? project = null,
        SearchMode mode = SearchMode.Hybrid,
        int maxResults = 10,
        CancellationToken ct = default);
}
```

---

## Implementation Plan

### Phase 1: Core Knowledge Storage (3-4 hours)
1. Create `KnowledgeEntry` model with decay fields
2. Add `knowledge` table to SQLite schema
3. Implement `SqliteKnowledgeRepository`
4. Add FTS5 indexing for knowledge
5. Implement `DecayCalculator` utility

### Phase 2: CLI Commands (3-4 hours)
1. Implement `RememberCommand`
2. Implement `RecallCommand`
3. Implement `ForgetCommand`
4. Implement `ReinforceCommand`
5. Implement `KnowledgeCommand` (list/stats/export/import/prune)

### Phase 3: Search Integration (2-3 hours)
1. Add knowledge to vector index (AJVI)
2. Extend `HybridSearcher` for unified search
3. Apply decay scoring to search results
4. Add `--include-knowledge` flag to search
5. Update search results display with decay indicators

### Phase 4: MCP Integration (2-3 hours)
1. Add `remember` MCP tool
2. Add `recall` MCP tool
3. Add `reinforce` MCP tool
4. Add `forget` MCP tool
5. Test with Claude Desktop

### Phase 5: Polish & Documentation (1-2 hours)
1. Add knowledge to skill file
2. Update CLAUDE.md
3. Add usage examples
4. Write tests

**Total Estimated Effort: 11-16 hours**

---

## Storage Options

### Option A: Extend Existing SQLite + Indexes
- Knowledge in SQLite with FTS5
- Vectors in existing AJVI index
- **Pros**: Reuses infrastructure, unified
- **Cons**: AJVI format needs knowledge flag

### Option B: Separate Knowledge Store
- Dedicated SQLite database for knowledge
- Separate vector index
- **Pros**: Clean separation
- **Cons**: More files, duplicate infrastructure

### Option C: SQLite with Embedded Vectors
- Knowledge + vectors all in SQLite
- Use BLOB for vectors, SIMD search in SQL
- **Pros**: Single file, atomic operations
- **Cons**: Slower vector search

**Recommendation**: Option A - extend existing infrastructure with a knowledge flag in AJVI entries.

---

## Differentiation: Knowledge vs Sessions

| Aspect | Sessions | Knowledge |
|--------|----------|-----------|
| Source | Auto-indexed from files | Explicitly stored |
| Mutability | Immutable | Mutable (CRUD) |
| Size | Large (full conversations) | Small (facts) |
| Structure | Messages with roles | Single text + metadata |
| Scope | Per-session | Global or per-project |
| Lifecycle | Permanent archive | Can be deleted |

---

## Example Workflows

### Agent Learns Project Conventions
```
User: "We use ESLint with Airbnb config"
Agent: "I'll remember that for future reference."
→ remember("Project uses ESLint with Airbnb configuration", tags=["tooling", "linting"])
```

### Agent Recalls Before Coding
```
Agent thinking: "Let me check what I know about this project's conventions..."
→ recall("conventions", project="current-project")
→ Returns: "Uses ESLint Airbnb", "Prefers functional components", "Tests in __tests__ folders"
```

### User Searches Everything
```bash
$ agent-journal search "linting" --include-knowledge

[1] Knowledge: abc123
    Score: 0.95
    Tags: tooling, linting
    Content: Project uses ESLint with Airbnb configuration

[2] Session: def456
    Score: 0.72
    Preview: ...fixed the ESLint errors by updating...
```

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Knowledge becomes stale | Add `updated_at`, allow updates |
| Too much noise in search | Separate `--include-knowledge` flag |
| Conflicting knowledge | Show source, allow delete |
| Storage bloat | Limit entry size, add cleanup |
| Cross-project leakage | Project scoping, global opt-in |

---

## Alignment Checklist

- [x] Reuses existing search infrastructure
- [x] Extends rather than replaces
- [x] Maintains backward compatibility
- [x] Works with all search modes
- [x] Integrates with MCP
- [x] Simple CLI UX

---

## Verdict

**Viable**: ✅ YES

The knowledge bank is a natural extension of agent-journal. It leverages existing infrastructure (SQLite, Lucene, AJVI, embeddings) and provides significant value for agent memory. The CLI commands (`remember`/`recall`/`forget`) are intuitive and the MCP tools enable seamless agent integration.

**Recommendation**: Proceed with implementation using Option A (extend existing infrastructure).

---

## Next Steps

1. Add `knowledge` table to SQLite schema
2. Create `IKnowledgeRepository` interface
3. Implement `SqliteKnowledgeRepository`
4. Add `RememberCommand` CLI
5. Add `RecallCommand` CLI
6. Extend search with knowledge integration
7. Add MCP tools for knowledge
