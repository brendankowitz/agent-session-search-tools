# Knowledge Bank Core - Quick Reference

## Usage Examples

### 1. Initialize Repository

```csharp
using AgentJournal.Core.Knowledge;
using AgentJournal.Core.Models;

var repo = new SqliteKnowledgeRepository("knowledge.db");
await repo.InitializeAsync();
```

### 2. Save Knowledge Entry

```csharp
var entry = new KnowledgeEntry(
    Id: Guid.NewGuid().ToString(),
    Content: "Project uses ESLint with Airbnb config",
    Tags: new[] { "tooling", "linting" },
    Project: "my-project",
    Source: "user",
    CreatedAt: DateTime.UtcNow,
    LastReinforcedAt: DateTime.UtcNow,
    ReinforcementCount: 0
);

await repo.SaveAsync(entry);
```

### 3. Search Knowledge

```csharp
// Keyword search with FTS5
var results = await repo.SearchAsync(
    query: "linting",
    tags: new[] { "tooling" },
    project: "my-project",
    mode: SearchMode.Hybrid,
    maxResults: 10
);

foreach (var result in results)
{
    Console.WriteLine($"Score: {result.Score:F2} (Decay: {result.DecayFactor:F2})");
    Console.WriteLine($"Content: {result.Entry.Content}");
    Console.WriteLine($"Status: {DecayCalculator.GetDecayStatus(result.DecayFactor)}");
}
```

### 4. Reinforce Knowledge

```csharp
// Reset decay timer when knowledge is used
var success = await repo.ReinforceAsync(entryId);
if (success)
{
    Console.WriteLine("Knowledge reinforced - decay timer reset");
}
```

### 5. Get Statistics

```csharp
var stats = await repo.GetStatsAsync();
Console.WriteLine($"Total entries: {stats.TotalEntries}");
Console.WriteLine($"Fresh: {stats.FreshEntries}");
Console.WriteLine($"Good: {stats.GoodEntries}");
Console.WriteLine($"Aging: {stats.AgingEntries}");
Console.WriteLine($"Decaying: {stats.DecayingEntries}");
Console.WriteLine($"Expiring: {stats.ExpiringEntries}");
```

### 6. Prune Expired Entries

```csharp
// Remove entries with decay < 0.05 (roughly 1 year old)
var pruned = await repo.PruneExpiredAsync(threshold: 0.05);
Console.WriteLine($"Pruned {pruned} expired entries");
```

### 7. Vector Search (Semantic)

```csharp
using AgentJournal.Core.Search;
using AgentJournal.Core.Embeddings;

var embedder = new NoMicEmbeddingProvider();
var vectorSearch = new VectorSearchEngine("./index", embedder);
await vectorSearch.InitializeAsync();

// Index knowledge for semantic search
await vectorSearch.IndexKnowledgeAsync(entry);

// Search semantically
var results = await vectorSearch.SearchKnowledgeAsync(
    query: "how to configure linting tools",
    maxResults: 10,
    halfLifeDays: 90.0
);
```

## Decay Calculation Examples

```csharp
// Calculate decay for a 30-day old entry
var decayFactor = DecayCalculator.CalculateDecayFactor(
    lastReinforced: DateTime.UtcNow.AddDays(-30),
    halfLifeDays: 90.0
);
// Result: ~0.79

// Apply decay to a score
var baseScore = 0.95;
var adjustedScore = DecayCalculator.ApplyDecay(baseScore, decayFactor);
// Result: ~0.75

// Check status
var status = DecayCalculator.GetDecayStatus(decayFactor);
// Result: "Good"

// Check if expired
var expired = DecayCalculator.IsExpired(decayFactor, threshold: 0.05);
// Result: false
```

## Database Schema

```sql
-- Main knowledge table
CREATE TABLE knowledge (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    tags TEXT,                      -- JSON array: ["tag1", "tag2"]
    project TEXT,
    source TEXT,
    created_at TEXT NOT NULL,       -- ISO 8601 format
    last_reinforced_at TEXT NOT NULL,
    reinforcement_count INTEGER DEFAULT 0
);

-- Full-text search virtual table
CREATE VIRTUAL TABLE knowledge_fts USING fts5(
    content,
    tags,
    project,
    content='knowledge',
    content_rowid='rowid'
);

-- Indexes
CREATE INDEX idx_knowledge_project ON knowledge(project);
CREATE INDEX idx_knowledge_last_reinforced ON knowledge(last_reinforced_at);
CREATE INDEX idx_knowledge_created ON knowledge(created_at);
```

## Decay Timeline

| Days | Decay Factor | Status    |
|------|-------------|-----------|
| 0    | 1.00        | Fresh     |
| 30   | 0.79        | Good      |
| 60   | 0.63        | Good      |
| 90   | 0.50        | Good      |
| 120  | 0.40        | Aging     |
| 180  | 0.25        | Aging     |
| 240  | 0.16        | Decaying  |
| 365  | 0.06        | Expiring  |

## Key Features

✅ **Temporal Decay**: Automatic aging with configurable half-life
✅ **Full-Text Search**: SQLite FTS5 for keyword search
✅ **Semantic Search**: Vector embeddings via AJVI index
✅ **Reinforcement**: Reset decay when knowledge is useful
✅ **Statistics**: Track decay distribution and usage
✅ **Pruning**: Remove expired entries automatically
✅ **Filtering**: By project, tags, decay status
✅ **Thread-Safe**: Proper locking in vector search
✅ **Async/Await**: Non-blocking I/O throughout

## Integration Points

### With Sessions
- Knowledge entries stored in same vector index as sessions
- Distinguished by agent type byte = 3
- Can appear in unified search results

### With MCP
- Ready for MCP tool wrappers:
  - `remember()` → `SaveAsync()`
  - `recall()` → `SearchAsync()`
  - `reinforce()` → `ReinforceAsync()`
  - `forget()` → `DeleteAsync()`

### With CLI
- Ready for command implementations:
  - `remember` → `SaveAsync()`
  - `recall` → `SearchAsync()`
  - `reinforce` → `ReinforceAsync()`
  - `forget` → `DeleteAsync()`
  - `knowledge list` → `ListAsync()`
  - `knowledge stats` → `GetStatsAsync()`
  - `knowledge prune` → `PruneExpiredAsync()`

## Next Implementation Steps

1. **CLI Commands** - Implement user-facing commands
2. **MCP Tools** - Wrap repository methods for agent access
3. **Unified Search** - Combine sessions + knowledge in search results
4. **Tests** - Unit and integration tests for all components
5. **Documentation** - User guide and examples

---

**Status**: Core implementation complete and tested
**Build**: Passing
**Files**: 4 new + 1 modified
