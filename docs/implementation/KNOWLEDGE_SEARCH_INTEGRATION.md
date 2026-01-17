# Knowledge Search Integration

## Overview

The search command has been enhanced to support unified searching across both agent sessions and knowledge entries. Users can now search for both types of content simultaneously and see results ranked by relevance.

## Changes Made

### 1. Unified Search Result Type

**File**: `src/AgentJournal.Core/Search/UnifiedSearchResult.cs`

Created a new record type that can hold both session and knowledge search results:

```csharp
public record UnifiedSearchResult
{
    public required string Id { get; init; }
    public required SearchResultType Type { get; init; }
    public required double Score { get; init; }
    public required object Data { get; init; }
    public double? DecayFactor { get; init; }
    public string? Highlight { get; init; }
    public IReadOnlyList<Message>? MatchingMessages { get; init; }
}

public enum SearchResultType { Session, Knowledge }
```

**Features**:
- Factory methods: `FromSession()` and `FromKnowledge()`
- Type-safe accessors: `AsSession()`, `AsKnowledge()`
- Try methods: `TryGetSession()`, `TryGetKnowledge()`

### 2. Enhanced SearchCommand

**File**: `src/AgentJournal/Commands/SearchCommand.cs`

#### New Option

Added `--include-knowledge` / `-k` flag (default: false):

```bash
agent-journal search "authentication" --include-knowledge
agent-journal search "JWT tokens" -k
```

#### Unified Search Flow

1. Execute session search
2. If `--include-knowledge` enabled, execute knowledge search
3. Merge and sort results by score
4. Take top N results across both types

#### Display Improvements

**Session Results** (unchanged):
```
[1] Session: def456
    Agent: claude-code
    Score: 0.90
    Started: 2026-01-15 10:30:00
    Preview: Implemented authentication...
```

**Knowledge Results** (new format):
```
[2] Knowledge: abc123
    Score: 0.85 (decay: 0.92 ██████████░)
    Tags: auth, security
    Project: /home/user/project
    Source: manual-entry
    Created: 2026-01-10 14:20:00
    Last reinforced: 2026-01-14 09:15:00 (3x)
    Content: Use JWT with 24h expiry...
```

#### Decay Visualization

The `RenderDecayBar()` helper method creates a visual representation of knowledge freshness:

- **Fresh** (>0.75): `(decay: 0.92 ██████████░)`
- **Good** (>0.50): `(decay: 0.65 ███████░░░)`
- **Aging** (>0.25): `(decay: 0.40 ████░░░░░░)`
- **Decaying** (<0.25): `(decay: 0.15 ██░░░░░░░░) ⚠️ decaying`

### 3. JSON Output Support

Robot mode (`--robot`) outputs structured JSON with discriminated types:

```json
[
  {
    "type": "session",
    "sessionId": "def456",
    "agentType": "claude-code",
    "score": 0.90,
    ...
  },
  {
    "type": "knowledge",
    "id": "abc123",
    "content": "Use JWT...",
    "tags": ["auth", "security"],
    "score": 0.85,
    "decayFactor": 0.92,
    ...
  }
]
```

## Usage Examples

### Basic Session Search
```bash
agent-journal search "authentication"
```

### Search Sessions and Knowledge
```bash
agent-journal search "authentication" --include-knowledge
```

### Semantic Search with Knowledge
```bash
agent-journal search "JWT tokens" -k -m semantic
```

### Project-Specific Search
```bash
agent-journal search "database migration" -k -p "api-service"
```

### JSON Output
```bash
agent-journal search "react hooks" -k --robot | jq '.[] | select(.type == "knowledge")'
```

## Implementation Details

### Dependency Injection

The `SearchCommand` now requires an optional `IKnowledgeRepository`:

```csharp
var knowledgeRepo = serviceProvider.GetService<IKnowledgeRepository>();
```

If the repository is not registered, knowledge search is skipped gracefully.

### Score Merging

Both session and knowledge results use the same score scale (0.0-1.0), allowing direct comparison:
- Session scores: Semantic similarity to query
- Knowledge scores: Semantic similarity × decay factor

### Filtering

- Agent type filter (`--agent`): Only applies to sessions
- Project filter (`--project`): Applies to both sessions and knowledge
- Max results (`--max`): Applied after merging all results

## Testing

### Build
```bash
dotnet build
```

### Test Commands
```bash
# Setup test data
agent-journal kb add "Use JWT for API authentication" --tags auth,security

# Search without knowledge
agent-journal search "JWT" -m semantic

# Search with knowledge
agent-journal search "JWT" -m semantic --include-knowledge

# JSON output
agent-journal search "authentication" -k --robot
```

## Future Enhancements

- [ ] Knowledge-specific filters (e.g., `--tags`, `--min-decay`)
- [ ] Separate score weighting for sessions vs knowledge
- [ ] Result grouping by type
- [ ] Search suggestions based on knowledge
- [ ] Auto-reinforce knowledge when found in search results

## Related Files

- `src/AgentJournal.Core/Search/UnifiedSearchResult.cs` - New unified result type
- `src/AgentJournal/Commands/SearchCommand.cs` - Enhanced command
- `src/AgentJournal.Core/Search/VectorSearchEngine.cs` - Existing search engine
- `src/AgentJournal.Core/Knowledge/IKnowledgeRepository.cs` - Knowledge repository interface
- `src/AgentJournal.Core/Knowledge/DecayCalculator.cs` - Decay calculation logic
