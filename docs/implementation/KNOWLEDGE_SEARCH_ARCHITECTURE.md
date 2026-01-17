# Knowledge Search Integration - Architecture

## Component Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         SearchCommand                           │
│                                                                 │
│  Options:                                                       │
│  • --query <text>                                               │
│  • --mode (lexical/semantic/hybrid)                             │
│  • --include-knowledge / -k  [NEW]                              │
│  • --project, --agent, --max, --robot                           │
└───────────────┬─────────────────────────────────────────────────┘
                │
                ├─────────────────────┬─────────────────────────┐
                │                     │                         │
                ▼                     ▼                         ▼
    ┌─────────────────────┐  ┌──────────────────┐  ┌─────────────────┐
    │  ISearchEngine      │  │ IKnowledgeRepo   │  │ UnifiedSearch   │
    │  (sessions)         │  │ (knowledge)      │  │ Result [NEW]    │
    │                     │  │                  │  │                 │
    │ • SearchAsync()     │  │ • SearchAsync()  │  │ • Type          │
    │   → SearchResult[]  │  │   → Knowledge    │  │ • Score         │
    │                     │  │      SearchResult│  │ • Data          │
    └─────────────────────┘  └──────────────────┘  │ • DecayFactor   │
                │                     │             │ • Highlight     │
                │                     │             └─────────────────┘
                │                     │
                └──────────┬──────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ Merge & Sort    │
                  │ by Score        │
                  └────────┬────────┘
                           │
                ┌──────────┴──────────┐
                │                     │
                ▼                     ▼
       ┌─────────────────┐   ┌─────────────────┐
       │ Session Results │   │ Knowledge       │
       │                 │   │ Results         │
       │ [1] Session     │   │ [2] Knowledge   │
       │     Score: 0.90 │   │     Score: 0.85 │
       │     Agent: ...  │   │     Decay: 0.92 │
       │     Started: .. │   │     Tags: ...   │
       └─────────────────┘   └─────────────────┘
```

## Data Flow

```
User Command
    │
    └─► agent-journal search "JWT" --include-knowledge
            │
            ├─► Parse arguments
            │   • query: "JWT"
            │   • includeKnowledge: true
            │   • mode: lexical (default)
            │
            ├─► Execute session search
            │   │
            │   └─► VectorSearchEngine.SearchAsync()
            │       └─► Returns: SearchResult[]
            │           │
            │           └─► UnifiedSearchResult.FromSession()
            │
            ├─► Execute knowledge search (if -k enabled)
            │   │
            │   └─► SqliteKnowledgeRepository.SearchAsync()
            │       └─► Returns: KnowledgeSearchResult[]
            │           │
            │           └─► UnifiedSearchResult.FromKnowledge()
            │
            ├─► Merge results
            │   └─► List<UnifiedSearchResult>
            │       • Sort by Score DESC
            │       • Take maxResults
            │
            └─► Display results
                │
                ├─► For each result:
                │   ├─► if (type == Session)
                │   │   └─► DisplaySessionResult()
                │   │
                │   └─► if (type == Knowledge)
                │       └─► DisplayKnowledgeResult()
                │           └─► RenderDecayBar()
                │
                └─► Output to console or JSON
```

## Type Hierarchy

```
UnifiedSearchResult (record)
├── Id: string
├── Type: SearchResultType (enum)
│   ├── Session
│   └── Knowledge
├── Score: double
├── Data: object
│   ├── Session (when Type == Session)
│   └── KnowledgeEntry (when Type == Knowledge)
├── DecayFactor: double? (only for Knowledge)
├── Highlight: string?
└── MatchingMessages: Message[]? (only for Session)

Factory Methods:
├── FromSession(SearchResult) → UnifiedSearchResult
└── FromKnowledge(KnowledgeSearchResult) → UnifiedSearchResult

Accessors:
├── AsSession() → Session (throws if wrong type)
├── AsKnowledge() → KnowledgeEntry (throws if wrong type)
├── TryGetSession(out Session?) → bool
└── TryGetKnowledge(out KnowledgeEntry?) → bool
```

## Decay Visualization

```
DecayFactor → Visual Bar → Display

1.00        →  ██████████  →  (decay: 1.00 ██████████)
0.92        →  █████████░  →  (decay: 0.92 █████████░)
0.75        →  ████████░░  →  (decay: 0.75 ████████░░)
0.50        →  █████░░░░░  →  (decay: 0.50 █████░░░░░)
0.25        →  ███░░░░░░░  →  (decay: 0.25 ███░░░░░░░)
0.15        →  ██░░░░░░░░  →  (decay: 0.15 ██░░░░░░░░) ⚠️ decaying
0.05        →  █░░░░░░░░░  →  (decay: 0.05 █░░░░░░░░░) ⚠️ decaying

Algorithm:
    filled = floor(decayFactor * 10)
    bar = repeat('█', filled) + repeat('░', 10 - filled)
    warning = decayFactor < 0.25 ? " ⚠️ decaying" : ""
    return "(decay: {decayFactor:F2} {bar}){warning}"
```

## Score Merging Strategy

```
Session Score:    Semantic similarity (0.0 - 1.0)
Knowledge Score:  Semantic similarity × decay factor (0.0 - 1.0)

Example:
┌─────────────────┬──────────────┬────────┬────────────┐
│ Result          │ Base Score   │ Decay  │ Final Score│
├─────────────────┼──────────────┼────────┼────────────┤
│ Session A       │ 0.90         │ N/A    │ 0.90       │
│ Knowledge B     │ 0.95         │ 0.92   │ 0.87       │
│ Session C       │ 0.85         │ N/A    │ 0.85       │
│ Knowledge D     │ 0.90         │ 0.50   │ 0.45       │
└─────────────────┴──────────────┴────────┴────────────┘

After sorting: [Session A, Knowledge B, Session C, Knowledge D]
```

## Display Format Comparison

```
┌─────────────────────────────────────────────────────────────────┐
│ SESSION RESULT                                                  │
├─────────────────────────────────────────────────────────────────┤
│ [1] Session: def456                                             │
│     Agent: claude-code                                          │
│     Score: 0.90                                                 │
│     Project: /home/user/api-service                             │
│     Started: 2026-01-15 10:30:00                                │
│     Matching messages:                                          │
│       [user] How do I implement JWT authentication?            │
│       [assistant] Here's how to implement JWT...               │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ KNOWLEDGE RESULT                                                │
├─────────────────────────────────────────────────────────────────┤
│ [2] Knowledge: abc123                                           │
│     Score: 0.85 (decay: 0.92 █████████░)                        │
│     Tags: auth, security, jwt                                   │
│     Project: /home/user/api-service                             │
│     Source: manual-entry                                        │
│     Created: 2026-01-10 14:20:00                                │
│     Last reinforced: 2026-01-14 09:15:00 (3x)                   │
│     Content: Use JWT tokens with 24h expiry. Store refresh...  │
└─────────────────────────────────────────────────────────────────┘
```

## Key Design Decisions

1. **Optional Knowledge**: Default is `false` to maintain backward compatibility
2. **Unified Scoring**: Both types use 0.0-1.0 scale for direct comparison
3. **Decay Applied**: Knowledge scores include decay factor in final score
4. **Type Safety**: Strong typing with helper methods to avoid casting errors
5. **Graceful Degradation**: Works even if IKnowledgeRepository not registered
6. **Visual Feedback**: Decay bars provide immediate visual understanding

## Integration Points

```
Program.cs (DI Container)
    │
    ├─► Register ISearchEngine
    ├─► Register IKnowledgeRepository (optional)
    │
    └─► SearchCommand.Create(serviceProvider)
            │
            └─► Resolves dependencies:
                ├─► GetRequiredService<ISearchEngine>()
                └─► GetService<IKnowledgeRepository>() [optional]
```

## Testing Scenarios

1. **No Knowledge Flag**: Search returns only sessions
2. **With Knowledge Flag**: Search returns merged results
3. **Knowledge Repo Missing**: Gracefully skips knowledge search
4. **Empty Results**: Handles no sessions and/or no knowledge
5. **JSON Output**: Both types serialized with type discriminator
6. **Decay Visualization**: All decay ranges display correctly
