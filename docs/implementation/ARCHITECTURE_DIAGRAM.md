# AgentJournal - Complete Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         AgentJournal CLI                                 │
│                         (Program.cs)                                     │
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │         Dependency Injection Container                          │   │
│  │  ┌──────────────────────────────────────────────────────────┐  │   │
│  │  │  Services                                                 │  │   │
│  │  │  • ConfigurationService (Singleton)                       │  │   │
│  │  │  • ClaudeCodeConnector (Singleton)                        │  │   │
│  │  │  • CopilotCliConnector (Singleton)                        │  │   │
│  │  │  • ISessionRepository → SqliteSessionRepository           │  │   │
│  │  │  • ISearchEngine → LuceneSearchEngine                     │  │   │
│  │  │  • IExporter[] → Html/Markdown/JsonExporter               │  │   │
│  │  └──────────────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┬────────────┐
                    │               │               │            │
                    ▼               ▼               ▼            ▼
        ┌───────────────┐ ┌────────────────┐ ┌────────────┐ ┌────────────┐
        │ IndexCommand  │ │ SearchCommand  │ │ExportCommand│ │ConfigCommand│
        │               │ │                │ │            │ │            │
        │ • --agent     │ │ • --mode       │ │ • --format │ │ • show     │
        │ • --watch     │ │ • --context    │ │ • --output │ │ • set      │
        │ • --rebuild   │ │ • --robot      │ │ • --stdout │ │ • agents   │
        └───────┬───────┘ └────────┬───────┘ └──────┬─────┘ └──────┬─────┘
                │                  │                 │              │
                │                  │                 │              │
        ┌───────▼──────────────────▼─────────────────▼──────────────▼─────┐
        │                    Configuration System                          │
        │  ┌────────────────────────────────────────────────────────────┐ │
        │  │  ConfigurationService                                      │ │
        │  │  • Load/Save JSON config (~/.agent-journal/config.json)   │ │
        │  │  • Caching                                                 │ │
        │  │  • Key-value updates                                       │ │
        │  └────────────────────────────────────────────────────────────┘ │
        │  ┌────────────────────────────────────────────────────────────┐ │
        │  │  AgentJournalConfig                                        │ │
        │  │  • DataPath: ~/.agent-journal                              │ │
        │  │  • DatabasePath: ~/.agent-journal/agent-journal.db         │ │
        │  │  • LuceneIndexPath: ~/.agent-journal/lucene-index          │ │
        │  │  • ClaudeProjectsPath                                      │ │
        │  │  • CopilotSessionsPath                                     │ │
        │  │  • DefaultSearchMode, DefaultContextMessages, etc.         │ │
        │  └────────────────────────────────────────────────────────────┘ │
        └────────────────────────────────────────────────────────────────┘
                │                     │                    │
        ┌───────▼──────┐     ┌────────▼────────┐     ┌────▼──────────┐
        │  Connectors  │     │   Repository    │     │  Search Engine │
        │              │     │                 │     │                │
        │ ┌──────────┐ │     │ ┌─────────────┐ │     │ ┌────────────┐ │
        │ │ Claude   │─┼────▶│ │   SQLite    │ │     │ │   Lucene   │ │
        │ │ Code     │ │     │ │             │ │     │ │            │ │
        │ └──────────┘ │     │ │ Sessions    │ │     │ │  Indexer   │ │
        │              │     │ │ Messages    │ │     │ │  Searcher  │ │
        │ ┌──────────┐ │     │ │ ToolCalls   │ │     │ └────────────┘ │
        │ │ Copilot  │─┼────▶│ │             │ │     │                │
        │ │ CLI      │ │     │ └─────────────┘ │     │  SearchResult  │
        │ └──────────┘ │     │                 │     │  • Session     │
        │              │     │  GetSession     │     │  • Score       │
        │ IAgentConn   │     │  SaveSession    │     │  • Highlight   │
        │ • ParseAsync │     │  GetAll         │     │  • Matches     │
        │ • GetPaths   │     │  Initialize     │     └────────────────┘
        └──────────────┘     └─────────────────┘              │
                                                               │
                                                       ┌───────▼────────┐
                                                       │   Exporters    │
                                                       │                │
                                                       │ ┌────────────┐ │
                                                       │ │   HTML     │ │
                                                       │ └────────────┘ │
                                                       │ ┌────────────┐ │
                                                       │ │  Markdown  │ │
                                                       │ └────────────┘ │
                                                       │ ┌────────────┐ │
                                                       │ │    JSON    │ │
                                                       │ └────────────┘ │
                                                       │                │
                                                       │ IExporter      │
                                                       │ • ExportAsync  │
                                                       │ • ToFile       │
                                                       └────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                         Data Flow Examples                               │
└─────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│ INDEX COMMAND FLOW                                                   │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  User: aj index --agent claude-code                                 │
│    │                                                                 │
│    ├─▶ IndexCommand.Execute()                                       │
│    │     │                                                           │
│    │     ├─▶ ConfigurationService.LoadConfig()                      │
│    │     │     └─▶ ~/.agent-journal/config.json                     │
│    │     │                                                           │
│    │     ├─▶ ClaudeCodeConnector.GetSessionPaths()                  │
│    │     │     └─▶ ~/.claude/projects/**/*.jsonl                    │
│    │     │                                                           │
│    │     ├─▶ ClaudeCodeConnector.ParseSessionsAsync()               │
│    │     │     └─▶ Session objects                                  │
│    │     │                                                           │
│    │     ├─▶ SqliteSessionRepository.SaveSessionAsync(session)      │
│    │     │     └─▶ ~/.agent-journal/agent-journal.db                │
│    │     │                                                           │
│    │     └─▶ LuceneSearchEngine.IndexSessionAsync(session)          │
│    │           └─▶ ~/.agent-journal/lucene-index/                   │
│    │                                                                 │
│    └─▶ Output: "Indexed 301 sessions"                               │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│ SEARCH COMMAND FLOW                                                  │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  User: aj search "authentication" --robot                           │
│    │                                                                 │
│    ├─▶ SearchCommand.Execute()                                      │
│    │     │                                                           │
│    │     ├─▶ ConfigurationService.LoadConfig()                      │
│    │     │                                                           │
│    │     ├─▶ LuceneSearchEngine.SearchAsync("authentication")       │
│    │     │     │                                                     │
│    │     │     ├─▶ Query Lucene index                               │
│    │     │     └─▶ SearchResult[] with scores                       │
│    │     │                                                           │
│    │     ├─▶ Filter by --agent, --project                           │
│    │     │                                                           │
│    │     └─▶ Format output (JSON or human-readable)                 │
│    │                                                                 │
│    └─▶ Output: JSON array of matching sessions                      │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│ EXPORT COMMAND FLOW                                                  │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  User: aj export session-123 --format html                          │
│    │                                                                 │
│    ├─▶ ExportCommand.Execute()                                      │
│    │     │                                                           │
│    │     ├─▶ SqliteSessionRepository.GetSessionAsync("session-123") │
│    │     │     └─▶ Session with all messages                        │
│    │     │                                                           │
│    │     ├─▶ Find HtmlExporter from IExporter[]                     │
│    │     │                                                           │
│    │     ├─▶ HtmlExporter.ExportToFileAsync(session, path)          │
│    │     │     └─▶ session-123.html                                 │
│    │     │                                                           │
│    │     └─▶ Output: "Export complete! 45 messages"                 │
│    │                                                                 │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│ CONFIG COMMAND FLOW                                                  │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  User: aj config set VerboseLogging true                            │
│    │                                                                 │
│    ├─▶ ConfigCommand.Set.Execute()                                  │
│    │     │                                                           │
│    │     ├─▶ ConfigurationService.LoadConfig()                      │
│    │     │                                                           │
│    │     ├─▶ ConfigurationService.SetConfigValueAsync(key, value)   │
│    │     │     │                                                     │
│    │     │     ├─▶ Update config object                             │
│    │     │     └─▶ SaveConfigAsync()                                │
│    │     │           └─▶ ~/.agent-journal/config.json               │
│    │     │                                                           │
│    │     └─▶ Output: "Configuration updated successfully!"          │
│    │                                                                 │
└──────────────────────────────────────────────────────────────────────┘

```

## Technology Stack

- **.NET 10** - Latest C# 13 features
- **System.CommandLine** - Modern CLI framework
- **Microsoft.Extensions.DependencyInjection** - DI container
- **Microsoft.Extensions.Configuration** - Config management
- **Lucene.Net** - Full-text search
- **System.Data.SQLite** - Local database
- **Scriban** - Template engine (for exporters)
- **xUnit** - Testing framework

## Key Design Patterns

1. **Dependency Injection** - All services injected via constructor
2. **Factory Pattern** - Command creation via static Create() methods
3. **Repository Pattern** - ISessionRepository abstraction
4. **Strategy Pattern** - IExporter, ISearchEngine interfaces
5. **Singleton Pattern** - Services registered as singletons
6. **Command Pattern** - System.CommandLine commands

## File Structure

```
agent-session-search-tools/
├── src/
│   ├── AgentJournal/                    (CLI Application)
│   │   ├── Program.cs                   ✅ DI Setup
│   │   ├── Configuration/
│   │   │   ├── AgentJournalConfig.cs    ✅ Config Model
│   │   │   └── ConfigurationService.cs  ✅ Config Service
│   │   └── Commands/
│   │       ├── IndexCommand.cs          ✅ Wired
│   │       ├── SearchCommand.cs         ✅ Wired
│   │       ├── ExportCommand.cs         ✅ Wired
│   │       ├── ConfigCommand.cs         ✅ Wired
│   │       └── README.md                ✅ Documentation
│   │
│   ├── AgentJournal.Core/               (Business Logic)
│   │   ├── Models/
│   │   │   ├── Session.cs               ✅ Implemented
│   │   │   ├── Message.cs               ✅ Implemented
│   │   │   ├── MessageRole.cs           ✅ Implemented
│   │   │   └── ToolCall.cs              ✅ Implemented
│   │   ├── Connectors/
│   │   │   ├── IAgentConnector.cs       ✅ Interface
│   │   │   ├── ClaudeCodeConnector.cs   ✅ Implemented
│   │   │   └── CopilotCliConnector.cs   ✅ Implemented
│   │   ├── Storage/
│   │   │   ├── ISessionRepository.cs    ✅ Interface
│   │   │   └── SqliteSessionRepository.cs ✅ Implemented
│   │   ├── Search/
│   │   │   ├── ISearchEngine.cs         ✅ Interface
│   │   │   ├── LuceneSearchEngine.cs    ✅ Implemented
│   │   │   └── VectorSearchEngine.cs    🚧 Stub
│   │   └── Export/
│   │       ├── IExporter.cs             ✅ Interface
│   │       ├── HtmlExporter.cs          ✅ Implemented
│   │       ├── MarkdownExporter.cs      ✅ Implemented
│   │       └── JsonExporter.cs          ✅ Implemented
│   │
│   └── AgentJournal.Tests/              (Unit Tests)
│       └── ModelTests.cs                ✅ Model tests
│
└── ~/.agent-journal/                    (User Data)
    ├── config.json                      (Configuration)
    ├── agent-journal.db                 (SQLite Database)
    └── lucene-index/                    (Search Index)
```

## Status Summary

### ✅ Fully Implemented (Ready to Use)
- CLI command structure
- Dependency injection
- Configuration system
- All command options and arguments
- Help system
- Error handling
- Progress indicators
- JSON output mode

### ✅ Implemented (Core Components)
- ClaudeCodeConnector (session parsing)
- CopilotCliConnector (session parsing)
- SqliteSessionRepository (CRUD operations)
- LuceneSearchEngine (indexing & search)
- All Exporters (HTML, Markdown, JSON)
- Data models (Session, Message, ToolCall)

### 🎯 Next Steps
- Integration testing with real data
- Performance optimization
- Additional export templates
- Semantic search implementation
- Advanced filtering options
