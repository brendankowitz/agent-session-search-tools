<div align="center">

# 📓 Agent Journal

**Index, search, and learn from your AI agent conversations**

[![.NET](https://img.shields.io/badge/.NET-10.0-512BD4?logo=dotnet)](https://dotnet.microsoft.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![NuGet](https://img.shields.io/badge/NuGet-agent--journal-blue?logo=nuget)](https://www.nuget.org/packages/AgentJournal)

*A .NET global tool and MCP server for managing AI agent session history with intelligent search and knowledge decay.*

</div>

---

## Overview

**Agent Journal** helps you build institutional memory from your AI coding sessions. It indexes conversations from Claude Code and GitHub Copilot CLI, enabling powerful search across your entire history. Use it to find past solutions, learn from previous debugging sessions, or build a personal knowledge base that grows with your work.

Key differentiators:
- **Temporal Decay** — Knowledge naturally ages using exponential decay (90-day half-life), keeping recent insights prominent while older content fades
- **Hybrid Search** — Combines lexical (Lucene) and semantic (vector) search for best-of-both-worlds retrieval
- **MCP Integration** — Expose your knowledge base directly to AI agents via Model Context Protocol
- **Content Indexing** — Index any markdown documentation alongside your sessions

---

## ✨ Key Features

### 🔍 Intelligent Search

- **Lexical Search** — Fast keyword-based search powered by Lucene.NET
- **Semantic Search** — Meaning-based search using ONNX embeddings
- **Hybrid Mode** — Combines both with Reciprocal Rank Fusion for optimal results
- **Context Window** — Retrieve N messages before/after matches for full context

### 🤖 Multi-Agent Support

- **Claude Code** — Index sessions from `~/.claude/projects/`
- **GitHub Copilot CLI** — Index sessions from `~/.copilot-cli/`
- **Extensible** — Clean connector architecture for adding more agents

### 🧠 Knowledge Management

- **Remember/Recall** — Store and retrieve knowledge snippets
- **Temporal Decay** — Exponential decay keeps knowledge fresh (configurable half-life)
- **Reinforcement** — Reset decay timer on important knowledge
- **Content Indexing** — Index markdown files from any directory

### 🔌 MCP Server

- **14 MCP Tools** — Full functionality exposed via Model Context Protocol
- **Claude Desktop** — Works seamlessly with Claude Desktop app
- **Any MCP Client** — Standard protocol works with any compatible client

### 📤 Export Options

- **HTML** — Beautifully formatted with syntax highlighting
- **Markdown** — Clean, portable format
- **JSON** — Full data export for processing

---

## 📦 Installation

### From NuGet (Recommended)

```bash
dotnet tool install -g agent-journal
```

### From Source

```bash
git clone https://github.com/brendankowitz/agent-session-search-tools.git
cd agent-session-search-tools
dotnet pack src/AgentJournal -c Release
dotnet tool install -g --add-source src/AgentJournal/nupkg AgentJournal
```

### Verify Installation

```bash
agent-journal --version
agent-journal --help
```

---

## 🚀 Quick Start

### 1. Index Your Sessions

```bash
# Index all agent sessions
agent-journal index

# Index specific agent type
agent-journal index --agent claude-code

# Watch for new sessions continuously
agent-journal index --watch
```

### 2. Search Your History

```bash
# Basic search
agent-journal search "error handling"

# Semantic search (meaning-based)
agent-journal search "how to fix authentication" --mode semantic

# Hybrid search with context
agent-journal search "database migration" --mode hybrid --context 5

# Filter by project
agent-journal search "api design" --project my-project
```

### 3. Build Your Knowledge Base

```bash
# Remember something important
agent-journal knowledge remember "Always use parameterized queries for SQL" --project security

# Recall knowledge
agent-journal knowledge recall "sql injection"

# Reinforce important knowledge (reset decay)
agent-journal knowledge reinforce <id>
```

### 4. Index Documentation

```bash
# Index markdown files from a directory
agent-journal content index ./docs --project my-project

# Add content directly
agent-journal content add --source "design-notes" --title "API Design" --content "..."

# Search indexed content
agent-journal content search "authentication flow"
```

---

## 💻 CLI Reference

### Commands

| Command | Description |
|---------|-------------|
| `search <query>` | Search sessions with lexical, semantic, or hybrid mode |
| `index` | Index sessions from configured agent paths |
| `export <id>` | Export a session to HTML, Markdown, or JSON |
| `knowledge` | Manage knowledge entries (remember, recall, reinforce, forget) |
| `content` | Manage indexed content (index, add, search, list, remove) |
| `config` | View and modify configuration |
| `mcp` | Start the MCP server |

### Search Options

```bash
agent-journal search <query> [options]

Options:
  -m, --mode <mode>       Search mode: lexical, semantic, hybrid (default: hybrid)
  --max <count>           Maximum results (default: 10)
  --context <n>           Messages before/after match (default: 0)
  -p, --project <name>    Filter by project
  -a, --agent <type>      Filter by agent: claude-code, copilot-cli
  --robot                 Output as JSON for scripting
```

### Content Options

```bash
agent-journal content index <path> [options]

Options:
  --filter <pattern>      Glob pattern (default: *.md)
  -p, --project <name>    Associate with project
  --recursive             Scan subdirectories (default: true)
  --rebuild               Clear and rebuild index

agent-journal content search <query> [options]

Options:
  --max <count>           Maximum results
  -p, --project <name>    Filter by project
  -s, --source-prefix     Filter by source path prefix
  -t, --tags <tags>       Filter by tags (comma-separated)
```

---

## 🔌 MCP Server Integration

### Start the Server

```bash
agent-journal mcp
```

### Claude Desktop Configuration

Add to your Claude Desktop config (`~/.config/claude/claude_desktop_config.json`):

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

### Available MCP Tools

| Tool | Description |
|------|-------------|
| `SearchSessions` | Search sessions with optional context window |
| `GetSession` | Get full session details by ID |
| `ListRecentSessions` | List recent sessions |
| `Remember` | Store knowledge with optional project/tags |
| `Recall` | Search knowledge base |
| `Reinforce` | Reset decay timer on knowledge |
| `Forget` | Remove knowledge entry |
| `IndexContent` | Index markdown files from directory |
| `AddContent` | Add content directly |
| `SearchContent` | Search indexed content |
| `ListContent` | List indexed content |
| `RemoveContent` | Remove content by criteria |
| `ReinforceContent` | Reset decay on content |
| `Search` | Unified search across sessions + knowledge |

---

## ⚙️ Configuration

Configuration stored in `~/.agent-journal/config.json`:

```json
{
  "DatabasePath": "~/.agent-journal/sessions.db",
  "IndexPath": "~/.agent-journal/index",
  "KnowledgePath": "~/.agent-journal/knowledge.db",
  "ContentPath": "~/.agent-journal/content.db",
  "Decay": {
    "HalfLifeDays": 90,
    "ExpirationThreshold": 0.05
  },
  "Agents": {
    "ClaudeCode": {
      "Enabled": true,
      "SessionsPath": "~/.claude/projects"
    },
    "CopilotCli": {
      "Enabled": true,
      "SessionsPath": "~/.copilot-cli"
    }
  },
  "Search": {
    "DefaultMode": "hybrid",
    "MaxResults": 10
  }
}
```

### View/Modify Configuration

```bash
# Show current config
agent-journal config show

# Set a value
agent-journal config set Decay.HalfLifeDays 60
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        CLI / MCP Server                      │
├─────────────────────────────────────────────────────────────┤
│  Commands: search, index, export, knowledge, content, mcp   │
├─────────────────────────────────────────────────────────────┤
│                      AgentJournal.Core                       │
├──────────────┬──────────────┬──────────────┬────────────────┤
│  Connectors  │    Search    │   Storage    │     Export     │
│ ─────────────│──────────────│──────────────│────────────────│
│ • Claude     │ • Lucene     │ • Sessions   │ • HTML         │
│ • Copilot    │ • Vector     │ • Knowledge  │ • Markdown     │
│              │ • Hybrid     │ • Content    │ • JSON         │
├──────────────┴──────────────┴──────────────┴────────────────┤
│                    SQLite + Lucene Index                     │
└─────────────────────────────────────────────────────────────┘
```

### Project Structure

```
src/
├── AgentJournal/              # CLI application
│   ├── Commands/              # CLI command handlers
│   └── Program.cs             # Entry point & DI
├── AgentJournal.Core/         # Core library
│   ├── Connectors/            # Agent session parsers
│   ├── Search/                # Lucene + Vector engines
│   ├── Storage/               # SQLite repositories
│   ├── Knowledge/             # Knowledge + Content repos
│   ├── Decay/                 # Temporal decay calculator
│   ├── Mcp/                   # MCP server & tools
│   └── Export/                # HTML, Markdown, JSON
└── AgentJournal.Tests/        # Unit tests
```

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request

For development details, see [CONTRIBUTING.md](CONTRIBUTING.md).

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Search powered by [Lucene.NET](https://lucenenet.apache.org/)
- CLI framework by [System.CommandLine](https://github.com/dotnet/command-line-api)
- Templates with [Scriban](https://github.com/scriban/scriban)
- MCP integration via [ModelContextProtocol](https://github.com/modelcontextprotocol)

---

<div align="center">

**Agent Journal** — *Your AI conversations, searchable and remembered.*

</div>
