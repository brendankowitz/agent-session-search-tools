# Feature: Agent Session Search Tool

**Status**: Exploring
**Created**: 2026-01-15

## Problem Statement

AI coding agents (GitHub Copilot CLI, Claude Code, Cursor, Aider, etc.) generate a wealth of conversational knowledge—debugging sessions, problem-solving threads, architectural discussions—but this knowledge is:
- **Scattered** across different tools with different storage formats
- **Unsearchable** without manual file inspection
- **Unshared** between agents and team members
- **Lost to context** when you need to recall past solutions

Users and AI agents need a way to index, search, and export historical conversations to leverage institutional knowledge and avoid re-solving problems.

## Solution Vision

A **.NET 10 global tool** that combines the best of:
- **[claude-run](https://github.com/kamranahmedse/claude-run)**: Beautiful HTML export and viewing of chat sessions
- **[coding_agent_session_search (cass)](https://github.com/Dicklesworthstone/coding_agent_session_search)**: Unified indexing, vector search, and multi-agent support

### Key Capabilities

1. **Multi-Agent Indexing**: Parse and normalize sessions from Copilot CLI, Claude Code, Cursor, Aider, Cline, Gemini CLI, etc.
2. **Vector Search**: Optional semantic search using local ONNX models (MiniLM); hash-based fallback when models unavailable
3. **Full-Text Search**: BM25-based lexical search for exact term matching
4. **HTML Export**: Generate beautiful, self-contained HTML files for any conversation
5. **Markdown Export**: Export threads to markdown for documentation or sharing
6. **Contextual Results**: Return 1-3 messages of context around matches with navigation (forward/back)
7. **Summarization**: Optional AI-powered summarization of search results
8. **Agent-Friendly**: JSON output mode for consumption by AI agents (robot mode)

## Constraints

### Technical Constraints
- **.NET 10** (latest LTS with enhanced global tool features)
- **Cross-platform**: Windows, macOS, Linux
- **Offline-first**: No cloud dependencies; all processing local
- **Optional ML**: Vector search works without ONNX model (hash-based fallback)
- **Single binary**: Distributed as a .NET global tool (`dotnet tool install -g`)

### Design Constraints
- **Privacy-first**: No telemetry, no data leaves the machine
- **Non-destructive**: Never modify source session files
- **Incremental indexing**: Watch for changes, update index efficiently
- **Memory-efficient**: Handle large histories without excessive RAM

### Supported Agent Formats (Initial)
| Agent | Location | Format |
|-------|----------|--------|
| Copilot CLI | `~/.copilot-cli/` | JSONL |
| Claude Code | `~/.claude/projects/` | JSONL |
| Cursor | Platform-specific | SQLite |
| Aider | `.aider.chat.history.md` | Markdown |
| Cline | VS Code storage | JSON |

## Investigations
| Investigation | Status | Summary |
|--------------|--------|---------|
| [dotnet-tool-architecture](investigations/dotnet-tool-architecture.md) | ✅ Recommended | .NET 10 global tool with Lucene.NET search, ONNX vectors, plugin connectors |

## Decision
*No ADR yet - investigations in progress*
