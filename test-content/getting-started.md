# Getting Started with Agent Journal

Agent Journal is a powerful CLI tool for indexing and searching AI agent conversations.

## Installation

Install the tool using dotnet:

```bash
dotnet tool install -g agent-journal
```

## Basic Usage

Index your agent sessions:

```bash
agent-journal index --agent all
```

Search for specific topics:

```bash
agent-journal search "database optimization"
```

## Features

- Full-text search with FTS5
- Vector-based semantic search
- Temporal decay for relevance
- Knowledge bank for facts
- Content indexing for markdown files
