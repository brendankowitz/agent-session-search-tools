# Architecture Overview

This document describes the architecture of Agent Journal.

## Core Components

### Storage Layer

- **SqliteSessionRepository**: Stores agent conversation sessions
- **SqliteKnowledgeRepository**: Manages the knowledge bank
- **SqliteContentRepository**: Indexes markdown content files

### Search Layer

- **LuceneSearchEngine**: Full-text search using Apache Lucene
- **VectorSearchEngine**: Semantic search using ONNX embeddings
- **HybridSearcher**: Combines both search modes

### Command Layer

All commands follow the System.CommandLine pattern:

- IndexCommand: Index agent sessions
- SearchCommand: Search sessions
- KnowledgeCommand: Manage knowledge bank
- ContentCommand: Index and search markdown content

## Design Patterns

The project follows SOLID principles and uses:

- Repository pattern for data access
- Dependency injection for service management
- Command pattern for CLI operations
