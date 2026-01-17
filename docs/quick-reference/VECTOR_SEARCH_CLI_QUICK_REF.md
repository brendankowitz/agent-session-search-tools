# Vector Search CLI Quick Reference

## Commands

### Models Management

#### List Models
```bash
agent-journal models list
```
Shows installed embedding models with status and size.

#### Download Model
```bash
agent-journal models download <name>
```
Download an embedding model (e.g., `minilm`).

#### Remove Model
```bash
agent-journal models remove <name>
```
Remove an installed embedding model.

### Search with Modes

#### Lexical Search (Default)
```bash
agent-journal search "query text"
```

#### Semantic Search
```bash
agent-journal search "query text" --mode semantic
agent-journal search "query text" -m semantic
```

#### Hybrid Search
```bash
agent-journal search "query text" --mode hybrid
agent-journal search "query text" -m hybrid
```

### Full Search Options
```bash
agent-journal search "query" \
  --mode hybrid \
  --context 5 \
  --max 10 \
  --agent claude-code \
  --project /path/to/project \
  --robot
```

## Search Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `lexical` | Keyword-based search using Lucene | Exact term matching, file names, error codes |
| `semantic` | Vector-based similarity search | Conceptual queries, natural language |
| `hybrid` | Combined lexical + semantic | Best of both worlds, most accurate |

## Models Directory

**Default location**: `~/.agent-journal/models/`

**Structure**:
```
~/.agent-journal/models/
└── minilm/
    ├── model.onnx
    └── tokenizer.json
```

## Configuration

**Models path** is defined in `AgentJournalConfig`:
```csharp
public string ModelsPath => Path.Combine(DataPath, "models");
```

## Quick Start

1. **List available models**:
   ```bash
   agent-journal models list
   ```

2. **Download a model** (manual install for now):
   - Create directory: `~/.agent-journal/models/minilm/`
   - Download `model.onnx` and `tokenizer.json` from HuggingFace
   - Place in the directory

3. **Verify installation**:
   ```bash
   agent-journal models list
   ```

4. **Search with semantic mode**:
   ```bash
   agent-journal search "authentication flow" --mode semantic
   ```

## Files Modified

1. **New**: `src/AgentJournal/Commands/ModelsCommand.cs`
2. **Updated**: `src/AgentJournal/Program.cs`
3. **Updated**: `src/AgentJournal/Configuration/AgentJournalConfig.cs`
4. **Unchanged**: `src/AgentJournal/Commands/SearchCommand.cs` (already had --mode)

## Build & Run

```bash
cd src/AgentJournal
dotnet build --configuration Release
cd bin/Release/net10.0
./AgentJournal --help
```

## Example Workflow

```bash
# 1. Check if models are installed
agent-journal models list

# 2. Search with different modes
agent-journal search "bug fix" --mode lexical
agent-journal search "performance issue" --mode semantic
agent-journal search "error handling" --mode hybrid

# 3. Filter results
agent-journal search "authentication" \
  --mode hybrid \
  --agent claude-code \
  --max 5

# 4. Export as JSON
agent-journal search "database query" \
  --mode semantic \
  --robot > results.json
```

## Troubleshooting

### Models not found
```bash
$ agent-journal search "query" --mode semantic

# If models aren't installed, falls back to hash-based embeddings
# To use semantic search, install models first
```

### Model incomplete
```
$ agent-journal models list
  minilm               ✗ Incomplete      45.2 MB
    Missing: tokenizer.json
```
Download missing file to complete installation.

## Integration Points

### Dependency Injection
```csharp
// In Program.cs ConfigureServices()
services.AddSingleton<IEmbeddingProvider>(sp =>
{
    var provider = EmbeddingProviderFactory.TryCreateAsync(config.ModelsPath)
        .GetAwaiter()
        .GetResult();
    return provider;
});
```

### Automatic Fallback
1. Tries ONNX provider if models exist
2. Falls back to hash-based if not
3. No errors, graceful degradation

## Next Steps

- [ ] Implement HTTP download in `models download`
- [ ] Add progress reporting for downloads
- [ ] Implement VectorSearchEngine.SearchAsync()
- [ ] Add hybrid ranking algorithm
- [ ] Support additional model types (e5-small, etc.)
