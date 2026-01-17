# Vector Search CLI - Usage Examples

## Getting Started

### 1. Check Current Status
```bash
# View all available commands
agent-journal --help

# Check installed models
agent-journal models list
```

### 2. Install an Embedding Model

Currently, model download requires manual installation:

```bash
# Create models directory
mkdir -p ~/.agent-journal/models/minilm

# Download model files from HuggingFace
# Visit: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
# Download: model.onnx, tokenizer.json

# Place files in: ~/.agent-journal/models/minilm/
```

Verify installation:
```bash
agent-journal models list

# Expected output:
# Installed Embedding Models:
# Models Path: ~/.agent-journal/models
#
#   minilm               ✓ Ready          89.5 MB
```

### 3. Index Your Sessions

```bash
# Index all agent sessions
agent-journal index

# Index specific agent type
agent-journal index --agent claude-code

# Rebuild index from scratch
agent-journal index --rebuild
```

## Search Examples

### Basic Lexical Search
```bash
# Default mode is lexical (keyword-based)
agent-journal search "authentication"
agent-journal search "database connection error"
```

### Semantic Search
```bash
# Use semantic mode for conceptual matching
agent-journal search "login problems" --mode semantic
agent-journal search "how to improve performance" -m semantic
```

### Hybrid Search
```bash
# Best of both worlds - combines lexical and semantic
agent-journal search "API rate limiting" --mode hybrid
agent-journal search "memory leak debugging" -m hybrid
```

### Advanced Search Options

#### Limit Results
```bash
agent-journal search "error handling" --max 5
agent-journal search "authentication" -n 10
```

#### Include Context Messages
```bash
# Include 5 surrounding messages
agent-journal search "bug fix" --context 5 -c 5
```

#### Filter by Agent Type
```bash
agent-journal search "typescript" --agent claude-code
agent-journal search "git workflow" -a copilot-cli
```

#### Filter by Project
```bash
agent-journal search "database" --project "/path/to/myapp"
agent-journal search "API" -p "backend"
```

#### JSON Output (for scripting)
```bash
agent-journal search "performance" --robot > results.json
agent-journal search "security" -r | jq '.[] | .sessionId'
```

#### Combined Options
```bash
agent-journal search "authentication issue" \
  --mode hybrid \
  --context 5 \
  --max 10 \
  --agent claude-code \
  --project "/home/user/myapp"
```

## Model Management Examples

### List Models
```bash
agent-journal models list

# Example output:
# Installed Embedding Models:
# Models Path: ~/.agent-journal/models
#
#   minilm               ✓ Ready          89.5 MB
#   e5-small             ✗ Incomplete     45.2 MB
#     Missing: tokenizer.json
```

### Download Model
```bash
# Note: Currently shows manual installation instructions
agent-journal models download minilm

# Expected output:
# Downloading model: minilm
# Target directory: ~/.agent-journal/models/minilm
# 
# Note: Model download is not yet implemented.
# 
# To manually install the model:
# 1. Download the model files from HuggingFace:
#    https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
# 2. Place the following files in: ~/.agent-journal/models/minilm
#    - model.onnx
#    - tokenizer.json
```

### Remove Model
```bash
agent-journal models remove minilm

# Confirmation prompt:
# Remove model: minilm
# Size: 89.5 MB
# Are you sure? (y/N):
```

## Real-World Scenarios

### Scenario 1: Finding Sessions About a Specific Topic
```bash
# Find all sessions discussing authentication
agent-journal search "authentication implementation" \
  --mode hybrid \
  --max 20

# Filter to specific project
agent-journal search "authentication" \
  --mode hybrid \
  --project "my-app" \
  --context 3
```

### Scenario 2: Debugging an Error
```bash
# Search for error messages
agent-journal search "TypeError: Cannot read property" \
  --mode lexical \
  --max 10

# Find similar errors semantically
agent-journal search "property access error" \
  --mode semantic \
  --context 5
```

### Scenario 3: Learning from Past Solutions
```bash
# Find how you solved similar problems
agent-journal search "optimize database queries" \
  --mode semantic \
  --agent claude-code

# Get detailed context
agent-journal search "caching strategy" \
  --mode hybrid \
  --context 10 \
  --max 5
```

### Scenario 4: Exporting Results for Analysis
```bash
# Export search results as JSON
agent-journal search "API design" \
  --mode hybrid \
  --robot \
  --max 50 > api_sessions.json

# Process with jq
agent-journal search "performance" -r | jq '
  .[] | {
    session: .sessionId,
    agent: .agentType,
    score: .score,
    project: .projectPath
  }
'
```

### Scenario 5: Cross-Agent Analysis
```bash
# Compare how different agents handle the same topic

# Claude Code sessions
agent-journal search "typescript generics" \
  --agent claude-code \
  --mode semantic

# Copilot CLI sessions
agent-journal search "typescript generics" \
  --agent copilot-cli \
  --mode semantic
```

## Scripting Examples

### Bash Script: Search Multiple Terms
```bash
#!/bin/bash

TERMS=(
  "authentication"
  "database"
  "performance"
  "error handling"
)

for term in "${TERMS[@]}"; do
  echo "=== Searching for: $term ==="
  agent-journal search "$term" \
    --mode hybrid \
    --max 5 \
    --robot | jq '.[] | {session: .sessionId, score: .score}'
  echo ""
done
```

### PowerShell Script: Export All Results
```powershell
# Search multiple topics and export
$topics = @(
    "authentication",
    "API design",
    "database optimization",
    "error handling"
)

foreach ($topic in $topics) {
    $filename = "search_$($topic -replace ' ', '_').json"
    
    agent-journal search $topic `
        --mode hybrid `
        --max 20 `
        --robot | Out-File $filename
    
    Write-Host "Exported: $filename"
}
```

### Python Script: Analyze Search Results
```python
#!/usr/bin/env python3
import subprocess
import json

def search_sessions(query, mode='hybrid', max_results=10):
    """Search sessions using agent-journal CLI"""
    result = subprocess.run(
        ['agent-journal', 'search', query, 
         '--mode', mode, 
         '--max', str(max_results),
         '--robot'],
        capture_output=True,
        text=True
    )
    return json.loads(result.stdout)

# Usage
results = search_sessions('authentication', mode='semantic')
for result in results:
    print(f"Session: {result['sessionId']}")
    print(f"Score: {result['score']}")
    print(f"Project: {result.get('projectPath', 'N/A')}")
    print()
```

## Performance Tips

### 1. Choose the Right Search Mode
- **Lexical**: Fastest, best for exact terms, error codes, file names
- **Semantic**: Slower but better for conceptual queries
- **Hybrid**: Best accuracy but slower, recommended for most searches

### 2. Limit Result Count
```bash
# Faster searches with fewer results
agent-journal search "query" --max 5

# Full results when needed
agent-journal search "query" --max 100
```

### 3. Use Filters Early
```bash
# Filter by agent and project to reduce search space
agent-journal search "query" \
  --agent claude-code \
  --project "specific-app" \
  --mode hybrid
```

## Configuration

### Models Directory Structure
```
~/.agent-journal/
├── models/
│   ├── minilm/
│   │   ├── model.onnx
│   │   └── tokenizer.json
│   └── e5-small/
│       ├── model.onnx
│       └── tokenizer.json
├── lucene-index/
├── agent-journal.db
└── config.json
```

### Default Search Mode
The default search mode is `lexical` for backward compatibility.
To change the default, you would modify the configuration (feature pending).

## Troubleshooting

### No Results Found
```bash
# Try different search modes
agent-journal search "query" --mode lexical
agent-journal search "query" --mode semantic
agent-journal search "query" --mode hybrid

# Try broader terms
agent-journal search "auth"  # instead of "authentication flow implementation"
```

### Models Not Working
```bash
# Check model installation
agent-journal models list

# Verify files exist
ls -lh ~/.agent-journal/models/minilm/

# Expected files:
# - model.onnx
# - tokenizer.json
```

### Slow Searches
```bash
# Use lexical mode for speed
agent-journal search "query" --mode lexical

# Reduce result count
agent-journal search "query" --max 5

# Rebuild index if corrupted
agent-journal index --rebuild
```

## Next Steps

1. **Index your sessions**: `agent-journal index`
2. **Install models**: Follow manual installation guide
3. **Try searches**: Start with lexical, then try semantic/hybrid
4. **Experiment**: Test different modes to see what works best

## Additional Resources

- **Implementation Details**: See `VECTOR_SEARCH_CLI_IMPLEMENTATION.md`
- **Quick Reference**: See `VECTOR_SEARCH_CLI_QUICK_REF.md`
- **Architecture**: See project documentation
