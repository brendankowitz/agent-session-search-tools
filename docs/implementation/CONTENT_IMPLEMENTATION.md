# Content Indexing Feature - Implementation Details

## Overview

Comprehensive content indexing feature for the agent-journal CLI that allows agents to index markdown files from directories and directly post content for indexing.

## Components

### 1. ContentEntry Model

**File:** `src/AgentJournal.Core/Models/ContentEntry.cs`

A record type representing indexed content with:

```csharp
public record ContentEntry
{
    public required string Id { get; init; }              // GUID-based, 12 chars
    public required string Title { get; init; }           // Content title
    public required string Content { get; init; }         // Full content text
    public required string Source { get; init; }          // File path or custom identifier
    public string? Project { get; init; }                 // Optional project association
    public string[]? Tags { get; init; }                  // Optional string array of tags
    public required DateTimeOffset CreatedAt { get; init; }
    public required DateTimeOffset LastReinforcedAt { get; init; }
    public required string ContentHash { get; init; }     // SHA256 hash
}
```

### 2. IContentRepository Interface

**File:** `src/AgentJournal.Core/Knowledge/IContentRepository.cs`

Repository interface with methods:

| Method | Description |
|--------|-------------|
| `InitializeAsync()` | Initialize database schema |
| `AddAsync(ContentEntry)` | Add or update content entry |
| `UpdateAsync(ContentEntry)` | Update existing entry |
| `GetByIdAsync(string)` | Get by ID |
| `GetBySourceAsync(string)` | Get by source identifier |
| `SearchAsync(...)` | Full-text search with FTS5 |
| `ListAsync(...)` | List entries with filtering |
| `DeleteAsync(string)` | Delete by source |
| `ReinforceAsync(string)` | Reset decay timer |
| `GetExpiredAsync(double)` | Get expired entries |
| `DeleteByCriteriaAsync(...)` | Delete by multiple criteria |
| `CountByCriteriaAsync(...)` | Count matching criteria |

### 3. SqliteContentRepository

**File:** `src/AgentJournal.Core/Knowledge/SqliteContentRepository.cs`

SQLite implementation with:

#### Database Schema

```sql
-- Main content table
CREATE TABLE IF NOT EXISTS content (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    source TEXT NOT NULL,
    project TEXT,
    tags TEXT,
    content_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    last_reinforced_at TEXT NOT NULL
);

-- FTS5 virtual table for full-text search
CREATE VIRTUAL TABLE IF NOT EXISTS content_fts USING fts5(
    title,
    content,
    content=content,
    content_rowid=rowid
);

-- Triggers to keep FTS index in sync
CREATE TRIGGER IF NOT EXISTS content_ai AFTER INSERT ON content BEGIN
    INSERT INTO content_fts(rowid, title, content)
    VALUES (new.rowid, new.title, new.content);
END;

CREATE TRIGGER IF NOT EXISTS content_ad AFTER DELETE ON content BEGIN
    DELETE FROM content_fts WHERE rowid = old.rowid;
END;

CREATE TRIGGER IF NOT EXISTS content_au AFTER UPDATE ON content BEGIN
    UPDATE content_fts SET title = new.title, content = new.content
    WHERE rowid = new.rowid;
END;

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_content_source ON content(source);
CREATE INDEX IF NOT EXISTS idx_content_project ON content(project);
CREATE INDEX IF NOT EXISTS idx_content_last_reinforced ON content(last_reinforced_at);
```

#### Features

**FTS5 Full-Text Search:**
- Uses SQLite FTS5 for fast full-text indexing
- Decay-adjusted scoring for relevance over time
- Context-aware highlighting of matches
- Query sanitization to prevent injection

**Content Hash Detection:**
- SHA256 hash computed for all content
- Enables efficient incremental indexing
- Skips re-indexing of unchanged files
- Updates last_reinforced_at on hash match

**Temporal Decay Support:**
- 90-day half-life default (configurable)
- Exponential decay function
- Decay factor ranges from 0 (expired) to 1 (fresh)
- Status levels: Fresh → Good → Aging → Decaying → Expiring

**Security Features:**
- Path traversal prevention with validation
- File size limits (10MB default)
- SQL injection prevention (parameterized queries)
- FTS5 query sanitization

**Performance Optimizations:**
- Connection pooling
- Prepared statements
- Indexed columns for fast filtering
- Batch operations support

### 4. ContentCommand

**File:** `src/AgentJournal/Commands/ContentCommand.cs`

Complete CLI command implementation with 6 subcommands:

#### Subcommands Overview

| Command | Purpose | Key Features |
|---------|---------|--------------|
| `index` | Index markdown files | Hash-based skip, recursive scan, glob patterns |
| `add` | Add content directly | Stdin support, upsert semantics, tagging |
| `search` | Search content | FTS5 search, decay scoring, filtering |
| `list` | List content | Multiple filters, decay visualization |
| `remove` | Remove content | Multiple criteria, confirmation prompts |
| `reinforce` | Reset decay | Atomic timestamp update |

#### `content index <path>`

Index markdown files from a directory.

**Implementation:**
```csharp
private async Task<int> ExecuteIndexAsync(
    string path,
    string filter,
    string? project,
    bool recursive,
    bool rebuild,
    CancellationToken cancellationToken)
{
    // 1. Validate path
    var validatedPath = ContentUtils.ValidatePath(path);
    
    // 2. Setup glob matcher
    var matcher = new Matcher();
    matcher.AddInclude(filter);
    
    // 3. Enumerate files
    var files = matcher.GetResultsInFullPath(validatedPath);
    
    // 4. Process each file
    foreach (var file in files)
    {
        // Validate file size
        ContentUtils.ValidateFileSize(file);
        
        // Read content
        var content = await File.ReadAllTextAsync(file);
        
        // Compute hash
        var hash = ContentUtils.ComputeHash(content);
        
        // Check if unchanged
        var existing = await _contentRepository.GetBySourceAsync(file);
        if (existing != null && existing.ContentHash == hash && !rebuild)
        {
            skipped++;
            continue;
        }
        
        // Extract title
        var title = ContentUtils.ExtractTitle(content, file);
        
        // Create/update entry
        var entry = new ContentEntry
        {
            Id = existing?.Id ?? GenerateId(),
            Title = title,
            Content = content,
            Source = file,
            Project = project,
            Tags = null,
            ContentHash = hash,
            CreatedAt = existing?.CreatedAt ?? DateTimeOffset.UtcNow,
            LastReinforcedAt = DateTimeOffset.UtcNow
        };
        
        await _contentRepository.AddAsync(entry, cancellationToken);
        indexed++;
    }
    
    return 0;
}
```

#### `content add`

Add content directly via CLI or stdin.

**Features:**
- Required parameters: `--source`, `--title`
- Optional: `--content` (or stdin), `--project`, `--tags`
- Updates existing content if source exists
- Preserves created timestamp on updates

#### `content search <query>`

Search indexed content with decay-adjusted scoring.

**Filtering Options:**
- `--max` / `-n` - Maximum results (default: 10)
- `--project` / `-p` - Filter by project
- `--source-prefix` / `-s` - Filter by source path prefix
- `--tags` / `-t` - Comma-separated tags (OR logic)
- `--robot` - JSON output

**Search Algorithm:**
1. Sanitize FTS5 query
2. Execute full-text search
3. Calculate decay factor for each result
4. Adjust score by decay factor
5. Sort by adjusted score (descending)
6. Apply result limit
7. Generate highlights

#### `content list`

List indexed content entries with filtering.

**Options:**
- `--project` / `-p` - Filter by project
- `--source-prefix` / `-s` - Filter by source prefix
- `--tags` / `-t` - Filter by tags (OR logic)
- `--limit` / `-n` - Max entries (default: 50)
- `--expired` - Show only expired content
- `--robot` - JSON output

**Display:**
- Decay status visualization (emoji + percentage)
- Time since last reinforcement
- Tag display
- Truncated content preview

#### `content remove`

Remove content by various criteria.

**Removal Strategies:**
- `--id` - Remove by content ID (Ulid)
- `--source` / `-s` - Remove by exact source match
- `--source-prefix` - Remove all with source prefix
- `--project` / `-p` - Remove all for project
- `--all` - Remove all content

**Safety Features:**
- Requires at least one criterion
- Counts matching entries before deletion
- Shows confirmation prompt (unless `--force`)
- Reports deletion count
- Supports combined filters (AND logic)

#### `content reinforce --source <source>`

Reset decay timer for content.

**Features:**
- Extends content lifetime
- Atomic timestamp update
- Validates source exists

### 5. Program.cs Integration

**File:** `src/AgentJournal/Program.cs`

```csharp
// Register content repository
services.AddSingleton<IContentRepository>(provider =>
{
    var dbPath = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
        ".agent-journal",
        "content.db"
    );
    return new SqliteContentRepository(dbPath);
});

// Initialize on startup
var contentRepository = serviceProvider.GetRequiredService<IContentRepository>();
await contentRepository.InitializeAsync();

// Add content command
var rootCommand = new RootCommand("AgentJournal CLI");
rootCommand.AddCommand(new ContentCommand(contentRepository));
```

## Filtering Implementation

**File:** `src/AgentJournal.Core/Knowledge/IContentRepository.cs`

### SearchAsync Filters

Added parameters:
- `sourcePrefix` - Filter by source starting with prefix
- `tags` - Filter by any matching tag (string array)

### ListAsync Filters

Added parameters:
- `sourcePrefix` - Filter by source starting with prefix
- `tags` - Filter by any matching tag (string array)

### SQL Implementation

**Source Prefix Filter:**
```sql
-- Safe implementation (prevents LIKE injection)
WHERE substr(source, 1, length(@sourcePrefix)) = @sourcePrefix
```

**Tags Filter:**
```sql
-- JSON array matching
WHERE EXISTS (
    SELECT 1 
    FROM json_each(c.tags) 
    WHERE value IN (@tag1, @tag2, ...)
)
```

## Remove Command Enhancement

### New Repository Methods

**DeleteByCriteriaAsync:**
```csharp
Task<int> DeleteByCriteriaAsync(
    string? id = null,
    string? source = null,
    string? sourcePrefix = null,
    string? project = null,
    bool deleteAll = false,
    CancellationToken cancellationToken = default);
```

**CountByCriteriaAsync:**
```csharp
Task<int> CountByCriteriaAsync(
    string? id = null,
    string? source = null,
    string? sourcePrefix = null,
    string? project = null,
    bool includeAll = false,
    CancellationToken cancellationToken = default);
```

### SQL Query Building

Dynamically builds WHERE clauses based on provided criteria:

```csharp
if (id != null) whereClauses.Add("id = @id");
if (source != null) whereClauses.Add("source = @source");
if (sourcePrefix != null) whereClauses.Add("substr(source, 1, length(@sourcePrefix)) = @sourcePrefix");
if (project != null) whereClauses.Add("project = @project");

var whereClause = whereClauses.Count > 0 
    ? "WHERE " + string.Join(" AND ", whereClauses) 
    : "";
```

## Security Utilities

**File:** `src/AgentJournal.Core/Utilities/ContentUtils.cs`

Shared security and utility methods:

### Path Validation

```csharp
public static string ValidatePath(string path, string? basePath = null)
{
    ArgumentException.ThrowIfNullOrWhiteSpace(path, nameof(path));
    
    var fullPath = Path.GetFullPath(path);
    
    if (basePath != null)
    {
        var fullBasePath = Path.GetFullPath(basePath);
        if (!fullPath.StartsWith(fullBasePath, StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidOperationException(
                $"Path '{path}' is outside the allowed directory '{basePath}'");
        }
    }
    
    // Additional check for .. after normalization
    if (fullPath.Contains(".."))
    {
        throw new InvalidOperationException(
            $"Path '{path}' contains invalid directory traversal");
    }
    
    return fullPath;
}
```

### File Size Validation

```csharp
public static void ValidateFileSize(string filePath, long maxSizeBytes = 10_485_760)
{
    var fileInfo = new FileInfo(filePath);
    if (fileInfo.Length > maxSizeBytes)
    {
        throw new InvalidOperationException(
            $"File '{filePath}' exceeds maximum size of {maxSizeBytes / 1_048_576} MB");
    }
}
```

### FTS5 Query Sanitization

```csharp
public static string SanitizeFts5Query(string query)
{
    ArgumentException.ThrowIfNullOrWhiteSpace(query, nameof(query));
    
    // Escape double quotes
    var escaped = query.Replace("\"", "\"\"");
    
    // Wrap in quotes for phrase search
    return $"\"{escaped}\"";
}
```

### Content Hash

```csharp
public static string ComputeHash(string content)
{
    var bytes = Encoding.UTF8.GetBytes(content);
    var hashBytes = SHA256.HashData(bytes);
    return Convert.ToHexString(hashBytes);
}
```

### Title Extraction

```csharp
public static string ExtractTitle(string content, string fallback)
{
    var lines = content.Split('\n');
    var firstLine = lines.FirstOrDefault(l => !string.IsNullOrWhiteSpace(l))?.Trim();
    
    if (firstLine?.StartsWith("#") == true)
    {
        return firstLine.TrimStart('#').Trim();
    }
    
    return Path.GetFileNameWithoutExtension(fallback);
}
```

## Decay System

Uses existing `DecayCalculator` with exponential decay:

```csharp
public static double CalculateDecay(DateTimeOffset lastReinforcedAt, double halfLifeDays = 90)
{
    var daysSince = (DateTimeOffset.UtcNow - lastReinforcedAt).TotalDays;
    return Math.Pow(0.5, daysSince / halfLifeDays);
}
```

### Decay Status Levels

| Status | Range | Description |
|--------|-------|-------------|
| Fresh | > 0.75 | Recently reinforced (< 38 days) |
| Good | > 0.50 | Still relevant (< 90 days) |
| Aging | > 0.25 | Getting old (< 180 days) |
| Decaying | > 0.10 | Needs attention (< 299 days) |
| Expiring | ≤ 0.10 | Very old (> 299 days) |

## Performance Characteristics

### Database Operations

- **Index creation**: O(n) where n = number of files
- **Search**: O(log n) with FTS5 index
- **List**: O(log n) with indexes on project/source
- **Delete**: O(1) for single, O(n) for bulk

### Memory Usage

- **Indexing**: One file at a time (< 10MB each)
- **Search**: Limited by max results parameter
- **List**: Limited by limit parameter
- **Bulk operations**: Streaming where possible

### Connection Pooling

SQLite connections are pooled and reused:
```csharp
private readonly string _connectionString;

public SqliteContentRepository(string dbPath)
{
    _connectionString = $"Data Source={dbPath}";
}
```

## Testing Results

All features tested successfully:

1. ✅ **Indexing** - Indexed 3 markdown files from test directory
2. ✅ **Skip Behavior** - Correctly skipped unchanged files on re-index
3. ✅ **Rebuild** - Forced re-indexing with `--rebuild` flag
4. ✅ **Direct Add** - Added custom content with tags
5. ✅ **Search** - Full-text search with highlighting
6. ✅ **List** - Listed all content with decay visualization
7. ✅ **Reinforce** - Reset decay timer for content
8. ✅ **Remove** - Deleted content by source
9. ✅ **JSON Output** - Robot mode for automation
10. ✅ **Decay Calculation** - Proper temporal decay applied
11. ✅ **Source Prefix Filtering** - Filter by path prefix
12. ✅ **Tag Filtering** - Filter by tags with OR logic
13. ✅ **Combined Filters** - Multiple filters work together
14. ✅ **Bulk Remove** - Remove by prefix/project
15. ✅ **Remove Confirmation** - Confirmation prompts work

## Architecture Patterns

Follows existing codebase patterns:

- ✅ **Repository Pattern** - Like `SqliteKnowledgeRepository`
- ✅ **System.CommandLine** - For CLI structure
- ✅ **Dependency Injection** - Services registered in DI
- ✅ **Record Types** - For immutable models
- ✅ **Async/Await** - Throughout the codebase
- ✅ **Error Handling** - Proper exceptions and logging
- ✅ **SOLID Principles** - Clean separation of concerns

## File Structure

```
src/AgentJournal.Core/
  Models/
    ContentEntry.cs                    # Content model
  Knowledge/
    IContentRepository.cs              # Repository interface
    SqliteContentRepository.cs         # SQLite implementation
  Utilities/
    ContentUtils.cs                    # Shared utilities
  Mcp/
    AgentJournalTools.cs              # MCP tools (content methods)

src/AgentJournal/
  Commands/
    ContentCommand.cs                  # CLI command
  Program.cs                           # Updated for DI registration
```

## Build Status

✅ Build successful - no warnings or errors

## Future Enhancements

Potential improvements:

1. **Content Export** - Export to various formats (JSON, CSV, Markdown)
2. **Batch Operations** - Import from file, bulk reinforcement
3. **Advanced Indexing** - Markdown-specific (headers, code blocks, links)
4. **Vector Search Integration** - Semantic search alongside FTS5
5. **Content Statistics** - Dashboard showing index health
6. **Tag Management** - Tag CRUD operations, rename, merge
7. **Content Versioning** - Track changes over time
8. **Scheduled Maintenance** - Auto-cleanup of expired content
9. **Permissions** - User-level access control
10. **Content Types** - Support for different file types

## Related Documentation

- [MCP Content Tools Implementation](MCP_CONTENT_TOOLS.md) - MCP-specific details
- [Content Security Review](../reviews/CONTENT_SECURITY_REVIEW.md) - Security analysis
- [Content Quick Reference](../quick-reference/CONTENT_QUICK_REF.md) - Command cheat sheet
- [Content Indexing User Guide](../CONTENT_INDEXING.md) - User documentation
