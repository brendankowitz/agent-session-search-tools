using AgentJournal.Core.Models;
using AgentJournal.Core.Utilities;
using Microsoft.Data.Sqlite;
using System.Text.Json;

namespace AgentJournal.Core.Knowledge;

/// <summary>
/// SQLite implementation of the content repository with FTS5 search
/// </summary>
public class SqliteContentRepository : IContentRepository
{
    private readonly string _connectionString;
    private readonly double _halfLifeDays;

    public SqliteContentRepository(string databasePath, double halfLifeDays = DecayCalculator.DefaultHalfLifeDays)
    {
        if (string.IsNullOrWhiteSpace(databasePath))
        {
            throw new ArgumentException("Database path cannot be null or empty", nameof(databasePath));
        }

        if (halfLifeDays <= 0)
        {
            throw new ArgumentException("Half-life must be positive", nameof(halfLifeDays));
        }

        var builder = new SqliteConnectionStringBuilder
        {
            DataSource = databasePath,
            Mode = SqliteOpenMode.ReadWriteCreate,
            Cache = SqliteCacheMode.Shared,
            Pooling = true
        };
        _connectionString = builder.ToString();
        _halfLifeDays = halfLifeDays;
    }

    /// <summary>
    /// Initializes the database schema
    /// </summary>
    public async Task InitializeAsync(CancellationToken ct = default)
    {
        await using var connection = new SqliteConnection(_connectionString);
        await connection.OpenAsync(ct);

        await using var command = connection.CreateCommand();
        command.CommandText = @"
            -- Content table
            CREATE TABLE IF NOT EXISTS content (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                source TEXT NOT NULL UNIQUE,
                project TEXT,
                tags TEXT,
                created_at TEXT NOT NULL,
                last_reinforced_at TEXT NOT NULL,
                content_hash TEXT NOT NULL
            );

            -- FTS5 virtual table for full-text search
            CREATE VIRTUAL TABLE IF NOT EXISTS content_fts USING fts5(
                title,
                content,
                project,
                content='content',
                content_rowid='rowid'
            );

            -- Triggers to keep FTS index in sync
            CREATE TRIGGER IF NOT EXISTS content_ai AFTER INSERT ON content BEGIN
                INSERT INTO content_fts(rowid, title, content, project)
                VALUES (new.rowid, new.title, new.content, new.project);
            END;

            CREATE TRIGGER IF NOT EXISTS content_ad AFTER DELETE ON content BEGIN
                INSERT INTO content_fts(content_fts, rowid, title, content, project)
                VALUES ('delete', old.rowid, old.title, old.content, old.project);
            END;

            CREATE TRIGGER IF NOT EXISTS content_au AFTER UPDATE ON content BEGIN
                INSERT INTO content_fts(content_fts, rowid, title, content, project)
                VALUES ('delete', old.rowid, old.title, old.content, old.project);
                INSERT INTO content_fts(rowid, title, content, project)
                VALUES (new.rowid, new.title, new.content, new.project);
            END;

            -- Indexes for performance
            CREATE INDEX IF NOT EXISTS idx_content_source ON content(source);
            CREATE INDEX IF NOT EXISTS idx_content_project ON content(project);
            CREATE INDEX IF NOT EXISTS idx_content_last_reinforced ON content(last_reinforced_at);
        ";

        await command.ExecuteNonQueryAsync(ct);

        // Enable WAL mode for concurrent read/write access
        var walCmd = connection.CreateCommand();
        walCmd.CommandText = "PRAGMA journal_mode=WAL; PRAGMA busy_timeout=5000;";
        await walCmd.ExecuteNonQueryAsync(ct);
    }

    /// <summary>
    /// Adds or updates a content entry
    /// </summary>
    public async Task<ContentEntry> AddAsync(ContentEntry entry, CancellationToken ct = default)
    {
        await using var connection = new SqliteConnection(_connectionString);
        await connection.OpenAsync(ct);

        await using var command = connection.CreateCommand();
        command.CommandText = @"
            INSERT INTO content (id, title, content, source, project, tags, created_at, last_reinforced_at, content_hash)
            VALUES (@id, @title, @content, @source, @project, @tags, @created_at, @last_reinforced_at, @content_hash)
            ON CONFLICT(source) DO UPDATE SET
                title = @title,
                content = @content,
                project = @project,
                tags = @tags,
                last_reinforced_at = @last_reinforced_at,
                content_hash = @content_hash;
        ";

        command.Parameters.AddWithValue("@id", entry.Id);
        command.Parameters.AddWithValue("@title", entry.Title);
        command.Parameters.AddWithValue("@content", entry.Content);
        command.Parameters.AddWithValue("@source", entry.Source);
        command.Parameters.AddWithValue("@project", (object?)entry.Project ?? DBNull.Value);
        command.Parameters.AddWithValue("@tags", entry.Tags != null ? JsonSerializer.Serialize(entry.Tags) : DBNull.Value);
        command.Parameters.AddWithValue("@created_at", entry.CreatedAt.ToString("O"));
        command.Parameters.AddWithValue("@last_reinforced_at", entry.LastReinforcedAt.ToString("O"));
        command.Parameters.AddWithValue("@content_hash", entry.ContentHash);

        await command.ExecuteNonQueryAsync(ct);

        return entry;
    }

    /// <summary>
    /// Updates an existing content entry
    /// </summary>
    public async Task<bool> UpdateAsync(ContentEntry entry, CancellationToken ct = default)
    {
        await using var connection = new SqliteConnection(_connectionString);
        await connection.OpenAsync(ct);

        await using var command = connection.CreateCommand();
        command.CommandText = @"
            UPDATE content
            SET title = @title,
                content = @content,
                project = @project,
                tags = @tags,
                last_reinforced_at = @last_reinforced_at,
                content_hash = @content_hash
            WHERE id = @id;
        ";

        command.Parameters.AddWithValue("@id", entry.Id);
        command.Parameters.AddWithValue("@title", entry.Title);
        command.Parameters.AddWithValue("@content", entry.Content);
        command.Parameters.AddWithValue("@project", (object?)entry.Project ?? DBNull.Value);
        command.Parameters.AddWithValue("@tags", entry.Tags != null ? JsonSerializer.Serialize(entry.Tags) : DBNull.Value);
        command.Parameters.AddWithValue("@last_reinforced_at", entry.LastReinforcedAt.ToString("O"));
        command.Parameters.AddWithValue("@content_hash", entry.ContentHash);

        var rowsAffected = await command.ExecuteNonQueryAsync(ct);
        return rowsAffected > 0;
    }

    /// <summary>
    /// Gets a content entry by its ID
    /// </summary>
    public async Task<ContentEntry?> GetByIdAsync(string id, CancellationToken ct = default)
    {
        await using var connection = new SqliteConnection(_connectionString);
        await connection.OpenAsync(ct);

        await using var command = connection.CreateCommand();
        command.CommandText = @"
            SELECT id, title, content, source, project, tags, created_at, last_reinforced_at, content_hash
            FROM content
            WHERE id = @id;
        ";
        command.Parameters.AddWithValue("@id", id);

        await using var reader = await command.ExecuteReaderAsync(ct);
        if (!await reader.ReadAsync(ct))
        {
            return null;
        }

        return ReadContentEntry(reader);
    }

    /// <summary>
    /// Gets a content entry by its source
    /// </summary>
    public async Task<ContentEntry?> GetBySourceAsync(string source, CancellationToken ct = default)
    {
        await using var connection = new SqliteConnection(_connectionString);
        await connection.OpenAsync(ct);

        await using var command = connection.CreateCommand();
        command.CommandText = @"
            SELECT id, title, content, source, project, tags, created_at, last_reinforced_at, content_hash
            FROM content
            WHERE source = @source;
        ";
        command.Parameters.AddWithValue("@source", source);

        await using var reader = await command.ExecuteReaderAsync(ct);
        if (!await reader.ReadAsync(ct))
        {
            return null;
        }

        return ReadContentEntry(reader);
    }

    /// <summary>
    /// Searches content entries using FTS5
    /// </summary>
    public async Task<IReadOnlyList<ContentSearchResult>> SearchAsync(
        string query,
        string? project = null,
        string? sourcePrefix = null,
        string[]? tags = null,
        int maxResults = 10,
        CancellationToken ct = default)
    {
        await using var connection = new SqliteConnection(_connectionString);
        await connection.OpenAsync(ct);

        var results = new List<ContentSearchResult>();

        var whereClauses = new List<string>();
        var parameters = new List<(string Name, object Value)>();

        // Project filter
        if (!string.IsNullOrWhiteSpace(project))
        {
            whereClauses.Add("c.project = @project");
            parameters.Add(("@project", project));
        }

        // Source prefix filter - use substr() to prevent LIKE injection
        if (!string.IsNullOrWhiteSpace(sourcePrefix))
        {
            whereClauses.Add("substr(c.source, 1, length(@sourcePrefix)) = @sourcePrefix");
            parameters.Add(("@sourcePrefix", sourcePrefix));
        }

        // Tags filter
        if (tags != null && tags.Length > 0)
        {
            var tagConditions = new List<string>();
            for (int i = 0; i < tags.Length; i++)
            {
                var paramName = $"@tag{i}";
                tagConditions.Add($"value = {paramName}");
                parameters.Add((paramName, tags[i]));
            }
            whereClauses.Add($"EXISTS (SELECT 1 FROM json_each(c.tags) WHERE {string.Join(" OR ", tagConditions)})");
        }

        // Full-text search query
        if (!string.IsNullOrWhiteSpace(query))
        {
            // Sanitize FTS5 query to prevent injection
            var sanitizedQuery = ContentUtils.SanitizeFts5Query(query);

            var sql = @"
                SELECT c.id, c.title, c.content, c.source, c.project, c.tags, c.created_at, c.last_reinforced_at, c.content_hash,
                       rank
                FROM content_fts
                JOIN content c ON content_fts.rowid = c.rowid
                WHERE content_fts MATCH @query" +
                (whereClauses.Count > 0 ? " AND " + string.Join(" AND ", whereClauses) : "") +
                " ORDER BY rank LIMIT @limit;";

            await using var command = connection.CreateCommand();
            command.CommandText = sql;
            command.Parameters.AddWithValue("@query", sanitizedQuery);
            command.Parameters.AddWithValue("@limit", maxResults);

            foreach (var (name, value) in parameters)
            {
                command.Parameters.AddWithValue(name, value);
            }

            await using var reader = await command.ExecuteReaderAsync(ct);
            while (await reader.ReadAsync(ct))
            {
                var entry = ReadContentEntry(reader);
                var rank = reader.GetDouble(9);

                // Calculate decay factor
                var decayFactor = DecayCalculator.CalculateDecayFactor(entry.LastReinforcedAt.DateTime, _halfLifeDays);

                // Apply decay to score (normalize rank to 0-1 range and apply decay)
                var normalizedScore = Math.Min(1.0, Math.Max(0.0, -rank / 10.0)); // FTS5 rank is negative
                var adjustedScore = DecayCalculator.ApplyDecay(normalizedScore, decayFactor);

                // Get highlight
                var highlight = GetHighlight(entry.Content, query);

                results.Add(new ContentSearchResult(entry, adjustedScore, decayFactor, highlight));
            }
        }
        else
        {
            // No query - just list with filters
            var sql = @"
                SELECT id, title, content, source, project, tags, created_at, last_reinforced_at, content_hash
                FROM content
                WHERE 1=1" +
                (whereClauses.Count > 0 ? " AND " + string.Join(" AND ", whereClauses) : "") +
                " ORDER BY last_reinforced_at DESC LIMIT @limit;";

            await using var command = connection.CreateCommand();
            command.CommandText = sql;
            command.Parameters.AddWithValue("@limit", maxResults);

            foreach (var (name, value) in parameters)
            {
                command.Parameters.AddWithValue(name, value);
            }

            await using var reader = await command.ExecuteReaderAsync(ct);
            while (await reader.ReadAsync(ct))
            {
                var entry = ReadContentEntry(reader);
                var decayFactor = DecayCalculator.CalculateDecayFactor(entry.LastReinforcedAt.DateTime, _halfLifeDays);

                results.Add(new ContentSearchResult(entry, decayFactor, decayFactor, null));
            }
        }

        // Sort by adjusted score and take top results
        return results
            .OrderByDescending(r => r.Score)
            .Take(maxResults)
            .ToList();
    }

    /// <summary>
    /// Lists content entries with optional filtering
    /// </summary>
    public async Task<IReadOnlyList<ContentEntry>> ListAsync(
        string? project = null,
        string? sourcePrefix = null,
        string[]? tags = null,
        int limit = 100,
        CancellationToken ct = default)
    {
        await using var connection = new SqliteConnection(_connectionString);
        await connection.OpenAsync(ct);

        var whereClauses = new List<string>();
        var parameters = new List<(string Name, object Value)>();

        // Project filter
        if (!string.IsNullOrWhiteSpace(project))
        {
            whereClauses.Add("project = @project");
            parameters.Add(("@project", project));
        }

        // Source prefix filter - use substr() to prevent LIKE injection
        if (!string.IsNullOrWhiteSpace(sourcePrefix))
        {
            whereClauses.Add("substr(source, 1, length(@sourcePrefix)) = @sourcePrefix");
            parameters.Add(("@sourcePrefix", sourcePrefix));
        }

        // Tags filter
        if (tags != null && tags.Length > 0)
        {
            var tagConditions = new List<string>();
            for (int i = 0; i < tags.Length; i++)
            {
                var paramName = $"@tag{i}";
                tagConditions.Add($"value = {paramName}");
                parameters.Add((paramName, tags[i]));
            }
            whereClauses.Add($"EXISTS (SELECT 1 FROM json_each(tags) WHERE {string.Join(" OR ", tagConditions)})");
        }

        var sql = @"
            SELECT id, title, content, source, project, tags, created_at, last_reinforced_at, content_hash
            FROM content" +
            (whereClauses.Count > 0 ? " WHERE " + string.Join(" AND ", whereClauses) : "") +
            " ORDER BY last_reinforced_at DESC LIMIT @limit;";

        await using var command = connection.CreateCommand();
        command.CommandText = sql;
        command.Parameters.AddWithValue("@limit", limit);

        foreach (var (name, value) in parameters)
        {
            command.Parameters.AddWithValue(name, value);
        }

        var results = new List<ContentEntry>();
        await using var reader = await command.ExecuteReaderAsync(ct);
        while (await reader.ReadAsync(ct))
        {
            results.Add(ReadContentEntry(reader));
        }

        return results;
    }

    /// <summary>
    /// Deletes a content entry by source
    /// </summary>
    public async Task<bool> DeleteAsync(string source, CancellationToken ct = default)
    {
        await using var connection = new SqliteConnection(_connectionString);
        await connection.OpenAsync(ct);

        await using var command = connection.CreateCommand();
        command.CommandText = "DELETE FROM content WHERE source = @source;";
        command.Parameters.AddWithValue("@source", source);

        var rowsAffected = await command.ExecuteNonQueryAsync(ct);
        return rowsAffected > 0;
    }

    /// <summary>
    /// Deletes content entries matching the specified criteria
    /// </summary>
    public async Task<int> DeleteByCriteriaAsync(
        string? id = null,
        string? source = null,
        string? sourcePrefix = null,
        string? project = null,
        bool deleteAll = false,
        CancellationToken ct = default)
    {
        await using var connection = new SqliteConnection(_connectionString);
        await connection.OpenAsync(ct);

        var whereClauses = new List<string>();
        var parameters = new List<(string Name, object Value)>();

        if (!deleteAll)
        {
            // Build WHERE clause based on criteria
            if (!string.IsNullOrWhiteSpace(id))
            {
                whereClauses.Add("id = @id");
                parameters.Add(("@id", id));
            }

            if (!string.IsNullOrWhiteSpace(source))
            {
                whereClauses.Add("source = @source");
                parameters.Add(("@source", source));
            }

            if (!string.IsNullOrWhiteSpace(sourcePrefix))
            {
                // Use substr() to prevent LIKE injection
                whereClauses.Add("substr(source, 1, length(@sourcePrefix)) = @sourcePrefix");
                parameters.Add(("@sourcePrefix", sourcePrefix));
            }

            if (!string.IsNullOrWhiteSpace(project))
            {
                whereClauses.Add("project = @project");
                parameters.Add(("@project", project));
            }

            if (whereClauses.Count == 0)
            {
                // No criteria specified and deleteAll is false
                return 0;
            }
        }

        var sql = "DELETE FROM content" +
                  (whereClauses.Count > 0 ? " WHERE " + string.Join(" AND ", whereClauses) : "") + ";";

        await using var command = connection.CreateCommand();
        command.CommandText = sql;

        foreach (var (name, value) in parameters)
        {
            command.Parameters.AddWithValue(name, value);
        }

        var rowsAffected = await command.ExecuteNonQueryAsync(ct);
        return rowsAffected;
    }

    /// <summary>
    /// Counts content entries matching the specified criteria
    /// </summary>
    public async Task<int> CountByCriteriaAsync(
        string? id = null,
        string? source = null,
        string? sourcePrefix = null,
        string? project = null,
        bool countAll = false,
        CancellationToken ct = default)
    {
        await using var connection = new SqliteConnection(_connectionString);
        await connection.OpenAsync(ct);

        var whereClauses = new List<string>();
        var parameters = new List<(string Name, object Value)>();

        if (!countAll)
        {
            // Build WHERE clause based on criteria
            if (!string.IsNullOrWhiteSpace(id))
            {
                whereClauses.Add("id = @id");
                parameters.Add(("@id", id));
            }

            if (!string.IsNullOrWhiteSpace(source))
            {
                whereClauses.Add("source = @source");
                parameters.Add(("@source", source));
            }

            if (!string.IsNullOrWhiteSpace(sourcePrefix))
            {
                // Use substr() to prevent LIKE injection
                whereClauses.Add("substr(source, 1, length(@sourcePrefix)) = @sourcePrefix");
                parameters.Add(("@sourcePrefix", sourcePrefix));
            }

            if (!string.IsNullOrWhiteSpace(project))
            {
                whereClauses.Add("project = @project");
                parameters.Add(("@project", project));
            }

            if (whereClauses.Count == 0)
            {
                // No criteria specified and countAll is false
                return 0;
            }
        }

        var sql = "SELECT COUNT(*) FROM content" +
                  (whereClauses.Count > 0 ? " WHERE " + string.Join(" AND ", whereClauses) : "") + ";";

        await using var command = connection.CreateCommand();
        command.CommandText = sql;

        foreach (var (name, value) in parameters)
        {
            command.Parameters.AddWithValue(name, value);
        }

        var result = await command.ExecuteScalarAsync(ct);
        return Convert.ToInt32(result);
    }

    /// <summary>
    /// Reinforces a content entry (resets decay timer)
    /// </summary>
    public async Task<bool> ReinforceAsync(string source, CancellationToken ct = default)
    {
        await using var connection = new SqliteConnection(_connectionString);
        await connection.OpenAsync(ct);

        await using var command = connection.CreateCommand();
        command.CommandText = @"
            UPDATE content
            SET last_reinforced_at = @now
            WHERE source = @source;
        ";
        command.Parameters.AddWithValue("@source", source);
        command.Parameters.AddWithValue("@now", DateTimeOffset.UtcNow.ToString("O"));

        var rowsAffected = await command.ExecuteNonQueryAsync(ct);
        return rowsAffected > 0;
    }

    /// <summary>
    /// Gets expired content entries below the threshold
    /// </summary>
    public async Task<IReadOnlyList<ContentEntry>> GetExpiredAsync(double threshold = 0.05, CancellationToken ct = default)
    {
        await using var connection = new SqliteConnection(_connectionString);
        await connection.OpenAsync(ct);

        // Calculate expiration date based on threshold
        var expirationDays = _halfLifeDays * Math.Log(threshold) / Math.Log(0.5);
        var expirationDate = DateTimeOffset.UtcNow.AddDays(-expirationDays);

        await using var command = connection.CreateCommand();
        command.CommandText = @"
            SELECT id, title, content, source, project, tags, created_at, last_reinforced_at, content_hash
            FROM content
            WHERE datetime(last_reinforced_at) < datetime(@expirationDate)
            ORDER BY last_reinforced_at ASC;
        ";
        command.Parameters.AddWithValue("@expirationDate", expirationDate.ToString("O"));

        var results = new List<ContentEntry>();
        await using var reader = await command.ExecuteReaderAsync(ct);
        while (await reader.ReadAsync(ct))
        {
            results.Add(ReadContentEntry(reader));
        }

        return results;
    }

    /// <summary>
    /// Helper method to read a content entry from a data reader
    /// </summary>
    private static ContentEntry ReadContentEntry(SqliteDataReader reader)
    {
        var id = reader.GetString(0);
        var title = reader.GetString(1);
        var content = reader.GetString(2);
        var source = reader.GetString(3);
        var project = reader.IsDBNull(4) ? null : reader.GetString(4);
        var tagsJson = reader.IsDBNull(5) ? null : reader.GetString(5);
        var tags = tagsJson != null ? JsonSerializer.Deserialize<string[]>(tagsJson) : null;
        var createdAtStr = reader.GetString(6);
        var createdAt = DateTimeOffset.Parse(createdAtStr, System.Globalization.CultureInfo.InvariantCulture);
        var lastReinforcedAtStr = reader.GetString(7);
        var lastReinforcedAt = DateTimeOffset.Parse(lastReinforcedAtStr, System.Globalization.CultureInfo.InvariantCulture);
        var contentHash = reader.GetString(8);

        return new ContentEntry(
            id,
            title,
            content,
            source,
            project,
            tags,
            createdAt,
            lastReinforcedAt,
            contentHash
        );
    }

    /// <summary>
    /// Creates a simple highlight by finding the query terms in the content
    /// </summary>
    private static string? GetHighlight(string content, string query, int maxLength = 200)
    {
        if (string.IsNullOrEmpty(content) || string.IsNullOrEmpty(query))
        {
            return null;
        }

        // Simple highlighting: find query terms and return context
        var queryTerms = query.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        var lowerContent = content.ToLowerInvariant();

        foreach (var term in queryTerms)
        {
            var lowerTerm = term.ToLowerInvariant();
            var index = lowerContent.IndexOf(lowerTerm, StringComparison.OrdinalIgnoreCase);

            if (index >= 0)
            {
                var start = Math.Max(0, index - 50);
                var end = Math.Min(content.Length, index + lowerTerm.Length + 150);
                var highlight = content.Substring(start, end - start);

                if (start > 0) highlight = "..." + highlight;
                if (end < content.Length) highlight += "...";

                return highlight;
            }
        }

        // If no match, return beginning of content
        return content.Length > maxLength
            ? content.Substring(0, maxLength) + "..."
            : content;
    }
}
