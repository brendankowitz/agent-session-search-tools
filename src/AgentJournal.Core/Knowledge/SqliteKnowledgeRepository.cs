using AgentJournal.Core.Models;
using AgentJournal.Core.Search;
using Microsoft.Data.Sqlite;
using System.Text.Json;

namespace AgentJournal.Core.Knowledge;

/// <summary>
/// SQLite implementation of the knowledge repository
/// </summary>
public class SqliteKnowledgeRepository : IKnowledgeRepository
{
    private readonly string _connectionString;
    private readonly double _halfLifeDays;

    public SqliteKnowledgeRepository(string databasePath, double halfLifeDays = DecayCalculator.DefaultHalfLifeDays)
    {
        if (string.IsNullOrWhiteSpace(databasePath))
        {
            throw new ArgumentException("Database path cannot be null or empty", nameof(databasePath));
        }

        if (halfLifeDays <= 0)
        {
            throw new ArgumentException("Half-life must be positive", nameof(halfLifeDays));
        }

        // Enable connection pooling and other optimizations
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
            -- Knowledge table
            CREATE TABLE IF NOT EXISTS knowledge (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                tags TEXT,
                project TEXT,
                source TEXT,
                created_at TEXT NOT NULL,
                last_reinforced_at TEXT NOT NULL,
                reinforcement_count INTEGER DEFAULT 0
            );

            -- FTS5 virtual table for full-text search
            CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_fts USING fts5(
                content,
                tags,
                project,
                content='knowledge',
                content_rowid='rowid'
            );

            -- Triggers to keep FTS index in sync
            CREATE TRIGGER IF NOT EXISTS knowledge_ai AFTER INSERT ON knowledge BEGIN
                INSERT INTO knowledge_fts(rowid, content, tags, project)
                VALUES (new.rowid, new.content, new.tags, new.project);
            END;

            CREATE TRIGGER IF NOT EXISTS knowledge_ad AFTER DELETE ON knowledge BEGIN
                INSERT INTO knowledge_fts(knowledge_fts, rowid, content, tags, project)
                VALUES ('delete', old.rowid, old.content, old.tags, old.project);
            END;

            CREATE TRIGGER IF NOT EXISTS knowledge_au AFTER UPDATE ON knowledge BEGIN
                INSERT INTO knowledge_fts(knowledge_fts, rowid, content, tags, project)
                VALUES ('delete', old.rowid, old.content, old.tags, old.project);
                INSERT INTO knowledge_fts(rowid, content, tags, project)
                VALUES (new.rowid, new.content, new.tags, new.project);
            END;

            -- Indexes for performance
            CREATE INDEX IF NOT EXISTS idx_knowledge_project ON knowledge(project);
            CREATE INDEX IF NOT EXISTS idx_knowledge_last_reinforced ON knowledge(last_reinforced_at);
            CREATE INDEX IF NOT EXISTS idx_knowledge_created ON knowledge(created_at);
        ";

        await command.ExecuteNonQueryAsync(ct);
    }

    /// <summary>
    /// Saves a knowledge entry to the database
    /// </summary>
    public async Task<KnowledgeEntry> SaveAsync(KnowledgeEntry entry, CancellationToken ct = default)
    {
        await using var connection = new SqliteConnection(_connectionString);
        await connection.OpenAsync(ct);

        await using var command = connection.CreateCommand();
        command.CommandText = @"
            INSERT INTO knowledge (id, content, tags, project, source, created_at, last_reinforced_at, reinforcement_count)
            VALUES (@id, @content, @tags, @project, @source, @created_at, @last_reinforced_at, @reinforcement_count)
            ON CONFLICT(id) DO UPDATE SET
                content = @content,
                tags = @tags,
                project = @project,
                source = @source,
                last_reinforced_at = @last_reinforced_at,
                reinforcement_count = @reinforcement_count;
        ";

        command.Parameters.AddWithValue("@id", entry.Id);
        command.Parameters.AddWithValue("@content", entry.Content);
        command.Parameters.AddWithValue("@tags", JsonSerializer.Serialize(entry.Tags));
        command.Parameters.AddWithValue("@project", (object?)entry.Project ?? DBNull.Value);
        command.Parameters.AddWithValue("@source", (object?)entry.Source ?? DBNull.Value);
        command.Parameters.AddWithValue("@created_at", entry.CreatedAt.ToString("O"));
        command.Parameters.AddWithValue("@last_reinforced_at", entry.LastReinforcedAt.ToString("O"));
        command.Parameters.AddWithValue("@reinforcement_count", entry.ReinforcementCount);

        await command.ExecuteNonQueryAsync(ct);

        return entry;
    }

    /// <summary>
    /// Gets a knowledge entry by its ID
    /// </summary>
    public async Task<KnowledgeEntry?> GetAsync(string id, CancellationToken ct = default)
    {
        await using var connection = new SqliteConnection(_connectionString);
        await connection.OpenAsync(ct);

        await using var command = connection.CreateCommand();
        command.CommandText = @"
            SELECT id, content, tags, project, source, created_at, last_reinforced_at, reinforcement_count
            FROM knowledge
            WHERE id = @id;
        ";
        command.Parameters.AddWithValue("@id", id);

        await using var reader = await command.ExecuteReaderAsync(ct);
        if (!await reader.ReadAsync(ct))
        {
            return null;
        }

        return ReadKnowledgeEntry(reader);
    }

    /// <summary>
    /// Searches knowledge entries using FTS5
    /// </summary>
    public async Task<IReadOnlyList<KnowledgeSearchResult>> SearchAsync(
        string query,
        IEnumerable<string>? tags = null,
        string? project = null,
        SearchMode mode = SearchMode.Hybrid,
        int maxResults = 10,
        CancellationToken ct = default)
    {
        await using var connection = new SqliteConnection(_connectionString);
        await connection.OpenAsync(ct);

        var results = new List<KnowledgeSearchResult>();

        // Build query with filters
        var whereClauses = new List<string>();
        var parameters = new List<(string Name, object Value)>();

        // Project filter
        if (!string.IsNullOrWhiteSpace(project))
        {
            whereClauses.Add("k.project = @project");
            parameters.Add(("@project", project));
        }

        // Tags filter
        var tagsList = tags?.ToList();
        if (tagsList != null && tagsList.Count > 0)
        {
            for (int i = 0; i < tagsList.Count; i++)
            {
                whereClauses.Add($"k.tags LIKE @tag{i} ESCAPE '\\'");
                var sanitized = SanitizeLikePattern(tagsList[i]);
                parameters.Add(($"@tag{i}", $"%\"{sanitized}\"%"));
            }
        }

        // Full-text search query
        if (!string.IsNullOrWhiteSpace(query))
        {
            // Use FTS5 for text search
            var sql = @"
                SELECT k.id, k.content, k.tags, k.project, k.source, k.created_at, k.last_reinforced_at, k.reinforcement_count,
                       rank
                FROM knowledge_fts
                JOIN knowledge k ON knowledge_fts.rowid = k.rowid
                WHERE knowledge_fts MATCH @query" +
                (whereClauses.Count > 0 ? " AND " + string.Join(" AND ", whereClauses) : "") +
                " ORDER BY rank LIMIT @limit;";

            await using var command = connection.CreateCommand();
            command.CommandText = sql;
            command.Parameters.AddWithValue("@query", query);
            command.Parameters.AddWithValue("@limit", maxResults);

            foreach (var (name, value) in parameters)
            {
                command.Parameters.AddWithValue(name, value);
            }

            await using var reader = await command.ExecuteReaderAsync(ct);
            while (await reader.ReadAsync(ct))
            {
                var entry = ReadKnowledgeEntry(reader);
                var rank = reader.GetDouble(8);

                // Calculate decay factor
                var decayFactor = DecayCalculator.CalculateDecayFactor(entry.LastReinforcedAt, _halfLifeDays);

                // Apply decay to score (normalize rank to 0-1 range and apply decay)
                var normalizedScore = Math.Min(1.0, Math.Max(0.0, -rank / 10.0)); // FTS5 rank is negative
                var adjustedScore = DecayCalculator.ApplyDecay(normalizedScore, decayFactor);

                // Get highlight
                var highlight = GetHighlight(entry.Content, query);

                results.Add(new KnowledgeSearchResult(entry, adjustedScore, decayFactor, highlight));
            }
        }
        else
        {
            // No query - just list with filters
            var sql = @"
                SELECT id, content, tags, project, source, created_at, last_reinforced_at, reinforcement_count
                FROM knowledge
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
                var entry = ReadKnowledgeEntry(reader);
                var decayFactor = DecayCalculator.CalculateDecayFactor(entry.LastReinforcedAt, _halfLifeDays);

                results.Add(new KnowledgeSearchResult(entry, decayFactor, decayFactor, null));
            }
        }

        // Sort by adjusted score and take top results
        // Re-sort because SQL LIMIT was applied before score adjustment
        return results
            .OrderByDescending(r => r.Score)
            .Take(maxResults)
            .ToList();
    }

    /// <summary>
    /// Deletes a knowledge entry
    /// </summary>
    public async Task<bool> DeleteAsync(string id, CancellationToken ct = default)
    {
        if (string.IsNullOrWhiteSpace(id))
        {
            throw new ArgumentException("ID cannot be null or empty", nameof(id));
        }

        await using var connection = new SqliteConnection(_connectionString);
        await connection.OpenAsync(ct);

        await using var command = connection.CreateCommand();
        command.CommandText = "DELETE FROM knowledge WHERE id = @id;";
        command.Parameters.AddWithValue("@id", id);

        var rowsAffected = await command.ExecuteNonQueryAsync(ct);
        return rowsAffected > 0;
    }

    /// <summary>
    /// Deletes multiple knowledge entries in a single transaction
    /// More efficient than calling DeleteAsync multiple times
    /// </summary>
    public async Task<int> DeleteManyAsync(IEnumerable<string> ids, CancellationToken ct = default)
    {
        var idList = ids?.ToList() ?? throw new ArgumentNullException(nameof(ids));
        if (idList.Count == 0)
        {
            return 0;
        }

        await using var connection = new SqliteConnection(_connectionString);
        await connection.OpenAsync(ct);

        await using var transaction = (SqliteTransaction)await connection.BeginTransactionAsync(ct);
        try
        {
            int deleted = 0;

            // Batch delete in chunks of 500 to avoid parameter limits
            const int batchSize = 500;
            for (int i = 0; i < idList.Count; i += batchSize)
            {
                var batch = idList.Skip(i).Take(batchSize).ToList();
                var placeholders = string.Join(",", batch.Select((_, idx) => $"@id{idx}"));

                await using var command = connection.CreateCommand();
                command.Transaction = transaction;
                command.CommandText = $"DELETE FROM knowledge WHERE id IN ({placeholders});";

                for (int j = 0; j < batch.Count; j++)
                {
                    command.Parameters.AddWithValue($"@id{j}", batch[j]);
                }

                deleted += await command.ExecuteNonQueryAsync(ct);
            }

            await transaction.CommitAsync(ct);
            return deleted;
        }
        catch
        {
            await transaction.RollbackAsync(ct);
            throw;
        }
    }

    /// <summary>
    /// Reinforces multiple knowledge entries in a single transaction
    /// More efficient than calling ReinforceAsync multiple times
    /// </summary>
    public async Task<int> ReinforceManyAsync(IEnumerable<string> ids, CancellationToken ct = default)
    {
        var idList = ids?.ToList() ?? throw new ArgumentNullException(nameof(ids));
        if (idList.Count == 0)
        {
            return 0;
        }

        await using var connection = new SqliteConnection(_connectionString);
        await connection.OpenAsync(ct);

        await using var transaction = (SqliteTransaction)await connection.BeginTransactionAsync(ct);
        try
        {
            int reinforced = 0;
            var now = DateTime.UtcNow.ToString("O");

            // Batch update in chunks of 500 to avoid parameter limits
            const int batchSize = 500;
            for (int i = 0; i < idList.Count; i += batchSize)
            {
                var batch = idList.Skip(i).Take(batchSize).ToList();
                var placeholders = string.Join(",", batch.Select((_, idx) => $"@id{idx}"));

                await using var command = connection.CreateCommand();
                command.Transaction = transaction;
                command.CommandText = $@"
                    UPDATE knowledge
                    SET last_reinforced_at = @now,
                        reinforcement_count = reinforcement_count + 1
                    WHERE id IN ({placeholders});
                ";
                command.Parameters.AddWithValue("@now", now);

                for (int j = 0; j < batch.Count; j++)
                {
                    command.Parameters.AddWithValue($"@id{j}", batch[j]);
                }

                reinforced += await command.ExecuteNonQueryAsync(ct);
            }

            await transaction.CommitAsync(ct);
            return reinforced;
        }
        catch
        {
            await transaction.RollbackAsync(ct);
            throw;
        }
    }

    /// <summary>
    /// Reinforces a knowledge entry (resets decay timer)
    /// Thread-safe: Uses SQL atomic increment to prevent race conditions
    /// </summary>
    public async Task<bool> ReinforceAsync(string id, CancellationToken ct = default)
    {
        if (string.IsNullOrWhiteSpace(id))
        {
            throw new ArgumentException("ID cannot be null or empty", nameof(id));
        }

        await using var connection = new SqliteConnection(_connectionString);
        await connection.OpenAsync(ct);

        await using var transaction = (SqliteTransaction)await connection.BeginTransactionAsync(ct);
        try
        {
            await using var command = connection.CreateCommand();
            command.Transaction = transaction;
            command.CommandText = @"
                UPDATE knowledge
                SET last_reinforced_at = @now,
                    reinforcement_count = reinforcement_count + 1
                WHERE id = @id;
            ";
            command.Parameters.AddWithValue("@id", id);
            command.Parameters.AddWithValue("@now", DateTime.UtcNow.ToString("O"));

            var rowsAffected = await command.ExecuteNonQueryAsync(ct);
            await transaction.CommitAsync(ct);

            return rowsAffected > 0;
        }
        catch
        {
            await transaction.RollbackAsync(ct);
            throw;
        }
    }

    /// <summary>
    /// Lists knowledge entries with optional filtering
    /// </summary>
    public async Task<IReadOnlyList<KnowledgeEntry>> ListAsync(
        string? project = null,
        IEnumerable<string>? tags = null,
        bool includeDecaying = true,
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

        // Tags filter
        var tagsList = tags?.ToList();
        if (tagsList != null && tagsList.Count > 0)
        {
            for (int i = 0; i < tagsList.Count; i++)
            {
                whereClauses.Add($"tags LIKE @tag{i} ESCAPE '\\'");
                var sanitized = SanitizeLikePattern(tagsList[i]);
                parameters.Add(($"@tag{i}", $"%\"{sanitized}\"%"));
            }
        }

        var sql = @"
            SELECT id, content, tags, project, source, created_at, last_reinforced_at, reinforcement_count
            FROM knowledge" +
            (whereClauses.Count > 0 ? " WHERE " + string.Join(" AND ", whereClauses) : "") +
            " ORDER BY last_reinforced_at DESC LIMIT @limit;";

        await using var command = connection.CreateCommand();
        command.CommandText = sql;
        command.Parameters.AddWithValue("@limit", limit);

        foreach (var (name, value) in parameters)
        {
            command.Parameters.AddWithValue(name, value);
        }

        var results = new List<KnowledgeEntry>();
        await using var reader = await command.ExecuteReaderAsync(ct);
        while (await reader.ReadAsync(ct))
        {
            var entry = ReadKnowledgeEntry(reader);

            // Filter by decay if requested
            if (!includeDecaying)
            {
                var decayFactor = DecayCalculator.CalculateDecayFactor(entry.LastReinforcedAt, _halfLifeDays);
                if (decayFactor < 0.5)
                {
                    continue;
                }
            }

            results.Add(entry);
        }

        return results;
    }

    /// <summary>
    /// Gets statistics about the knowledge bank
    /// Optimized to stream results without loading all into memory
    /// </summary>
    public async Task<KnowledgeStats> GetStatsAsync(CancellationToken ct = default)
    {
        await using var connection = new SqliteConnection(_connectionString);
        await connection.OpenAsync(ct);

        // Get lightweight data needed for stats (no content)
        await using var command = connection.CreateCommand();
        command.CommandText = @"
            SELECT tags, project, last_reinforced_at
            FROM knowledge;
        ";

        var entriesByProject = new Dictionary<string, int>();
        var entriesByTag = new Dictionary<string, int>();

        int total = 0;
        int fresh = 0;
        int good = 0;
        int aging = 0;
        int decaying = 0;
        int expiring = 0;

        await using var reader = await command.ExecuteReaderAsync(ct);
        while (await reader.ReadAsync(ct))
        {
            total++;

            // Calculate decay
            var lastReinforcedStr = reader.GetString(2);
            var lastReinforced = DateTime.ParseExact(lastReinforcedStr, "O", System.Globalization.CultureInfo.InvariantCulture);
            var decayFactor = DecayCalculator.CalculateDecayFactor(lastReinforced, _halfLifeDays);

            if (decayFactor > 0.75) fresh++;
            else if (decayFactor > 0.50) good++;
            else if (decayFactor > 0.25) aging++;
            else if (decayFactor > 0.10) decaying++;
            else expiring++;

            // Count by project
            var projectKey = reader.IsDBNull(1) ? "(global)" : reader.GetString(1);
            entriesByProject[projectKey] = entriesByProject.GetValueOrDefault(projectKey, 0) + 1;

            // Count by tags
            if (!reader.IsDBNull(0))
            {
                var tagsJson = reader.GetString(0);
                var tags = JsonSerializer.Deserialize<string[]>(tagsJson) ?? Array.Empty<string>();
                foreach (var tag in tags)
                {
                    entriesByTag[tag] = entriesByTag.GetValueOrDefault(tag, 0) + 1;
                }
            }
        }

        return new KnowledgeStats(
            TotalEntries: total,
            FreshEntries: fresh,
            GoodEntries: good,
            AgingEntries: aging,
            DecayingEntries: decaying,
            ExpiringEntries: expiring,
            EntriesByProject: entriesByProject,
            EntriesByTag: entriesByTag
        );
    }

    /// <summary>
    /// Prunes expired knowledge entries below the threshold
    /// </summary>
    public async Task<int> PruneExpiredAsync(double threshold = 0.05, CancellationToken ct = default)
    {
        await using var connection = new SqliteConnection(_connectionString);
        await connection.OpenAsync(ct);

        // Calculate expiration date based on threshold
        // If threshold is 0.05, and half-life is 90 days, solve: 0.05 = 0.5^(days/90)
        // days = 90 * log(threshold) / log(0.5)
        var expirationDays = _halfLifeDays * Math.Log(threshold) / Math.Log(0.5);
        var expirationDate = DateTime.UtcNow.AddDays(-expirationDays);

        // Delete in a single SQL operation with transaction
        await using var transaction = (SqliteTransaction)await connection.BeginTransactionAsync(ct);
        try
        {
            await using var deleteCommand = connection.CreateCommand();
            deleteCommand.Transaction = transaction;
            deleteCommand.CommandText = @"
                DELETE FROM knowledge 
                WHERE datetime(last_reinforced_at) < datetime(@expirationDate);
            ";
            deleteCommand.Parameters.AddWithValue("@expirationDate", expirationDate.ToString("O"));

            var deleted = await deleteCommand.ExecuteNonQueryAsync(ct);
            await transaction.CommitAsync(ct);

            return deleted;
        }
        catch
        {
            await transaction.RollbackAsync(ct);
            throw;
        }
    }

    /// <summary>
    /// Sanitizes input for use in SQL LIKE patterns to prevent SQL injection
    /// </summary>
    private static string SanitizeLikePattern(string input)
    {
        if (string.IsNullOrEmpty(input)) return input;
        if (input.Length > 100 || input.Any(c => char.IsControl(c)))
            throw new ArgumentException("Invalid tag format", nameof(input));
        return input
            .Replace("\\", "\\\\")
            .Replace("%", "\\%")
            .Replace("_", "\\_")
            .Replace("[", "\\[");
    }

    /// <summary>
    /// Helper method to read a knowledge entry from a data reader
    /// Uses explicit culture-invariant parsing for DateTime values
    /// </summary>
    private static KnowledgeEntry ReadKnowledgeEntry(SqliteDataReader reader)
    {
        var id = reader.GetString(0);
        var content = reader.GetString(1);
        var tagsJson = reader.IsDBNull(2) ? "[]" : reader.GetString(2);
        var tags = JsonSerializer.Deserialize<string[]>(tagsJson) ?? Array.Empty<string>();
        var project = reader.IsDBNull(3) ? null : reader.GetString(3);
        var source = reader.IsDBNull(4) ? null : reader.GetString(4);

        // Use ParseExact with InvariantCulture for consistent DateTime parsing
        var createdAtStr = reader.GetString(5);
        var createdAt = DateTime.ParseExact(createdAtStr, "O", System.Globalization.CultureInfo.InvariantCulture);

        var lastReinforcedAtStr = reader.GetString(6);
        var lastReinforced = DateTime.ParseExact(lastReinforcedAtStr, "O", System.Globalization.CultureInfo.InvariantCulture);

        var reinforcementCount = reader.GetInt32(7);

        return new KnowledgeEntry(
            id,
            content,
            tags,
            project,
            source,
            createdAt,
            lastReinforced,
            reinforcementCount
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
