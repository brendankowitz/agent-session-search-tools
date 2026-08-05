using Microsoft.Data.Sqlite;

namespace AgentJournal.Core.Storage;

/// <summary>
/// Builds connection strings and opens connections with the settings every Agent Journal
/// database depends on.
/// </summary>
/// <remarks>
/// Two settings here are load-bearing and were previously missing:
/// <list type="bullet">
/// <item>
/// <description>
/// <c>ForeignKeys</c>: SQLite ignores <c>ON DELETE CASCADE</c> unless foreign key enforcement is
/// switched on per connection. Without it, deleting a session's messages orphaned its
/// <c>tool_calls</c> rows, and re-indexing an updated session then failed with a primary key
/// collision on those orphans.
/// </description>
/// </item>
/// <item>
/// <description>
/// <c>busy_timeout</c>: a PRAGMA is connection-scoped, so setting it once during schema creation
/// left every subsequent connection at the default of 0. Concurrent readers and writers (the CLI
/// indexing while an MCP server serves queries) failed immediately with "database is locked"
/// instead of waiting.
/// </description>
/// </item>
/// </list>
/// Shared cache mode is also deliberately avoided: combined with WAL it produces table-level
/// SQLITE_LOCKED errors that no busy timeout can retry away.
/// </remarks>
public static class SqliteConnectionFactory
{
    /// <summary>
    /// How long a connection waits for a competing writer before failing, in milliseconds.
    /// </summary>
    public const int BusyTimeoutMs = 10_000;

    /// <summary>
    /// Builds a connection string for an Agent Journal database file.
    /// </summary>
    /// <param name="databasePath">Path to the SQLite database file.</param>
    public static string BuildConnectionString(string databasePath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(databasePath);

        return new SqliteConnectionStringBuilder
        {
            DataSource = databasePath,
            Mode = SqliteOpenMode.ReadWriteCreate,
            Pooling = true,
            ForeignKeys = true
        }.ToString();
    }

    /// <summary>
    /// Opens a connection and applies the per-connection PRAGMAs.
    /// </summary>
    /// <param name="connectionString">Connection string produced by <see cref="BuildConnectionString"/>.</param>
    /// <param name="ct">Cancellation token.</param>
    public static async Task<SqliteConnection> OpenAsync(string connectionString, CancellationToken ct = default)
    {
        var connection = new SqliteConnection(connectionString);
        try
        {
            await connection.OpenAsync(ct);

            await using var pragma = connection.CreateCommand();
            pragma.CommandText = $"PRAGMA busy_timeout={BusyTimeoutMs};";
            await pragma.ExecuteNonQueryAsync(ct);

            return connection;
        }
        catch
        {
            await connection.DisposeAsync();
            throw;
        }
    }
}
