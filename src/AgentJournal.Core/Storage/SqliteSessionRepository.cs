using AgentJournal.Core.Models;
using Microsoft.Data.Sqlite;
using System.Runtime.CompilerServices;

namespace AgentJournal.Core.Storage;

/// <summary>
/// SQLite implementation of the session repository
/// </summary>
public class SqliteSessionRepository : ISessionRepository
{
    private readonly string _connectionString;

    public SqliteSessionRepository(string databasePath)
    {
        _connectionString = $"Data Source={databasePath}";
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
            -- Sessions table
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                agent_type TEXT NOT NULL,
                project_path TEXT,
                git_branch TEXT,
                agent_version TEXT,
                started_at TEXT NOT NULL,
                ended_at TEXT,
                summary TEXT
            );

            -- Messages table
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                role INTEGER NOT NULL,
                content TEXT NOT NULL,
                raw_content TEXT,
                timestamp TEXT NOT NULL,
                parent_id TEXT,
                model TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
            );

            -- Tool calls table
            CREATE TABLE IF NOT EXISTS tool_calls (
                id TEXT PRIMARY KEY,
                message_id TEXT NOT NULL,
                name TEXT NOT NULL,
                arguments TEXT,
                result TEXT,
                success INTEGER,
                FOREIGN KEY (message_id) REFERENCES messages(id) ON DELETE CASCADE
            );

            -- Indexes for performance
            CREATE INDEX IF NOT EXISTS idx_sessions_agent_type ON sessions(agent_type);
            CREATE INDEX IF NOT EXISTS idx_sessions_project_path ON sessions(project_path);
            CREATE INDEX IF NOT EXISTS idx_sessions_started_at ON sessions(started_at);
            CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id);
            CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp);
            CREATE INDEX IF NOT EXISTS idx_tool_calls_message_id ON tool_calls(message_id);
        ";

        await command.ExecuteNonQueryAsync(ct);

        // Migration: Add last_modified column if it doesn't exist
        try 
        {
            var alterCmd = connection.CreateCommand();
            alterCmd.CommandText = "ALTER TABLE sessions ADD COLUMN last_modified TEXT;";
            await alterCmd.ExecuteNonQueryAsync(ct);
        }
        catch (SqliteException) 
        {
            // Column likely already exists, ignore
        }
    }

    /// <summary>
    /// Saves a session to the database
    /// </summary>
    public async Task SaveSessionAsync(Session session, CancellationToken ct = default)
    {
        await using var connection = new SqliteConnection(_connectionString);
        await connection.OpenAsync(ct);

        await using var transaction = (SqliteTransaction)await connection.BeginTransactionAsync(ct);

        try
        {
            // Upsert session
            await using (var command = connection.CreateCommand())
            {
                command.Transaction = transaction;
                command.CommandText = @"
                    INSERT INTO sessions (id, agent_type, project_path, git_branch, agent_version, started_at, ended_at, last_modified, summary)
                    VALUES (@id, @agent_type, @project_path, @git_branch, @agent_version, @started_at, @ended_at, @last_modified, @summary)
                    ON CONFLICT(id) DO UPDATE SET
                        agent_type = @agent_type,
                        project_path = @project_path,
                        git_branch = @git_branch,
                        agent_version = @agent_version,
                        started_at = @started_at,
                        ended_at = @ended_at,
                        last_modified = @last_modified,
                        summary = @summary;
                ";

                command.Parameters.AddWithValue("@id", session.Id);
                command.Parameters.AddWithValue("@agent_type", session.AgentType);
                command.Parameters.AddWithValue("@project_path", (object?)session.ProjectPath ?? DBNull.Value);
                command.Parameters.AddWithValue("@git_branch", (object?)session.GitBranch ?? DBNull.Value);
                command.Parameters.AddWithValue("@agent_version", (object?)session.AgentVersion ?? DBNull.Value);
                command.Parameters.AddWithValue("@started_at", session.StartedAt.ToString("O"));
                command.Parameters.AddWithValue("@ended_at", session.EndedAt?.ToString("O") ?? (object)DBNull.Value);
                command.Parameters.AddWithValue("@last_modified", session.LastModified?.ToString("O") ?? (object)DBNull.Value);
                command.Parameters.AddWithValue("@summary", (object?)session.Summary ?? DBNull.Value);

                await command.ExecuteNonQueryAsync(ct);
            }

            // Delete existing messages and tool calls for this session to avoid duplicates
            await using (var command = connection.CreateCommand())
            {
                command.Transaction = transaction;
                command.CommandText = "DELETE FROM messages WHERE session_id = @session_id;";
                command.Parameters.AddWithValue("@session_id", session.Id);
                await command.ExecuteNonQueryAsync(ct);
            }

            // Insert messages
            foreach (var message in session.Messages)
            {
                await using var command = connection.CreateCommand();
                command.Transaction = transaction;
                command.CommandText = @"
                    INSERT INTO messages (id, session_id, role, content, raw_content, timestamp, parent_id, model)
                    VALUES (@id, @session_id, @role, @content, @raw_content, @timestamp, @parent_id, @model);
                ";

                command.Parameters.AddWithValue("@id", message.Id);
                command.Parameters.AddWithValue("@session_id", message.SessionId);
                command.Parameters.AddWithValue("@role", (int)message.Role);
                command.Parameters.AddWithValue("@content", message.Content);
                command.Parameters.AddWithValue("@raw_content", (object?)message.RawContent ?? DBNull.Value);
                command.Parameters.AddWithValue("@timestamp", message.Timestamp.ToString("O"));
                command.Parameters.AddWithValue("@parent_id", (object?)message.ParentId ?? DBNull.Value);
                command.Parameters.AddWithValue("@model", (object?)message.Model ?? DBNull.Value);

                await command.ExecuteNonQueryAsync(ct);

                // Insert tool calls for this message
                if (message.ToolCalls != null)
                {
                    foreach (var toolCall in message.ToolCalls)
                    {
                        await using var tcCommand = connection.CreateCommand();
                        tcCommand.Transaction = transaction;
                        tcCommand.CommandText = @"
                            INSERT INTO tool_calls (id, message_id, name, arguments, result, success)
                            VALUES (@id, @message_id, @name, @arguments, @result, @success);
                        ";

                        tcCommand.Parameters.AddWithValue("@id", toolCall.Id);
                        tcCommand.Parameters.AddWithValue("@message_id", toolCall.MessageId);
                        tcCommand.Parameters.AddWithValue("@name", toolCall.Name);
                        tcCommand.Parameters.AddWithValue("@arguments", (object?)toolCall.Arguments ?? DBNull.Value);
                        tcCommand.Parameters.AddWithValue("@result", (object?)toolCall.Result ?? DBNull.Value);
                        tcCommand.Parameters.AddWithValue("@success", toolCall.Success.HasValue ? (object)(toolCall.Success.Value ? 1 : 0) : DBNull.Value);

                        await tcCommand.ExecuteNonQueryAsync(ct);
                    }
                }
            }

            await transaction.CommitAsync(ct);
        }
        catch
        {
            await transaction.RollbackAsync(ct);
            throw;
        }
    }

    /// <summary>
    /// Saves multiple sessions to the database
    /// </summary>
    public async Task SaveSessionsAsync(IEnumerable<Session> sessions, CancellationToken ct = default)
    {
        foreach (var session in sessions)
        {
            await SaveSessionAsync(session, ct);
        }
    }

    /// <summary>
    /// Gets a session by its ID
    /// </summary>
    public async Task<Session?> GetSessionAsync(string sessionId, CancellationToken ct = default)
    {
        await using var connection = new SqliteConnection(_connectionString);
        await connection.OpenAsync(ct);

        // Get session
        await using var sessionCommand = connection.CreateCommand();
        sessionCommand.CommandText = @"
            SELECT id, agent_type, project_path, git_branch, agent_version, started_at, ended_at, summary, last_modified
            FROM sessions
            WHERE id = @id;
        ";
        sessionCommand.Parameters.AddWithValue("@id", sessionId);

        await using var sessionReader = await sessionCommand.ExecuteReaderAsync(ct);
        if (!await sessionReader.ReadAsync(ct))
        {
            return null;
        }

        var id = sessionReader.GetString(0);
        var agentType = sessionReader.GetString(1);
        var projectPath = sessionReader.IsDBNull(2) ? null : sessionReader.GetString(2);
        var gitBranch = sessionReader.IsDBNull(3) ? null : sessionReader.GetString(3);
        var agentVersion = sessionReader.IsDBNull(4) ? null : sessionReader.GetString(4);
        var startedAt = DateTime.Parse(sessionReader.GetString(5)).ToUniversalTime();
        var endedAt = sessionReader.IsDBNull(6) ? (DateTime?)null : DateTime.Parse(sessionReader.GetString(6)).ToUniversalTime();
        var summary = sessionReader.IsDBNull(7) ? null : sessionReader.GetString(7);
        var lastModified = sessionReader.IsDBNull(8) ? (DateTime?)null : DateTime.Parse(sessionReader.GetString(8)).ToUniversalTime();

        await sessionReader.CloseAsync();

        // Get messages
        var messages = new List<Message>();
        await using var messageCommand = connection.CreateCommand();
        messageCommand.CommandText = @"
            SELECT id, session_id, role, content, raw_content, timestamp, parent_id, model
            FROM messages
            WHERE session_id = @session_id
            ORDER BY timestamp;
        ";
        messageCommand.Parameters.AddWithValue("@session_id", sessionId);

        await using var messageReader = await messageCommand.ExecuteReaderAsync(ct);
        while (await messageReader.ReadAsync(ct))
        {
            var messageId = messageReader.GetString(0);
            var msgSessionId = messageReader.GetString(1);
            var role = (MessageRole)messageReader.GetInt32(2);
            var content = messageReader.GetString(3);
            var rawContent = messageReader.IsDBNull(4) ? null : messageReader.GetString(4);
            var timestamp = DateTime.Parse(messageReader.GetString(5));
            var parentId = messageReader.IsDBNull(6) ? null : messageReader.GetString(6);
            var model = messageReader.IsDBNull(7) ? null : messageReader.GetString(7);

            // Get tool calls for this message
            var toolCalls = await GetToolCallsForMessageAsync(connection, messageId, ct);

            messages.Add(new Message(
                messageId,
                msgSessionId,
                role,
                content,
                rawContent,
                timestamp,
                parentId,
                model,
                toolCalls
            ));
        }

        return new Session(
            id,
            agentType,
            projectPath,
            gitBranch,
            agentVersion,
            startedAt,
            endedAt,
            lastModified,
            summary,
            messages
        );
    }

    /// <summary>
    /// Gets the last modified timestamp for a session
    /// </summary>
    public async Task<DateTime?> GetSessionLastModifiedAsync(string sessionId, CancellationToken ct = default)
    {
        await using var connection = new SqliteConnection(_connectionString);
        await connection.OpenAsync(ct);

        await using var command = connection.CreateCommand();
        command.CommandText = @"
            SELECT last_modified
            FROM sessions
            WHERE id = @id;
        ";
        command.Parameters.AddWithValue("@id", sessionId);

        var result = await command.ExecuteScalarAsync(ct);
        
        if (result == null || result == DBNull.Value)
        {
            return null;
        }

        // Ensure we return UTC for consistent comparison
        return DateTime.Parse((string)result).ToUniversalTime();
    }

    /// <summary>
    /// Gets all sessions from the database
    /// </summary>
    public async IAsyncEnumerable<Session> GetAllSessionsAsync([EnumeratorCancellation] CancellationToken ct = default)
    {
        await using var connection = new SqliteConnection(_connectionString);
        await connection.OpenAsync(ct);

        await using var command = connection.CreateCommand();
        command.CommandText = @"
            SELECT id
            FROM sessions
            ORDER BY started_at DESC;
        ";

        await using var reader = await command.ExecuteReaderAsync(ct);
        while (await reader.ReadAsync(ct))
        {
            var sessionId = reader.GetString(0);
            var session = await GetSessionAsync(sessionId, ct);
            if (session != null)
            {
                yield return session;
            }
        }
    }

    /// <summary>
    /// Gets sessions filtered by agent type
    /// </summary>
    public async IAsyncEnumerable<Session> GetSessionsByAgentTypeAsync(string agentType, [EnumeratorCancellation] CancellationToken ct = default)
    {
        await using var connection = new SqliteConnection(_connectionString);
        await connection.OpenAsync(ct);

        await using var command = connection.CreateCommand();
        command.CommandText = @"
            SELECT id
            FROM sessions
            WHERE agent_type = @agent_type
            ORDER BY started_at DESC;
        ";
        command.Parameters.AddWithValue("@agent_type", agentType);

        await using var reader = await command.ExecuteReaderAsync(ct);
        while (await reader.ReadAsync(ct))
        {
            var sessionId = reader.GetString(0);
            var session = await GetSessionAsync(sessionId, ct);
            if (session != null)
            {
                yield return session;
            }
        }
    }

    /// <summary>
    /// Deletes a session from the database
    /// </summary>
    public async Task DeleteSessionAsync(string sessionId, CancellationToken ct = default)
    {
        await using var connection = new SqliteConnection(_connectionString);
        await connection.OpenAsync(ct);

        await using var command = connection.CreateCommand();
        command.CommandText = @"
            DELETE FROM sessions WHERE id = @id;
        ";
        command.Parameters.AddWithValue("@id", sessionId);

        await command.ExecuteNonQueryAsync(ct);
    }

    /// <summary>
    /// Helper method to get tool calls for a message
    /// </summary>
    private static async Task<IReadOnlyList<ToolCall>?> GetToolCallsForMessageAsync(
        SqliteConnection connection,
        string messageId,
        CancellationToken ct)
    {
        var toolCalls = new List<ToolCall>();

        await using var command = connection.CreateCommand();
        command.CommandText = @"
            SELECT id, message_id, name, arguments, result, success
            FROM tool_calls
            WHERE message_id = @message_id;
        ";
        command.Parameters.AddWithValue("@message_id", messageId);

        await using var reader = await command.ExecuteReaderAsync(ct);
        while (await reader.ReadAsync(ct))
        {
            var id = reader.GetString(0);
            var msgId = reader.GetString(1);
            var name = reader.GetString(2);
            var arguments = reader.IsDBNull(3) ? null : reader.GetString(3);
            var result = reader.IsDBNull(4) ? null : reader.GetString(4);
            bool? success = reader.IsDBNull(5) ? null : reader.GetInt32(5) == 1;

            toolCalls.Add(new ToolCall(id, msgId, name, arguments, result, success));
        }

        return toolCalls.Count > 0 ? toolCalls : null;
    }
}
