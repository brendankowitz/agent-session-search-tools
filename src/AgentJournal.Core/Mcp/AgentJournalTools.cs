using System.ComponentModel;
using System.Security.Cryptography;
using System.Text;
using AgentJournal.Core.Knowledge;
using AgentJournal.Core.Models;
using AgentJournal.Core.Search;
using AgentJournal.Core.Storage;
using AgentJournal.Core.Utilities;
using ModelContextProtocol.Server;
using Microsoft.Extensions.FileSystemGlobbing;

namespace AgentJournal.Core.Mcp;

/// <summary>
/// MCP tools for Agent Journal - provides session and knowledge search capabilities
/// </summary>
[McpServerToolType]
public class AgentJournalTools
{
    private readonly ISearchEngine _searchEngine;
    private readonly ISessionRepository _sessionRepository;
    private readonly IKnowledgeRepository _knowledgeRepository;
    private readonly IContentRepository _contentRepository;

    public AgentJournalTools(
        ISearchEngine searchEngine,
        ISessionRepository sessionRepository,
        IKnowledgeRepository knowledgeRepository,
        IContentRepository contentRepository)
    {
        _searchEngine = searchEngine;
        _sessionRepository = sessionRepository;
        _knowledgeRepository = knowledgeRepository;
        _contentRepository = contentRepository;
    }

    #region Session Tools

    /// <summary>
    /// Search agent session history for relevant conversations.
    /// </summary>
    /// <param name="query">Search query to find relevant sessions</param>
    /// <param name="mode">Search mode: lexical (keyword), semantic (meaning), or hybrid (both)</param>
    /// <param name="project">Filter by project path or name</param>
    /// <param name="limit">Maximum number of results to return</param>
    /// <param name="around">Number of messages before and after matches to include for context (0 to disable)</param>
    [McpServerTool]
    public async Task<SearchSessionsResult> SearchSessions(
        string query,
        string mode = "hybrid",
        string? project = null,
        int limit = 10,
        int around = 0)
    {
        var searchMode = mode.ToLowerInvariant() switch
        {
            "semantic" => SearchMode.Semantic,
            "lexical" => SearchMode.Lexical,
            _ => SearchMode.Hybrid
        };

        var contextCount = Math.Clamp(around, 0, 50);
        var results = await _searchEngine.SearchAsync(query, searchMode, Math.Clamp(limit, 1, 100), contextCount);

        // Filter by project if specified
        var filtered = results.AsEnumerable();
        if (!string.IsNullOrWhiteSpace(project))
        {
            filtered = filtered.Where(r => 
                r.Session.ProjectPath?.Contains(project, StringComparison.OrdinalIgnoreCase) == true);
        }

        var sessions = filtered.Take(limit).Select(r => new SessionSummary
        {
            Id = r.Session.Id,
            AgentType = r.Session.AgentType,
            ProjectPath = r.Session.ProjectPath,
            StartedAt = r.Session.StartedAt,
            MessageCount = r.Session.MessageCount,
            Score = r.Score,
            Preview = GetSessionPreview(r.Session, r.Highlight),
            MatchingMessages = r.MatchingMessages?.Select(m => new MessageSummary
            {
                Role = m.Role.ToString(),
                Content = m.Content,
                Timestamp = m.Timestamp
            }).ToList()
        }).ToList();

        return new SearchSessionsResult
        {
            Query = query,
            Mode = searchMode.ToString(),
            TotalResults = sessions.Count,
            Sessions = sessions
        };
    }

    /// <summary>
    /// Get full session details including all messages.
    /// </summary>
    /// <param name="id">Session ID to retrieve</param>
    [McpServerTool]
    public async Task<SessionDetails> GetSession(string id)
    {
        var session = await _sessionRepository.GetSessionAsync(id);
        if (session == null)
        {
            throw new ArgumentException($"Session with ID '{id}' not found.", nameof(id));
        }

        return new SessionDetails
        {
            Id = session.Id,
            AgentType = session.AgentType,
            ProjectPath = session.ProjectPath,
            GitBranch = session.GitBranch,
            AgentVersion = session.AgentVersion,
            StartedAt = session.StartedAt,
            EndedAt = session.EndedAt,
            Summary = session.Summary,
            MessageCount = session.MessageCount,
            Messages = session.Messages.Select(m => new MessageInfo
            {
                Role = m.Role.ToString(),
                Content = m.Content,
                Timestamp = m.Timestamp,
                ToolCallCount = m.ToolCalls?.Count ?? 0
            }).ToList()
        };
    }

    /// <summary>
    /// List recent sessions with optional filtering.
    /// </summary>
    /// <param name="limit">Maximum number of sessions to return</param>
    /// <param name="project">Filter by project path or name</param>
    [McpServerTool]
    public async Task<ListSessionsResult> ListRecentSessions(
        int limit = 10,
        string? project = null)
    {
        var sessions = new List<Session>();
        await foreach (var session in _sessionRepository.GetAllSessionsAsync())
        {
            if (!string.IsNullOrWhiteSpace(project))
            {
                if (session.ProjectPath?.Contains(project, StringComparison.OrdinalIgnoreCase) != true)
                {
                    continue;
                }
            }
            sessions.Add(session);
        }

        var recent = sessions
            .OrderByDescending(s => s.StartedAt)
            .Take(Math.Clamp(limit, 1, 100))
            .Select(s => new SessionSummary
            {
                Id = s.Id,
                AgentType = s.AgentType,
                ProjectPath = s.ProjectPath,
                StartedAt = s.StartedAt,
                MessageCount = s.MessageCount,
                Score = 1.0,
                Preview = s.Summary ?? GetSessionPreview(s, null)
            })
            .ToList();

        return new ListSessionsResult
        {
            TotalResults = recent.Count,
            Sessions = recent
        };
    }

    #endregion

    #region Knowledge Tools

    /// <summary>
    /// Store knowledge in the knowledge bank.
    /// </summary>
    /// <param name="content">Content to remember</param>
    /// <param name="tags">Tags for categorization (comma-separated)</param>
    /// <param name="project">Associated project path</param>
    /// <param name="source">Source of this knowledge</param>
    [McpServerTool]
    public async Task<RememberResult> Remember(
        string content,
        string? tags = null,
        string? project = null,
        string? source = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(content, nameof(content));

        var tagArray = tags?.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            ?? Array.Empty<string>();

        var entry = new KnowledgeEntry(
            Id: Guid.NewGuid().ToString(),
            Content: content,
            Tags: tagArray,
            Project: project,
            Source: source,
            CreatedAt: DateTime.UtcNow,
            LastReinforcedAt: DateTime.UtcNow,
            ReinforcementCount: 0
        );

        var saved = await _knowledgeRepository.SaveAsync(entry);

        return new RememberResult
        {
            Id = saved.Id,
            Success = true,
            Message = "Knowledge stored successfully"
        };
    }

    /// <summary>
    /// Search knowledge bank for relevant information.
    /// </summary>
    /// <param name="query">Search query to find relevant knowledge</param>
    /// <param name="tags">Filter by tags (comma-separated)</param>
    /// <param name="project">Filter by project path</param>
    /// <param name="mode">Search mode: lexical, semantic, or hybrid</param>
    /// <param name="limit">Maximum number of results</param>
    [McpServerTool]
    public async Task<RecallResult> Recall(
        string query,
        string? tags = null,
        string? project = null,
        string mode = "hybrid",
        int limit = 10)
    {
        var searchMode = mode.ToLowerInvariant() switch
        {
            "semantic" => SearchMode.Semantic,
            "lexical" => SearchMode.Lexical,
            _ => SearchMode.Hybrid
        };

        var tagArray = tags?.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);

        var results = await _knowledgeRepository.SearchAsync(
            query,
            tagArray,
            project,
            searchMode,
            Math.Clamp(limit, 1, 100)
        );

        var entries = results.Select(r => new KnowledgeInfo
        {
            Id = r.Entry.Id,
            Content = r.Entry.Content,
            Tags = r.Entry.Tags,
            Project = r.Entry.Project,
            Source = r.Entry.Source,
            CreatedAt = r.Entry.CreatedAt,
            LastReinforcedAt = r.Entry.LastReinforcedAt,
            ReinforcementCount = r.Entry.ReinforcementCount,
            Score = r.Score,
            DecayFactor = r.DecayFactor,
            DaysSinceReinforcement = r.Entry.DaysSinceReinforcement
        }).ToList();

        return new RecallResult
        {
            Query = query,
            Mode = searchMode.ToString(),
            TotalResults = entries.Count,
            Entries = entries
        };
    }

    /// <summary>
    /// Reinforce knowledge entries to prevent decay.
    /// </summary>
    /// <param name="ids">Knowledge entry IDs to reinforce (comma-separated)</param>
    [McpServerTool]
    public async Task<ReinforceResult> Reinforce(string ids)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(ids, nameof(ids));

        var idArray = ids.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        var reinforced = new List<string>();
        var failed = new List<string>();

        foreach (var id in idArray)
        {
            var success = await _knowledgeRepository.ReinforceAsync(id);
            if (success)
            {
                reinforced.Add(id);
            }
            else
            {
                failed.Add(id);
            }
        }

        return new ReinforceResult
        {
            ReinforcedCount = reinforced.Count,
            FailedCount = failed.Count,
            ReinforcedIds = reinforced,
            FailedIds = failed
        };
    }

    /// <summary>
    /// Remove knowledge from the knowledge bank.
    /// </summary>
    /// <param name="id">Knowledge entry ID to remove</param>
    [McpServerTool]
    public async Task<ForgetResult> Forget(string id)
    {
        var deleted = await _knowledgeRepository.DeleteAsync(id);

        return new ForgetResult
        {
            Id = id,
            Success = deleted,
            Message = deleted ? "Knowledge removed successfully" : "Knowledge entry not found"
        };
    }

    #endregion

    #region Content Tools

    /// <summary>
    /// Index markdown files from a directory path.
    /// </summary>
    /// <param name="path">Directory path to scan for markdown files</param>
    /// <param name="filter">Glob pattern for file matching (default: *.md)</param>
    /// <param name="project">Project name to associate with indexed content</param>
    /// <param name="recursive">Recursively scan subdirectories (default: true)</param>
    [McpServerTool]
    public async Task<IndexContentResult> IndexContent(
        string path,
        string filter = "*.md",
        string? project = null,
        bool recursive = true)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path, nameof(path));

        // Validate path to prevent directory traversal
        var validatedPath = ContentUtils.ValidatePath(path);

        if (!Directory.Exists(validatedPath))
        {
            throw new ArgumentException($"Directory not found: {validatedPath}", nameof(path));
        }

        var matcher = new Matcher();
        matcher.AddInclude(filter);

        var searchOption = recursive ? SearchOption.AllDirectories : SearchOption.TopDirectoryOnly;
        var files = Directory.GetFiles(validatedPath, "*", searchOption);

        var matchedFiles = matcher.Match(Path.GetFullPath(validatedPath), files.Select(f => Path.GetRelativePath(validatedPath, f)))
            .Files
            .Select(f => Path.Combine(validatedPath, f.Path))
            .ToList();

        int indexed = 0;
        int skipped = 0;
        var errors = new List<string>();

        foreach (var file in matchedFiles)
        {
            try
            {
                // Validate file size before reading
                ContentUtils.ValidateFileSize(file);

                var fileContent = await File.ReadAllTextAsync(file);
                var contentHash = ContentUtils.ComputeHash(fileContent);

                // Check if file already indexed and unchanged
                var existing = await _contentRepository.GetBySourceAsync(file);
                if (existing != null && existing.ContentHash == contentHash)
                {
                    skipped++;
                    continue;
                }

                // Extract title from first line or filename
                var title = ContentUtils.ExtractTitle(fileContent, file);

                var entry = new ContentEntry(
                    Id: Guid.NewGuid().ToString("N")[..12],
                    Title: title,
                    Content: fileContent,
                    Source: file,
                    Project: project,
                    Tags: null,
                    CreatedAt: existing?.CreatedAt ?? DateTimeOffset.UtcNow,
                    LastReinforcedAt: DateTimeOffset.UtcNow,
                    ContentHash: contentHash
                );

                await _contentRepository.AddAsync(entry);
                indexed++;
            }
            catch (Exception ex)
            {
                errors.Add($"{Path.GetFileName(file)}: {ex.Message}");
            }
        }

        return new IndexContentResult
        {
            TotalFiles = matchedFiles.Count,
            Indexed = indexed,
            Skipped = skipped,
            Errors = errors,
            Success = errors.Count == 0
        };
    }

    /// <summary>
    /// Add content directly without reading from a file.
    /// </summary>
    /// <param name="source">Source identifier for this content</param>
    /// <param name="title">Content title</param>
    /// <param name="content">Content text</param>
    /// <param name="project">Project name</param>
    /// <param name="tags">Tags for categorization</param>
    [McpServerTool]
    public async Task<AddContentResult> AddContent(
        string source,
        string title,
        string content,
        string? project = null,
        string[]? tags = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(source, nameof(source));
        ArgumentException.ThrowIfNullOrWhiteSpace(title, nameof(title));
        ArgumentException.ThrowIfNullOrWhiteSpace(content, nameof(content));

        var contentHash = ContentUtils.ComputeHash(content);

        // Check if source already exists
        var existing = await _contentRepository.GetBySourceAsync(source);

        var entry = new ContentEntry(
            Id: existing?.Id ?? Guid.NewGuid().ToString("N")[..12],
            Title: title,
            Content: content,
            Source: source,
            Project: project,
            Tags: tags,
            CreatedAt: existing?.CreatedAt ?? DateTimeOffset.UtcNow,
            LastReinforcedAt: DateTimeOffset.UtcNow,
            ContentHash: contentHash
        );

        await _contentRepository.AddAsync(entry);

        return new AddContentResult
        {
            Id = entry.Id,
            Source = source,
            IsUpdate = existing != null,
            Success = true
        };
    }

    /// <summary>
    /// Search indexed content using full-text search.
    /// </summary>
    /// <param name="query">Search query text</param>
    /// <param name="maxResults">Maximum number of results (default: 10)</param>
    /// <param name="project">Filter by project</param>
    /// <param name="sourcePrefix">Filter by source path prefix</param>
    /// <param name="tags">Filter by tags</param>
    [McpServerTool]
    public async Task<SearchContentResult> SearchContent(
        string query,
        int maxResults = 10,
        string? project = null,
        string? sourcePrefix = null,
        string[]? tags = null)
    {
        maxResults = Math.Clamp(maxResults, 1, 100);

        var results = await _contentRepository.SearchAsync(
            query,
            project,
            sourcePrefix,
            tags,
            maxResults
        );

        var items = results.Select(r => new ContentInfo
        {
            Id = r.Entry.Id,
            Title = r.Entry.Title,
            Source = r.Entry.Source,
            Project = r.Entry.Project,
            Tags = r.Entry.Tags ?? Array.Empty<string>(),
            Score = r.Score,
            DecayFactor = r.DecayFactor,
            DaysSinceReinforcement = r.Entry.DaysSinceReinforcement,
            Highlight = r.Highlight,
            CreatedAt = r.Entry.CreatedAt,
            LastReinforcedAt = r.Entry.LastReinforcedAt
        }).ToList();

        return new SearchContentResult
        {
            Query = query,
            TotalResults = items.Count,
            Results = items
        };
    }

    /// <summary>
    /// List indexed content with optional filtering.
    /// </summary>
    /// <param name="project">Filter by project</param>
    /// <param name="sourcePrefix">Filter by source path prefix</param>
    /// <param name="tags">Filter by tags</param>
    /// <param name="limit">Maximum number of entries to return (default: 50)</param>
    /// <param name="expiredOnly">Show only expired content (default: false)</param>
    [McpServerTool]
    public async Task<ListContentResult> ListContent(
        string? project = null,
        string? sourcePrefix = null,
        string[]? tags = null,
        int limit = 50,
        bool expiredOnly = false)
    {
        limit = Math.Clamp(limit, 1, 1000);

        IReadOnlyList<ContentEntry> entries;

        if (expiredOnly)
        {
            entries = await _contentRepository.GetExpiredAsync(0.05);
        }
        else
        {
            entries = await _contentRepository.ListAsync(project, sourcePrefix, tags, limit);
        }

        var items = entries.Select(e => new ContentInfo
        {
            Id = e.Id,
            Title = e.Title,
            Source = e.Source,
            Project = e.Project,
            Tags = e.Tags ?? Array.Empty<string>(),
            Score = 0,
            DecayFactor = DecayCalculator.CalculateDecayFactor(e.LastReinforcedAt.DateTime),
            DaysSinceReinforcement = e.DaysSinceReinforcement,
            Highlight = null,
            CreatedAt = e.CreatedAt,
            LastReinforcedAt = e.LastReinforcedAt
        }).ToList();

        return new ListContentResult
        {
            TotalResults = items.Count,
            IsExpiredOnly = expiredOnly,
            Content = items
        };
    }

    /// <summary>
    /// Remove content by various criteria.
    /// </summary>
    /// <param name="id">Remove by content ID</param>
    /// <param name="source">Remove by exact source match</param>
    /// <param name="sourcePrefix">Remove all content where source starts with prefix</param>
    /// <param name="project">Remove all content for a project</param>
    [McpServerTool]
    public async Task<RemoveContentResult> RemoveContent(
        string? id = null,
        string? source = null,
        string? sourcePrefix = null,
        string? project = null)
    {
        // Validate that at least one criteria is specified
        if (string.IsNullOrWhiteSpace(id) && string.IsNullOrWhiteSpace(source) &&
            string.IsNullOrWhiteSpace(sourcePrefix) && string.IsNullOrWhiteSpace(project))
        {
            throw new ArgumentException("At least one removal criteria must be specified (id, source, sourcePrefix, or project)");
        }

        // Count and delete
        var count = await _contentRepository.CountByCriteriaAsync(id, source, sourcePrefix, project);
        var deleted = await _contentRepository.DeleteByCriteriaAsync(id, source, sourcePrefix, project);

        return new RemoveContentResult
        {
            RemovedCount = deleted,
            Success = deleted > 0
        };
    }

    /// <summary>
    /// Reinforce content to reset its decay timer.
    /// </summary>
    /// <param name="source">Source identifier of content to reinforce</param>
    [McpServerTool]
    public async Task<ReinforceContentResult> ReinforceContent(string source)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(source, nameof(source));

        var reinforced = await _contentRepository.ReinforceAsync(source);

        return new ReinforceContentResult
        {
            Source = source,
            Success = reinforced,
            Message = reinforced ? "Content reinforced successfully" : "Content not found"
        };
    }

    #endregion

    #region Unified Search

    /// <summary>
    /// Search both sessions and knowledge in a unified way.
    /// </summary>
    /// <param name="query">Search query</param>
    /// <param name="mode">Search mode: lexical, semantic, or hybrid</param>
    /// <param name="includeKnowledge">Include knowledge entries in results</param>
    /// <param name="project">Filter by project path</param>
    /// <param name="limit">Maximum total results</param>
    [McpServerTool]
    public async Task<UnifiedSearchResult> Search(
        string query,
        string mode = "hybrid",
        bool includeKnowledge = true,
        string? project = null,
        int limit = 20)
    {
        var searchMode = mode.ToLowerInvariant() switch
        {
            "semantic" => SearchMode.Semantic,
            "lexical" => SearchMode.Lexical,
            _ => SearchMode.Hybrid
        };

        var results = new List<object>();

        // Search sessions
        var sessionResults = await _searchEngine.SearchAsync(query, searchMode, limit);
        var filtered = sessionResults.AsEnumerable();
        
        if (!string.IsNullOrWhiteSpace(project))
        {
            filtered = filtered.Where(r => 
                r.Session.ProjectPath?.Contains(project, StringComparison.OrdinalIgnoreCase) == true);
        }

        results.AddRange(filtered.Select(r => new
        {
            type = "session",
            id = r.Session.Id,
            agentType = r.Session.AgentType,
            projectPath = r.Session.ProjectPath,
            startedAt = r.Session.StartedAt,
            messageCount = r.Session.MessageCount,
            score = r.Score,
            preview = GetSessionPreview(r.Session, r.Highlight)
        }));

        // Search knowledge if requested
        if (includeKnowledge)
        {
            var knowledgeResults = await _knowledgeRepository.SearchAsync(
                query,
                null,
                project,
                searchMode,
                limit
            );

            results.AddRange(knowledgeResults.Select(r => new
            {
                type = "knowledge",
                id = r.Entry.Id,
                content = r.Entry.Content,
                tags = r.Entry.Tags,
                project = r.Entry.Project,
                source = r.Entry.Source,
                createdAt = r.Entry.CreatedAt,
                score = r.Score,
                decayFactor = r.DecayFactor
            }));
        }

        return new UnifiedSearchResult
        {
            Query = query,
            Mode = searchMode.ToString(),
            TotalResults = results.Count,
            Results = results.OrderByDescending(r =>
            {
                var obj = r as dynamic;
                return obj?.score ?? 0.0;
            }).Take(limit).ToList()
        };
    }

    #endregion

    #region Helper Methods

    private static string GetSessionPreview(Session session, string? highlight)
    {
        if (!string.IsNullOrWhiteSpace(highlight))
        {
            return highlight.Length > 200 ? highlight[..200] + "..." : highlight;
        }

        if (!string.IsNullOrWhiteSpace(session.Summary))
        {
            return session.Summary.Length > 200 ? session.Summary[..200] + "..." : session.Summary;
        }

        // Get first user or assistant message as preview
        var firstMessage = session.Messages.FirstOrDefault(m => 
            m.Role == MessageRole.User || m.Role == MessageRole.Assistant);
        
        if (firstMessage != null)
        {
            var content = firstMessage.Content;
            return content.Length > 200 ? content[..200] + "..." : content;
        }

        return $"Session with {session.MessageCount} messages";
    }



    #endregion
}

#region Result Types

public class SearchSessionsResult
{
    public required string Query { get; init; }
    public required string Mode { get; init; }
    public required int TotalResults { get; init; }
    public required List<SessionSummary> Sessions { get; init; }
}

public class SessionSummary
{
    public required string Id { get; init; }
    public required string AgentType { get; init; }
    public required string? ProjectPath { get; init; }
    public required DateTime StartedAt { get; init; }
    public required int MessageCount { get; init; }
    public required double Score { get; init; }
    public required string Preview { get; init; }
    public List<MessageSummary>? MatchingMessages { get; init; }
}

public class SessionDetails
{
    public required string Id { get; init; }
    public required string AgentType { get; init; }
    public required string? ProjectPath { get; init; }
    public required string? GitBranch { get; init; }
    public required string? AgentVersion { get; init; }
    public required DateTime StartedAt { get; init; }
    public required DateTime? EndedAt { get; init; }
    public required string? Summary { get; init; }
    public required int MessageCount { get; init; }
    public required List<MessageInfo> Messages { get; init; }
}

public class MessageInfo
{
    public required string Role { get; init; }
    public required string Content { get; init; }
    public required DateTime Timestamp { get; init; }
    public required int ToolCallCount { get; init; }
}

public class MessageSummary
{
    public required string Role { get; init; }
    public required string Content { get; init; }
    public required DateTime Timestamp { get; init; }
}

public class ListSessionsResult
{
    public required int TotalResults { get; init; }
    public required List<SessionSummary> Sessions { get; init; }
}

public class RememberResult
{
    public required string Id { get; init; }
    public required bool Success { get; init; }
    public required string Message { get; init; }
}

public class RecallResult
{
    public required string Query { get; init; }
    public required string Mode { get; init; }
    public required int TotalResults { get; init; }
    public required List<KnowledgeInfo> Entries { get; init; }
}

public class KnowledgeInfo
{
    public required string Id { get; init; }
    public required string Content { get; init; }
    public required string[] Tags { get; init; }
    public required string? Project { get; init; }
    public required string? Source { get; init; }
    public required DateTime CreatedAt { get; init; }
    public required DateTime LastReinforcedAt { get; init; }
    public required int ReinforcementCount { get; init; }
    public required double Score { get; init; }
    public required double DecayFactor { get; init; }
    public required double DaysSinceReinforcement { get; init; }
}

public class ReinforceResult
{
    public required int ReinforcedCount { get; init; }
    public required int FailedCount { get; init; }
    public required List<string> ReinforcedIds { get; init; }
    public required List<string> FailedIds { get; init; }
}

public class ForgetResult
{
    public required string Id { get; init; }
    public required bool Success { get; init; }
    public required string Message { get; init; }
}

public class UnifiedSearchResult
{
    public required string Query { get; init; }
    public required string Mode { get; init; }
    public required int TotalResults { get; init; }
    public required List<object> Results { get; init; }
}

public class IndexContentResult
{
    public required int TotalFiles { get; init; }
    public required int Indexed { get; init; }
    public required int Skipped { get; init; }
    public required List<string> Errors { get; init; }
    public required bool Success { get; init; }
}

public class AddContentResult
{
    public required string Id { get; init; }
    public required string Source { get; init; }
    public required bool IsUpdate { get; init; }
    public required bool Success { get; init; }
}

public class SearchContentResult
{
    public required string Query { get; init; }
    public required int TotalResults { get; init; }
    public required List<ContentInfo> Results { get; init; }
}

public class ListContentResult
{
    public required int TotalResults { get; init; }
    public required bool IsExpiredOnly { get; init; }
    public required List<ContentInfo> Content { get; init; }
}

public class ContentInfo
{
    public required string Id { get; init; }
    public required string Title { get; init; }
    public required string Source { get; init; }
    public required string? Project { get; init; }
    public required string[] Tags { get; init; }
    public required double Score { get; init; }
    public required double DecayFactor { get; init; }
    public required double DaysSinceReinforcement { get; init; }
    public required string? Highlight { get; init; }
    public required DateTimeOffset CreatedAt { get; init; }
    public required DateTimeOffset LastReinforcedAt { get; init; }
}

public class RemoveContentResult
{
    public required int RemovedCount { get; init; }
    public required bool Success { get; init; }
}

public class ReinforceContentResult
{
    public required string Source { get; init; }
    public required bool Success { get; init; }
    public required string Message { get; init; }
}

#endregion
