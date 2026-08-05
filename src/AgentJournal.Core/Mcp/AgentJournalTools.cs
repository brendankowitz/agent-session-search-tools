using System.ComponentModel;
using System.Security.Cryptography;
using System.Text;
using AgentJournal.Core.Knowledge;
using AgentJournal.Core.Models;
using AgentJournal.Core.Search;
using AgentJournal.Core.Storage;
using AgentJournal.Core.Tasks;
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
    [Description("Search agent session history across all projects for relevant past conversations. " +
        "Use this to find how a problem was solved before.")]
    public async Task<SearchSessionsResult> SearchSessions(
        [Description("Search query to find relevant sessions")] string query,
        [Description("Search mode: lexical (keyword), semantic (meaning), or hybrid (both)")] string mode = "hybrid",
        [Description("Filter by project path or name")] string? project = null,
        [Description("Maximum number of results to return")] int limit = 10,
        [Description("Number of messages before and after matches to include for context (0 to disable)")] int around = 0)
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
    /// <param name="last">
    /// When set, return only the last N messages. Use this to read the tail of a long session
    /// without pulling the whole transcript into context.
    /// </param>
    [McpServerTool]
    [Description("Get full session details including all messages.")]
    public async Task<SessionDetails> GetSession(
        [Description("Session ID to retrieve")] string id,
        [Description("When set, return only the last N messages. Use this to read the tail of a long session without pulling the whole transcript into context.")] int? last = null)
    {
        var session = await _sessionRepository.GetSessionAsync(id);
        if (session == null)
        {
            throw new ArgumentException($"Session with ID '{id}' not found.", nameof(id));
        }

        if (last is <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(last), last, "last must be greater than zero.");
        }

        var totalMessages = session.MessageCount;
        var messages = last.HasValue ? session.WithLastMessages(last.Value).Messages : session.Messages;

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
            MessageCount = totalMessages,
            ReturnedMessageCount = messages.Count,
            Truncated = messages.Count < totalMessages,
            Messages = messages.Select(m => new MessageInfo
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
    [Description("List recent agent sessions, most recent first, with optional project filtering.")]
    public async Task<ListSessionsResult> ListRecentSessions(
        [Description("Maximum number of sessions to return")] int limit = 10,
        [Description("Filter by project path or name")] string? project = null)
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
    [Description("Store a durable fact, convention, or learning in the knowledge bank so it can be recalled in future sessions.")]
    public async Task<RememberResult> Remember(
        [Description("Content to remember")] string content,
        [Description("Tags for categorization (comma-separated)")] string? tags = null,
        [Description("Associated project path")] string? project = null,
        [Description("Source of this knowledge")] string? source = null)
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
    [Description("Search the knowledge bank for previously stored facts, conventions, and learnings.")]
    public async Task<RecallResult> Recall(
        [Description("Search query to find relevant knowledge")] string query,
        [Description("Filter by tags (comma-separated)")] string? tags = null,
        [Description("Filter by project path")] string? project = null,
        [Description("Search mode: lexical, semantic, or hybrid")] string mode = "hybrid",
        [Description("Maximum number of results")] int limit = 10)
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
    [Description("Reinforce knowledge entries so they do not decay. Call this when a recalled entry proved useful.")]
    public async Task<ReinforceResult> Reinforce(
        [Description("Knowledge entry IDs to reinforce (comma-separated)")] string ids)
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
    [Description("Remove an entry from the knowledge bank.")]
    public async Task<ForgetResult> Forget(
        [Description("Knowledge entry ID to remove")] string id)
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
    [Description("Index markdown files from a directory so their content becomes searchable.")]
    public async Task<IndexContentResult> IndexContent(
        [Description("Directory path to scan for markdown files")] string path,
        [Description("Glob pattern for file matching (default: *.md)")] string filter = "*.md",
        [Description("Project name to associate with indexed content")] string? project = null,
        [Description("Recursively scan subdirectories (default: true)")] bool recursive = true)
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
    [Description("Add a searchable content document directly, without reading it from a file on disk.")]
    public async Task<AddContentResult> AddContent(
        [Description("Source identifier for this content")] string source,
        [Description("Content title")] string title,
        [Description("Content text")] string content,
        [Description("Project name")] string? project = null,
        [Description("Tags for categorization")] string[]? tags = null)
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
    [Description("Search indexed content documents (for example, indexed markdown docs) using full-text search.")]
    public async Task<SearchContentResult> SearchContent(
        [Description("Search query text")] string query,
        [Description("Maximum number of results (default: 10)")] int maxResults = 10,
        [Description("Filter by project")] string? project = null,
        [Description("Filter by source path prefix")] string? sourcePrefix = null,
        [Description("Filter by tags")] string[]? tags = null)
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
    [Description("List indexed content documents with optional filtering.")]
    public async Task<ListContentResult> ListContent(
        [Description("Filter by project")] string? project = null,
        [Description("Filter by source path prefix")] string? sourcePrefix = null,
        [Description("Filter by tags")] string[]? tags = null,
        [Description("Maximum number of entries to return (default: 50)")] int limit = 50,
        [Description("Show only expired content (default: false)")] bool expiredOnly = false)
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
    [Description("Remove indexed content by ID, exact source, source prefix, or project. At least one criterion is required.")]
    public async Task<RemoveContentResult> RemoveContent(
        [Description("Remove by content ID")] string? id = null,
        [Description("Remove by exact source match")] string? source = null,
        [Description("Remove all content where source starts with this prefix")] string? sourcePrefix = null,
        [Description("Remove all content for a project")] string? project = null)
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
    [Description("Reinforce indexed content to reset its decay timer.")]
    public async Task<ReinforceContentResult> ReinforceContent(
        [Description("Source identifier of content to reinforce")] string source)
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
    [Description("Unified search across both past sessions and the knowledge bank. Prefer this when you are not sure which store holds the answer.")]
    public async Task<UnifiedSearchResult> Search(
        [Description("Search query")] string query,
        [Description("Search mode: lexical, semantic, or hybrid")] string mode = "hybrid",
        [Description("Include knowledge entries in results")] bool includeKnowledge = true,
        [Description("Filter by project path")] string? project = null,
        [Description("Maximum total results")] int limit = 20)
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

    #region Task Journal

    [McpServerTool]
    [Description("Get the current state of a task journal, including which task to resume. " +
                 "Use this after context loss to find out where execution of a plan stopped.")]
    public async Task<TaskJournalResult> TaskStatus(
        [Description("Journal name. Omit when only one journal exists in the repository.")] string? name = null,
        [Description("Path inside the repository holding the journal. Defaults to the server's working directory, which is often not the repository - pass this explicitly when running as an MCP server.")] string? repositoryPath = null)
    {
        var store = TaskJournalStore.ForRepository(repositoryPath);
        var resolved = await ResolveJournalNameAsync(store, name);
        return ToResult(await store.LoadAsync(resolved));
    }

    [McpServerTool]
    [Description("Create a task journal for a plan file so progress survives context loss. " +
                 "Returns the journal state and the list of tasks.")]
    public async Task<TaskJournalResult> TaskInit(
        [Description("Path to the plan file being executed")] string planPath,
        [Description("Number of tasks. Omit to count '## Task N' headings in the plan.")] int? taskCount = null,
        [Description("Journal name. Defaults to the plan file name.")] string? name = null,
        [Description("Path inside the repository that should hold the journal. Defaults to the server's working directory, which is often not the repository - pass this explicitly when running as an MCP server.")] string? repositoryPath = null)
    {
        var store = TaskJournalStore.ForRepository(repositoryPath);
        return ToResult(await store.InitAsync(planPath, taskCount, name));
    }

    [McpServerTool]
    [Description("Record a task state change in the journal. Valid states: started, complete, fix. " +
                 "Call this immediately after a task finishes so progress is never held only in conversation context.")]
    public async Task<TaskJournalResult> TaskRecord(
        [Description("Task number")] int taskNumber,
        [Description("New state: 'started', 'complete', or 'fix' (reopens a task after review found problems)")] string state,
        [Description("Short note recorded alongside the state change")] string? note = null,
        [Description("Journal name. Omit when only one journal exists in the repository.")] string? name = null,
        [Description("Path inside the repository holding the journal. Defaults to the server's working directory.")] string? repositoryPath = null)
    {
        var journalState = ParseState(state);

        var store = TaskJournalStore.ForRepository(repositoryPath);
        var resolved = await ResolveJournalNameAsync(store, name);
        return ToResult(await store.AppendAsync(resolved, taskNumber, journalState, note));
    }

    [McpServerTool]
    [Description("Store a task brief or report in the journal. " +
                 "Tell a subagent to fetch it with TaskReadArtifact rather than pasting the content, " +
                 "so the coordinator's context stays small.")]
    public async Task<TaskArtifactResult> TaskWriteArtifact(
        [Description("Task number")] int taskNumber,
        [Description("Artifact kind: 'brief' (instructions for a subagent) or 'report' (what the subagent did)")] string kind,
        [Description("Markdown content to store")] string content,
        [Description("Journal name. Omit when only one journal exists in the repository.")] string? name = null,
        [Description("Path inside the repository holding the journal. Defaults to the server's working directory.")] string? repositoryPath = null)
    {
        var artifactKind = ParseArtifactKind(kind);

        var store = TaskJournalStore.ForRepository(repositoryPath);
        var resolved = await ResolveJournalNameAsync(store, name);

        await store.WriteArtifactAsync(resolved, taskNumber, artifactKind, content);

        return new TaskArtifactResult
        {
            Journal = resolved,
            TaskNumber = taskNumber,
            Kind = artifactKind.ToString().ToLowerInvariant(),
            Stored = true
        };
    }

    [McpServerTool]
    [Description("Read a task brief or report back out of the journal. " +
                 "A subagent calls this to collect its instructions without the coordinator having to paste them.")]
    public async Task<TaskArtifactResult> TaskReadArtifact(
        [Description("Task number")] int taskNumber,
        [Description("Artifact kind: 'brief' or 'report'")] string kind,
        [Description("Journal name. Omit when only one journal exists in the repository.")] string? name = null,
        [Description("Path inside the repository holding the journal. Defaults to the server's working directory.")] string? repositoryPath = null)
    {
        var artifactKind = ParseArtifactKind(kind);

        var store = TaskJournalStore.ForRepository(repositoryPath);
        var resolved = await ResolveJournalNameAsync(store, name);

        var content = await store.ReadArtifactAsync(resolved, taskNumber, artifactKind);

        return new TaskArtifactResult
        {
            Journal = resolved,
            TaskNumber = taskNumber,
            Kind = artifactKind.ToString().ToLowerInvariant(),
            Stored = content is not null,
            Content = content
        };
    }

    [McpServerTool]
    [Description("List the task journals present in this repository.")]
    public async Task<TaskJournalListResult> TaskList(
        [Description("Path inside the repository holding the journals. Defaults to the server's working directory.")] string? repositoryPath = null)
    {
        var store = TaskJournalStore.ForRepository(repositoryPath);
        return new TaskJournalListResult
        {
            Root = store.TasksRoot,
            Journals = (await store.ListAsync()).ToList()
        };
    }

    [McpServerTool]
    [Description("Search task journal progress notes and artifacts (briefs and reports) in this repository. " +
        "Use this to find prior work on a task before redoing it. Scoped to the current repository, " +
        "unlike SearchSessions and RecallKnowledge which cover all projects.")]
    public async Task<TaskSearchToolResult> TaskSearch(
        [Description("Search query. Matched as literal words; FTS5 operators are not interpreted.")] string query,
        [Description("Maximum number of results. Defaults to 10.")] int limit = 10,
        [Description("Path inside the repository holding the journals. Defaults to the server's working directory.")] string? repositoryPath = null)
    {
        var store = TaskJournalStore.ForRepository(repositoryPath);
        var results = await store.SearchAsync(query, limit);
        return new TaskSearchToolResult
        {
            Root = store.TasksRoot,
            Results = results.Select(r => new TaskSearchToolItem
            {
                JournalName = r.JournalName,
                TaskNumber = r.TaskNumber,
                Kind = r.Kind,
                Excerpt = r.Excerpt,
                Score = r.Score
            }).ToList()
        };
    }

    private static TaskJournalState ParseState(string state) => state.Trim().ToLowerInvariant() switch
    {
        "started" or "start" or "inprogress" or "in-progress" => TaskJournalState.InProgress,
        "complete" or "completed" or "done" => TaskJournalState.Complete,
        "fix" or "fixround" or "fix-round" or "reopen" => TaskJournalState.FixRound,
        _ => throw new ArgumentException(
            $"Unknown state '{state}'. Use 'started', 'complete', or 'fix'.", nameof(state))
    };

    private static TaskArtifactKind ParseArtifactKind(string kind) => kind.Trim().ToLowerInvariant() switch
    {
        "brief" => TaskArtifactKind.Brief,
        "report" => TaskArtifactKind.Report,
        _ => throw new ArgumentException($"Unknown kind '{kind}'. Use 'brief' or 'report'.", nameof(kind))
    };

    private static async Task<string> ResolveJournalNameAsync(TaskJournalStore store, string? name)
    {
        if (!string.IsNullOrWhiteSpace(name))
        {
            return name;
        }

        var journals = await store.ListAsync();

        return journals.Count switch
        {
            0 => throw new InvalidOperationException(
                "No task journals exist in this repository. Call TaskInit first."),
            1 => journals[0],
            _ => throw new InvalidOperationException(
                $"{journals.Count} task journals exist ({string.Join(", ", journals)}); specify one by name.")
        };
    }

    private static TaskJournalResult ToResult(TaskJournalSnapshot snapshot) => new()
    {
        Name = snapshot.Name,
        PlanPath = snapshot.PlanPath,
        DatabasePath = snapshot.DatabasePath,
        TotalTasks = snapshot.Tasks.Count,
        CompletedTasks = snapshot.CompletedCount,
        IsComplete = snapshot.IsComplete,
        NextTask = snapshot.NextTask is null ? null : ToResult(snapshot.NextTask),
        Tasks = snapshot.Tasks.Select(ToResult).ToList()
    };

    private static TaskJournalTaskResult ToResult(TaskJournalTask task) => new()
    {
        Number = task.Number,
        State = task.State.ToString(),
        FixRound = task.FixRound,
        LastNote = task.LastNote,
        HasBrief = task.HasBrief,
        HasReport = task.HasReport
    };

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

    /// <summary>Total messages in the stored session, regardless of how many are returned.</summary>
    public required int MessageCount { get; init; }

    /// <summary>
    /// Number of messages actually present in <see cref="Messages"/>. Differs from
    /// <see cref="MessageCount"/> when the caller requested only the tail of the session.
    /// </summary>
    public required int ReturnedMessageCount { get; init; }

    /// <summary>True when <see cref="Messages"/> holds only the tail of the session.</summary>
    public required bool Truncated { get; init; }

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

public class TaskJournalResult
{
    public required string Name { get; init; }

    /// <summary>Absolute path to the plan file, safe to open from any working directory.</summary>
    public required string PlanPath { get; init; }

    public required string DatabasePath { get; init; }
    public required int TotalTasks { get; init; }
    public required int CompletedTasks { get; init; }
    public required bool IsComplete { get; init; }

    /// <summary>The task to resume at, or null when every task is complete.</summary>
    public required TaskJournalTaskResult? NextTask { get; init; }

    public required List<TaskJournalTaskResult> Tasks { get; init; }
}

public class TaskJournalTaskResult
{
    public required int Number { get; init; }
    public required string State { get; init; }
    public required int FixRound { get; init; }
    public required string? LastNote { get; init; }

    /// <summary>True when a brief is stored. Fetch it with TaskReadArtifact when it is needed.</summary>
    public required bool HasBrief { get; init; }

    /// <summary>True when a report is stored. Fetch it with TaskReadArtifact when it is needed.</summary>
    public required bool HasReport { get; init; }
}

public class TaskArtifactResult
{
    public required string Journal { get; init; }
    public required int TaskNumber { get; init; }
    public required string Kind { get; init; }

    /// <summary>True when the artifact exists in the journal.</summary>
    public required bool Stored { get; init; }

    /// <summary>Artifact body. Populated by reads; null on writes and when nothing is stored.</summary>
    public string? Content { get; init; }
}

public class TaskJournalListResult
{
    public required string Root { get; init; }
    public required List<string> Journals { get; init; }
}

public class TaskSearchToolResult
{
    public required string Root { get; init; }
    public required List<TaskSearchToolItem> Results { get; init; }
}

public class TaskSearchToolItem
{
    public required string JournalName { get; init; }
    public required int TaskNumber { get; init; }
    public required string Kind { get; init; }
    public required string Excerpt { get; init; }
    public required double Score { get; init; }
}

#endregion
