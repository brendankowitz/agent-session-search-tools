using AgentJournal.Core.Models;
using Lucene.Net.Analysis;
using Lucene.Net.Analysis.Standard;
using Lucene.Net.Documents;
using Lucene.Net.Index;
using Lucene.Net.QueryParsers.Classic;
using Lucene.Net.Search;
using Lucene.Net.Search.Similarities;
using Lucene.Net.Store;
using Lucene.Net.Util;
using System.Collections.Concurrent;

namespace AgentJournal.Core.Search;

/// <summary>
/// Lucene.NET-based lexical search engine with BM25 scoring
/// </summary>
public class LuceneSearchEngine : ISearchEngine, IDisposable
{
    private const LuceneVersion LUCENE_VERSION = LuceneVersion.LUCENE_48;

    // Field names
    private const string FIELD_ID = "id";
    private const string FIELD_SESSION_ID = "session_id";
    private const string FIELD_AGENT_TYPE = "agent_type";
    private const string FIELD_PROJECT_PATH = "project_path";
    private const string FIELD_ROLE = "role";
    private const string FIELD_CONTENT = "content";
    private const string FIELD_ALL_CONTENT = "all_content"; // Combined content from all session messages
    private const string FIELD_TIMESTAMP = "timestamp";

    private readonly string _indexPath;
    private readonly ReaderWriterLockSlim _indexLock = new(LockRecursionPolicy.NoRecursion);
    private readonly ConcurrentDictionary<string, Session> _sessionCache = new();

    private FSDirectory? _directory;
    private Analyzer? _analyzer;
    private IndexWriter? _writer;
    private SearcherManager? _searcherManager;
    private bool _disposed;

    public IReadOnlyList<SearchMode> SupportedModes { get; } = new[] { SearchMode.Lexical };

    public LuceneSearchEngine(string? indexPath = null)
    {
        _indexPath = indexPath ?? Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".agent-journal",
            "lucene-index");
    }

    /// <summary>
    /// Initializes the Lucene index and creates the directory if needed
    /// </summary>
    public async Task InitializeAsync(CancellationToken ct = default)
    {
        _indexLock.EnterWriteLock();
        try
        {
            if (_directory != null)
            {
                return; // Already initialized
            }

            // Create index directory if it doesn't exist
            if (!System.IO.Directory.Exists(_indexPath))
            {
                System.IO.Directory.CreateDirectory(_indexPath);
            }

            _directory = FSDirectory.Open(_indexPath);
            _analyzer = new StandardAnalyzer(LUCENE_VERSION);

            var indexConfig = new IndexWriterConfig(LUCENE_VERSION, _analyzer)
            {
                OpenMode = OpenMode.CREATE_OR_APPEND,
                // Use BM25 similarity for better ranking
                Similarity = new BM25Similarity()
            };

            _writer = new IndexWriter(_directory, indexConfig);
            _writer.Commit(); // Ensure index is created

            _searcherManager = new SearcherManager(_writer, true, null);

            await Task.CompletedTask; // Satisfy async signature
        }
        finally
        {
            _indexLock.ExitWriteLock();
        }
    }

    /// <summary>
    /// Indexes a single session and all its messages
    /// </summary>
    public async Task IndexSessionAsync(Session session, CancellationToken ct = default)
    {
        EnsureInitialized();

        _indexLock.EnterWriteLock();
        try
        {
            // Delete any existing documents for this session first to handle updates
            var sessionTerm = new Term(FIELD_SESSION_ID, session.Id);
            _writer!.DeleteDocuments(sessionTerm);

            // Cache the session for retrieval during search
            _sessionCache[session.Id] = session;

            // Create combined content from all messages for session-level searching
            var allContent = string.Join(" ", session.Messages.Select(m => m.Content ?? ""));

            // Index each message in the session
            foreach (var message in session.Messages)
            {
                var doc = CreateDocument(session, message, allContent);
                _writer!.AddDocument(doc);
            }

            // Commit changes and refresh searcher (blocking to ensure visibility)
            _writer!.Commit();
            _searcherManager?.MaybeRefreshBlocking();

            await Task.CompletedTask; // Satisfy async signature
        }
        finally
        {
            _indexLock.ExitWriteLock();
        }
    }

    /// <summary>
    /// Indexes multiple sessions in batch
    /// </summary>
    public async Task IndexSessionsAsync(IEnumerable<Session> sessions, CancellationToken ct = default)
    {
        EnsureInitialized();

        _indexLock.EnterWriteLock();
        try
        {
            foreach (var session in sessions)
            {
                ct.ThrowIfCancellationRequested();

                // Delete any existing documents for this session first
                var sessionTerm = new Term(FIELD_SESSION_ID, session.Id);
                _writer!.DeleteDocuments(sessionTerm);

                // Cache the session
                _sessionCache[session.Id] = session;

                // Create combined content from all messages for session-level searching
                var allContent = string.Join(" ", session.Messages.Select(m => m.Content ?? ""));

                // Index each message
                foreach (var message in session.Messages)
                {
                    var doc = CreateDocument(session, message, allContent);
                    _writer!.AddDocument(doc);
                }
            }

            // Commit changes and refresh searcher (blocking to ensure visibility)
            _writer!.Commit();
            _searcherManager?.MaybeRefreshBlocking();

            await Task.CompletedTask; // Satisfy async signature
        }
        finally
        {
            _indexLock.ExitWriteLock();
        }
    }

    /// <summary>
    /// Performs full-text search with BM25 scoring
    /// </summary>
    public async Task<IReadOnlyList<SearchResult>> SearchAsync(
        string query,
        SearchMode mode = SearchMode.Lexical,
        int maxResults = 10,
        int contextCount = 0,
        CancellationToken ct = default)
    {
        if (mode != SearchMode.Lexical)
        {
            throw new NotSupportedException($"LuceneSearchEngine does not support {mode} search mode.");
        }

        if (string.IsNullOrWhiteSpace(query))
        {
            return Array.Empty<SearchResult>();
        }

        EnsureInitialized();

        _searcherManager?.MaybeRefresh();
        var searcher = _searcherManager?.Acquire();

        try
        {
            if (searcher == null)
            {
                return Array.Empty<SearchResult>();
            }

            // Parse the query - search in the combined all_content field for session-level matching
            var parser = new QueryParser(LUCENE_VERSION, FIELD_ALL_CONTENT, _analyzer!);
            parser.DefaultOperator = Operator.AND;

            Query luceneQuery;
            try
            {
                luceneQuery = parser.Parse(query);
            }
            catch (ParseException)
            {
                // If parsing fails, try as phrase query
                luceneQuery = parser.Parse($"\"{QueryParserBase.Escape(query)}\"");
            }

            // Execute search
            var topDocs = searcher.Search(luceneQuery, maxResults * 10); // Get more to allow for dedup
            var results = new List<SearchResult>();
            var seenSessions = new HashSet<string>();

            foreach (var scoreDoc in topDocs.ScoreDocs)
            {
                var doc = searcher.Doc(scoreDoc.Doc);
                var sessionId = doc.Get(FIELD_SESSION_ID);

                // Skip if we've already seen this session (deduplicate by session)
                if (!seenSessions.Add(sessionId))
                {
                    continue;
                }

                // Get the session from cache, or create minimal session from doc
                Session session;
                IReadOnlyList<Message>? matchingMessages = null;

                if (_sessionCache.TryGetValue(sessionId, out var cachedSession))
                {
                    session = cachedSession;
                    var messageId = doc.Get(FIELD_ID);
                    var matchedMessages = session.Messages
                        .Where(m => m.Id == messageId)
                        .ToList();

                    // Expand with context if requested
                    matchingMessages = ExpandWithContext(session.Messages, matchedMessages, contextCount);
                }
                else
                {
                    // Create minimal session from Lucene document data
                    // Note: Timestamp is stored as DateTime.Ticks
                    var timestamp = long.TryParse(doc.Get(FIELD_TIMESTAMP), out var ticks) && ticks > 0
                        ? new DateTime(ticks, DateTimeKind.Utc)
                        : DateTime.MinValue;

                    session = new Session(
                        Id: sessionId,
                        AgentType: doc.Get(FIELD_AGENT_TYPE) ?? "unknown",
                        ProjectPath: doc.Get(FIELD_PROJECT_PATH),
                        GitBranch: null,
                        AgentVersion: null,
                        StartedAt: timestamp,
                        EndedAt: null,
                        LastModified: null,
                        Summary: null,
                        Messages: Array.Empty<Message>()
                    );
                }

                // Get highlights if available
                var content = doc.Get(FIELD_CONTENT);
                var highlight = GetHighlight(content, query);

                results.Add(new SearchResult(
                    Session: session,
                    Score: scoreDoc.Score,
                    MatchingMessages: matchingMessages,
                    Highlight: highlight
                ));

                // Stop if we have enough results
                if (results.Count >= maxResults)
                {
                    break;
                }
            }

            return results;
        }
        finally
        {
            if (searcher != null)
            {
                _searcherManager?.Release(searcher);
            }
        }
    }

    /// <summary>
    /// Clears all documents from the index
    /// </summary>
    public async Task ClearIndexAsync(CancellationToken ct = default)
    {
        EnsureInitialized();

        _indexLock.EnterWriteLock();
        try
        {
            _writer!.DeleteAll();
            _writer.Commit();
            _sessionCache.Clear();
            _searcherManager?.MaybeRefresh();

            await Task.CompletedTask; // Satisfy async signature
        }
        finally
        {
            _indexLock.ExitWriteLock();
        }
    }

    /// <summary>
    /// Deletes all messages for a specific session
    /// </summary>
    public async Task DeleteSessionAsync(string sessionId, CancellationToken ct = default)
    {
        EnsureInitialized();

        _indexLock.EnterWriteLock();
        try
        {
            var term = new Term(FIELD_SESSION_ID, sessionId);
            _writer!.DeleteDocuments(term);
            _writer.Commit();
            _sessionCache.TryRemove(sessionId, out _);
            _searcherManager?.MaybeRefresh();

            await Task.CompletedTask; // Satisfy async signature
        }
        finally
        {
            _indexLock.ExitWriteLock();
        }
    }

    /// <summary>
    /// Gets index statistics
    /// </summary>
    public async Task<IndexStats> GetIndexStatsAsync(CancellationToken ct = default)
    {
        EnsureInitialized();

        _indexLock.EnterReadLock();
        try
        {
            var docCount = _writer!.NumDocs;
            var maxDoc = _writer.MaxDoc;

            // Calculate directory size
            var directoryInfo = new DirectoryInfo(_indexPath);
            var sizeBytes = directoryInfo.Exists
                ? directoryInfo.EnumerateFiles("*", SearchOption.AllDirectories).Sum(f => f.Length)
                : 0;

            await Task.CompletedTask; // Satisfy async signature

            return new IndexStats(
                DocumentCount: docCount,
                MaxDocuments: maxDoc,
                SizeBytes: sizeBytes,
                SessionCount: _sessionCache.Count
            );
        }
        finally
        {
            _indexLock.ExitReadLock();
        }
    }

    private Document CreateDocument(Session session, Message message, string allContent)
    {
        var doc = new Document();

        // Store fields (not analyzed)
        doc.Add(new StringField(FIELD_ID, message.Id, Field.Store.YES));
        doc.Add(new StringField(FIELD_SESSION_ID, session.Id, Field.Store.YES));
        doc.Add(new StringField(FIELD_AGENT_TYPE, session.AgentType, Field.Store.YES));

        if (!string.IsNullOrEmpty(session.ProjectPath))
        {
            doc.Add(new StringField(FIELD_PROJECT_PATH, session.ProjectPath, Field.Store.YES));
        }

        doc.Add(new StringField(FIELD_ROLE, message.Role.ToString(), Field.Store.YES));

        // Timestamp as sortable long field
        var timestampTicks = message.Timestamp.Ticks;
        doc.Add(new Int64Field(FIELD_TIMESTAMP, timestampTicks, Field.Store.YES));

        // Content field (analyzed and stored for highlighting)
        if (!string.IsNullOrEmpty(message.Content))
        {
            doc.Add(new TextField(FIELD_CONTENT, message.Content, Field.Store.YES));
        }

        // All content field (analyzed but not stored) - for session-level searching
        if (!string.IsNullOrEmpty(allContent))
        {
            doc.Add(new TextField(FIELD_ALL_CONTENT, allContent, Field.Store.NO));
        }

        return doc;
    }

    private string? GetHighlight(string content, string query, int maxLength = 200)
    {
        if (string.IsNullOrEmpty(content) || string.IsNullOrEmpty(query))
        {
            return null;
        }

        // Simple highlighting: find the query term and return context
        var queryTerms = query.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        var lowerContent = content.ToLowerInvariant();

        foreach (var term in queryTerms)
        {
            var lowerTerm = term.ToLowerInvariant().Trim('"', '\'');
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

    /// <summary>
    /// Expands matching messages to include N messages before and after
    /// </summary>
    private static IReadOnlyList<Message> ExpandWithContext(
        IReadOnlyList<Message> allMessages,
        IReadOnlyList<Message> matchedMessages,
        int contextCount)
    {
        if (contextCount <= 0 || matchedMessages.Count == 0)
        {
            return matchedMessages;
        }

        var messagesToInclude = new HashSet<Message>();

        foreach (var matched in matchedMessages)
        {
            // Find the index of this matched message in the full list
            var matchIndex = -1;
            for (int i = 0; i < allMessages.Count; i++)
            {
                if (allMessages[i].Id == matched.Id)
                {
                    matchIndex = i;
                    break;
                }
            }

            if (matchIndex < 0)
            {
                continue; // Message not found, skip
            }

            // Include context messages before and after
            var startIndex = Math.Max(0, matchIndex - contextCount);
            var endIndex = Math.Min(allMessages.Count - 1, matchIndex + contextCount);

            for (int i = startIndex; i <= endIndex; i++)
            {
                messagesToInclude.Add(allMessages[i]);
            }
        }

        // Return messages ordered by their original position (timestamp)
        return allMessages
            .Where(m => messagesToInclude.Contains(m))
            .ToList();
    }

    private void EnsureInitialized()
    {
        if (_directory == null || _writer == null || _analyzer == null)
        {
            throw new InvalidOperationException("LuceneSearchEngine must be initialized before use. Call InitializeAsync first.");
        }
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _searcherManager?.Dispose();
        _writer?.Dispose();
        _analyzer?.Dispose();
        _directory?.Dispose();
        _indexLock?.Dispose();

        _disposed = true;
        GC.SuppressFinalize(this);
    }
}

/// <summary>
/// Index statistics
/// </summary>
public record IndexStats(
    int DocumentCount,
    int MaxDocuments,
    long SizeBytes,
    int SessionCount
)
{
    public double SizeMB => SizeBytes / (1024.0 * 1024.0);
};
