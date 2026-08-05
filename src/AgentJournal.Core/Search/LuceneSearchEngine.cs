using AgentJournal.Core.Models;
using AgentJournal.Core.Storage;
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
    private const string FIELD_ALL_CONTENT = "all_content"; // Combined content, on the session document only
    private const string FIELD_DOC_TYPE = "doc_type";
    private const string DOC_TYPE_SESSION = "session";
    private const string DOC_TYPE_MESSAGE = "message";
    private const string FIELD_TIMESTAMP = "timestamp";

    /// <summary>
    /// Caps how many matching messages are collected per session so a single long session cannot
    /// dominate a result payload. Context expansion is applied on top of this.
    /// </summary>
    private const int MAX_MATCHING_MESSAGES_PER_SESSION = 20;

    /// <summary>
    /// Upper bound on documents scanned for session grouping. A term present in most of the corpus
    /// would otherwise allocate one <c>ScoreDoc</c> per message in the whole index.
    /// </summary>
    private const int MAX_HITS_SCANNED = 10_000;

    private readonly string _indexPath;
    private readonly ISessionRepository? _sessionRepository;
    private readonly ReaderWriterLockSlim _indexLock = new(LockRecursionPolicy.NoRecursion);
    private readonly ConcurrentDictionary<string, Session> _sessionCache = new();

    private FSDirectory? _directory;
    private Analyzer? _analyzer;
    private IndexWriter? _writer;
    private SearcherManager? _searcherManager;
    private bool _readOnly;
    private bool _disposed;

    public IReadOnlyList<SearchMode> SupportedModes { get; } = new[] { SearchMode.Lexical };

    /// <param name="indexPath">Directory holding the Lucene index.</param>
    /// <param name="sessionRepository">
    /// Used to hydrate sessions that were indexed by an earlier process. Without it the in-memory
    /// cache is cold on every fresh invocation, so no result can carry matching messages or the
    /// message list that <c>--context</c> expands over.
    /// </param>
    public LuceneSearchEngine(string? indexPath = null, ISessionRepository? sessionRepository = null)
    {
        _sessionRepository = sessionRepository;
        _indexPath = indexPath ?? Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".agent-journal",
            "lucene-index");
    }

    /// <summary>
    /// Initializes the Lucene index in read-only mode.
    /// Write access is acquired lazily on the first write operation.
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

            // Start in read-only mode; upgrade to write on demand
            _writer = null;
            _readOnly = true;

            if (DirectoryReader.IndexExists(_directory))
            {
                _searcherManager = new SearcherManager(_directory, null);
            }

            await Task.CompletedTask;
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
        EnsureWritable();

        _indexLock.EnterWriteLock();
        try
        {
            // Delete any existing documents for this session first to handle updates
            var sessionTerm = new Term(FIELD_SESSION_ID, session.Id);
            _writer!.DeleteDocuments(sessionTerm);

            // Cache the session for retrieval during search
            _sessionCache[session.Id] = session;

            AddSessionDocuments(session);

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
        EnsureWritable();

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

                AddSessionDocuments(session);
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

            // Session-level matching runs against all_content, which now lives only on the single
            // session document. One hit per matching session, so ranking and budgeting are per
            // session rather than per message.
            var parser = new QueryParser(LUCENE_VERSION, FIELD_ALL_CONTENT, _analyzer!);
            parser.DefaultOperator = Operator.AND;

            // Matching messages come from a separate query over the per-message content field.
            // all_content spans the whole session, so a session-level match says nothing about
            // which individual messages mention the query terms.
            var messageParser = new QueryParser(LUCENE_VERSION, FIELD_CONTENT, _analyzer!);
            messageParser.DefaultOperator = Operator.AND;

            Query luceneQuery;
            Query messageQuery;
            try
            {
                luceneQuery = parser.Parse(query);
                messageQuery = messageParser.Parse(query);
            }
            catch (ParseException)
            {
                // If parsing fails, try as phrase query
                var escaped = $"\"{QueryParserBase.Escape(query)}\"";
                luceneQuery = parser.Parse(escaped);
                messageQuery = messageParser.Parse(escaped);
            }

            // Select sessions. Session documents are one per session, so maxResults hits is exactly
            // the number of sessions wanted - no oversampling, and no chatty session can crowd out
            // the rest the way per-message hits did.
            var sessionDocs = searcher.Search(luceneQuery, Math.Max(maxResults, 1));

            var hitsBySession = new Dictionary<string, SessionHits>(StringComparer.Ordinal);
            var sessionOrder = new List<string>();

            foreach (var scoreDoc in sessionDocs.ScoreDocs)
            {
                var doc = searcher.Doc(scoreDoc.Doc);
                var sessionId = doc.Get(FIELD_SESSION_ID);
                if (string.IsNullOrEmpty(sessionId) || hitsBySession.ContainsKey(sessionId))
                {
                    continue;
                }

                hitsBySession[sessionId] = new SessionHits(doc, scoreDoc.Score);
                sessionOrder.Add(sessionId);
            }

            if (sessionOrder.Count > 0)
            {
                // Collect matching messages for the selected sessions. Message hits are ranked
                // independently of the session ranking, so a single verbose session can consume the
                // head of this list; size the fetch against the real total and bound it so a very
                // common term cannot allocate wildly.
                var totalHitCollector = new TotalHitCountCollector();
                searcher.Search(messageQuery, totalHitCollector);

                var desiredHits = sessionOrder.Count * MAX_MATCHING_MESSAGES_PER_SESSION;
                var hitBudget = Math.Clamp(totalHitCollector.TotalHits, desiredHits, MAX_HITS_SCANNED);

                var messageDocs = searcher.Search(messageQuery, Math.Max(hitBudget, 1));

                foreach (var scoreDoc in messageDocs.ScoreDocs)
                {
                    var doc = searcher.Doc(scoreDoc.Doc);
                    var sessionId = doc.Get(FIELD_SESSION_ID);
                    if (string.IsNullOrEmpty(sessionId) ||
                        !hitsBySession.TryGetValue(sessionId, out var hits))
                    {
                        continue;
                    }

                    if (hits.MessageIds.Count >= MAX_MATCHING_MESSAGES_PER_SESSION)
                    {
                        continue;
                    }

                    var messageId = doc.Get(FIELD_ID);
                    if (string.IsNullOrEmpty(messageId))
                    {
                        continue;
                    }

                    hits.MessageIds.Add(messageId);

                    // Highlight from a message that actually matched, rather than from the session
                    // document, which stores no message text at all.
                    hits.OfferMessageMatchDoc(doc);
                }
            }

            var results = new List<SearchResult>(sessionOrder.Count);

            foreach (var sessionId in sessionOrder)
            {
                var hits = hitsBySession[sessionId];
                var doc = hits.TopDoc;

                Session session;
                IReadOnlyList<Message>? matchingMessages = null;

                // The cache only holds sessions this process indexed. Every ordinary invocation
                // searches an index built earlier, so fall back to the repository - otherwise no
                // result can ever carry matching messages or the messages --context expands over.
                if (!_sessionCache.TryGetValue(sessionId, out var resolvedSession) &&
                    _sessionRepository != null)
                {
                    resolvedSession = await _sessionRepository
                        .GetSessionAsync(sessionId, ct)
                        .ConfigureAwait(false);
                }

                if (resolvedSession != null)
                {
                    session = resolvedSession;

                    var matchedIds = hits.MessageIds.ToHashSet(StringComparer.Ordinal);
                    var matchedMessages = session.Messages
                        .Where(m => matchedIds.Contains(m.Id))
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
                    Score: hits.TopScore,
                    MatchingMessages: matchingMessages,
                    Highlight: highlight,
                    MatchedMessageIds: hits.MessageIds
                ));
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
        EnsureWritable();

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
        EnsureWritable();

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
            int docCount = 0;
            int maxDoc = 0;

            if (_writer != null)
            {
                docCount = _writer.NumDocs;
                maxDoc = _writer.MaxDoc;
            }
            else if (_searcherManager != null)
            {
                var searcher = _searcherManager.Acquire();
                try
                {
                    docCount = searcher.IndexReader.NumDocs;
                    maxDoc = searcher.IndexReader.MaxDoc;
                }
                finally
                {
                    _searcherManager.Release(searcher);
                }
            }

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

    /// <summary>
    /// Writes one session-level document plus one document per message.
    /// </summary>
    /// <remarks>
    /// The combined <c>all_content</c> text lives on a single session document. It used to be
    /// copied onto every message document, which made the index grow with
    /// messages x session length - a few hundred ordinary messages produced hundreds of megabytes
    /// of postings, so indexing a real corpus never finished.
    /// </remarks>
    private void AddSessionDocuments(Session session)
    {
        var allContent = string.Join(" ", session.Messages.Select(m => m.Content ?? ""));
        _writer!.AddDocument(CreateSessionDocument(session, allContent));

        foreach (var message in session.Messages)
        {
            _writer!.AddDocument(CreateDocument(session, message));
        }
    }

    private static Document CreateSessionDocument(Session session, string allContent)
    {
        var doc = new Document();

        doc.Add(new StringField(FIELD_DOC_TYPE, DOC_TYPE_SESSION, Field.Store.YES));
        doc.Add(new StringField(FIELD_SESSION_ID, session.Id, Field.Store.YES));
        doc.Add(new StringField(FIELD_AGENT_TYPE, session.AgentType, Field.Store.YES));

        if (!string.IsNullOrEmpty(session.ProjectPath))
        {
            doc.Add(new StringField(FIELD_PROJECT_PATH, session.ProjectPath, Field.Store.YES));
        }

        // Session start time, so a result built purely from the index still carries a timestamp.
        doc.Add(new Int64Field(FIELD_TIMESTAMP, session.StartedAt.Ticks, Field.Store.YES));

        if (!string.IsNullOrEmpty(allContent))
        {
            doc.Add(new TextField(FIELD_ALL_CONTENT, allContent, Field.Store.NO));
        }

        return doc;
    }

    private static Document CreateDocument(Session session, Message message)
    {
        var doc = new Document();

        // Store fields (not analyzed)
        doc.Add(new StringField(FIELD_DOC_TYPE, DOC_TYPE_MESSAGE, Field.Store.YES));
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
        if (_directory == null || _analyzer == null)
        {
            throw new InvalidOperationException("LuceneSearchEngine must be initialized before use. Call InitializeAsync first.");
        }
    }

    private void EnsureWritable()
    {
        EnsureInitialized();

        if (!_readOnly && _writer != null)
        {
            return; // Already in write mode
        }

        // Upgrade from read-only to read-write
        _indexLock.EnterWriteLock();
        try
        {
            if (!_readOnly && _writer != null)
            {
                return; // Double-check after acquiring lock
            }

            // Dispose the read-only SearcherManager before upgrading
            _searcherManager?.Dispose();
            _searcherManager = null;

            // Do NOT force-unlock here. IndexWriter.Unlock() cannot distinguish a stale lock left
            // by a crash from a live lock held by another agent-journal process, and stealing a
            // live lock puts two IndexWriters on one directory, which corrupts the index. Let the
            // LockObtainFailedException below surface as an actionable error instead.

            var indexConfig = new IndexWriterConfig(LUCENE_VERSION, _analyzer!)
            {
                OpenMode = OpenMode.CREATE_OR_APPEND,
                Similarity = new BM25Similarity()
            };

            _writer = new IndexWriter(_directory!, indexConfig);
            _writer.Commit();
            _searcherManager = new SearcherManager(_writer, true, null);
            _readOnly = false;
        }
        catch (LockObtainFailedException)
        {
            // Re-establish read-only SearcherManager if upgrade failed
            if (_searcherManager == null && DirectoryReader.IndexExists(_directory!))
            {
                _searcherManager = new SearcherManager(_directory!, null);
            }
            throw new InvalidOperationException(
                "Cannot acquire write lock — another process (e.g., MCP server) holds it. " +
                "Stop the other agent-journal process to enable writes.");
        }
        finally
        {
            _indexLock.ExitWriteLock();
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

    /// <summary>
    /// Accumulates the per-message Lucene hits belonging to one session while searching.
    /// </summary>
    private sealed class SessionHits
    {
        public SessionHits(Document topDoc, float topScore)
        {
            TopDoc = topDoc;
            TopScore = topScore;
        }

        /// <summary>The highest-scoring document for the session, used for session metadata and highlighting.</summary>
        public Document TopDoc { get; private set; }

        /// <summary>The score of <see cref="TopDoc"/>, used as the session's overall relevance.</summary>
        public float TopScore { get; }

        /// <summary>
        /// Whether <see cref="TopDoc"/> is a document whose own message content matched, rather than
        /// one pulled in only by the session-level <c>all_content</c> field.
        /// </summary>
        public bool TopDocIsMessageMatch { get; private set; }

        /// <summary>Message ids that matched, in descending score order.</summary>
        public List<string> MessageIds { get; } = new();

        /// <summary>
        /// Promotes a matching message document to be the highlight source. The session document
        /// stores no message text, so without this there would be nothing to highlight from.
        /// </summary>
        public void OfferMessageMatchDoc(Document doc)
        {
            if (TopDocIsMessageMatch)
            {
                return;
            }

            TopDoc = doc;
            TopDocIsMessageMatch = true;
        }
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
