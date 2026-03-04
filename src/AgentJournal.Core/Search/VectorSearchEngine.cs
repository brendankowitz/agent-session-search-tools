using System.Collections.Concurrent;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using AgentJournal.Core.Embeddings;
using AgentJournal.Core.Models;
using AgentJournal.Core.Knowledge;

namespace AgentJournal.Core.Search;

/// <summary>
/// Vector-based semantic search engine using AJVI index
/// </summary>
public class VectorSearchEngine : ISearchEngine, IDisposable
{
    private readonly string _indexPath;
    private readonly IEmbeddingProvider _embedder;
    private readonly ConcurrentDictionary<string, Session> _sessionCache = new();
    private readonly ConcurrentDictionary<Guid, string> _messageToSessionMap = new();
    private readonly ConcurrentDictionary<string, KnowledgeEntry> _knowledgeCache = new();
    private readonly ConcurrentDictionary<Guid, string> _messageToKnowledgeMap = new();
    private readonly SemaphoreSlim _initLock = new(1, 1);
    private readonly ReaderWriterLockSlim _rwLock = new(LockRecursionPolicy.NoRecursion);

    private AjviIndex? _index;
    private bool _initialized;
    private bool _disposed;

    public IReadOnlyList<SearchMode> SupportedModes { get; } = [SearchMode.Semantic, SearchMode.Hybrid];

    public VectorSearchEngine(string indexPath, IEmbeddingProvider embedder)
    {
        _indexPath = indexPath;
        _embedder = embedder;
    }

    /// <summary>
    /// Initializes the vector search engine and creates or opens the AJVI index
    /// </summary>
    public async Task InitializeAsync(CancellationToken ct = default)
    {
        if (_initialized) return;

        await _initLock.WaitAsync(ct).ConfigureAwait(false);
        try
        {
            if (_initialized) return; // Double-check

            var ajviPath = Path.Combine(_indexPath, "index.ajvi");

            if (File.Exists(ajviPath))
            {
                _index = AjviIndex.Open(ajviPath, readOnly: false);
                await LoadMappingsAsync(ct);
            }
            else
            {
                Directory.CreateDirectory(_indexPath);
                _index = AjviIndex.Create(ajviPath, _embedder.Dimensions, AjviIndex.VectorPrecision.Float16);
            }

            _initialized = true;
        }
        finally
        {
            _initLock.Release();
        }
    }

    /// <summary>
    /// Indexes a single session and all its messages
    /// </summary>
    public async Task IndexSessionAsync(Session session, CancellationToken ct = default)
    {
        EnsureInitialized();

        // Cache the session for retrieval during search
        _sessionCache[session.Id] = session;

        // Process each message that has content
        var messagesToEmbed = session.Messages
            .Where(m => !string.IsNullOrWhiteSpace(m.Content))
            .ToList();

        if (messagesToEmbed.Count == 0)
        {
            return;
        }

        // Batch embed all messages
        var contents = messagesToEmbed.Select(m => m.Content).ToList();
        var embeddings = await _embedder.EmbedBatchAsync(contents, ct);

        _rwLock.EnterWriteLock();
        try
        {
            // Add each embedding to the index
            for (int i = 0; i < messagesToEmbed.Count; i++)
            {
                var message = messagesToEmbed[i];
                var embedding = embeddings[i];

                // Calculate content hash for deduplication
                var contentHash = ComputeContentHash(message.Content);

                // Skip if already indexed (deduplication)
                if (_index!.ContainsHash(contentHash))
                {
                    continue;
                }

                // Normalize the embedding vector
                _embedder.Normalize(embedding);

                // Map agent type to byte: copilot-cli = 0, claude-code = 1, others = 2
                byte agentType = session.AgentType.ToLowerInvariant() switch
                {
                    "copilot-cli" => 0,
                    "claude-code" => 1,
                    _ => 2
                };

                // Convert timestamp to Unix milliseconds
                var timestamp = new DateTimeOffset(message.Timestamp).ToUnixTimeMilliseconds();

                // Create deterministic GUID from message ID for consistent indexing
                var messageGuid = Guid.TryParse(message.Id, out var guid)
                    ? guid
                    : CreateGuidFromString(message.Id);

                // Add entry to index
                _index.AddEntry(contentHash, messageGuid, agentType, timestamp, embedding);

                // Map message ID to session ID for reverse lookup
                _messageToSessionMap[messageGuid] = session.Id;
            }
        }
        finally
        {
            _rwLock.ExitWriteLock();
        }
    }

    /// <summary>
    /// Creates a deterministic GUID from a string using MD5 hash
    /// </summary>
    private static Guid CreateGuidFromString(string input)
    {
        using var md5 = System.Security.Cryptography.MD5.Create();
        var hash = md5.ComputeHash(Encoding.UTF8.GetBytes(input));
        return new Guid(hash);
    }

    /// <summary>
    /// Indexes multiple sessions in batch with progress tracking
    /// </summary>
    public async Task IndexSessionsAsync(IEnumerable<Session> sessions, CancellationToken ct = default)
    {
        EnsureInitialized();

        // Phase 1: Embed all sessions OUTSIDE the lock (async-safe)
        var sessionEmbeddings = new List<(Session Session, List<Message> Messages, float[][] Embeddings)>();

        foreach (var session in sessions)
        {
            ct.ThrowIfCancellationRequested();

            // Cache the session for retrieval during search
            _sessionCache[session.Id] = session;

            // Process each message that has content
            var messagesToEmbed = session.Messages
                .Where(m => !string.IsNullOrWhiteSpace(m.Content))
                .ToList();

            if (messagesToEmbed.Count == 0)
            {
                continue;
            }

            // Batch embed all messages (outside the lock)
            var contents = messagesToEmbed.Select(m => m.Content).ToList();
            var embeddings = await _embedder.EmbedBatchAsync(contents, ct);

            sessionEmbeddings.Add((session, messagesToEmbed, embeddings));
        }

        // Phase 2: Mutate the index INSIDE the lock (synchronous only)
        _rwLock.EnterWriteLock();
        try
        {
            foreach (var (session, messagesToEmbed, embeddings) in sessionEmbeddings)
            {
                // Add each embedding to the index
                for (int i = 0; i < messagesToEmbed.Count; i++)
                {
                    var message = messagesToEmbed[i];
                    var embedding = embeddings[i];

                    // Calculate content hash for deduplication
                    var contentHash = ComputeContentHash(message.Content);

                    // Skip if already indexed (deduplication)
                    if (_index!.ContainsHash(contentHash))
                    {
                        continue;
                    }

                    // Normalize the embedding vector
                    _embedder.Normalize(embedding);

                    // Map agent type to byte: copilot-cli = 0, claude-code = 1, others = 2
                    byte agentType = session.AgentType.ToLowerInvariant() switch
                    {
                        "copilot-cli" => 0,
                        "claude-code" => 1,
                        _ => 2
                    };

                    // Convert timestamp to Unix milliseconds
                    var timestamp = new DateTimeOffset(message.Timestamp).ToUnixTimeMilliseconds();

                    // Create deterministic GUID from message ID for consistent indexing
                    var messageGuid = Guid.TryParse(message.Id, out var guid)
                        ? guid
                        : CreateGuidFromString(message.Id);

                    // Add entry to index
                    _index.AddEntry(contentHash, messageGuid, agentType, timestamp, embedding);

                    // Map message ID to session ID for reverse lookup
                    _messageToSessionMap[messageGuid] = session.Id;
                }
            }
        }
        finally
        {
            _rwLock.ExitWriteLock();
        }
    }

    /// <summary>
    /// Indexes a knowledge entry for semantic search
    /// </summary>
    public async Task IndexKnowledgeAsync(KnowledgeEntry entry, CancellationToken ct = default)
    {
        EnsureInitialized();

        // Cache the knowledge entry for retrieval during search
        _knowledgeCache[entry.Id] = entry;

        if (string.IsNullOrWhiteSpace(entry.Content))
        {
            return;
        }

        // Embed the content
        var embedding = await _embedder.EmbedAsync(entry.Content, ct);

        _rwLock.EnterWriteLock();
        try
        {
            // Calculate content hash for deduplication
            var contentHash = ComputeContentHash(entry.Content);

            // Skip if already indexed (deduplication)
            if (_index!.ContainsHash(contentHash))
            {
                return;
            }

            // Normalize the embedding vector
            _embedder.Normalize(embedding);

            // Use agent type byte = 3 for knowledge entries to distinguish from sessions
            byte agentType = 3;

            // Convert timestamp to Unix milliseconds
            var timestamp = new DateTimeOffset(entry.CreatedAt).ToUnixTimeMilliseconds();

            // Create deterministic GUID from knowledge ID with "knowledge:" prefix
            var knowledgeGuid = CreateGuidFromString("knowledge:" + entry.Id);

            // Add entry to index
            _index.AddEntry(contentHash, knowledgeGuid, agentType, timestamp, embedding);

            // Map knowledge GUID to knowledge ID for reverse lookup
            _messageToKnowledgeMap[knowledgeGuid] = entry.Id;
        }
        finally
        {
            _rwLock.ExitWriteLock();
        }
    }

    /// <summary>
    /// Searches knowledge entries using semantic search
    /// </summary>
    public async Task<IReadOnlyList<KnowledgeSearchResult>> SearchKnowledgeAsync(
        string query,
        int maxResults = 10,
        double halfLifeDays = DecayCalculator.DefaultHalfLifeDays,
        CancellationToken ct = default)
    {
        if (string.IsNullOrWhiteSpace(query))
        {
            return Array.Empty<KnowledgeSearchResult>();
        }

        EnsureInitialized();

        // Embed the query
        var queryEmbedding = await _embedder.EmbedAsync(query, ct);
        _embedder.Normalize(queryEmbedding);

        var results = new List<KnowledgeSearchResult>();

        _rwLock.EnterReadLock();
        try
        {
            // Search the index for similar vectors
            var topK = Math.Max(maxResults * 2, 20);
            var searchResults = _index!.Search(queryEmbedding, topK);

            foreach (var (index, score) in searchResults)
            {
                var messageId = _index.GetMessageId(index);

                // Check if this is a knowledge entry (not a session message)
                if (_messageToKnowledgeMap.TryGetValue(messageId, out var knowledgeId))
                {
                    if (_knowledgeCache.TryGetValue(knowledgeId, out var entry))
                    {
                        // Calculate decay factor
                        var decayFactor = DecayCalculator.CalculateDecayFactor(entry.LastReinforcedAt, halfLifeDays);

                        // Apply decay to score
                        var adjustedScore = DecayCalculator.ApplyDecay(score, decayFactor);

                        // Get highlight
                        var highlight = GetHighlight(entry.Content, query);

                        results.Add(new KnowledgeSearchResult(entry, adjustedScore, decayFactor, highlight));
                    }
                }
            }
        }
        finally
        {
            _rwLock.ExitReadLock();
        }

        // Sort by adjusted score and take top results
        return results
            .OrderByDescending(r => r.Score)
            .Take(maxResults)
            .ToList();
    }

    /// <summary>
    /// Performs semantic search by embedding the query and finding similar vectors
    /// </summary>
    public async Task<IReadOnlyList<SearchResult>> SearchAsync(
        string query,
        SearchMode mode = SearchMode.Semantic,
        int maxResults = 10,
        int contextCount = 0,
        CancellationToken ct = default)
    {
        if (mode != SearchMode.Semantic && mode != SearchMode.Hybrid)
        {
            throw new NotSupportedException($"VectorSearchEngine does not support {mode} search mode.");
        }

        if (string.IsNullOrWhiteSpace(query))
        {
            return Array.Empty<SearchResult>();
        }

        EnsureInitialized();

        // Embed the query
        var queryEmbedding = await _embedder.EmbedAsync(query, ct);
        _embedder.Normalize(queryEmbedding);

        // Group results by session and aggregate scores
        var sessionScores = new Dictionary<string, (double Score, List<Guid> MessageIds)>();

        _rwLock.EnterReadLock();
        try
        {
            // Search the index for similar vectors
            var topK = Math.Max(maxResults * 3, 50); // Get more results for better session aggregation
            var searchResults = _index!.Search(queryEmbedding, topK);

            foreach (var (index, score) in searchResults)
            {
                var messageId = _index.GetMessageId(index);

                if (_messageToSessionMap.TryGetValue(messageId, out var sessionId))
                {
                    if (!sessionScores.TryGetValue(sessionId, out var sessionData))
                    {
                        sessionData = (0.0, new List<Guid>());
                        sessionScores[sessionId] = sessionData;
                    }

                    // Add message ID to the list
                    sessionData.MessageIds.Add(messageId);

                    // Update score (use max score for the session)
                    if (score > sessionData.Score)
                    {
                        sessionScores[sessionId] = (score, sessionData.MessageIds);
                    }
                }
            }
        }
        finally
        {
            _rwLock.ExitReadLock();
        }

        // Build search results
        var results = new List<SearchResult>();

        foreach (var (sessionId, (score, messageIds)) in sessionScores
            .OrderByDescending(kvp => kvp.Value.Score)
            .Take(maxResults))
        {
            if (_sessionCache.TryGetValue(sessionId, out var session))
            {
                // Find matching messages by converting message IDs to GUIDs
                var matchedMessages = session.Messages
                    .Where(m =>
                    {
                        var msgGuid = Guid.TryParse(m.Id, out var g) ? g : CreateGuidFromString(m.Id);
                        return messageIds.Contains(msgGuid);
                    })
                    .ToList();

                // Expand with context if requested
                var matchingMessages = ExpandWithContext(session.Messages, matchedMessages, contextCount);

                // Get highlight from the best matching message
                string? highlight = null;
                if (matchedMessages.Count > 0)
                {
                    var bestMessage = matchedMessages.First();
                    highlight = GetHighlight(bestMessage.Content, query);
                }

                results.Add(new SearchResult(
                    Session: session,
                    Score: score,
                    MatchingMessages: matchingMessages,
                    Highlight: highlight
                ));
            }
        }

        return results;
    }

    /// <summary>
    /// Clears all indexed data and recreates the index
    /// </summary>
    public async Task ClearIndexAsync(CancellationToken ct = default)
    {
        _rwLock.EnterWriteLock();
        try
        {
            if (_index != null)
            {
                _index.Dispose();
                _index = null;
            }

            var indexFile = Path.Combine(_indexPath, "index.ajvi");
            if (File.Exists(indexFile))
            {
                File.Delete(indexFile);
            }

            // Also delete mappings files
            var mappingsFile = Path.Combine(_indexPath, "mappings.json");
            if (File.Exists(mappingsFile))
            {
                File.Delete(mappingsFile);
            }
            var sessionsFile = Path.Combine(_indexPath, "sessions.json");
            if (File.Exists(sessionsFile))
            {
                File.Delete(sessionsFile);
            }
            var knowledgeFile = Path.Combine(_indexPath, "knowledge.json");
            if (File.Exists(knowledgeFile))
            {
                File.Delete(knowledgeFile);
            }

            _sessionCache.Clear();
            _messageToSessionMap.Clear();
            _knowledgeCache.Clear();
            _messageToKnowledgeMap.Clear();
            _initialized = false;
        }
        finally
        {
            _rwLock.ExitWriteLock();
        }

        // Reinitialize
        await InitializeAsync(ct);
    }

    /// <summary>
    /// Computes SHA256 hash of the message content for deduplication
    /// </summary>
    private static byte[] ComputeContentHash(string content)
    {
        var bytes = Encoding.UTF8.GetBytes(content);
        return SHA256.HashData(bytes);
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

    private void EnsureInitialized()
    {
        if (!_initialized || _index == null)
        {
            throw new InvalidOperationException("VectorSearchEngine must be initialized before use. Call InitializeAsync first.");
        }
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        // Save mappings before disposing
        SaveMappingsSync();

        _index?.Dispose();
        _index = null;
        _initLock.Dispose();
        _rwLock.Dispose();
        _disposed = true;
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Saves the message-to-session mappings to disk
    /// </summary>
    private void SaveMappingsSync()
    {
        try
        {
            var mappingsPath = Path.Combine(_indexPath, "mappings.json");
            var mappingsData = _messageToSessionMap.ToDictionary(
                kvp => kvp.Key.ToString(),
                kvp => kvp.Value);
            var json = JsonSerializer.Serialize(mappingsData);
            File.WriteAllText(mappingsPath, json);

            // Also save session cache (minimal version)
            var sessionsPath = Path.Combine(_indexPath, "sessions.json");
            var sessionsData = _sessionCache.Values.Select(s => new
            {
                s.Id,
                s.AgentType,
                s.Summary,
                s.ProjectPath,
                s.StartedAt,
                s.EndedAt,
                s.MessageCount
            }).ToList();
            var sessionsJson = JsonSerializer.Serialize(sessionsData);
            File.WriteAllText(sessionsPath, sessionsJson);

            // Save knowledge mappings
            var knowledgeMappingsPath = Path.Combine(_indexPath, "knowledge_mappings.json");
            var knowledgeMappingsData = _messageToKnowledgeMap.ToDictionary(
                kvp => kvp.Key.ToString(),
                kvp => kvp.Value);
            var knowledgeMappingsJson = JsonSerializer.Serialize(knowledgeMappingsData);
            File.WriteAllText(knowledgeMappingsPath, knowledgeMappingsJson);

            // Save knowledge cache
            var knowledgePath = Path.Combine(_indexPath, "knowledge.json");
            var knowledgeData = _knowledgeCache.Values.Select(k => new
            {
                k.Id,
                k.Content,
                k.Tags,
                k.Project,
                k.Source,
                k.CreatedAt,
                k.LastReinforcedAt,
                k.ReinforcementCount
            }).ToList();
            var knowledgeJson = JsonSerializer.Serialize(knowledgeData);
            File.WriteAllText(knowledgePath, knowledgeJson);
        }
        catch
        {
            // Ignore save errors - mappings will be rebuilt on next index
        }
    }

    /// <summary>
    /// Loads the message-to-session mappings from disk
    /// </summary>
    private async Task LoadMappingsAsync(CancellationToken ct)
    {
        try
        {
            var mappingsPath = Path.Combine(_indexPath, "mappings.json");
            if (File.Exists(mappingsPath))
            {
                var json = await File.ReadAllTextAsync(mappingsPath, ct);
                var mappingsData = JsonSerializer.Deserialize<Dictionary<string, string>>(json);
                if (mappingsData != null)
                {
                    foreach (var kvp in mappingsData)
                    {
                        if (Guid.TryParse(kvp.Key, out var guid))
                        {
                            _messageToSessionMap[guid] = kvp.Value;
                        }
                    }
                }
            }

            // Load minimal session data for result building
            var sessionsPath = Path.Combine(_indexPath, "sessions.json");
            if (File.Exists(sessionsPath))
            {
                var json = await File.ReadAllTextAsync(sessionsPath, ct);
                using var doc = JsonDocument.Parse(json);
                foreach (var element in doc.RootElement.EnumerateArray())
                {
                    var id = element.GetProperty("Id").GetString();
                    if (id == null) continue;

                    var session = new Session(
                        Id: id,
                        AgentType: element.TryGetProperty("AgentType", out var at) ? at.GetString() ?? "unknown" : "unknown",
                        ProjectPath: element.TryGetProperty("ProjectPath", out var pp) ? pp.GetString() : null,
                        GitBranch: null,
                        AgentVersion: null,
                        StartedAt: element.TryGetProperty("StartedAt", out var st) ? st.GetDateTime() : DateTime.MinValue,
                        EndedAt: element.TryGetProperty("EndedAt", out var et) && et.ValueKind != JsonValueKind.Null ? et.GetDateTime() : null,
                        LastModified: null,
                        Summary: element.TryGetProperty("Summary", out var s) ? s.GetString() : null,
                        Messages: Array.Empty<Message>()
                    );
                    _sessionCache[id] = session;
                }
            }

            // Load knowledge mappings
            var knowledgeMappingsPath = Path.Combine(_indexPath, "knowledge_mappings.json");
            if (File.Exists(knowledgeMappingsPath))
            {
                var json = await File.ReadAllTextAsync(knowledgeMappingsPath, ct);
                var mappingsData = JsonSerializer.Deserialize<Dictionary<string, string>>(json);
                if (mappingsData != null)
                {
                    foreach (var kvp in mappingsData)
                    {
                        if (Guid.TryParse(kvp.Key, out var guid))
                        {
                            _messageToKnowledgeMap[guid] = kvp.Value;
                        }
                    }
                }
            }

            // Load knowledge cache
            var knowledgePath = Path.Combine(_indexPath, "knowledge.json");
            if (File.Exists(knowledgePath))
            {
                var json = await File.ReadAllTextAsync(knowledgePath, ct);
                using var doc = JsonDocument.Parse(json);
                foreach (var element in doc.RootElement.EnumerateArray())
                {
                    var id = element.GetProperty("Id").GetString();
                    if (id == null) continue;

                    var tagsJson = element.TryGetProperty("Tags", out var tagsElement)
                        ? tagsElement.EnumerateArray().Select(t => t.GetString() ?? "").ToArray()
                        : Array.Empty<string>();

                    var entry = new KnowledgeEntry(
                        Id: id,
                        Content: element.TryGetProperty("Content", out var c) ? c.GetString() ?? "" : "",
                        Tags: tagsJson,
                        Project: element.TryGetProperty("Project", out var p) && p.ValueKind != JsonValueKind.Null ? p.GetString() : null,
                        Source: element.TryGetProperty("Source", out var src) && src.ValueKind != JsonValueKind.Null ? src.GetString() : null,
                        CreatedAt: element.TryGetProperty("CreatedAt", out var ca) ? ca.GetDateTime() : DateTime.MinValue,
                        LastReinforcedAt: element.TryGetProperty("LastReinforcedAt", out var lra) ? lra.GetDateTime() : DateTime.MinValue,
                        ReinforcementCount: element.TryGetProperty("ReinforcementCount", out var rc) ? rc.GetInt32() : 0
                    );
                    _knowledgeCache[id] = entry;
                }
            }
        }
        catch
        {
            // Ignore load errors - mappings will be empty and rebuilt on index
        }
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
}
