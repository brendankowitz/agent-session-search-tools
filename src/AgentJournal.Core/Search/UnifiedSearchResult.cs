using AgentJournal.Core.Models;
using AgentJournal.Core.Knowledge;
using AgentJournal.Core.Tasks;

namespace AgentJournal.Core.Search;

/// <summary>
/// Type of search result
/// </summary>
public enum SearchResultType
{
    /// <summary>
    /// Result is an agent session
    /// </summary>
    Session,

    /// <summary>
    /// Result is a knowledge entry
    /// </summary>
    Knowledge,

    /// <summary>
    /// Result is a task journal note or artifact
    /// </summary>
    Task
}

/// <summary>
/// Unified search result that can hold both sessions and knowledge entries
/// </summary>
public record UnifiedSearchResult
{
    /// <summary>
    /// Unique identifier for the result
    /// </summary>
    public required string Id { get; init; }

    /// <summary>
    /// Type of search result (Session or Knowledge)
    /// </summary>
    public required SearchResultType Type { get; init; }

    /// <summary>
    /// Relevance score
    /// </summary>
    public required double Score { get; init; }

    /// <summary>
    /// Underlying data (either Session or KnowledgeEntry)
    /// </summary>
    public required object Data { get; init; }

    /// <summary>
    /// Optional decay factor for knowledge entries
    /// </summary>
    public double? DecayFactor { get; init; }

    /// <summary>
    /// Optional highlight text
    /// </summary>
    public string? Highlight { get; init; }

    /// <summary>
    /// Optional matching messages for session results
    /// </summary>
    public IReadOnlyList<Message>? MatchingMessages { get; init; }

    /// <summary>
    /// Ids of the messages that actually matched the query. <see cref="MatchingMessages"/> is
    /// context-expanded, so this is the only way to tell a real match from surrounding context.
    /// </summary>
    public IReadOnlyList<string>? MatchedMessageIds { get; init; }

    /// <summary>
    /// Whether the supplied message actually matched the query rather than being included as context.
    /// </summary>
    public bool IsMatch(Message message) =>
        MatchedMessageIds != null && MatchedMessageIds.Contains(message.Id);

    /// <summary>
    /// Creates a unified search result from a session search result
    /// </summary>
    public static UnifiedSearchResult FromSession(SearchResult result)
    {
        return new UnifiedSearchResult
        {
            Id = result.Session.Id,
            Type = SearchResultType.Session,
            Score = result.Score,
            Data = result.Session,
            Highlight = result.Highlight,
            MatchingMessages = result.MatchingMessages,
            MatchedMessageIds = result.MatchedMessageIds
        };
    }

    /// <summary>
    /// Creates a unified search result from a knowledge search result
    /// </summary>
    public static UnifiedSearchResult FromKnowledge(KnowledgeSearchResult result)
    {
        return new UnifiedSearchResult
        {
            Id = result.Entry.Id,
            Type = SearchResultType.Knowledge,
            Score = result.Score,
            Data = result.Entry,
            DecayFactor = result.DecayFactor,
            Highlight = result.Highlight
        };
    }

    /// <summary>
    /// Creates a unified search result from a task journal search result
    /// </summary>
    public static UnifiedSearchResult FromTask(TaskSearchResult result)
    {
        return new UnifiedSearchResult
        {
            // Task rows have no global id, so compose one that is stable and lets a caller run
            // `task show` against the right journal.
            Id = $"{result.JournalName}#{result.TaskNumber}:{result.Kind}",
            Type = SearchResultType.Task,
            Score = result.Score,
            Data = result,
            Highlight = result.Excerpt
        };
    }

    /// <summary>
    /// Gets the underlying session (throws if not a session result)
    /// </summary>
    public Session AsSession()
    {
        if (Type != SearchResultType.Session || Data is not Session session)
        {
            throw new InvalidOperationException("Result is not a session");
        }
        return session;
    }

    /// <summary>
    /// Gets the underlying knowledge entry (throws if not a knowledge result)
    /// </summary>
    public KnowledgeEntry AsKnowledge()
    {
        if (Type != SearchResultType.Knowledge || Data is not KnowledgeEntry entry)
        {
            throw new InvalidOperationException("Result is not a knowledge entry");
        }
        return entry;
    }

    /// <summary>
    /// Tries to get the underlying session
    /// </summary>
    public bool TryGetSession(out Session? session)
    {
        if (Type == SearchResultType.Session && Data is Session s)
        {
            session = s;
            return true;
        }
        session = null;
        return false;
    }

    /// <summary>
    /// Tries to get the underlying knowledge entry
    /// </summary>
    public bool TryGetKnowledge(out KnowledgeEntry? entry)
    {
        if (Type == SearchResultType.Knowledge && Data is KnowledgeEntry e)
        {
            entry = e;
            return true;
        }
        entry = null;
        return false;
    }

    /// <summary>
    /// Tries to get the underlying task journal result
    /// </summary>
    public bool TryGetTask(out TaskSearchResult? task)
    {
        if (Type == SearchResultType.Task && Data is TaskSearchResult t)
        {
            task = t;
            return true;
        }
        task = null;
        return false;
    }
}
