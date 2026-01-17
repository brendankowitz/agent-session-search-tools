using AgentJournal.Core.Models;
using AgentJournal.Core.Knowledge;

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
    Knowledge
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
            MatchingMessages = result.MatchingMessages
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
}
