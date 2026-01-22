namespace AgentJournal.Core.Models;

/// <summary>
/// Represents a discrete piece of knowledge or fact stored in the knowledge bank
/// </summary>
public record KnowledgeEntry(
    string Id,
    string Content,
    string[] Tags,
    string? Project,
    string? Source,
    DateTime CreatedAt,
    DateTime LastReinforcedAt,
    int ReinforcementCount
)
{
    /// <summary>
    /// Gets the time since last reinforcement
    /// </summary>
    public TimeSpan TimeSinceReinforcement => DateTime.UtcNow - LastReinforcedAt;

    /// <summary>
    /// Gets the number of days since last reinforcement
    /// </summary>
    public double DaysSinceReinforcement => TimeSinceReinforcement.TotalDays;
}
