namespace AgentJournal.Core.Models;

/// <summary>
/// Represents indexed content from markdown files or direct posts
/// </summary>
public record ContentEntry(
    string Id,
    string Title,
    string Content,
    string Source,
    string? Project,
    string[]? Tags,
    DateTimeOffset CreatedAt,
    DateTimeOffset LastReinforcedAt,
    string ContentHash
)
{
    /// <summary>
    /// Gets the time since last reinforcement
    /// </summary>
    public TimeSpan TimeSinceReinforcement => DateTimeOffset.UtcNow - LastReinforcedAt;

    /// <summary>
    /// Gets the number of days since last reinforcement
    /// </summary>
    public double DaysSinceReinforcement => TimeSinceReinforcement.TotalDays;
}
