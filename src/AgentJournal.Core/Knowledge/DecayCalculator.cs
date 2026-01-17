namespace AgentJournal.Core.Knowledge;

/// <summary>
/// Calculates temporal decay for knowledge entries using exponential decay formula
/// </summary>
public static class DecayCalculator
{
    /// <summary>
    /// Default half-life in days (90 days)
    /// </summary>
    public const double DefaultHalfLifeDays = 90.0;

    /// <summary>
    /// Calculates the decay factor based on time since last reinforcement
    /// Formula: 0.5^(days_since_reinforced / half_life)
    /// </summary>
    /// <param name="lastReinforced">When the knowledge was last reinforced</param>
    /// <param name="halfLifeDays">Half-life in days (default: 90)</param>
    /// <returns>Decay factor between 0 and 1</returns>
    public static double CalculateDecayFactor(DateTime lastReinforced, double halfLifeDays = DefaultHalfLifeDays)
    {
        if (halfLifeDays <= 0)
        {
            throw new ArgumentException("Half-life must be positive", nameof(halfLifeDays));
        }

        var daysSinceReinforced = (DateTime.UtcNow - lastReinforced).TotalDays;
        
        // Ensure we don't have future dates (clock skew protection)
        if (daysSinceReinforced < 0)
        {
            daysSinceReinforced = 0;
        }

        return Math.Pow(0.5, daysSinceReinforced / halfLifeDays);
    }

    /// <summary>
    /// Applies decay to a base score
    /// </summary>
    /// <param name="baseScore">Original score</param>
    /// <param name="decayFactor">Decay factor from CalculateDecayFactor</param>
    /// <returns>Score with decay applied</returns>
    public static double ApplyDecay(double baseScore, double decayFactor)
    {
        return baseScore * decayFactor;
    }

    /// <summary>
    /// Applies decay to a base score using last reinforced timestamp
    /// </summary>
    /// <param name="baseScore">Original score</param>
    /// <param name="lastReinforced">When the knowledge was last reinforced</param>
    /// <param name="halfLifeDays">Half-life in days (default: 90)</param>
    /// <returns>Score with decay applied</returns>
    public static double ApplyDecay(double baseScore, DateTime lastReinforced, double halfLifeDays = DefaultHalfLifeDays)
    {
        var decayFactor = CalculateDecayFactor(lastReinforced, halfLifeDays);
        return ApplyDecay(baseScore, decayFactor);
    }

    /// <summary>
    /// Determines decay status based on decay factor
    /// </summary>
    /// <param name="decayFactor">Decay factor between 0 and 1</param>
    /// <returns>Status string (Fresh, Good, Aging, Decaying, Expiring)</returns>
    public static string GetDecayStatus(double decayFactor)
    {
        return decayFactor switch
        {
            > 0.75 => "Fresh",
            > 0.50 => "Good",
            > 0.25 => "Aging",
            > 0.10 => "Decaying",
            _ => "Expiring"
        };
    }

    /// <summary>
    /// Checks if knowledge should be considered expired
    /// </summary>
    /// <param name="decayFactor">Decay factor between 0 and 1</param>
    /// <param name="threshold">Expiration threshold (default: 0.05)</param>
    /// <returns>True if expired</returns>
    public static bool IsExpired(double decayFactor, double threshold = 0.05)
    {
        return decayFactor < threshold;
    }
}
