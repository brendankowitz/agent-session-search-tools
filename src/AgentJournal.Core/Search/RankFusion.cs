namespace AgentJournal.Core.Search;

/// <summary>
/// Merges ranked result lists that come from different search backends.
/// </summary>
/// <remarks>
/// Sessions are ranked by Lucene, while knowledge entries and task journals are ranked by SQLite
/// FTS5 <c>bm25()</c>. Those scores are not comparable: bm25 depends on corpus statistics, and in a
/// small per-repository corpus its magnitude collapses towards zero. Sorting a combined list by raw
/// score therefore buries exact matches from the smaller sources underneath weak session matches.
/// Fusing on rank sidesteps the problem because only positions are compared.
/// </remarks>
public static class RankFusion
{
    /// <summary>
    /// Constant for Reciprocal Rank Fusion, matching <see cref="HybridSearcher"/>. Larger values
    /// flatten the advantage held by top ranks.
    /// </summary>
    public const int DefaultK = 60;

    /// <summary>
    /// Merges already-ranked result lists using Reciprocal Rank Fusion.
    /// </summary>
    /// <param name="resultsBySource">
    /// One list per source, each already ordered best-first by that source.
    /// </param>
    /// <param name="maxResults">Maximum number of results to return.</param>
    /// <param name="k">Reciprocal Rank Fusion constant.</param>
    /// <returns>
    /// The fused results, best-first. A single source is returned in its original order, because
    /// 1/(k+rank) is strictly decreasing in rank - so ordinary single-source search is unchanged.
    /// </returns>
    public static List<T> Fuse<T>(
        IReadOnlyList<IReadOnlyList<T>> resultsBySource,
        int maxResults,
        int k = DefaultK)
    {
        ArgumentNullException.ThrowIfNull(resultsBySource);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxResults);
        ArgumentOutOfRangeException.ThrowIfNegative(k);

        var scored = new List<(double Score, int Source, int Position, T Result)>();

        for (int source = 0; source < resultsBySource.Count; source++)
        {
            var results = resultsBySource[source];
            for (int position = 0; position < results.Count; position++)
            {
                scored.Add((1.0 / (k + position + 1), source, position, results[position]));
            }
        }

        return scored
            .OrderByDescending(x => x.Score)
            // Sources tie whenever they hold a result at the same position. Break ties on source
            // then position so the output is deterministic rather than reliant on sort stability.
            .ThenBy(x => x.Source)
            .ThenBy(x => x.Position)
            .Take(maxResults)
            .Select(x => x.Result)
            .ToList();
    }
}
