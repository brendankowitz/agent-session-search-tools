using AgentJournal.Core.Search;

namespace AgentJournal.Tests;

/// <summary>
/// Pins the cross-source merge used by `search`. The behaviour that matters is that a source with
/// small absolute scores (FTS5 bm25 over a small corpus) is not buried underneath a source with
/// larger absolute scores (Lucene).
/// </summary>
public class RankFusionTests
{
    [Fact]
    public void Fuse_PreservesOrderForASingleSource()
    {
        // Ordinary session-only search must be completely unaffected by fusion.
        var source = new[] { "a", "b", "c", "d" };

        var fused = RankFusion.Fuse(new[] { source }, maxResults: 10);

        Assert.Equal(source, fused);
    }

    [Fact]
    public void Fuse_InterleavesSourcesByRankNotByScore()
    {
        // The regression this exists for: the top hit of a low-scoring source must appear near the
        // top of the merged list, not after every hit of a high-scoring source.
        var sessions = new[] { "session-1", "session-2", "session-3" };
        var tasks = new[] { "task-1" };

        var fused = RankFusion.Fuse(new[] { sessions, tasks }, maxResults: 10);

        Assert.Equal("session-1", fused[0]);
        Assert.Equal("task-1", fused[1]);
        Assert.Equal("session-2", fused[2]);
    }

    [Fact]
    public void Fuse_IsDeterministicWhenSourcesTie()
    {
        var first = new[] { "a1", "a2" };
        var second = new[] { "b1", "b2" };

        var once = RankFusion.Fuse(new[] { first, second }, maxResults: 10);
        var twice = RankFusion.Fuse(new[] { first, second }, maxResults: 10);

        Assert.Equal(once, twice);
        Assert.Equal(new[] { "a1", "b1", "a2", "b2" }, once);
    }

    [Fact]
    public void Fuse_RespectsMaxResults()
    {
        var sessions = new[] { "s1", "s2", "s3" };
        var tasks = new[] { "t1", "t2", "t3" };

        var fused = RankFusion.Fuse(new[] { sessions, tasks }, maxResults: 2);

        Assert.Equal(2, fused.Count);
    }

    [Fact]
    public void Fuse_IgnoresEmptySources()
    {
        var sessions = new[] { "s1", "s2" };
        var tasks = Array.Empty<string>();

        var fused = RankFusion.Fuse(new[] { sessions, tasks }, maxResults: 10);

        Assert.Equal(sessions, fused);
    }

    [Fact]
    public void Fuse_ReturnsEmptyWhenNoSourcesHaveResults()
    {
        var fused = RankFusion.Fuse(new[] { Array.Empty<string>() }, maxResults: 10);

        Assert.Empty(fused);
    }

    [Fact]
    public void Fuse_RejectsNonPositiveMaxResults()
    {
        Assert.Throws<ArgumentOutOfRangeException>(
            () => RankFusion.Fuse(new[] { new[] { "a" } }, maxResults: 0));
    }
}
