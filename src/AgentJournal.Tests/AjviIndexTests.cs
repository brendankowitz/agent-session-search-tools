using System.Security.Cryptography;
using System.Text;
using AgentJournal.Core.Search;
using Xunit;

namespace AgentJournal.Tests;

/// <summary>
/// Round-trip and search tests for the AJVI on-disk vector index.
/// </summary>
public class AjviIndexTests : IDisposable
{
    private const int Dimensions = 384;

    private readonly string _directory;
    private readonly string _indexPath;

    public AjviIndexTests()
    {
        _directory = Path.Combine(Path.GetTempPath(), "ajvi-tests-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_directory);
        _indexPath = Path.Combine(_directory, "index.ajvi");
    }

    public void Dispose()
    {
        try
        {
            if (Directory.Exists(_directory))
            {
                Directory.Delete(_directory, recursive: true);
            }
        }
        catch (IOException)
        {
            // A leaked handle should not fail the test run.
        }

        GC.SuppressFinalize(this);
    }

    [Fact]
    public void AddEntry_ShouldRoundTripAllFields()
    {
        var contentHash = HashOf("entry 0");
        var messageId = Guid.NewGuid();
        var vector = CreateNormalizedVector(seed: 7);
        const byte agentType = 1;
        const long timestamp = 1_700_000_000_000L;

        using var index = AjviIndex.Create(_indexPath, Dimensions);
        index.AddEntry(contentHash, messageId, agentType, timestamp, vector);

        Assert.Equal(1, index.EntryCount);
        Assert.Equal(messageId, index.GetMessageId(0));
        Assert.Equal(agentType, index.GetAgentType(0));
        Assert.Equal(timestamp, index.GetTimestamp(0));
        Assert.Equal(contentHash, index.GetContentHash(0));
        Assert.Equal(Dimensions, index.GetVector(0).Length);
    }

    [Fact]
    public void Search_ShouldRankTheExactMatchFirst()
    {
        var target = CreateNormalizedVector(seed: 3);

        using var index = AjviIndex.Create(_indexPath, Dimensions);
        for (var i = 0; i < 10; i++)
        {
            index.AddEntry(
                HashOf($"entry {i}"),
                Guid.NewGuid(),
                agentType: (byte)(i % 2),
                timestamp: i,
                vector: CreateNormalizedVector(seed: i));
        }

        var results = index.Search(target, topK: 5);

        Assert.Equal(5, results.Count);
        Assert.Equal(3, results[0].Index);
        // Cosine similarity of a unit vector with itself is 1; Float16 quantisation loses a little.
        Assert.True(results[0].Score > 0.99f, $"Expected self-similarity near 1 but got {results[0].Score}");

        // Results must be ordered by descending score.
        for (var i = 1; i < results.Count; i++)
        {
            Assert.True(
                results[i - 1].Score >= results[i].Score,
                $"Result {i} scored {results[i].Score}, higher than the preceding {results[i - 1].Score}");
        }
    }

    [Fact]
    public void Search_ShouldRejectQueryWithWrongDimensions()
    {
        using var index = AjviIndex.Create(_indexPath, Dimensions);
        index.AddEntry(HashOf("only"), Guid.NewGuid(), 0, 0, CreateNormalizedVector(seed: 1));

        // Search takes a ReadOnlySpan, so the query has to be materialised outside the lambda.
        var shortQuery = new float[Dimensions - 1];
        Assert.Throws<ArgumentException>(() => index.Search(shortQuery, topK: 1));
    }

    [Fact]
    public void ContainsHash_ShouldDistinguishKnownFromUnknownContent()
    {
        var known = HashOf("known content");

        using var index = AjviIndex.Create(_indexPath, Dimensions);
        index.AddEntry(known, Guid.NewGuid(), 0, 0, CreateNormalizedVector(seed: 2));

        Assert.True(index.ContainsHash(known));
        Assert.False(index.ContainsHash(HashOf("never added")));
    }

    [Fact]
    public void Open_ShouldSeePersistedEntriesAndHeader()
    {
        var messageId = Guid.NewGuid();

        using (var index = AjviIndex.Create(_indexPath, Dimensions, AjviIndex.VectorPrecision.Float16))
        {
            index.AddEntry(HashOf("persisted"), messageId, 1, 42, CreateNormalizedVector(seed: 5));
        }

        using var reopened = AjviIndex.Open(_indexPath, readOnly: true);

        Assert.Equal(1, reopened.EntryCount);
        Assert.Equal(Dimensions, reopened.Dimensions);
        Assert.Equal(AjviIndex.VectorPrecision.Float16, reopened.Precision);
        Assert.Equal(messageId, reopened.GetMessageId(0));
    }

    [Fact]
    public void GetVector_ShouldRejectOutOfRangeIndex()
    {
        using var index = AjviIndex.Create(_indexPath, Dimensions);
        index.AddEntry(HashOf("only"), Guid.NewGuid(), 0, 0, CreateNormalizedVector(seed: 1));

        Assert.Throws<ArgumentOutOfRangeException>(() => index.GetVector(1));
        Assert.Throws<ArgumentOutOfRangeException>(() => index.GetMessageId(-1));
    }

    private static byte[] HashOf(string content) => SHA256.HashData(Encoding.UTF8.GetBytes(content));

    private static float[] CreateNormalizedVector(int seed)
    {
        var random = new Random(seed);
        var vector = new float[Dimensions];
        float sumSquares = 0;

        for (var i = 0; i < Dimensions; i++)
        {
            vector[i] = (float)((random.NextDouble() * 2) - 1);
            sumSquares += vector[i] * vector[i];
        }

        var magnitude = MathF.Sqrt(sumSquares);
        for (var i = 0; i < Dimensions; i++)
        {
            vector[i] /= magnitude;
        }

        return vector;
    }
}
