using System.Text.RegularExpressions;

namespace AgentJournal.Core.Embeddings;

/// <summary>
/// Hash-based fallback embedding provider using FNV-1a algorithm.
/// Not semantic, but provides consistent vector representations for keyword matching.
/// </summary>
public sealed partial class HashEmbeddingProvider : IEmbeddingProvider
{
    private const int VectorDimensions = 384; // Match MiniLM dimensions
    private const int ActivationsPerWord = 8;
    private const uint FnvOffsetBasis = 2166136261u;
    private const uint FnvPrime = 16777619u;

    [GeneratedRegex(@"\w+", RegexOptions.Compiled)]
    private static partial Regex WordTokenizer();

    public int Dimensions => VectorDimensions;

    public bool IsSemanticModel => false;

    public Task<float[]> EmbedAsync(string text, CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(text);

        var vector = new float[VectorDimensions];
        var words = TokenizeText(text);

        if (words.Count == 0)
        {
            return Task.FromResult(vector);
        }

        var tfWeight = 1.0f / words.Count;

        foreach (var word in words)
        {
            var hash = ComputeFnv1aHash(word.ToLowerInvariant());
            var random = new Random((int)hash);

            for (int i = 0; i < ActivationsPerWord; i++)
            {
                var dimension = random.Next(VectorDimensions);
                vector[dimension] += tfWeight;
            }
        }

        Normalize(vector);
        return Task.FromResult(vector);
    }

    public async Task<float[][]> EmbedBatchAsync(IReadOnlyList<string> texts, CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(texts);

        var results = new float[texts.Count][];

        for (int i = 0; i < texts.Count; i++)
        {
            ct.ThrowIfCancellationRequested();
            results[i] = await EmbedAsync(texts[i], ct);
        }

        return results;
    }

    public void Normalize(Span<float> vector)
    {
        var sumOfSquares = 0.0f;

        for (int i = 0; i < vector.Length; i++)
        {
            sumOfSquares += vector[i] * vector[i];
        }

        if (sumOfSquares > 0)
        {
            var magnitude = MathF.Sqrt(sumOfSquares);

            for (int i = 0; i < vector.Length; i++)
            {
                vector[i] /= magnitude;
            }
        }
    }

    private static List<string> TokenizeText(string text)
    {
        var matches = WordTokenizer().Matches(text);
        var words = new List<string>(matches.Count);

        foreach (Match match in matches)
        {
            words.Add(match.Value);
        }

        return words;
    }

    private static uint ComputeFnv1aHash(string text)
    {
        var hash = FnvOffsetBasis;

        foreach (var c in text)
        {
            hash ^= c;
            hash *= FnvPrime;
        }

        return hash;
    }
}
