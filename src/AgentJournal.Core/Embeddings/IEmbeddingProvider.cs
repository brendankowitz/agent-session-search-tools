namespace AgentJournal.Core.Embeddings;

/// <summary>
/// Provides text embedding capabilities
/// </summary>
public interface IEmbeddingProvider
{
    /// <summary>Number of dimensions in output vectors</summary>
    int Dimensions { get; }
    
    /// <summary>Whether this provider uses a real ML model</summary>
    bool IsSemanticModel { get; }
    
    /// <summary>Generate embedding for a single text</summary>
    Task<float[]> EmbedAsync(string text, CancellationToken ct = default);
    
    /// <summary>Generate embeddings for multiple texts (batched)</summary>
    Task<float[][]> EmbedBatchAsync(IReadOnlyList<string> texts, CancellationToken ct = default);
    
    /// <summary>L2-normalize a vector in-place</summary>
    void Normalize(Span<float> vector);
}
