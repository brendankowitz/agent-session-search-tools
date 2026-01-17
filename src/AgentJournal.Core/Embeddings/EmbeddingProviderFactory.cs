namespace AgentJournal.Core.Embeddings;

/// <summary>
/// Factory for creating appropriate embedding providers based on available resources.
/// </summary>
public static class EmbeddingProviderFactory
{
    /// <summary>
    /// Attempts to create an embedding provider, preferring ONNX-based semantic models
    /// but falling back to hash-based embeddings if models are unavailable.
    /// </summary>
    /// <param name="modelsPath">Path to the directory containing model files</param>
    /// <param name="ct">Cancellation token</param>
    /// <returns>An embedding provider (never null)</returns>
    public static async Task<IEmbeddingProvider> TryCreateAsync(
        string? modelsPath = null, 
        CancellationToken ct = default)
    {
        // Try ONNX provider if models path is specified
        if (!string.IsNullOrWhiteSpace(modelsPath))
        {
            var onnxProvider = await OnnxEmbeddingProvider.TryCreateAsync(modelsPath, ct);
            if (onnxProvider is not null)
            {
                return onnxProvider;
            }
        }

        // Fallback to hash-based embeddings
        return new HashEmbeddingProvider();
    }

    /// <summary>
    /// Creates a hash-based embedding provider (non-semantic fallback).
    /// </summary>
    public static IEmbeddingProvider CreateHashProvider()
    {
        return new HashEmbeddingProvider();
    }

    /// <summary>
    /// Attempts to create an ONNX-based semantic embedding provider.
    /// Returns null if model files are not found.
    /// </summary>
    public static async Task<IEmbeddingProvider?> CreateOnnxProviderAsync(
        string modelsPath, 
        CancellationToken ct = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelsPath);
        return await OnnxEmbeddingProvider.TryCreateAsync(modelsPath, ct);
    }
}
