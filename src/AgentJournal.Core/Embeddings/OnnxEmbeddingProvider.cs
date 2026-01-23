using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.Tokenizers;

namespace AgentJournal.Core.Embeddings;

/// <summary>
/// ONNX-based semantic embedding provider using MiniLM model.
/// </summary>
public sealed class OnnxEmbeddingProvider : IEmbeddingProvider, IDisposable
{
    private const int VectorDimensions = 384;
    private const int MaxSequenceLength = 256;
    private const int BatchSize = 32;

    private readonly InferenceSession _session;
    private readonly Tokenizer _tokenizer;
    private readonly SemaphoreSlim _inferenceLock = new(1, 1); // ONNX Session is not thread-safe

    public int Dimensions => VectorDimensions;

    public bool IsSemanticModel => true;

    /// <summary>
    /// The execution provider being used (e.g., "DirectML (GPU)" or "CPU")
    /// </summary>
    public string ExecutionProvider { get; }

    private OnnxEmbeddingProvider(InferenceSession session, Tokenizer tokenizer, string executionProvider)
    {
        _session = session ?? throw new ArgumentNullException(nameof(session));
        _tokenizer = tokenizer ?? throw new ArgumentNullException(nameof(tokenizer));
        ExecutionProvider = executionProvider;
    }

    /// <summary>
    /// Attempts to create an ONNX embedding provider from the specified models path.
    /// Falls back to bundled model if not found on disk.
    /// </summary>
    public static async Task<OnnxEmbeddingProvider?> TryCreateAsync(string modelsPath, CancellationToken ct = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelsPath);

        var modelPath = Path.Combine(modelsPath, "minilm", "model.onnx");
        var tokenizerPath = Path.Combine(modelsPath, "minilm", "tokenizer.json");

        // If model doesn't exist on disk, extract bundled model
        if (!File.Exists(modelPath))
        {
            var extracted = await ExtractBundledModelAsync(modelsPath, ct);
            if (!extracted)
            {
                return null;
            }
        }

        try
        {
            // Create session with auto-detected execution provider (GPU/NPU -> CPU fallback)
            var (session, executionProvider) = CreateSessionWithBestProvider(modelPath);

            // Load tokenizer - BertTokenizer needs vocab.txt for MiniLM/BERT models
            var vocabPath = Path.Combine(modelsPath, "minilm", "vocab.txt");
            Tokenizer tokenizer;

            if (File.Exists(vocabPath))
            {
                // Use vocab.txt for proper BERT/WordPiece tokenization
                tokenizer = await Task.Run(() => BertTokenizer.Create(vocabPath), ct);
            }
            else if (File.Exists(tokenizerPath))
            {
                // Try tokenizer.json as fallback
                try
                {
                    tokenizer = await Task.Run(() => BertTokenizer.Create(tokenizerPath), ct);
                }
                catch
                {
                    // Last resort fallback - will produce incorrect token IDs!
                    tokenizer = TiktokenTokenizer.CreateForModel("gpt-4");
                }
            }
            else
            {
                tokenizer = TiktokenTokenizer.CreateForModel("gpt-4");
            }

            return new OnnxEmbeddingProvider(session, tokenizer, executionProvider);
        }
        catch (Exception ex) when (ex is FileNotFoundException
                                or DirectoryNotFoundException
                                or InvalidDataException
                                or Microsoft.ML.OnnxRuntime.OnnxRuntimeException)
        {
            // Model not available or corrupted - fall back to hash embeddings
            return null;
        }
    }

    /// <summary>
    /// Tries to create an InferenceSession with the best available execution provider.
    /// Attempts DirectML (GPU) first, then falls back to CPU.
    /// </summary>
    private static (InferenceSession Session, string Provider) CreateSessionWithBestProvider(string modelPath)
    {
        // Try DirectML (Windows GPU - works with AMD, Intel, NVIDIA)
        try
        {
            var options = new SessionOptions();
            options.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR; // Suppress warnings
            options.AppendExecutionProvider_DML(0); // Device ID 0
            options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
            var session = new InferenceSession(modelPath, options);
            return (session, "DirectML (GPU)");
        }
        catch
        {
            // DirectML not available, continue to CPU fallback
        }

        // Fallback to CPU with optimizations
        {
            var options = new SessionOptions();
            options.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR; // Suppress warnings
            options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
            options.InterOpNumThreads = Environment.ProcessorCount;
            options.IntraOpNumThreads = Environment.ProcessorCount;
            var session = new InferenceSession(modelPath, options);
            return (session, "CPU");
        }
    }

    public async Task<float[]> EmbedAsync(string text, CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(text);

        var encoding = _tokenizer.EncodeToIds(text);
        var tokenIds = encoding.Take(MaxSequenceLength).ToList();
        var attentionMask = Enumerable.Repeat(1L, tokenIds.Count).ToList();

        // Pad to max length if needed
        while (tokenIds.Count < MaxSequenceLength)
        {
            tokenIds.Add(0);
            attentionMask.Add(0);
        }

        // Create input tensors
        var inputIdsTensor = new DenseTensor<long>(
            tokenIds.Select(id => (long)id).ToArray(),
            new[] { 1, MaxSequenceLength }
        );

        var attentionMaskTensor = new DenseTensor<long>(
            attentionMask.ToArray(),
            new[] { 1, MaxSequenceLength }
        );

        // Token type IDs (all zeros for single sentence)
        var tokenTypeIds = new long[MaxSequenceLength];
        var tokenTypeIdsTensor = new DenseTensor<long>(
            tokenTypeIds,
            new[] { 1, MaxSequenceLength }
        );

        // Run inference with lock (InferenceSession is not thread-safe)
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input_ids", inputIdsTensor),
            NamedOnnxValue.CreateFromTensor("attention_mask", attentionMaskTensor),
            NamedOnnxValue.CreateFromTensor("token_type_ids", tokenTypeIdsTensor)
        };

        await _inferenceLock.WaitAsync(ct).ConfigureAwait(false);
        try
        {
            using var results = _session.Run(inputs);
            var output = results.First().AsTensor<float>();

            // Mean pooling
            var embedding = MeanPooling(output, attentionMask.ToArray());

            // Normalize
            Normalize(embedding);

            return embedding;
        }
        finally
        {
            _inferenceLock.Release();
        }
    }

    public async Task<float[][]> EmbedBatchAsync(IReadOnlyList<string> texts, CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(texts);

        var results = new float[texts.Count][];

        for (int i = 0; i < texts.Count; i += BatchSize)
        {
            ct.ThrowIfCancellationRequested();

            var batchEnd = Math.Min(i + BatchSize, texts.Count);
            var batchSize = batchEnd - i;

            for (int j = 0; j < batchSize; j++)
            {
                results[i + j] = await EmbedAsync(texts[i + j], ct);
            }
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

    private static float[] MeanPooling(Tensor<float> output, long[] attentionMask)
    {
        var sequenceLength = attentionMask.Length;
        var hiddenSize = VectorDimensions;

        var pooled = new float[hiddenSize];
        var maskSum = 0L;

        for (int i = 0; i < sequenceLength; i++)
        {
            if (attentionMask[i] > 0)
            {
                maskSum++;
                for (int j = 0; j < hiddenSize; j++)
                {
                    // Handle different tensor shapes - most models output [batch, sequence, hidden]
                    var index = new[] { 0, i, j };
                    pooled[j] += output[index];
                }
            }
        }

        if (maskSum > 0)
        {
            for (int i = 0; i < hiddenSize; i++)
            {
                pooled[i] /= maskSum;
            }
        }

        return pooled;
    }

    public void Dispose()
    {
        _session?.Dispose();
        (_tokenizer as IDisposable)?.Dispose();
        _inferenceLock.Dispose();
    }

    /// <summary>
    /// Extracts the bundled MiniLM model to the models path
    /// </summary>
    private static async Task<bool> ExtractBundledModelAsync(string modelsPath, CancellationToken ct)
    {
        try
        {
            var assembly = typeof(OnnxEmbeddingProvider).Assembly;
            var modelDir = Path.Combine(modelsPath, "minilm");
            Directory.CreateDirectory(modelDir);

            // Extract model.onnx
            var modelStream = assembly.GetManifestResourceStream("minilm.model.onnx");
            if (modelStream == null)
            {
                return false; // Bundled model not found
            }

            var modelPath = Path.Combine(modelDir, "model.onnx");
            await using (var fileStream = File.Create(modelPath))
            {
                await modelStream.CopyToAsync(fileStream, ct);
            }
            modelStream.Dispose();

            // Extract vocab.txt (required for BertTokenizer)
            var vocabStream = assembly.GetManifestResourceStream("minilm.vocab.txt");
            if (vocabStream != null)
            {
                var vocabPath = Path.Combine(modelDir, "vocab.txt");
                await using (var fileStream = File.Create(vocabPath))
                {
                    await vocabStream.CopyToAsync(fileStream, ct);
                }
                vocabStream.Dispose();
            }

            // Extract tokenizer.json (optional fallback)
            var tokenizerStream = assembly.GetManifestResourceStream("minilm.tokenizer.json");
            if (tokenizerStream != null)
            {
                var tokenizerPath = Path.Combine(modelDir, "tokenizer.json");
                await using (var fileStream = File.Create(tokenizerPath))
                {
                    await tokenizerStream.CopyToAsync(fileStream, ct);
                }
                tokenizerStream.Dispose();
            }

            return true;
        }
        catch
        {
            return false;
        }
    }
}
