# Embedding Infrastructure

This directory contains the embedding infrastructure for the agent-journal vector search feature.

## Overview

The embedding infrastructure provides text-to-vector conversion capabilities with two implementations:

1. **OnnxEmbeddingProvider** - Semantic embeddings using ONNX-based MiniLM model (384 dimensions)
2. **HashEmbeddingProvider** - Fast hash-based fallback for keyword matching (384 dimensions)

## Components

### IEmbeddingProvider

The main interface for embedding providers with the following capabilities:

- `EmbedAsync(text)` - Generate embedding for a single text
- `EmbedBatchAsync(texts)` - Generate embeddings for multiple texts (batched)
- `Normalize(vector)` - L2-normalize a vector in-place
- `Dimensions` - Number of dimensions in output vectors (384)
- `IsSemanticModel` - Whether the provider uses a real ML model

### OnnxEmbeddingProvider

**Semantic embedding provider** using Microsoft.ML.OnnxRuntime:

- **Model**: MiniLM ONNX model (expected at `{modelsPath}/minilm/model.onnx`)
- **Tokenizer**: TiktokenTokenizer (GPT-4 tokenizer as fallback)
- **Dimensions**: 384
- **Max Sequence Length**: 256 tokens
- **Batch Size**: 32 (for batch processing)
- **Features**:
  - Semantic similarity search
  - Mean pooling of token embeddings
  - L2 normalization
  - Implements `IDisposable` for proper resource cleanup

**Usage**:
```csharp
// Try to create ONNX provider
var provider = await OnnxEmbeddingProvider.TryCreateAsync(modelsPath);
if (provider is not null)
{
    var embedding = await provider.EmbedAsync("sample text");
    // Use embedding...
    provider.Dispose();
}
```

### HashEmbeddingProvider

**Fast hash-based fallback provider** using FNV-1a algorithm:

- **Algorithm**: FNV-1a hash with random dimension activation
- **Dimensions**: 384 (matches MiniLM)
- **Activations per word**: 8 dimensions
- **Features**:
  - Deterministic (same input always produces same output)
  - Fast computation (no ML model required)
  - TF-like weighting (1/wordCount per activation)
  - L2 normalization
  - Good for keyword matching, not semantic similarity

**Usage**:
```csharp
var provider = new HashEmbeddingProvider();
var embedding = await provider.EmbedAsync("sample text");
```

### EmbeddingProviderFactory

Factory for creating the appropriate embedding provider:

- `TryCreateAsync(modelsPath)` - Attempts to create ONNX provider, falls back to hash provider
- `CreateHashProvider()` - Creates hash-based provider directly
- `CreateOnnxProviderAsync(modelsPath)` - Attempts to create ONNX provider, returns null if not found

**Usage**:
```csharp
// Automatic fallback
var provider = await EmbeddingProviderFactory.TryCreateAsync(modelsPath);
if (provider.IsSemanticModel)
{
    Console.WriteLine("Using semantic ONNX embeddings");
}
else
{
    Console.WriteLine("Using hash-based embeddings (fallback)");
}
```

## Dependencies

The following NuGet packages are required:

- **Microsoft.ML.OnnxRuntime** (v1.20.0) - ONNX inference engine
- **Microsoft.ML.Tokenizers** (v1.0.0) - Text tokenization
- **System.Numerics.Tensors** (v10.0.1) - Tensor operations

## Model Setup

To use semantic embeddings, place the MiniLM ONNX model at:

```
{modelsPath}/
  └── minilm/
      ├── model.onnx       (required)
      └── tokenizer.json   (optional)
```

If the model is not found, the system automatically falls back to hash-based embeddings.

## Performance Considerations

### OnnxEmbeddingProvider
- **First call**: Includes model loading overhead (~100-500ms)
- **Subsequent calls**: ~10-50ms per text
- **Memory**: ~100-200MB for model
- **Best for**: Semantic similarity, finding related content

### HashEmbeddingProvider
- **Performance**: ~1-5ms per text
- **Memory**: Minimal overhead
- **Best for**: Exact keyword matching, fast indexing

## Example

```csharp
// Create provider with automatic fallback
var provider = await EmbeddingProviderFactory.TryCreateAsync("/path/to/models");

// Embed single text
var embedding = await provider.EmbedAsync("How do I implement vector search?");

// Embed multiple texts (batched)
var texts = new[] { "text1", "text2", "text3" };
var embeddings = await provider.EmbedBatchAsync(texts);

// Compute similarity
float Similarity(float[] a, float[] b)
{
    return a.Zip(b, (x, y) => x * y).Sum(); // Dot product (vectors are normalized)
}

// Clean up if using ONNX
if (provider is IDisposable disposable)
{
    disposable.Dispose();
}
```

## Architecture Notes

- Both providers output **384-dimensional** vectors for compatibility
- All vectors are **L2-normalized** for cosine similarity via dot product
- The factory pattern ensures graceful degradation when models are unavailable
- Hash-based provider uses **FNV-1a** for good distribution properties
- ONNX provider uses **mean pooling** over token embeddings (standard for sentence transformers)

## Future Enhancements

- [ ] Support for custom tokenizer loading from JSON
- [ ] Batch inference optimization for ONNX provider
- [ ] GPU acceleration support
- [ ] Model quantization for reduced memory footprint
- [ ] Alternative hash functions for improved distribution
