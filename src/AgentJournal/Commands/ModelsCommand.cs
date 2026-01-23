using System.CommandLine;
using Microsoft.Extensions.DependencyInjection;
using AgentJournal.Configuration;
using AgentJournal.Core.Embeddings;

namespace AgentJournal.Commands;

/// <summary>
/// Command to manage embedding models
/// </summary>
public class ModelsCommand : Command
{
    public ModelsCommand() : base("models", "Manage embedding models")
    {
        AddCommand(new ListModelsCommand());
        AddCommand(new DownloadModelCommand());
        AddCommand(new RemoveModelCommand());
    }

    public static Command Create(IServiceProvider serviceProvider)
    {
        var command = new ModelsCommand();

        // Wire up subcommands with service provider
        foreach (var subCommand in command.Subcommands)
        {
            if (subCommand is ListModelsCommand listCmd)
            {
                listCmd.SetupHandler(serviceProvider);
            }
            else if (subCommand is DownloadModelCommand downloadCmd)
            {
                downloadCmd.SetupHandler(serviceProvider);
            }
            else if (subCommand is RemoveModelCommand removeCmd)
            {
                removeCmd.SetupHandler(serviceProvider);
            }
        }

        return command;
    }
}

/// <summary>
/// List installed embedding models
/// </summary>
public class ListModelsCommand : Command
{
    public ListModelsCommand() : base("list", "List installed embedding models")
    {
    }

    public void SetupHandler(IServiceProvider serviceProvider)
    {
        this.SetHandler(async () =>
        {
            var configService = serviceProvider.GetRequiredService<ConfigurationService>();
            var embeddingProvider = serviceProvider.GetRequiredService<IEmbeddingProvider>();
            await ExecuteAsync(configService, embeddingProvider, CancellationToken.None);
        });
    }

    private static async Task ExecuteAsync(
        ConfigurationService configService,
        IEmbeddingProvider embeddingProvider,
        CancellationToken ct)
    {
        var config = await configService.LoadConfigAsync(ct);
        var modelsPath = GetModelsPath(config);

        Console.WriteLine("Installed Embedding Models:");
        Console.WriteLine($"Models Path: {modelsPath}");

        // Show active execution provider
        if (embeddingProvider is OnnxEmbeddingProvider onnxProvider)
        {
            Console.WriteLine($"Execution Provider: {onnxProvider.ExecutionProvider}");
        }
        Console.WriteLine();

        if (!Directory.Exists(modelsPath))
        {
            Console.WriteLine("No models directory found.");
            Console.WriteLine($"Create directory at: {modelsPath}");
            return;
        }

        var modelDirs = Directory.GetDirectories(modelsPath);

        if (modelDirs.Length == 0)
        {
            Console.WriteLine("No models installed.");
            Console.WriteLine();
            Console.WriteLine("To download a model, run:");
            Console.WriteLine("  agent-journal models download minilm");
            return;
        }

        foreach (var modelDir in modelDirs)
        {
            var modelName = Path.GetFileName(modelDir);
            var modelFile = Path.Combine(modelDir, "model.onnx");
            var tokenizerFile = Path.Combine(modelDir, "tokenizer.json");

            var hasModel = File.Exists(modelFile);
            var hasTokenizer = File.Exists(tokenizerFile);
            var status = (hasModel && hasTokenizer) ? "✓ Ready" : "✗ Incomplete";

            var size = ModelCommandHelpers.GetDirectorySize(modelDir);
            var sizeStr = ModelCommandHelpers.FormatBytes(size);

            Console.WriteLine($"  {modelName,-20} {status,-15} {sizeStr,10}");

            if (!hasModel)
            {
                Console.WriteLine($"    Missing: model.onnx");
            }
            if (!hasTokenizer)
            {
                Console.WriteLine($"    Missing: tokenizer.json");
            }
        }
    }

    private static string GetModelsPath(AgentJournalConfig config)
    {
        return Path.Combine(config.DataPath, "models");
    }
}

/// <summary>
/// Download an embedding model
/// </summary>
public class DownloadModelCommand : Command
{
    private Argument<string> _nameArgument = null!;

    public DownloadModelCommand() : base("download", "Download an embedding model")
    {
        _nameArgument = new Argument<string>(
            name: "name",
            description: "Model name (e.g., 'minilm')");
        AddArgument(_nameArgument);
    }

    public void SetupHandler(IServiceProvider serviceProvider)
    {
        this.SetHandler(async (name) =>
        {
            var configService = serviceProvider.GetRequiredService<ConfigurationService>();
            await ExecuteAsync(name, configService, CancellationToken.None);
        }, _nameArgument);
    }

    private static async Task ExecuteAsync(
        string name,
        ConfigurationService configService,
        CancellationToken ct)
    {
        name = ModelCommandHelpers.ValidateModelName(name);

        var config = await configService.LoadConfigAsync(ct);
        var modelsPath = Path.Combine(config.DataPath, "models");
        var modelDir = Path.Combine(modelsPath, name);

        Console.WriteLine($"Downloading model: {name}");
        Console.WriteLine($"Target directory: {modelDir}");
        Console.WriteLine();

        // Create models directory if needed
        Directory.CreateDirectory(modelsPath);

        if (Directory.Exists(modelDir))
        {
            Console.WriteLine($"Model '{name}' already exists.");
            Console.Write("Overwrite? (y/N): ");
            var response = Console.ReadLine()?.Trim().ToLowerInvariant();

            if (response != "y" && response != "yes")
            {
                Console.WriteLine("Download cancelled.");
                return;
            }

            Directory.Delete(modelDir, recursive: true);
        }

        Directory.CreateDirectory(modelDir);

        // Model download URLs (these would be actual URLs in production)
        var modelUrls = GetModelUrls(name);

        if (modelUrls is null)
        {
            Console.WriteLine($"Unknown model: {name}");
            Console.WriteLine();
            Console.WriteLine("Available models:");
            Console.WriteLine("  minilm  - all-MiniLM-L6-v2 (384 dimensions, ~90MB)");
            return;
        }

        Console.WriteLine("Note: Model download is not yet implemented.");
        Console.WriteLine();
        Console.WriteLine("To manually install the model:");
        Console.WriteLine($"1. Download the model files from HuggingFace:");
        Console.WriteLine($"   https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2");
        Console.WriteLine($"2. Place the following files in: {modelDir}");
        Console.WriteLine("   - model.onnx");
        Console.WriteLine("   - tokenizer.json");
        Console.WriteLine();
        Console.WriteLine("Alternatively, use a pre-converted ONNX model from:");
        Console.WriteLine("   https://github.com/microsoft/onnxruntime/tree/main/onnxruntime/test/testdata");

        // TODO: Implement actual download from HuggingFace or GitHub releases
        // This would use HttpClient to download the model files with progress reporting
    }

    private static Dictionary<string, string>? GetModelUrls(string name)
    {
        return name.ToLowerInvariant() switch
        {
            "minilm" => new Dictionary<string, string>
            {
                ["model.onnx"] = "https://example.com/minilm/model.onnx",
                ["tokenizer.json"] = "https://example.com/minilm/tokenizer.json"
            },
            _ => null
        };
    }
}

/// <summary>
/// Remove an installed embedding model
/// </summary>
public class RemoveModelCommand : Command
{
    private Argument<string> _nameArgument = null!;

    public RemoveModelCommand() : base("remove", "Remove an installed embedding model")
    {
        _nameArgument = new Argument<string>(
            name: "name",
            description: "Model name to remove");
        AddArgument(_nameArgument);
    }

    public void SetupHandler(IServiceProvider serviceProvider)
    {
        this.SetHandler(async (name) =>
        {
            var configService = serviceProvider.GetRequiredService<ConfigurationService>();
            await ExecuteAsync(name, configService, CancellationToken.None);
        }, _nameArgument);
    }

    private static async Task ExecuteAsync(
        string name,
        ConfigurationService configService,
        CancellationToken ct)
    {
        name = ModelCommandHelpers.ValidateModelName(name);

        var config = await configService.LoadConfigAsync(ct);
        var modelsPath = Path.Combine(config.DataPath, "models");
        var modelDir = Path.Combine(modelsPath, name);

        if (!Directory.Exists(modelDir))
        {
            Console.WriteLine($"Model '{name}' not found.");
            return;
        }

        var size = ModelCommandHelpers.GetDirectorySize(modelDir);
        var sizeStr = ModelCommandHelpers.FormatBytes(size);

        Console.WriteLine($"Remove model: {name}");
        Console.WriteLine($"Size: {sizeStr}");
        Console.Write("Are you sure? (y/N): ");

        var response = Console.ReadLine()?.Trim().ToLowerInvariant();

        if (response != "y" && response != "yes")
        {
            Console.WriteLine("Removal cancelled.");
            return;
        }

        try
        {
            Directory.Delete(modelDir, recursive: true);
            Console.WriteLine($"Model '{name}' removed successfully.");
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Error removing model: {ex.Message}");
        }
    }
}

/// <summary>
/// Helper utilities for model commands
/// </summary>
internal static class ModelCommandHelpers
{
    /// <summary>
    /// Validates a model name to prevent path traversal attacks
    /// </summary>
    public static string ValidateModelName(string name)
    {
        if (string.IsNullOrWhiteSpace(name))
            throw new ArgumentException("Model name is required.", nameof(name));

        // Prevent path traversal
        if (name.Contains("..", StringComparison.Ordinal) ||
            name.IndexOfAny([Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar]) >= 0 ||
            Path.IsPathRooted(name))
        {
            throw new ArgumentException("Model name must not contain path separators or '..'.", nameof(name));
        }

        // Ensure name is filesystem safe
        var safeName = Path.GetFileName(name);
        if (!string.Equals(name, safeName, StringComparison.Ordinal))
        {
            throw new ArgumentException("Invalid model name.", nameof(name));
        }

        return name;
    }

    /// <summary>
    /// Gets the size of a directory recursively
    /// </summary>
    public static long GetDirectorySize(string path)
    {
        var dirInfo = new DirectoryInfo(path);
        return dirInfo.EnumerateFiles("*", SearchOption.AllDirectories)
            .Sum(file => file.Length);
    }

    /// <summary>
    /// Formats bytes as human-readable string
    /// </summary>
    public static string FormatBytes(long bytes)
    {
        string[] sizes = ["B", "KB", "MB", "GB"];
        double len = bytes;
        int order = 0;

        while (len >= 1024 && order < sizes.Length - 1)
        {
            order++;
            len /= 1024;
        }

        return $"{len:0.##} {sizes[order]}";
    }
}
