using System.CommandLine;
using Microsoft.Extensions.DependencyInjection;
using AgentJournal.Commands;
using AgentJournal.Configuration;
using AgentJournal.Core.Connectors;
using AgentJournal.Core.Storage;
using AgentJournal.Core.Search;
using AgentJournal.Core.Export;
using AgentJournal.Core.Embeddings;
using AgentJournal.Core.Knowledge;

namespace AgentJournal;

/// <summary>
/// Main entry point for the Agent Journal CLI tool
/// </summary>
public class Program
{
    public static async Task<int> Main(string[] args)
    {
        // Load configuration before DI setup (quick file read, acceptable for CLI startup)
        var configService = new ConfigurationService();
        var config = await configService.LoadConfigAsync();

        // Ensure data directory exists
        Directory.CreateDirectory(config.DataPath);

        // Set up dependency injection with loaded config
        var services = new ServiceCollection();
        ConfigureServices(services, config);

        // Use using to ensure proper disposal of services (including LuceneSearchEngine)
        await using var serviceProvider = services.BuildServiceProvider();

        // Initialize repository and search engine
        var repository = serviceProvider.GetRequiredService<ISessionRepository>();
        await repository.InitializeAsync();

        var searchEngine = serviceProvider.GetRequiredService<ISearchEngine>();
        await searchEngine.InitializeAsync();

        // Initialize knowledge repository
        var knowledgeRepository = serviceProvider.GetRequiredService<IKnowledgeRepository>();
        await knowledgeRepository.InitializeAsync();

        // Initialize content repository
        var contentRepository = serviceProvider.GetRequiredService<IContentRepository>();
        await contentRepository.InitializeAsync();

        var rootCommand = new RootCommand("Agent Journal - Index, search, and export AI agent conversation sessions")
        {
            IndexCommand.Create(serviceProvider),
            SearchCommand.Create(serviceProvider),
            ExportCommand.Create(serviceProvider),
            ConfigCommand.Create(serviceProvider),
            ModelsCommand.Create(serviceProvider),
            RememberCommand.Create(serviceProvider),
            RecallCommand.Create(serviceProvider),
            ForgetCommand.Create(serviceProvider),
            ReinforceCommand.Create(serviceProvider),
            KnowledgeCommand.Create(serviceProvider),
            ContentCommand.Create(serviceProvider),
            McpCommand.Create(serviceProvider)
        };

        return await rootCommand.InvokeAsync(args);
    }

    private static void ConfigureServices(IServiceCollection services, AgentJournalConfig config)
    {
        // Configuration - register the already-loaded config instance
        services.AddSingleton(config);
        services.AddSingleton<ConfigurationService>();

        // Connectors
        services.AddSingleton<ClaudeCodeConnector>();
        services.AddSingleton<CopilotCliConnector>();
        services.AddSingleton<IEnumerable<IAgentConnector>>(sp => new IAgentConnector[]
        {
            sp.GetRequiredService<ClaudeCodeConnector>(),
            sp.GetRequiredService<CopilotCliConnector>()
        });

        // Storage - SQLite repository (using pre-loaded config)
        services.AddSingleton<ISessionRepository>(sp =>
            new SqliteSessionRepository(config.DatabasePath));

        // Knowledge Bank - SQLite repository
        services.AddSingleton<IKnowledgeRepository>(sp =>
            new SqliteKnowledgeRepository(Path.Combine(config.DataPath, "knowledge.db")));

        // Content Repository - SQLite repository
        services.AddSingleton<IContentRepository>(sp =>
            new SqliteContentRepository(Path.Combine(config.DataPath, "content.db")));

        // Embeddings - Try to create ONNX provider, fallback to hash-based
        services.AddSingleton<IEmbeddingProvider>(sp =>
        {
            var provider = EmbeddingProviderFactory.TryCreateAsync(config.ModelsPath).GetAwaiter().GetResult();
            return provider;
        });

        // Search engines
        services.AddSingleton<LuceneSearchEngine>(sp =>
            new LuceneSearchEngine(config.LuceneIndexPath));

        services.AddSingleton<VectorSearchEngine>(sp =>
            new VectorSearchEngine(
                Path.Combine(config.DataPath, "vector-index"),
                sp.GetRequiredService<IEmbeddingProvider>()));

        // HybridSearcher as main ISearchEngine (supports all modes)
        services.AddSingleton<ISearchEngine>(sp =>
            new HybridSearcher(
                sp.GetRequiredService<LuceneSearchEngine>(),
                sp.GetRequiredService<VectorSearchEngine>()));

        // Exporters
        services.AddSingleton<HtmlExporter>();
        services.AddSingleton<MarkdownExporter>();
        services.AddSingleton<JsonExporter>();
        services.AddSingleton<IEnumerable<IExporter>>(sp => new IExporter[]
        {
            sp.GetRequiredService<HtmlExporter>(),
            sp.GetRequiredService<MarkdownExporter>(),
            sp.GetRequiredService<JsonExporter>()
        });
    }
}
