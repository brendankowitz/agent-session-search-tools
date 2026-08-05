using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using AgentJournal.Core.Knowledge;
using AgentJournal.Core.Search;
using AgentJournal.Core.Storage;
using ModelContextProtocol.Server;

namespace AgentJournal.Core.Mcp;

/// <summary>
/// MCP (Model Context Protocol) server for Agent Journal
/// Provides stdio-based interface for AI agents to query session history and knowledge
/// </summary>
public static class AgentJournalMcpServer
{
    /// <summary>
    /// Creates and configures the MCP server host
    /// </summary>
    public static async Task<IHost> CreateHostAsync(
        ISearchEngine searchEngine,
        ISessionRepository sessionRepository,
        IKnowledgeRepository knowledgeRepository,
        IContentRepository contentRepository)
    {
        var builder = Host.CreateApplicationBuilder();

        // The stdio transport uses stdout exclusively for JSON-RPC framing. The default host
        // console logger also writes to stdout, which interleaves log text with protocol
        // messages and corrupts the stream. Force every log level to stderr instead.
        builder.Logging.AddConsole(options =>
        {
            options.LogToStandardErrorThreshold = LogLevel.Trace;
        });

        // Register Agent Journal services
        builder.Services.AddSingleton(searchEngine);
        builder.Services.AddSingleton(sessionRepository);
        builder.Services.AddSingleton(knowledgeRepository);
        builder.Services.AddSingleton(contentRepository);

        // Register MCP tools
        builder.Services.AddSingleton<AgentJournalTools>();

        // Configure MCP server with stdio transport
        builder.Services
            .AddMcpServer()
            .WithStdioServerTransport()
            .WithToolsFromAssembly();

        return builder.Build();
    }

    /// <summary>
    /// Runs the MCP server until cancellation
    /// </summary>
    public static async Task RunAsync(
        ISearchEngine searchEngine,
        ISessionRepository sessionRepository,
        IKnowledgeRepository knowledgeRepository,
        IContentRepository contentRepository,
        CancellationToken cancellationToken = default)
    {
        var host = await CreateHostAsync(searchEngine, sessionRepository, knowledgeRepository, contentRepository);
        await host.RunAsync(cancellationToken);
    }
}
