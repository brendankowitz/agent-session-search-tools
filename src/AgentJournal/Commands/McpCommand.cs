using System.CommandLine;
using Microsoft.Extensions.DependencyInjection;
using AgentJournal.Core.Knowledge;
using AgentJournal.Core.Mcp;
using AgentJournal.Core.Search;
using AgentJournal.Core.Storage;

namespace AgentJournal.Commands;

/// <summary>
/// Command to start the MCP (Model Context Protocol) server
/// </summary>
public class McpCommand : Command
{
    private McpCommand() : base("mcp", "Start MCP server for AI agent integration via stdio")
    {
        // MCP server operates on stdio protocol - no additional options needed
    }

    public static Command Create(IServiceProvider serviceProvider)
    {
        var command = new McpCommand();
        
        command.SetHandler(async () =>
        {
            var searchEngine = serviceProvider.GetRequiredService<ISearchEngine>();
            var sessionRepository = serviceProvider.GetRequiredService<ISessionRepository>();
            var knowledgeRepository = serviceProvider.GetRequiredService<IKnowledgeRepository>();
            var contentRepository = serviceProvider.GetRequiredService<IContentRepository>();

            await ExecuteAsync(searchEngine, sessionRepository, knowledgeRepository, contentRepository, CancellationToken.None);
        });

        return command;
    }

    private static async Task ExecuteAsync(
        ISearchEngine searchEngine,
        ISessionRepository sessionRepository,
        IKnowledgeRepository knowledgeRepository,
        IContentRepository contentRepository,
        CancellationToken cancellationToken)
    {
        // Run MCP server on stdio
        // No console output - only MCP protocol messages
        await AgentJournalMcpServer.RunAsync(
            searchEngine,
            sessionRepository,
            knowledgeRepository,
            contentRepository,
            cancellationToken);
    }
}
