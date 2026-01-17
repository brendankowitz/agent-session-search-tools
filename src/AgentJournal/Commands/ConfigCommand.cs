using System.CommandLine;
using Microsoft.Extensions.DependencyInjection;
using AgentJournal.Configuration;
using AgentJournal.Core.Connectors;

namespace AgentJournal.Commands;

/// <summary>
/// Command to manage configuration
/// </summary>
public class ConfigCommand : Command
{
    private ConfigCommand() : base("config", "Manage agent journal configuration")
    {
    }

    public static Command Create(IServiceProvider serviceProvider)
    {
        var command = new ConfigCommand();
        command.AddCommand(ShowSubCommand.Create(serviceProvider));
        command.AddCommand(SetSubCommand.Create(serviceProvider));
        command.AddCommand(AgentsSubCommand.Create(serviceProvider));
        return command;
    }

    private class ShowSubCommand : Command
    {
        private ShowSubCommand() : base("show", "Show current configuration")
        {
        }

        public static Command Create(IServiceProvider serviceProvider)
        {
            var command = new ShowSubCommand();
            command.SetHandler(async () =>
            {
                var configService = serviceProvider.GetRequiredService<ConfigurationService>();
                await ExecuteAsync(configService, CancellationToken.None);
            });
            return command;
        }

        private static async Task ExecuteAsync(ConfigurationService configService, CancellationToken ct)
        {
            var config = await configService.LoadConfigAsync(ct);

            Console.WriteLine("Agent Journal Configuration");
            Console.WriteLine("===========================");
            Console.WriteLine();
            Console.WriteLine($"Configuration file: {configService.ConfigPath}");
            Console.WriteLine();
            Console.WriteLine("Settings:");
            Console.WriteLine($"  DataPath: {config.DataPath}");
            Console.WriteLine($"  DatabasePath: {config.DatabasePath}");
            Console.WriteLine($"  LuceneIndexPath: {config.LuceneIndexPath}");
            Console.WriteLine();
            Console.WriteLine("Agent Paths:");
            Console.WriteLine($"  ClaudeProjectsPath: {config.ClaudeProjectsPath ?? "(not configured)"}");
            Console.WriteLine($"  CopilotSessionsPath: {config.CopilotSessionsPath ?? "(not configured)"}");
            Console.WriteLine();
            Console.WriteLine("Search Settings:");
            Console.WriteLine($"  DefaultSearchMode: {config.DefaultSearchMode}");
            Console.WriteLine($"  DefaultContextMessages: {config.DefaultContextMessages}");
            Console.WriteLine($"  DefaultMaxResults: {config.DefaultMaxResults}");
            Console.WriteLine();
            Console.WriteLine("Other:");
            Console.WriteLine($"  VerboseLogging: {config.VerboseLogging}");
            Console.WriteLine();
            Console.WriteLine("To change a setting, use: aj config set <key> <value>");
        }
    }

    private class SetSubCommand : Command
    {
        private SetSubCommand() : base("set", "Set a configuration value")
        {
            var keyArgument = new Argument<string>(
                name: "key",
                description: "Configuration key");

            var valueArgument = new Argument<string>(
                name: "value",
                description: "Configuration value");

            this.AddArgument(keyArgument);
            this.AddArgument(valueArgument);
        }

        public static Command Create(IServiceProvider serviceProvider)
        {
            var command = new SetSubCommand();
            command.SetHandler(async (key, value) =>
            {
                var configService = serviceProvider.GetRequiredService<ConfigurationService>();
                await ExecuteAsync(key, value, configService, CancellationToken.None);
            }, 
            command.Arguments[0] as Argument<string> ?? throw new InvalidOperationException("Missing key argument"),
            command.Arguments[1] as Argument<string> ?? throw new InvalidOperationException("Missing value argument"));
            return command;
        }

        private static async Task ExecuteAsync(
            string key, 
            string value, 
            ConfigurationService configService, 
            CancellationToken ct)
        {
            Console.WriteLine($"Setting configuration: {key} = {value}");

            var success = await configService.SetConfigValueAsync(key, value, ct);

            if (success)
            {
                Console.WriteLine("Configuration updated successfully!");
                Console.WriteLine($"Saved to: {configService.ConfigPath}");
            }
            else
            {
                Console.Error.WriteLine($"Error: Invalid configuration key '{key}'");
                Console.Error.WriteLine();
                Console.Error.WriteLine("Valid configuration keys:");
                Console.Error.WriteLine("  DataPath - Base data directory");
                Console.Error.WriteLine("  ClaudeProjectsPath - Path to Claude Code sessions");
                Console.Error.WriteLine("  CopilotSessionsPath - Path to Copilot CLI sessions");
                Console.Error.WriteLine("  DefaultSearchMode - lexical, semantic, or hybrid");
                Console.Error.WriteLine("  DefaultContextMessages - Number (e.g., 3)");
                Console.Error.WriteLine("  DefaultMaxResults - Number (e.g., 10)");
                Console.Error.WriteLine("  VerboseLogging - true or false");
            }
        }
    }

    private class AgentsSubCommand : Command
    {
        private AgentsSubCommand() : base("agents", "List available agent connectors")
        {
        }

        public static Command Create(IServiceProvider serviceProvider)
        {
            var command = new AgentsSubCommand();
            command.SetHandler(async () =>
            {
                var connectors = serviceProvider.GetRequiredService<IEnumerable<IAgentConnector>>();
                await ExecuteAsync(connectors, CancellationToken.None);
            });
            return command;
        }

        private static async Task ExecuteAsync(
            IEnumerable<IAgentConnector> connectors, 
            CancellationToken ct)
        {
            Console.WriteLine("Available Agent Connectors");
            Console.WriteLine("=========================");
            Console.WriteLine();

            foreach (var connector in connectors)
            {
                Console.WriteLine($"Agent Type: {connector.AgentType}");
                
                try
                {
                    var sessionPaths = connector.GetSessionPaths().ToList();
                    Console.WriteLine($"  Found: {sessionPaths.Count} session path(s)");
                    
                    if (sessionPaths.Count > 0)
                    {
                        Console.WriteLine($"  Example path: {sessionPaths.First()}");
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"  Status: Not available ({ex.Message})");
                }
                
                Console.WriteLine();
            }

            Console.WriteLine("To index sessions from an agent, use: aj index --agent <type>");
            
            await Task.CompletedTask;
        }
    }
}
