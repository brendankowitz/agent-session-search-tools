using System.CommandLine;
using System.Diagnostics;
using Microsoft.Extensions.DependencyInjection;
using AgentJournal.Configuration;
using AgentJournal.Core.Connectors;
using AgentJournal.Core.Storage;
using AgentJournal.Core.Search;
using AgentJournal.Core.Embeddings;

namespace AgentJournal.Commands;

/// <summary>
/// Command to index agent sessions
/// </summary>
public class IndexCommand : Command
{
    private IndexCommand() : base("index", "Index agent sessions for searching")
    {
        var agentOption = new Option<string?>(
            name: "--agent",
            getDefaultValue: () => "all",
            description: "Agent type to index (claude-code, copilot-cli, or all)");
        agentOption.AddAlias("-a");

        var watchOption = new Option<bool>(
            name: "--watch",
            description: "Watch for new sessions and index them automatically");
        watchOption.AddAlias("-w");

        var rebuildOption = new Option<bool>(
            name: "--rebuild",
            description: "Clear existing index and rebuild from scratch");
        rebuildOption.AddAlias("-r");

        this.AddOption(agentOption);
        this.AddOption(watchOption);
        this.AddOption(rebuildOption);
    }

    public static Command Create(IServiceProvider serviceProvider)
    {
        var command = new IndexCommand();

        command.SetHandler(async (agentType, watch, rebuild) =>
        {
            var configService = serviceProvider.GetRequiredService<ConfigurationService>();
            var repository = serviceProvider.GetRequiredService<ISessionRepository>();
            var searchEngine = serviceProvider.GetRequiredService<ISearchEngine>();
            var connectors = serviceProvider.GetRequiredService<IEnumerable<IAgentConnector>>();
            var embeddingProvider = serviceProvider.GetRequiredService<IEmbeddingProvider>();

            await ExecuteAsync(
                agentType,
                watch,
                rebuild,
                configService,
                repository,
                searchEngine,
                connectors,
                embeddingProvider,
                CancellationToken.None);
        },
        command.Options[0] as Option<string?> ?? throw new InvalidOperationException("Missing agent option"),
        command.Options[1] as Option<bool> ?? throw new InvalidOperationException("Missing watch option"),
        command.Options[2] as Option<bool> ?? throw new InvalidOperationException("Missing rebuild option"));

        return command;
    }

    private static async Task ExecuteAsync(
        string? agentType,
        bool watch,
        bool rebuild,
        ConfigurationService configService,
        ISessionRepository repository,
        ISearchEngine searchEngine,
        IEnumerable<IAgentConnector> allConnectors,
        IEmbeddingProvider embeddingProvider,
        CancellationToken ct)
    {
        var config = await configService.LoadConfigAsync(ct);

        // Determine parallelism based on execution provider
        var useParallelism = embeddingProvider is OnnxEmbeddingProvider onnx &&
                             onnx.ExecutionProvider.Contains("GPU", StringComparison.OrdinalIgnoreCase);
        var maxParallelism = useParallelism ? Environment.ProcessorCount : 1;

        Console.WriteLine($"Agent Journal - Indexing Sessions");
        Console.WriteLine($"Agent type: {agentType ?? "all"}");
        Console.WriteLine($"Database: {config.DatabasePath}");
        Console.WriteLine($"Index: {config.LuceneIndexPath}");
        if (embeddingProvider is OnnxEmbeddingProvider onnxProvider)
        {
            Console.WriteLine($"Embeddings: {onnxProvider.ExecutionProvider}");
            if (useParallelism)
            {
                Console.WriteLine($"Parallelism: {maxParallelism} threads");
            }
        }
        Console.WriteLine();

        if (rebuild)
        {
            Console.WriteLine("Clearing existing index...");
            await searchEngine.ClearIndexAsync(ct);
        }

        // Filter connectors based on agent type
        var connectorsToUse = agentType?.ToLowerInvariant() switch
        {
            "claude-code" or "claude" => allConnectors.Where(c => c.AgentType == "claude-code"),
            "copilot-cli" or "copilot" => allConnectors.Where(c => c.AgentType == "copilot-cli"),
            _ => allConnectors
        };

        // Index sessions from each connector
        var totalIndexed = 0;
        var totalSkipped = 0;
        var totalErrors = 0;
        var stopwatch = Stopwatch.StartNew();

        foreach (var connector in connectorsToUse)
        {
            Console.WriteLine($"Indexing {connector.AgentType} sessions...");

            try
            {
                var sessionPaths = connector.GetSessionPaths().ToList();
                Console.WriteLine($"  Found {sessionPaths.Count} session paths");

                // Collect all sessions first for parallel processing
                var sessions = new List<Core.Models.Session>();
                await foreach (var session in connector.ParseSessionsAsync(ct))
                {
                    sessions.Add(session);
                }

                var indexed = 0;
                var skipped = 0;
                var errors = 0;
                var indexLock = new object();

                if (useParallelism && sessions.Count > 1)
                {
                    // Parallel indexing for GPU acceleration
                    var options = new ParallelOptions
                    {
                        MaxDegreeOfParallelism = maxParallelism,
                        CancellationToken = ct
                    };

                    await Parallel.ForEachAsync(sessions, options, async (session, token) =>
                    {
                        try
                        {
                            // Check if session needs indexing
                            var shouldIndex = true;
                            if (!rebuild && session.LastModified.HasValue)
                            {
                                var dbLastMod = await repository.GetSessionLastModifiedAsync(session.Id, token);
                                if (dbLastMod.HasValue)
                                {
                                    // Debug output
                                    // Console.WriteLine($"Session: {session.LastModified.Value:O}, DB: {dbLastMod.Value:O}");

                                    // Truncate to seconds for comparison to avoid precision issues
                                    var sessionTime = session.LastModified.Value;
                                    var dbTime = dbLastMod.Value;

                                    // Allow for small difference (e.g. if DB loses precision)
                                    if (sessionTime <= dbTime.AddMilliseconds(100))
                                    {
                                        shouldIndex = false;
                                    }
                                }
                            }

                            if (!shouldIndex)
                            {
                                lock (indexLock)
                                {
                                    skipped++;
                                    totalSkipped++;
                                    if (config.VerboseLogging)
                                    {
                                        Console.WriteLine($"  - Skipped: {session.Id} (Up to date)");
                                    }
                                    else if ((indexed + skipped) % 10 == 0)
                                    {
                                        Console.Write($"\r  Processed: {indexed + skipped}/{sessions.Count} (Indexed: {indexed}, Skipped: {skipped})...");
                                    }
                                }
                                return;
                            }

                            // Save to repository (thread-safe via SQLite)
                            await repository.SaveSessionAsync(session, token);

                            // Index in search engine (thread-safe with locks)
                            await searchEngine.IndexSessionAsync(session, token);

                            lock (indexLock)
                            {
                                indexed++;
                                totalIndexed++;

                                if (config.VerboseLogging)
                                {
                                    Console.WriteLine($"  ✓ Indexed: {session.Id} ({session.MessageCount} messages)");
                                }
                                else if ((indexed + skipped) % 10 == 0)
                                {
                                    Console.Write($"\r  Processed: {indexed + skipped}/{sessions.Count} (Indexed: {indexed}, Skipped: {skipped})...");
                                }
                            }
                        }
                        catch (Exception ex)
                        {
                            lock (indexLock)
                            {
                                errors++;
                                totalErrors++;
                            }
                            if (config.VerboseLogging)
                            {
                                Console.Error.WriteLine($"  ✗ Error indexing session {session.Id}: {ex.Message}");
                            }
                        }
                    });
                }
                else
                {
                    // Sequential indexing
                    foreach (var session in sessions)
                    {
                        try
                        {
                            // Check if session needs indexing
                            var shouldIndex = true;
                            if (!rebuild && session.LastModified.HasValue)
                            {
                                var dbLastMod = await repository.GetSessionLastModifiedAsync(session.Id, ct);
                                if (dbLastMod.HasValue)
                                {
                                    // Truncate to seconds for comparison to avoid precision issues
                                    var sessionTime = session.LastModified.Value.ToUniversalTime();
                                    var dbTime = dbLastMod.Value.ToUniversalTime();

                                    // Allow for small difference (e.g. if DB loses precision)
                                    if (sessionTime <= dbTime.AddMilliseconds(100))
                                    {
                                        shouldIndex = false;
                                    }
                                }
                            }

                            if (!shouldIndex)
                            {
                                skipped++;
                                totalSkipped++;
                                if (config.VerboseLogging)
                                {
                                    Console.WriteLine($"  - Skipped: {session.Id} (Up to date)");
                                }
                                else if ((indexed + skipped) % 10 == 0)
                                {
                                    Console.Write($"\r  Processed: {indexed + skipped}/{sessions.Count} (Indexed: {indexed}, Skipped: {skipped})...");
                                }
                                continue;
                            }

                            await repository.SaveSessionAsync(session, ct);
                            await searchEngine.IndexSessionAsync(session, ct);

                            indexed++;
                            totalIndexed++;

                            if (config.VerboseLogging)
                            {
                                Console.WriteLine($"  ✓ Indexed: {session.Id} ({session.MessageCount} messages)");
                            }
                            else if ((indexed + skipped) % 10 == 0)
                            {
                                Console.Write($"\r  Processed: {indexed + skipped}/{sessions.Count} (Indexed: {indexed}, Skipped: {skipped})...");
                            }
                        }
                        catch (Exception ex)
                        {
                            errors++;
                            totalErrors++;
                            Console.Error.WriteLine($"  ✗ Error indexing session {session.Id}: {ex.Message}");
                        }
                    }
                }

                if (!config.VerboseLogging && (indexed + skipped) > 0)
                {
                    Console.WriteLine($"\r  Processed: {indexed + skipped} sessions (Indexed: {indexed}, Skipped: {skipped})    ");
                }

                if (errors > 0)
                {
                    Console.WriteLine($"  Errors: {errors}");
                }
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"  Error accessing {connector.AgentType} sessions: {ex.Message}");
            }

            Console.WriteLine();
        }

        stopwatch.Stop();
        Console.WriteLine($"Indexing complete!");
        Console.WriteLine($"  Total sessions processed: {totalIndexed + totalSkipped}");
        Console.WriteLine($"  Indexed: {totalIndexed}");
        Console.WriteLine($"  Skipped: {totalSkipped}");
        Console.WriteLine($"  Time elapsed: {stopwatch.Elapsed.TotalSeconds:F1}s");
        if (totalErrors > 0)
        {
            Console.WriteLine($"  Total errors: {totalErrors}");
        }

        if (watch)
        {
            Console.WriteLine();
            Console.WriteLine("Watch mode enabled - monitoring for new sessions...");
            Console.WriteLine("Press Ctrl+C to stop watching");

            // Simple polling implementation (could be improved with FileSystemWatcher)
            while (!ct.IsCancellationRequested)
            {
                await Task.Delay(TimeSpan.FromSeconds(10), ct);

                // Re-index (this is a simple approach; production would track what's new)
                foreach (var connector in connectorsToUse)
                {
                    await foreach (var session in connector.ParseSessionsAsync(ct))
                    {
                        try
                        {
                            // Check if session needs indexing
                            var shouldIndex = true;
                            if (session.LastModified.HasValue)
                            {
                                var dbLastMod = await repository.GetSessionLastModifiedAsync(session.Id, ct);
                                if (dbLastMod.HasValue && session.LastModified.Value <= dbLastMod.Value)
                                {
                                    shouldIndex = false;
                                }
                            }

                            if (!shouldIndex) continue;

                            await repository.SaveSessionAsync(session, ct);
                            await searchEngine.IndexSessionAsync(session, ct);

                            if (config.VerboseLogging)
                            {
                                Console.WriteLine($"  ✓ Updated: {session.Id}");
                            }
                        }
                        catch
                        {
                            // Silently ignore errors in watch mode
                        }
                    }
                }
            }
        }
    }
}
