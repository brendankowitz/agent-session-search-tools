using System.Text.Json;
using AgentJournal.Core.Search;

namespace AgentJournal.Configuration;

/// <summary>
/// Service for managing Agent Journal configuration
/// </summary>
public class ConfigurationService
{
    private readonly string _configPath;
    private AgentJournalConfig? _cachedConfig;

    public ConfigurationService()
    {
        var dataPath = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".agent-journal"
        );
        _configPath = Path.Combine(dataPath, "config.json");
    }

    /// <summary>
    /// Loads the configuration from disk, creating default if it doesn't exist
    /// </summary>
    public async Task<AgentJournalConfig> LoadConfigAsync(CancellationToken ct = default)
    {
        if (_cachedConfig != null)
        {
            return _cachedConfig;
        }

        if (!File.Exists(_configPath))
        {
            _cachedConfig = new AgentJournalConfig();
            await SaveConfigAsync(_cachedConfig, ct);
            return _cachedConfig;
        }

        try
        {
            var json = await File.ReadAllTextAsync(_configPath, ct);
            _cachedConfig = JsonSerializer.Deserialize<AgentJournalConfig>(json, new JsonSerializerOptions
            {
                PropertyNameCaseInsensitive = true,
                WriteIndented = true
            }) ?? new AgentJournalConfig();
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Warning: Failed to load config from {_configPath}: {ex.Message}");
            Console.Error.WriteLine("Using default configuration.");
            _cachedConfig = new AgentJournalConfig();
        }

        return _cachedConfig;
    }

    /// <summary>
    /// Saves the configuration to disk
    /// </summary>
    public async Task SaveConfigAsync(AgentJournalConfig config, CancellationToken ct = default)
    {
        var directory = Path.GetDirectoryName(_configPath);
        if (!string.IsNullOrEmpty(directory))
        {
            Directory.CreateDirectory(directory);
        }

        var json = JsonSerializer.Serialize(config, new JsonSerializerOptions
        {
            PropertyNameCaseInsensitive = true,
            WriteIndented = true
        });

        await File.WriteAllTextAsync(_configPath, json, ct);
        _cachedConfig = config;
    }

    /// <summary>
    /// Updates a specific configuration value
    /// </summary>
    public async Task<bool> SetConfigValueAsync(string key, string value, CancellationToken ct = default)
    {
        var config = await LoadConfigAsync(ct);

        switch (key.ToLowerInvariant())
        {
            case "datapath":
                config.DataPath = value;
                break;
            case "claudeprojectspath":
                config.ClaudeProjectsPath = value;
                break;
            case "copilotsessionspath":
                config.CopilotSessionsPath = value;
                break;
            case "defaultsearchmode":
                if (Enum.TryParse<SearchMode>(value, true, out var searchMode))
                {
                    config.DefaultSearchMode = searchMode;
                }
                else
                {
                    return false;
                }
                break;
            case "defaultcontextmessages":
                if (int.TryParse(value, out var contextMessages))
                {
                    config.DefaultContextMessages = contextMessages;
                }
                else
                {
                    return false;
                }
                break;
            case "defaultmaxresults":
                if (int.TryParse(value, out var maxResults))
                {
                    config.DefaultMaxResults = maxResults;
                }
                else
                {
                    return false;
                }
                break;
            case "verboselogging":
                if (bool.TryParse(value, out var verboseLogging))
                {
                    config.VerboseLogging = verboseLogging;
                }
                else
                {
                    return false;
                }
                break;
            default:
                return false;
        }

        await SaveConfigAsync(config, ct);
        return true;
    }

    /// <summary>
    /// Gets the configuration file path
    /// </summary>
    public string ConfigPath => _configPath;
}
