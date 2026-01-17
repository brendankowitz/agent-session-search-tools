using AgentJournal.Core.Search;

namespace AgentJournal.Configuration;

/// <summary>
/// Configuration for the Agent Journal application
/// </summary>
public class AgentJournalConfig
{
    /// <summary>
    /// Base data directory for all Agent Journal files
    /// </summary>
    public string DataPath { get; set; } = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
        ".agent-journal"
    );

    /// <summary>
    /// Path to the SQLite database file
    /// </summary>
    public string DatabasePath => Path.Combine(DataPath, "agent-journal.db");

    /// <summary>
    /// Path to the Lucene search index directory
    /// </summary>
    public string LuceneIndexPath => Path.Combine(DataPath, "lucene-index");

    /// <summary>
    /// Path to the embedding models directory
    /// </summary>
    public string ModelsPath => Path.Combine(DataPath, "models");

    /// <summary>
    /// Path to Claude Code project sessions (if available)
    /// </summary>
    public string? ClaudeProjectsPath { get; set; }

    /// <summary>
    /// Path to Copilot CLI sessions (if available)
    /// </summary>
    public string? CopilotSessionsPath { get; set; }

    /// <summary>
    /// Default search mode for queries
    /// </summary>
    public SearchMode DefaultSearchMode { get; set; } = SearchMode.Lexical;

    /// <summary>
    /// Default number of context messages to include in search results
    /// </summary>
    public int DefaultContextMessages { get; set; } = 3;

    /// <summary>
    /// Default maximum number of search results
    /// </summary>
    public int DefaultMaxResults { get; set; } = 10;

    /// <summary>
    /// Enable verbose logging
    /// </summary>
    public bool VerboseLogging { get; set; } = false;
}
