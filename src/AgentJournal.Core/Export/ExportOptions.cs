namespace AgentJournal.Core.Export;

/// <summary>
/// Options for controlling export behavior
/// </summary>
public record ExportOptions(
    bool IncludeToolCalls = true,
    bool IncludeTimestamps = true,
    string Theme = "dark",
    int? MaxToolResultLength = 500,
    bool PrettyPrint = true,
    bool CollapseToolCallsByDefault = true
)
{
    /// <summary>
    /// Default export options
    /// </summary>
    public static readonly ExportOptions Default = new();

    /// <summary>
    /// Export options with light theme
    /// </summary>
    public static readonly ExportOptions Light = new ExportOptions() with { Theme = "light" };

    /// <summary>
    /// Export options with no tool calls included
    /// </summary>
    public static readonly ExportOptions NoToolCalls = new ExportOptions() with { IncludeToolCalls = false };
}
