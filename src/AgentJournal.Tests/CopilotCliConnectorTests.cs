using AgentJournal.Core.Connectors;
using Xunit;

namespace AgentJournal.Tests;

public class CopilotCliConnectorTests
{
    [Fact]
    public void GetSessionPaths_UsesConfiguredSessionsPath()
    {
        // The configured CopilotSessionsPath used to be ignored entirely: the connector always
        // probed ~/.copilot/session-state and ~/.copilot-cli/sessions, so `config set
        // CopilotSessionsPath` was a no-op and indexing walked the real corpus regardless.
        var root = Path.Combine(Path.GetTempPath(), $"aj-copilot-{Guid.NewGuid():N}");
        var sessionDirectory = Path.Combine(root, "session-1");
        Directory.CreateDirectory(sessionDirectory);
        File.WriteAllText(Path.Combine(sessionDirectory, "events.jsonl"), "{}");

        // A sibling directory without events.jsonl must not be reported as a session.
        Directory.CreateDirectory(Path.Combine(root, "not-a-session"));

        try
        {
            var connector = new CopilotCliConnector(root);

            var paths = connector.GetSessionPaths().ToList();

            Assert.Equal(new[] { sessionDirectory }, paths);
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    [Fact]
    public void GetSessionPaths_ReturnsEmptyWhenConfiguredPathIsMissing()
    {
        // A configured-but-absent directory must yield nothing rather than silently falling back
        // to the default locations and indexing an unrelated corpus.
        var missing = Path.Combine(Path.GetTempPath(), $"aj-copilot-missing-{Guid.NewGuid():N}");

        var connector = new CopilotCliConnector(missing);

        Assert.Empty(connector.GetSessionPaths());
    }
}
