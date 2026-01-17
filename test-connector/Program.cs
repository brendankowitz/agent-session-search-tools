using AgentJournal.Core.Connectors;
using AgentJournal.Core.Models;

Console.WriteLine("Testing CopilotCliConnector...\n");

var connector = new CopilotCliConnector();

// Test 1: Simple session
Console.WriteLine("=== Test 1: Simple Session ===");
var testPath1 = @"E:\data\src\agent-session-search-tools\test-data\copilot-cli-test";
await TestSession(connector, testPath1);

// Test 2: Complex session with multiple messages and tools
Console.WriteLine("\n=== Test 2: Complex Session ===");
var testPath2 = @"E:\data\src\agent-session-search-tools\test-data\copilot-cli-complex";
await TestSession(connector, testPath2);

// Test 3: GetSessionPaths
Console.WriteLine("\n=== Test 3: GetSessionPaths ===");
var sessionPaths = connector.GetSessionPaths().ToList();
Console.WriteLine($"Found {sessionPaths.Count} session paths");

Console.WriteLine("\n✅ All tests completed successfully!");
return 0;

static async Task TestSession(CopilotCliConnector connector, string testPath)
{
    Console.WriteLine($"Parsing session from: {testPath}");
    var session = await connector.ParseSessionAsync(testPath);

    if (session == null)
    {
        Console.WriteLine("❌ Failed to parse session");
        return;
    }

    Console.WriteLine($"✅ Session parsed successfully!");
    Console.WriteLine($"   Session ID: {session.Id}");
    Console.WriteLine($"   Agent Type: {session.AgentType}");
    Console.WriteLine($"   Agent Version: {session.AgentVersion}");
    Console.WriteLine($"   Started At: {session.StartedAt}");
    Console.WriteLine($"   Ended At: {session.EndedAt}");
    Console.WriteLine($"   Duration: {session.Duration?.TotalSeconds:F1}s");
    Console.WriteLine($"   Messages: {session.MessageCount}");
    Console.WriteLine($"   User Messages: {session.UserMessageCount}");
    Console.WriteLine($"   Assistant Messages: {session.AssistantMessageCount}");
    Console.WriteLine($"   Tool Calls: {session.ToolCallCount}");

    Console.WriteLine("\n📨 Messages:");
    foreach (var msg in session.Messages)
    {
        Console.WriteLine($"   [{msg.Timestamp:HH:mm:ss}] [{msg.Role}] {msg.Content}");
        if (msg.ToolCalls != null && msg.ToolCalls.Count > 0)
        {
            foreach (var tool in msg.ToolCalls)
            {
                Console.WriteLine($"      🔧 Tool: {tool.Name}");
                Console.WriteLine($"         ID: {tool.Id}");
                Console.WriteLine($"         Args: {tool.Arguments}");
                Console.WriteLine($"         Result: {tool.Result}");
                Console.WriteLine($"         Success: {tool.Success}");
                Console.WriteLine($"         Completed: {tool.IsCompleted}");
            }
        }
    }
}
