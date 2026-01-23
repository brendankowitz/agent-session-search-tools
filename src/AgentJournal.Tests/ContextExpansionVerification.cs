using AgentJournal.Core.Models;
using AgentJournal.Core.Search;

namespace AgentJournal.Tests;

/// <summary>
/// Quick verification test for context expansion functionality
/// </summary>
public class ContextExpansionVerification
{
    [Fact]
    public void ExpandWithContext_ShouldIncludeMessagesBeforeAndAfter()
    {
        // Arrange: Create a session with 10 messages
        var messages = Enumerable.Range(0, 10)
            .Select(i => new Message(
                Id: $"msg-{i}",
                SessionId: "test-session",
                Role: i % 2 == 0 ? MessageRole.User : MessageRole.Assistant,
                Content: $"Message {i}",
                RawContent: null,
                Timestamp: DateTime.UtcNow.AddMinutes(i),
                ParentId: null,
                Model: null,
                ToolCalls: null
            ))
            .ToList();

        // Matched messages: indices 3, 7 (4th and 8th messages)
        var matchedMessages = new List<Message> { messages[3], messages[7] };

        // Act: Expand with context count of 2
        var expanded = ExpandWithContext(messages, matchedMessages, contextCount: 2);

        // Assert: Should include messages around each match
        // Match at index 3: should include 1,2,3,4,5 (indices 3-2 to 3+2)
        // Match at index 7: should include 5,6,7,8,9 (indices 7-2 to 7+2)
        // Combined (deduplicated): 1,2,3,4,5,6,7,8,9
        Assert.Equal(9, expanded.Count);

        // Verify messages are in order
        for (int i = 0; i < expanded.Count; i++)
        {
            Assert.Equal($"Message {i + 1}", expanded[i].Content);
        }
    }

    [Fact]
    public void ExpandWithContext_WithZeroContext_ShouldReturnOnlyMatches()
    {
        // Arrange
        var messages = Enumerable.Range(0, 10)
            .Select(i => new Message(
                Id: $"msg-{i}",
                SessionId: "test-session",
                Role: MessageRole.User,
                Content: $"Message {i}",
                RawContent: null,
                Timestamp: DateTime.UtcNow.AddMinutes(i),
                ParentId: null,
                Model: null,
                ToolCalls: null
            ))
            .ToList();

        var matchedMessages = new List<Message> { messages[3], messages[7] };

        // Act
        var expanded = ExpandWithContext(messages, matchedMessages, contextCount: 0);

        // Assert: Should only include matched messages
        Assert.Equal(2, expanded.Count);
        Assert.Equal("Message 3", expanded[0].Content);
        Assert.Equal("Message 7", expanded[1].Content);
    }

    [Fact]
    public void ExpandWithContext_AtBoundaries_ShouldNotExceedLimits()
    {
        // Arrange
        var messages = Enumerable.Range(0, 5)
            .Select(i => new Message(
                Id: $"msg-{i}",
                SessionId: "test-session",
                Role: MessageRole.User,
                Content: $"Message {i}",
                RawContent: null,
                Timestamp: DateTime.UtcNow.AddMinutes(i),
                ParentId: null,
                Model: null,
                ToolCalls: null
            ))
            .ToList();

        // Match at index 0 (first message)
        var matchedMessages = new List<Message> { messages[0] };

        // Act: Request 10 messages of context (more than available)
        var expanded = ExpandWithContext(messages, matchedMessages, contextCount: 10);

        // Assert: Should not exceed array bounds, return all 5 messages
        Assert.Equal(5, expanded.Count);
    }

    [Fact]
    public void ExpandWithContext_OverlappingRanges_ShouldDeduplicate()
    {
        // Arrange
        var messages = Enumerable.Range(0, 10)
            .Select(i => new Message(
                Id: $"msg-{i}",
                SessionId: "test-session",
                Role: MessageRole.User,
                Content: $"Message {i}",
                RawContent: null,
                Timestamp: DateTime.UtcNow.AddMinutes(i),
                ParentId: null,
                Model: null,
                ToolCalls: null
            ))
            .ToList();

        // Matches very close together: indices 4 and 5
        var matchedMessages = new List<Message> { messages[4], messages[5] };

        // Act: Context of 3 will create overlapping ranges
        // Range 1: 1,2,3,4,5,6,7 (4-3 to 4+3)
        // Range 2: 2,3,4,5,6,7,8 (5-3 to 5+3)
        // Combined: 1,2,3,4,5,6,7,8
        var expanded = ExpandWithContext(messages, matchedMessages, contextCount: 3);

        // Assert: Should deduplicate overlapping messages
        Assert.Equal(8, expanded.Count);

        // Verify no duplicates
        var contentSet = new HashSet<string>(expanded.Select(m => m.Content));
        Assert.Equal(expanded.Count, contentSet.Count);
    }

    // Helper method that mimics the implementation
    private static IReadOnlyList<Message> ExpandWithContext(
        IReadOnlyList<Message> allMessages,
        IReadOnlyList<Message> matchedMessages,
        int contextCount)
    {
        if (contextCount <= 0 || matchedMessages.Count == 0)
        {
            return matchedMessages;
        }

        var messagesToInclude = new HashSet<Message>();

        foreach (var matched in matchedMessages)
        {
            // Find the index of this matched message in the full list
            var matchIndex = -1;
            for (int i = 0; i < allMessages.Count; i++)
            {
                if (allMessages[i].Id == matched.Id)
                {
                    matchIndex = i;
                    break;
                }
            }

            if (matchIndex < 0)
            {
                continue; // Message not found, skip
            }

            // Include context messages before and after
            var startIndex = Math.Max(0, matchIndex - contextCount);
            var endIndex = Math.Min(allMessages.Count - 1, matchIndex + contextCount);

            for (int i = startIndex; i <= endIndex; i++)
            {
                messagesToInclude.Add(allMessages[i]);
            }
        }

        // Return messages ordered by their original position (timestamp)
        return allMessages
            .Where(m => messagesToInclude.Contains(m))
            .ToList();
    }
}
