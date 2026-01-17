# Quick Fix Reference - Lucene Session-Level Search

## Files Modified
- `src/AgentJournal.Core/Search/LuceneSearchEngine.cs`

## Changes Summary

### 1. Added Field Constant (Line 29)
```csharp
private const string FIELD_ALL_CONTENT = "all_content"; // Combined content from all session messages
```

### 2. Updated IndexSessionAsync (Lines 109-116)
```csharp
// Create combined content from all messages for session-level searching
var allContent = string.Join(" ", session.Messages.Select(m => m.Content ?? ""));

// Index each message in the session
foreach (var message in session.Messages)
{
    var doc = CreateDocument(session, message, allContent);
    _writer!.AddDocument(doc);
}
```

### 3. Updated IndexSessionsAsync (Lines 150-158)
```csharp
// Create combined content from all messages for session-level searching
var allContent = string.Join(" ", session.Messages.Select(m => m.Content ?? ""));

// Index each message
foreach (var message in session.Messages)
{
    var doc = CreateDocument(session, message, allContent);
    _writer!.AddDocument(doc);
}
```

### 4. Updated CreateDocument Method (Lines 347, 373-377)
```csharp
// Changed signature
private Document CreateDocument(Session session, Message message, string allContent)
{
    // ... existing fields ...
    
    // NEW: All content field (analyzed but not stored) - for session-level searching
    if (!string.IsNullOrEmpty(allContent))
    {
        doc.Add(new TextField(FIELD_ALL_CONTENT, allContent, Field.Store.NO));
    }

    return doc;
}
```

### 5. Updated SearchAsync QueryParser (Line 196)
```csharp
// Changed from FIELD_CONTENT to FIELD_ALL_CONTENT
var parser = new QueryParser(LUCENE_VERSION, FIELD_ALL_CONTENT, _analyzer!);
```

## Testing
Run: `dotnet test --filter "LuceneSearchEngineTests"`

Expected: All tests pass, including:
- IndexSessionAsync_AddsSessionToIndex
- SearchAsync_BooleanQuery_AND
- SearchAsync_FindsRelevantContent
