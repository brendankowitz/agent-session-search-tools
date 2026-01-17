# Lucene Search Test Fix Summary

## Problem Identified

The failing tests were:
1. `IndexSessionAsync_AddsSessionToIndex` - searched for "Lucene" and got empty results
2. `SearchAsync_BooleanQuery_AND` - searched for "Lucene AND text" and got empty results

But `SearchAsync_FindsRelevantContent` which searched for "Lucene search" PASSED.

## Root Cause

The issue was **NOT** with case sensitivity (StandardAnalyzer handles that correctly). 

The real problem was **document-level vs session-level searching**:

- Each message in a session was indexed as a **separate Lucene document**
- The QueryParser with `DefaultOperator = Operator.AND` searches for documents containing ALL terms
- When searching "Lucene AND text", Lucene looked for a single document with BOTH terms
- But the test data had:
  - Message 1 (User): "Lucene search engine" - contains "lucene" but NOT "text"
  - Message 2 (Assistant): "Full text search" - contains "text" but NOT "lucene"
- Since no single document contained both terms, the search returned 0 results

The passing test worked because "Lucene full-text search" was in a single message, so both terms existed in one document.

## Solution Implemented

Added session-level searching by creating a combined content field:

### Changes Made to `LuceneSearchEngine.cs`:

1. **Added new field constant** (line 29):
   ```csharp
   private const string FIELD_ALL_CONTENT = "all_content"; // Combined content from all session messages
   ```

2. **Updated `IndexSessionAsync` method** (lines 109-110):
   ```csharp
   // Create combined content from all messages for session-level searching
   var allContent = string.Join(" ", session.Messages.Select(m => m.Content ?? ""));
   ```

3. **Updated `IndexSessionsAsync` method** (lines 142-143):
   ```csharp
   // Create combined content from all messages for session-level searching
   var allContent = string.Join(" ", session.Messages.Select(m => m.Content ?? ""));
   ```

4. **Updated `CreateDocument` signature and implementation** (lines 347, 373-377):
   ```csharp
   private Document CreateDocument(Session session, Message message, string allContent)
   {
       // ... existing fields ...
       
       // All content field (analyzed but not stored) - for session-level searching
       if (!string.IsNullOrEmpty(allContent))
       {
           doc.Add(new TextField(FIELD_ALL_CONTENT, allContent, Field.Store.NO));
       }
   }
   ```

5. **Updated QueryParser to search the combined field** (line 196):
   ```csharp
   var parser = new QueryParser(LUCENE_VERSION, FIELD_ALL_CONTENT, _analyzer!);
   ```

## How It Works Now

1. When indexing a session, all message content is combined into a single string
2. Each message document gets both:
   - `FIELD_CONTENT` - the individual message content (stored for highlighting)
   - `FIELD_ALL_CONTENT` - all session messages combined (not stored, just indexed for searching)
3. Searches now query the `FIELD_ALL_CONTENT` field, which contains all session messages
4. Boolean queries like "Lucene AND text" now work because the combined field contains content from all messages

## Test Results Expected

All tests should now pass:

- ✅ `IndexSessionAsync_AddsSessionToIndex` - "Lucene" found in combined content
- ✅ `SearchAsync_BooleanQuery_AND` - "Lucene AND text" found in combined content
- ✅ `SearchAsync_FindsRelevantContent` - continues to work as before

## Benefits

- Session-level searching: Boolean queries work across all messages in a session
- Message-level precision: Individual message content still stored for highlighting
- Performance: `FIELD_ALL_CONTENT` not stored, saving index space
- Backward compatible: Existing functionality preserved
