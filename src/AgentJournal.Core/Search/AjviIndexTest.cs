using System.Security.Cryptography;

namespace AgentJournal.Core.Search;

/// <summary>
/// Simple test/demo for AjviIndex - demonstrates usage and validates core functionality
/// </summary>
public static class AjviIndexTest
{
    /// <summary>
    /// Runs a simple test of the AJVI index functionality
    /// </summary>
    public static void RunBasicTest()
    {
        const string testIndexPath = "test_index.ajvi";
        const int dimensions = 384; // Common for all-MiniLM-L6-v2

        try
        {
            // Clean up any existing test file
            if (File.Exists(testIndexPath))
            {
                File.Delete(testIndexPath);
            }

            // Create a new index
            using (var index = AjviIndex.Create(testIndexPath, dimensions, AjviIndex.VectorPrecision.Float16))
            {
                Console.WriteLine($"Created index with {dimensions} dimensions");

                // Add some test entries
                for (int i = 0; i < 10; i++)
                {
                    var contentHash = SHA256.HashData(System.Text.Encoding.UTF8.GetBytes($"test content {i}"));
                    var messageId = Guid.NewGuid();
                    byte agentType = (byte)(i % 2); // Alternate between copilot (0) and claude (1)
                    long timestamp = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();

                    // Create a random normalized vector
                    var vector = CreateRandomNormalizedVector(dimensions, seed: i);

                    index.AddEntry(contentHash, messageId, agentType, timestamp, vector);
                }

                Console.WriteLine($"Added {index.EntryCount} entries to the index");

                // Test retrieval
                var firstMessageId = index.GetMessageId(0);
                Console.WriteLine($"First message ID: {firstMessageId}");

                var firstVector = index.GetVector(0);
                Console.WriteLine($"First vector has {firstVector.Length} dimensions");

                // Test search
                var queryVector = CreateRandomNormalizedVector(dimensions, seed: 0);
                var results = index.Search(queryVector, topK: 5);

                Console.WriteLine($"\nSearch results (top 5):");
                foreach (var (idx, score) in results)
                {
                    Console.WriteLine($"  Index {idx}: Score {score:F4}");
                }

                // Test deduplication
                var testHash = SHA256.HashData(System.Text.Encoding.UTF8.GetBytes("test content 5"));
                bool exists = index.ContainsHash(testHash);
                Console.WriteLine($"\nContent hash exists: {exists}");
            }

            // Test reopening the index
            using (var index = AjviIndex.Open(testIndexPath, readOnly: true))
            {
                Console.WriteLine($"\nReopened index with {index.EntryCount} entries");
                Console.WriteLine($"Dimensions: {index.Dimensions}, Precision: {index.Precision}");
            }

            Console.WriteLine("\n✓ All tests passed!");
        }
        finally
        {
            // Clean up
            if (File.Exists(testIndexPath))
            {
                File.Delete(testIndexPath);
            }
        }
    }

    private static float[] CreateRandomNormalizedVector(int dimensions, int seed)
    {
        var random = new Random(seed);
        var vector = new float[dimensions];
        float sumSquares = 0;

        // Generate random values
        for (int i = 0; i < dimensions; i++)
        {
            vector[i] = (float)(random.NextDouble() * 2 - 1); // Random value between -1 and 1
            sumSquares += vector[i] * vector[i];
        }

        // Normalize to unit length
        float magnitude = MathF.Sqrt(sumSquares);
        for (int i = 0; i < dimensions; i++)
        {
            vector[i] /= magnitude;
        }

        return vector;
    }
}
