using System.Security.Cryptography;
using System.Text;

namespace AgentJournal.Core.Utilities;

/// <summary>
/// Utility methods for content processing
/// </summary>
public static class ContentUtils
{
    private const long MaxFileSizeBytes = 10 * 1024 * 1024; // 10MB

    /// <summary>
    /// Computes SHA256 hash of content
    /// </summary>
    /// <param name="content">Content to hash</param>
    /// <returns>Hexadecimal hash string</returns>
    public static string ComputeHash(string content)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(content, nameof(content));

        using var sha256 = SHA256.Create();
        var bytes = Encoding.UTF8.GetBytes(content);
        var hash = sha256.ComputeHash(bytes);
        return Convert.ToHexString(hash);
    }

    /// <summary>
    /// Extracts title from markdown content or falls back to filename
    /// </summary>
    /// <param name="content">Markdown content</param>
    /// <param name="filePath">File path for fallback</param>
    /// <returns>Extracted title</returns>
    public static string ExtractTitle(string content, string filePath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(content, nameof(content));
        ArgumentException.ThrowIfNullOrWhiteSpace(filePath, nameof(filePath));

        // Try to extract title from first markdown header
        var lines = content.Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        foreach (var line in lines)
        {
            if (line.StartsWith("# "))
            {
                return line.Substring(2).Trim();
            }
        }

        // Fallback to filename without extension
        return Path.GetFileNameWithoutExtension(filePath);
    }

    /// <summary>
    /// Validates path to prevent directory traversal attacks
    /// </summary>
    /// <param name="path">Path to validate</param>
    /// <param name="allowedBasePath">Optional allowed base path to restrict to</param>
    /// <returns>Validated full path</returns>
    /// <exception cref="ArgumentException">If path is invalid or attempts traversal</exception>
    public static string ValidatePath(string path, string? allowedBasePath = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path, nameof(path));

        try
        {
            // Get full path to resolve any relative path components
            var fullPath = Path.GetFullPath(path);

            // If allowedBasePath is specified, ensure path is under it
            if (!string.IsNullOrWhiteSpace(allowedBasePath))
            {
                var fullBasePath = Path.GetFullPath(allowedBasePath);

                // Ensure the path is under the allowed base path
                if (!fullPath.StartsWith(fullBasePath, StringComparison.OrdinalIgnoreCase))
                {
                    throw new ArgumentException(
                        $"Path '{path}' is outside allowed directory '{allowedBasePath}'",
                        nameof(path));
                }
            }

            // Additional check: Ensure the path doesn't contain path traversal sequences after normalization
            if (fullPath.Contains(".."))
            {
                throw new ArgumentException(
                    $"Path '{path}' contains invalid path traversal sequences",
                    nameof(path));
            }

            return fullPath;
        }
        catch (Exception ex) when (ex is not ArgumentException)
        {
            throw new ArgumentException($"Invalid path: {path}", nameof(path), ex);
        }
    }

    /// <summary>
    /// Validates file size before reading
    /// </summary>
    /// <param name="filePath">Path to file</param>
    /// <param name="maxSizeBytes">Maximum allowed file size in bytes (default 10MB)</param>
    /// <exception cref="InvalidOperationException">If file exceeds size limit</exception>
    public static void ValidateFileSize(string filePath, long maxSizeBytes = MaxFileSizeBytes)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(filePath, nameof(filePath));

        if (!File.Exists(filePath))
        {
            throw new FileNotFoundException($"File not found: {filePath}", filePath);
        }

        var fileInfo = new FileInfo(filePath);
        if (fileInfo.Length > maxSizeBytes)
        {
            throw new InvalidOperationException(
                $"File '{filePath}' exceeds maximum size limit of {maxSizeBytes / 1024 / 1024}MB " +
                $"(actual size: {fileInfo.Length / 1024 / 1024}MB)");
        }
    }

    /// <summary>
    /// Escapes special characters in LIKE patterns to prevent SQL injection
    /// </summary>
    /// <param name="pattern">Pattern to escape</param>
    /// <returns>Escaped pattern</returns>
    public static string EscapeLikePattern(string pattern)
    {
        if (string.IsNullOrEmpty(pattern))
        {
            return pattern;
        }

        // Escape % and _ which are wildcards in SQL LIKE
        return pattern.Replace("%", "\\%").Replace("_", "\\_");
    }

    /// <summary>
    /// Sanitizes FTS5 query to prevent query syntax errors.
    /// If query is wrapped in quotes, treats as phrase search.
    /// Otherwise, splits words and joins with OR for inclusive search.
    /// </summary>
    /// <param name="query">Query to sanitize</param>
    /// <returns>Sanitized query</returns>
    public static string SanitizeFts5Query(string query)
    {
        if (string.IsNullOrWhiteSpace(query))
        {
            return query;
        }

        var trimmed = query.Trim();

        // If user explicitly wrapped in quotes, treat as phrase search
        if (trimmed.StartsWith('"') && trimmed.EndsWith('"') && trimmed.Length > 2)
        {
            // Escape any internal quotes and return as-is (already quoted)
            var inner = trimmed[1..^1];
            var escaped = inner.Replace("\"", "\"\"");
            return $"\"{escaped}\"";
        }

        // Otherwise, split into words and join with OR for inclusive search
        // Remove FTS5 special characters from each term
        var terms = trimmed.Split(' ', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        var sanitizedTerms = new List<string>();

        foreach (var term in terms)
        {
            // Remove FTS5 operators and special chars, keep alphanumeric and common punctuation
            var sanitized = SanitizeFts5Term(term);
            if (!string.IsNullOrWhiteSpace(sanitized))
            {
                sanitizedTerms.Add(sanitized);
            }
        }

        if (sanitizedTerms.Count == 0)
        {
            return "\"\""; // Empty query
        }

        if (sanitizedTerms.Count == 1)
        {
            return sanitizedTerms[0];
        }

        // Join with OR for inclusive search
        return string.Join(" OR ", sanitizedTerms);
    }

    /// <summary>
    /// Sanitizes a single FTS5 term by removing special characters
    /// </summary>
    private static string SanitizeFts5Term(string term)
    {
        // Remove characters that are FTS5 operators or could cause syntax errors
        // Keep: letters, digits, hyphens, underscores
        var sb = new StringBuilder();
        foreach (var c in term)
        {
            if (char.IsLetterOrDigit(c) || c == '-' || c == '_')
            {
                sb.Append(c);
            }
        }
        return sb.ToString();
    }
}
