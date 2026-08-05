namespace AgentJournal;

/// <summary>
/// Carries a non-zero exit code out of a command handler.
/// </summary>
/// <remarks>
/// System.CommandLine's <c>SetHandler</c> overloads used in this project accept handlers that
/// return <see cref="Task"/> rather than <c>Task&lt;int&gt;</c>, so a handler has no direct way to
/// influence the process exit code. Handlers previously reported failures such as "session not
/// found" on stderr and then returned normally, which meant the process still exited 0 - any agent
/// or shell script driving the CLI saw those failures as success.
/// Handlers call <see cref="Fail"/>, and <c>Program.Main</c> folds the recorded code into its
/// return value.
/// </remarks>
public static class CommandOutcome
{
    /// <summary>Exit code used when a command fails for an ordinary, expected reason.</summary>
    public const int GeneralFailure = 1;

    /// <summary>Exit code used when a requested item does not exist.</summary>
    public const int NotFound = 2;

    /// <summary>Exit code used when a command completed but some items failed to process.</summary>
    public const int PartialFailure = 3;

    /// <summary>
    /// The exit code recorded by the most recent failing handler, or 0 if none failed.
    /// </summary>
    public static int ExitCode => Volatile.Read(ref _exitCode);

    private static int _exitCode;

    /// <summary>
    /// Records a failure. The first non-zero code wins so the original cause is not masked.
    /// </summary>
    /// <param name="exitCode">The exit code to report.</param>
    /// <remarks>
    /// Commands such as <c>index</c> fan work out across tasks, so this can be called from several
    /// threads at once. The compare-and-swap keeps "first non-zero code wins" true under that
    /// concurrency; a plain read-then-assign could let a later failure overwrite the first.
    /// </remarks>
    public static void Fail(int exitCode = GeneralFailure)
    {
        if (exitCode == 0)
        {
            return;
        }

        Interlocked.CompareExchange(ref _exitCode, exitCode, 0);
    }
}
