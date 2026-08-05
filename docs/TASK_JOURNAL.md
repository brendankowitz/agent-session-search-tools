# Task Journal

A SQLite-backed ledger that lets an agent resume a multi-task plan after its conversation context is
compacted, truncated, or lost entirely.

## The problem

An agent executing a long plan holds its progress in conversation context. When that context is
compacted the agent loses track of which tasks are done, and typically either redoes completed work
or skips work it believes it finished. Neither failure is visible until much later.

The fix is to keep progress somewhere that does not live in context. The task journal writes it to
a database, so the answer to "where was I?" is a query rather than a memory.

This design follows the pattern in
[obra/superpowers `subagent-driven-development`](https://github.com/obra/superpowers/tree/main/skills/subagent-driven-development),
adapted to a CLI plus MCP surface.

## Storage

All journals in a repository share one database, in a directory that is already git-ignored:

```
.agent-journal/tasks/journals.db
```

```sql
task_journals  (name PK, plan_path, task_count, created_at)
task_entries   (id PK AUTOINCREMENT, journal_name FK, task_number, state, fix_round, note, created_at)
task_artifacts (journal_name FK, task_number, kind, content, updated_at,
                PRIMARY KEY (journal_name, task_number, kind))
```

Three properties make this work:

- **A journal names its plan.** `plan_path` binds a journal to one plan file, so it cannot silently
  be reused for different work.
- **`task_entries` is append-only.** Rows are inserted, never updated, so the table doubles as an
  audit trail.
- **The last entry for a task wins.** Ordering is by `id` — insertion order, not timestamp, so ties
  are impossible. A task that was completed and then reopened resolves to incomplete and becomes
  the resume point again.

### Why SQLite rather than files

This started as a Markdown ledger. It moved to SQLite for correctness, not speed — at ledger scale
a file read is likely *faster*. The two arguments that decided it:

- **Concurrent fix rounds.** Computing the next fix round means read-max, increment, append. As
  separate file operations that is an unguarded read-modify-write, and two agents reopening tasks
  at once silently collapse to one round. In SQLite it is one transaction, verified by
  `ConcurrentFixRounds_EachGetADistinctRoundNumber`.
- **Silent parse failures.** The Markdown ledger dropped malformed lines and mis-parsed headers
  without reporting anything. A typed schema removes that failure class entirely.

The genuine cost is that a journal is no longer repairable in a text editor. `task show` and
`--robot` output exist to offset that, but they do not fully replace it.

## Resume semantics

`NextTask` is the lowest-numbered task without a terminal `complete` entry. That single rule covers
every case:

| Ledger state | Next task |
|---|---|
| Nothing recorded | Task 1 |
| Task 1 complete | Task 2 |
| Task 2 complete, task 1 untouched | Task 1 |
| Task 1 complete then reopened | Task 1 |
| All complete | none — `isComplete` is true |

## CLI

```bash
# Bind a journal to a plan. Task count comes from '## Task N' headings unless --tasks is given.
agent-journal task init docs/plans/refactor.md --name refactor

# Where was I?
agent-journal task next --robot

# Record progress as it happens
agent-journal task start 1 --brief ./brief.md
agent-journal task complete 1 --report ./report.md --note "extracted the parser"

# Review found a problem after the task was marked complete
agent-journal task fix 1 --note "leak in the disposal path"

# Read a stored artifact back
agent-journal task show brief 1
agent-journal task show report 1 --out ./report.md

agent-journal task status
agent-journal task list
```

`--brief` and `--report` accept a file path or `-` to read stdin.

`--name` may be omitted whenever the repository contains exactly one journal.

Artifacts have no filesystem path, so `task show` is how they are retrieved — either to stdout or,
with `--out`, exported to a file.

### Exit codes

| Code | Meaning |
|---|---|
| 0 | Success |
| 1 | Failure (bad arguments, task number out of range, plan mismatch, ambiguous journal) |
| 2 | Journal, plan file, or artifact not found |

### Machine-readable output

Every subcommand accepts `--robot` and emits JSON. This is the form an agent should use:

```json
{
  "isComplete": false,
  "planPath": "/repo/docs/plans/refactor.md",
  "nextTask": {
    "number": 1,
    "state": "FixRound",
    "fixRound": 1,
    "lastNote": "leak in the disposal path",
    "hasBrief": true,
    "hasReport": true
  }
}
```

Notes are stored verbatim, so a multi-line note round-trips through `--robot` unchanged. Human
output flattens notes to one line for display only.

## MCP tools

| Tool | Purpose |
|---|---|
| `TaskInit` | Bind a journal to a plan file |
| `TaskStatus` | Full snapshot, including the resume point |
| `TaskRecord` | Record `started`, `complete`, or `fix` for a task |
| `TaskWriteArtifact` | Store a brief or report |
| `TaskReadArtifact` | Read a brief or report back |
| `TaskSearch` | Full-text search across notes, briefs, and reports |
| `TaskList` | List journals in the repository |

Every tool takes an optional `repositoryPath`. An MCP server's working directory is frequently not
the repository being worked on, so pass it explicitly rather than relying on the default.

## Searching the journal

Notes, briefs, and reports accumulate into a record of why the work went the way it did. That record
is searchable:

```bash
agent-journal search "disposal path" --include-tasks
```

The index is a SQLite FTS5 table inside `journals.db` itself, maintained by triggers, so a write and
its indexing happen in the same transaction — the index cannot drift from the data.

**This index is deliberately repo-local, unlike session and knowledge search.** Sessions and
knowledge live in a user-global index under `~/.agent-journal/`. Task journals do not, and must not:
if repo-local task content were written into the global index, running `index --rebuild` from one
repository would delete another repository's task documents with no way to recover them, because the
source data is not reachable from there. Keeping the index beside the data it describes makes that
class of failure impossible.

Consequences worth knowing:

- `--include-tasks` only works inside a repository, and only searches that repository.
- Matching is lexical (FTS5), regardless of `--mode`.
- `index --rebuild` rebuilds the task index for the current repository, when run inside one.
- Search terms are quoted and escaped before reaching `MATCH`, so `:`, `"`, `*`, and `-` are treated
  as literal text rather than FTS5 operators.

## Why briefs and reports are stored, not pasted

A coordinator that inlines each subagent's report grows its own context with every completed task
and hits compaction sooner — the exact failure the journal exists to survive.

So the coordinator writes a brief with `TaskWriteArtifact` and tells the subagent to fetch it with
`TaskReadArtifact`. Neither the brief nor the report ever passes through the coordinator's context.
Coordinator context stays flat regardless of plan length.

## Recommended loop

```
1. task init <plan>
2. task next --robot          -> task number N
3. TaskWriteArtifact N brief  -> write instructions for the subagent
4. task start N
5. dispatch subagent, telling it to call TaskReadArtifact N brief
6. subagent stores its report via TaskWriteArtifact N report
7. coordinator reads it with TaskReadArtifact N report and reviews
   - good     -> task complete N --note "..."
   - problems -> task fix N --note "what was wrong", then back to step 4
8. repeat from step 2 until isComplete
```

Steps 2 and 4 are what make the loop restartable. An agent that resumes with no memory of steps
1–7 can run `task next` and rejoin at exactly the right place.

## Validation

The store rejects operations that would corrupt the journal rather than accepting them silently:

- Task numbers outside `1..taskCount` are rejected for both entries and artifacts. A phantom entry
  would never appear in a snapshot, so the write would look successful but do nothing.
- `init` fails when the plan file does not exist — including when `--tasks N` is supplied, so a
  typo'd path cannot create a journal pointing at nothing.
- `init` against an existing journal with a different plan path fails. Rebinding would resume the
  wrong work.
- `init` against an existing journal with the same plan path is idempotent and preserves progress.
  If the plan has grown, `task_count` is raised; it is never lowered, because lowering it would
  orphan recorded entries.
- `ForRepository` throws when no repository is found, rather than falling back to the current
  directory and writing a journal somewhere unexpected.

