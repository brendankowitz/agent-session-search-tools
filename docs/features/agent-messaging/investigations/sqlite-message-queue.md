# Investigation: SQLite Message Queue

**Feature**: agent-messaging  
**Status**: Proposed  
**Created**: 2026-08-04

## Approach

Store messages in the same SQLite database that already backs the task journal
(`.agent-journal/tasks/journals.db`), adding a `messages` table alongside `task_journals`,
`task_entries`, and `task_artifacts`.

Delivery is an `INSERT`. Collection is an `UPDATE` that stamps `collected_at` inside a
transaction. No daemon, no network, no filesystem scanning.

### Schema

```sql
CREATE TABLE messages (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    recipient    TEXT NOT NULL,          -- agent name, or '*' for broadcast
    sender       TEXT NOT NULL,
    severity     TEXT NOT NULL,          -- 'info' | 'warning' | 'blocker'
    subject      TEXT NOT NULL,
    body         TEXT NOT NULL,
    reply_to     INTEGER REFERENCES messages(id) ON DELETE SET NULL,
    created_at   TEXT NOT NULL,
    collected_at TEXT                     -- NULL while pending
);

CREATE INDEX idx_messages_pending ON messages (recipient, collected_at, id);
```

`collected_at IS NULL` is the entire read/unread model — the state a file-based design has to
encode as a directory move. Broadcast is handled by matching `recipient IN (@me, '*')`, which
means a broadcast is one row rather than one row per recipient.

### Operations

| Command | SQL |
|---------|-----|
| `message send` | `INSERT INTO messages (...)` |
| `message inbox` | `SELECT ... WHERE recipient IN (@me,'*') AND collected_at IS NULL ORDER BY id` |
| `message collect` | `UPDATE messages SET collected_at=@now WHERE id=@id AND collected_at IS NULL` in a transaction, returning the row |
| `message wait` | poll `inbox` on an interval; SQLite has no native notification |
| `message thread` | recursive CTE over `reply_to` |

The `AND collected_at IS NULL` guard in `collect` makes collection idempotent and race-free: two
agents racing to collect the same message produce exactly one winner, decided by the database
rather than by filesystem move semantics.

## Tradeoffs

| Pros | Cons |
|------|------|
| Concurrency correctness is the engine's problem, not ours — WAL plus a transactional guarded `UPDATE` removes the entire class of interleaved-writer bugs | A message is no longer readable or repairable with a text editor; recovery needs `sqlite3` or a CLI subcommand |
| Reuses infrastructure that already exists and is already tested (`SqliteConnectionFactory`, WAL, `busy_timeout`, foreign keys) | Adds write contention against a database also used by the task journal |
| Querying is expressive: filter by severity, sender, age, or thread with an index behind it, instead of listing a directory and parsing every file | Requires a schema migration path once the table ships; files need none |
| Read/unread is one nullable column, not a two-directory protocol whose invariant lives in prose | Loses the "just `ls` the inbox" affordance that makes a file mailbox trivially debuggable |
| Broadcast is a single row, so a 10-agent broadcast does not become 10 files to reconcile | Cross-machine use is off the table (already a non-goal, but SQLite makes it structural — network filesystems and SQLite are a known-bad pairing) |
| Consistent with the task journal's storage decision, so agents learn one mental model | A corrupt database takes down messaging *and* the task journal; separate files fail independently |

## Alignment

- [x] **No daemon** — delivery is a local `INSERT`; nothing needs to be running
- [x] **Offline and local** — single file in the repository, no network transport
- [x] **Crash-safe** — WAL plus transactions; a killed process cannot leave a torn message
- [x] **Concurrent-safe** — the guarded `UPDATE` makes double-collection impossible by construction
- [ ] **Human-inspectable** — *not met.* A mailbox is no longer editable in a text editor. Mitigated,
      not solved, by `message list --robot` and `message export`
- [x] **Append-only** — messages are inserted once; only `collected_at` mutates, and only NULL→timestamp
- [ ] **Paths over payloads** — *deliberately dropped.* Bodies live in the `body` column. The task
      journal migration already abandoned artifact paths for the same reason: a path is a second
      thing that can go missing
- [x] **Separable from the journal** — a distinct table; the ledger's append-only entry flow is untouched
- [x] **Non-destructive** — collection stamps a timestamp; nothing is deleted, so the audit trail survives

Two constraints in the feature readme are not met. Both were written when a file-based mailbox was
the assumed design, and both should be renegotiated rather than treated as failures of this
approach — see Verdict.

## Evidence

The task journal migrated from files to SQLite in this same branch, which makes the comparison
concrete rather than theoretical:

- **The concurrency win is real and measured.** `ConcurrentFixRounds_EachGetADistinctRoundNumber`
  and `ConcurrentWritersFromSeparateStores_AllEntriesSurvive` (20 concurrent writers) both pass
  against the SQLite store. The file-based predecessor computed its fix round by reading the
  ledger, incrementing, and appending — a read-modify-write with no lock, which loses rounds under
  exactly the multi-agent concurrency this feature exists to enable.
- **Parsing was the larger defect source, not speed.** The file-backed journal silently dropped
  malformed ledger lines and mis-parsed headers (three separate findings in review). None of those
  failure modes survive in a typed schema. Messaging would have inherited the same class of bug:
  every message file is a parse boundary.
- **Speed is not the argument.** At mailbox scale (hundreds of messages, kilobytes each) a
  directory read is likely *faster* than opening a database connection. The case for SQLite is
  correctness under concurrency and elimination of silent parse failures, not throughput. Claiming
  a performance win here would not survive measurement.
- **The infrastructure is already paid for.** `SqliteConnectionFactory` centralises WAL,
  `busy_timeout=10000`, and `ForeignKeys=true`. A new table inherits all of it. A file mailbox
  would need its own atomicity protocol, and the task journal's history shows that protocol is
  where the bugs live.

## Verdict

**Recommended**, with two constraints in the feature readme requiring amendment:

1. **"Human-inspectable"** should be downgraded from a hard constraint to a requirement satisfied
   by tooling (`message list`, `message export`). It was a genuine property of the file design and
   is genuinely lost here — this is a real cost, not a technicality, and the readme should say so
   rather than quietly dropping the bullet.
2. **"Paths over payloads"** should be removed. It described the task journal's file-era
   brief/report handover, which no longer exists.

The decisive argument is that messaging's entire purpose is *concurrent* multi-agent coordination.
A design whose correctness rests on hand-rolled filesystem atomicity is at its weakest precisely
where this feature is most used. The task journal's own migration demonstrated that cost in this
repository, not in the abstract.

Remaining open question for the ADR: whether messages belong in `journals.db` or a sibling
`messages.db`. A separate file isolates corruption blast radius and lets messaging be deleted
without touching task history; a shared file keeps one connection and permits future joins between
a message and the task it concerns. Current lean is a **separate `messages.db`** — the two have no
transactional relationship, and nothing in the capability list requires an atomic write spanning
both.
