# Feature: Agent Messaging

**Status**: Exploring
**Created**: 2026-08-04

## Problem Statement

The task journal (`docs/TASK_JOURNAL.md`) solves coordination *across time* — a single agent
resuming its own work after context compaction. It does not solve coordination *across agents*
running concurrently.

There is currently no way for:

- a subagent to report a blocker to its coordinator without terminating,
- two agents editing different parts of a repository to warn each other about a shared interface
  change,
- a long-running agent to leave a finding for whichever agent picks up the work next,
- a human to inject a correction into a running agent's queue.

The only available workaround is to funnel every signal back through the coordinator's conversation
context — which is precisely the scarcest resource, and the one whose exhaustion the task journal
was built to survive.

## Solution Vision

A durable, transactional message store backed by SQLite, requiring no daemon and no network.
Delivery is an `INSERT`; collection is a guarded `UPDATE` that stamps a collection timestamp.

This mirrors the storage decision already made for the task journal, which migrated from a
file-backed ledger to SQLite in the same branch. The motivation is the same: messaging exists to
coordinate *concurrent* agents, and a design whose correctness depends on hand-rolled filesystem
atomicity is weakest exactly where it is used most.

### Key Capabilities

1. **Addressed delivery**: send a message to a named agent, or broadcast to all known recipients
2. **Read/unread tracking**: a nullable `collected_at` column, updated transactionally
3. **Severity triage**: `info` / `warning` / `blocker`, so a recipient can decide whether a message
   justifies interrupting its current task without reading every message first
4. **Threading**: `reply_to` links a response to its cause
5. **Blocking wait**: an agent can park until a message arrives rather than busy-polling
6. **Agent-friendly**: `--robot` JSON for machine consumption, with `message export` for bodies
   large enough to be worth keeping out of an agent's context

## Constraints

### Technical Constraints
- **No daemon**: nothing may need to be running for delivery to succeed
- **Offline and local**: repository filesystem only; no network transport
- **Crash-safe**: a killed process must not lose or corrupt delivered messages
- **Concurrent-safe**: multiple agents may send and collect simultaneously
- **Inspectable via tooling**: a mailbox must be dumpable and exportable through the CLI.
  *Amended from an earlier "human-inspectable with a text editor" constraint — see
  [sqlite-message-queue](investigations/sqlite-message-queue.md). Direct editability is a real
  cost of the SQLite decision, accepted knowingly rather than overlooked.*

### Design Constraints
- **Append-only**: a message is written once and never edited; only `collected_at` mutates, and
  only from NULL to a timestamp
- **Separable from the journal**: messaging must not compromise the task journal's append-only
  entry flow, and should not share its database file
- **Non-destructive**: collecting a message must not destroy the audit trail

### Non-Goals
- Cross-machine or networked transport
- Authentication between agents — anything that can write to the repository can send as any sender
- Delivery guarantees beyond the collection stamp
- Streaming or partial message delivery

## Investigations

| Investigation | Status | Summary |
|--------------|--------|---------|
| [sqlite-message-queue](investigations/sqlite-message-queue.md) | ✅ Recommended | `messages` table with a guarded transactional collect; reuses the task journal's SQLite infrastructure |

### Rejected alternatives

- **file-based-mailbox** — inbox/read directory pair per recipient; delivery by file create,
  collection by atomic move. Rejected: its correctness rests on filesystem move semantics, and the
  task journal's file-era implementation produced exactly the concurrency and silent-parse defects
  this feature cannot afford. Its one genuine advantage — text-editor repairability — is
  acknowledged as lost.

### Candidate investigations (not yet started)

- **journal-embedded-messages** — put messages in the task journal's own database and entry stream.
  Fewest new concepts; risks coupling two features with no transactional relationship, and widens
  the blast radius of a corrupt file.
- **named-pipe-transport** — OS-level IPC for low-latency delivery. Removes polling for
  `message wait`, but requires both agents alive simultaneously and loses durability, so it is at
  best a complement to a durable store rather than a replacement.

## Decision

*No ADR yet. The investigation recommends SQLite, but this feature is **specified only — not
implemented**. Two open questions remain for the ADR: whether messages live in a sibling
`messages.db` (current lean) or share `journals.db`, and how `message wait` polls without burning
CPU.*

