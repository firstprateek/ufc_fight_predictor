# Adopt agent-git-workflow with no core paths

Status: Accepted once merged · Date: 2026-11-17

## Context

The repo is a personal project built mostly by coding agents. The owner wants agents to land changes
without waiting, and to approve only what they ask to see.

## Decision

Use agent-git-workflow with `core: none`. Every change goes through a PR with proof and a fresh-context
adversarial review, then auto-merges. Check: `make check`. Proof: test output, plus a command transcript
with row counts for pipeline changes.

## Alternatives

- Core paths (schemas, CI): not needed yet; paths can be added later from evidence.

## Consequences

Only changes the owner explicitly asks to approve are root. The `agent-git-workflow` ruleset is enabled once
the CI check exists on `master`.

## Rollback

Delete the gate workflow and AGENTS.md section, and disable the ruleset in repo settings.

## Proof

The setup PR itself passes `agent-gate`.
