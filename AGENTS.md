# Agent notes

Design and plan: `docs/design.md`; the 2018 project in `legacy/2018/` is history only and nothing new imports from it. Both arrive in the scaffold PR (next); until then `make check` cannot pass (no `pyproject.toml` yet).

## Agent git workflow

Agents land every change through a pull request, following [agent-git-workflow](https://github.com/firstprateek/agent-git-workflow).

- check: `make check` (ruff lint, ruff format check, pytest; needs `uv`). It must pass before a PR is opened.
- proof: test output for library code. For pipeline or CLI changes, also a transcript of the command run with row counts. No personal data, emails or home paths in proof (the repo is public).
- core: none. Nothing needs the owner's approval unless they ask for it on a change.
- owner: @firstprateek. Agents act on GitHub as the owner.
- Decision records are in `docs/decisions/`.
- Leaf PRs auto-merge after a fresh-context review and green checks. Root PRs (with `core: none`, only
  changes the owner asked to approve) also need a decision record, and the owner approves it by adding
  the `owner-approved` label. Agents never add or remove that label, and never push to the default
  branch.
- Waiting for the owner: `is:open is:pr label:tier:root -label:owner-approved`.
