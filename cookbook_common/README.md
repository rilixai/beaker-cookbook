# cookbook-common

Shared, recipe-agnostic helpers for the rilixai cookbook benchmark recipes
(`apex_agents`, `hotpotqa`):

- `local_eval` — the SDK-only local evaluation loop (Shape B) used by each
  recipe's `cli.py evaluate`.
- `cli_support` — local CLI plumbing: candidate-JSON loading, spec-validation
  logging, and eval-report serialization.

This is an internal workspace member, not a published package.
