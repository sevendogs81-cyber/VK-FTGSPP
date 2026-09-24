# Maintainer tree

This directory is **not** part of the user path (`scripts/setup.sh`,
`ftgspp`, and `examples/`). Keep it for CI, agents, and local research.
Unsupported `tools/` are omitted from release source distributions.

- `tests/` — CPU contracts and GPU engineering scripts. Official evaluation
  is still `ftgspp eval`.
- `tools/` — unsupported research helpers; they may change without notice.
