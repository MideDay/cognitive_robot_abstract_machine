# refactor/dataclass_pure_kwargs_semantic_digital_twin

## Plan (resolve 2026-11-27)
- Amend the working-tree step_simulation fix into the methods commit `ace404badc` (ask first).
- Tag the state `kwonly-semdt-rebased5`, then replay onto main `913a92c231` (55 commits, coraplex #1076 included): conflict hunks to main's side with `keep_head.py`, then `catchup_fields.py` / `catchup_methods.py` and the Markdown passes, with a fresh `baseline_tree_new5`.
- Checks: `audit_conflicts`, `unit_losses`, `ast_compare`, mypy `call-arg --check-untyped-defs` diff, 0 positional lines in semDT and coraplex.
- Tests with `CI=true`: CI-gated batches, then the batches main's new commits touch, coraplex, experiments, robokudo and the MongoDB files.
- Ask before pushing to `work`; the user opens the draft PR with `api-break`.

## Done
- Stack pass: success, nothing to restack (`work/main` at `913a92c231`). Upstream: no pull request for the branch.

## Next
- Ask to amend the fix, then replay.
