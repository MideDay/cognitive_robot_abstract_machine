# refactor/dataclass_pure_kwargs_semantic_digital_twin

## Plan (resolve 2026-11-27)
- Replay onto main `913a92c231` (55 commits, coraplex #1076 included): conflict hunks to main's side with `keep_head.py`, then the catch-up codemods and Markdown passes against `baseline_tree_new5`.
- Checks: `audit_conflicts`, `unit_losses`, `ast_compare`, mypy `call-arg --check-untyped-defs` diff, 0 positional lines in semDT.
- Tests with `CI=true`: CI-gated batches, then every batch and the MongoDB files.
- Ask before pushing to `work`; the user opens the draft PR with `api-break`.

## Done
- Stack pass: success, nothing to restack (`work/main` at `913a92c231`). Upstream: no pull request for the branch.
- step_simulation fix amended into the methods commit (`e5761ae5a3`); old head tagged `kwonly-semdt-rebased5`.
- Fields commit replayed as `f6d56ceb3d`: 10 conflicts on main's side; 5+1 `kw_only` decorators, 10 call edits; main's docstring drift kept as it is. 0 positional dataclasses in semDT; audit losses are only classes main removed.

## Next
- Methods commit: 42 conflicts on main's side; catch-up codemod running (3 flagged semDT methods: `FindParts._apply`, `SplitPartFromShape._apply`, `AnnotationPartFinder._add`).
