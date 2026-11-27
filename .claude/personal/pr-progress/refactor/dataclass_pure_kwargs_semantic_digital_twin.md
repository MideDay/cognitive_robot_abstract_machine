# refactor/dataclass_pure_kwargs_semantic_digital_twin

## Plan (resolve 2026-11-27)
- Replay onto main `913a92c231` (coraplex #1076 included) with the catch-up codemods - done.
- Tests with `CI=true`: every batch, the robokudo MongoDB files, `test_integration_pr2`, the launcher test (`run_rebased6.sh`, `runs/rebased6*`).
- Ask before pushing to `work`; the user opens the draft PR with `api-break` (text in `pr_body.md`).

## Done
- Stack pass: success, nothing to restack (`work/main` at `913a92c231`). Upstream: no pull request for the branch.
- step_simulation fix amended into the methods commit; old head tagged `kwonly-semdt-rebased5`.
- Fields `f6d56ceb3d`: 10 conflicts, 6 `kw_only` decorators, 10 call edits.
- Methods `7f6a14fca8`: 42 conflicts, 3 new keyword-only methods, 78 call edits, 6 Markdown calls and 1 test call by hand.
- Checks read at `7f6a14fca8`: 0 positional lines in semDT; `unit_losses` empty; audit losses only code main removed or moved; mypy `call-arg` diff only the known `connect` change; import sweep matches the baseline (64); ORM regenerated.

## Next
- Read `runs/rebased6*`; then ask to push.
