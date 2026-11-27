# refactor/dataclass_pure_kwargs_semantic_digital_twin

## Plan
- The user opens the draft PR from `~/.claude/plans/i-am-currently-working-peppy-bumblebee.files/pr_body.md`, with labels `api-break` and `maintenance`, and records its number on the plan item.
- Before review: discuss the break with the semDT and downstream code owners (`.github/CODEOWNERS`) or in the Cramgineers channel.
- If main moves before review: replay again with the catch-up codemods (plan peppy-bumblebee).

## Done
- Replayed onto main `913a92c231` after coraplex #1076 merged: fields `f6d56ceb3d`, methods `112c015e2f` (with the step_simulation and MeshDecomposer stand-in fixes amended in).
- Checks read at `112c015e2f`'s tree: 0 positional lines in semDT; `unit_losses` empty; mypy `call-arg` diff only the known `connect` change; import sweep matches the baseline; `krrood check --since 913a92c231`: 815 old violations, 0 positional.
- `runs/rebased6` (CI=true) passed every batch, the MongoDB files, `test_integration_pr2` and the launcher test, apart from the 6 known gazebo errors; the Stretch flake passed when rerun alone.
- Pushed to `work` (`112c015e2f`) on 2026-11-27.

## Next
- Open the draft PR (the user).
