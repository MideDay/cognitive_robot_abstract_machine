# refactor/dataclass_pure_kwargs_semantic_digital_twin

## Plan
- Draft PR cram2 #1084 is open (head `112c015e2f`). The user adds the labels `api-break` and `maintenance`, since this account cannot label on cram2.
- Before review: discuss the break with the semDT and downstream code owners (`.github/CODEOWNERS`) or in the Cramgineers channel.
- If main moves before review: replay again with the catch-up codemods (plan peppy-bumblebee).

## Done
- Replayed onto main `913a92c231` after coraplex #1076 merged: fields `f6d56ceb3d`, methods `112c015e2f` (with the step_simulation and MeshDecomposer stand-in fixes amended in).
- Checks read at `112c015e2f`'s tree: 0 positional lines in semDT; `unit_losses` empty; mypy `call-arg` diff only the known `connect` change; import sweep matches the baseline; `krrood check --since 913a92c231`: 815 old violations, 0 positional.
- `runs/rebased6` (CI=true) passed every batch, the MongoDB files, `test_integration_pr2` and the launcher test, apart from the 6 known gazebo errors; the Stretch flake passed when rerun alone.
- Pushed to `work` (`112c015e2f`) and opened draft PR #1084 on 2026-11-27.

## Next
- Labels and the code-owner discussion (the user); replay again if main moves before review.
