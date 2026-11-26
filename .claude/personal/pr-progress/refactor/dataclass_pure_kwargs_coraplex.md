## Plan
- Make coraplex's dataclass fields and method parameters keyword-only (follow-up to #923, #940, #967, #1062), with experiments' callers and overrides.
- Helpers and logs: `~/.claude/plans/i-am-working-on-compiled-possum.files/`; plan `~/.claude/plans/i-am-working-on-compiled-possum-of-117b08c1-7741-46f3-87ee-ed3e0dd8487c.md`.

## Done
- Pushed to `work`, head `1c309c7af2`: fields `0a5f2b5533`, methods `7540ec92f5`, pre-existing rule violations `efc87cea2b`, merge of main `2f3af7d491` (#1047, #986) as `1c309c7af2`.
- Read at the merge: `krrood check --since origin/main` clean; mypy same as main; coraplex 1152 and experiments 372 tests passed (needs iai_maps at origin/ros-jazzy built in ~/workspaces/cram).

## Next
- The user opens the draft PR to cram2 from `.files/pr_body.md` and adds the `api-break` label.
- When semDT makes `RobotSpecification.spawn` and `MultiSim.step_simulation` keyword-only, add `*` to the coraplex stand-ins `RecordingRobotSpecification.spawn` and `SimulationThatCountsItsSteps.step_simulation`.
