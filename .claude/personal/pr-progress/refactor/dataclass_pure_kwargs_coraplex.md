## Plan
- Make coraplex's dataclass fields and method parameters keyword-only (follow-up to #923, #940, #967, #1062), with experiments' callers and overrides.
- Helpers and logs: `~/.claude/plans/i-am-working-on-compiled-possum.files/`; plan `~/.claude/plans/i-am-working-on-compiled-possum-of-117b08c1-7741-46f3-87ee-ed3e0dd8487c.md`.

## Done
- Three commits pushed to `work`, head `efc87cea2b`: fields `0a5f2b5533`, methods `7540ec92f5`, pre-existing rule violations `efc87cea2b` (renames `for_source`→`from_source`, `with_friction`→`contact_parameters`, `_with_z_up_and_x_towards`→`_frame_with_z_up_and_x_towards`; unused `validate` removed).
- Read at `efc87cea2b`: `krrood check --since origin/main` clean; coraplex 1136 and experiments 372 tests passed as on main.

## Next
- The user opens the draft PR to cram2 from `.files/pr_body.md` and adds the `api-break` label.
- The branch is 38 commits behind `main`: check main for new coraplex code and catch up before review.
- When semDT makes `RobotSpecification.spawn` and `MultiSim.step_simulation` keyword-only, add `*` to the coraplex stand-ins `RecordingRobotSpecification.spawn` and `SimulationThatCountsItsSteps.step_simulation`.
