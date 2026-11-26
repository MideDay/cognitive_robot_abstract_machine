## Plan
- Make coraplex's dataclass fields and method parameters keyword-only (plan keyword-only-migration, item coraplex), with experiments' callers and overrides.
- Helpers and logs: `~/.claude/plans/i-am-working-on-compiled-possum.files/`.

## Done
- Pushed to `work`, head `1c309c7af2`: fields `0a5f2b5533`, methods `7540ec92f5`, pre-existing rule violations `efc87cea2b`, merge of main as `1c309c7af2`; coraplex 1152 and experiments 372 tests passed there.
- Read 2026-11-26 at `1c309c7af2`: merge probe CLEAN against main `81bf3ace24` (8 newer commits, none in coraplex or experiments); no PR on cram2 or the fork.

## Next
- Draft PR #1076 on cram2 is open: the user adds `api-break` and discusses the break with the coraplex code owners before requesting review.
- When semDT makes `RobotSpecification.spawn` and `MultiSim.step_simulation` keyword-only, add `*` to the coraplex stand-ins `RecordingRobotSpecification.spawn` and `SimulationThatCountsItsSteps.step_simulation`.
