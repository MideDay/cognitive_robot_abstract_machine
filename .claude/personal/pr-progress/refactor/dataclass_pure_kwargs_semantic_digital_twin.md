# refactor/dataclass_pure_kwargs_semantic_digital_twin

## Plan
- No local test runs for this PR (the user's call); the full CI runs once the user marks #1084 ready for review, and then its checks are watched and each failure reported.
- Keep checking main; when it moves, replay with the catch-up codemods and the static checks, then ask before force-pushing.

## Done
- Replayed onto main `bae41bf22c` (random_events #1082, #1060): fields `60167ecf0e`, methods `29f598dbda`; 5 conflicts combined by hand; static checks clean (0 positional, mypy `call-arg` diff empty, import sweep as main).
- Force-pushed to `work` with a lease on 2026-11-27 (approved; `maintenance push` refuses non-fast-forwards). Merge probe at `29f598dbda` against `bae41bf22c`: clean.
- The local run `runs/rebased7` was stopped after giskardpy; every batch up to it had passed.

## Next
- The user marks #1084 ready; then watch CI. Labels: `api-break` set, `maintenance` missing.
