# refactor/dataclass_pure_kwargs_semantic_digital_twin

## Plan (resolve 2026-11-27, afternoon)
- Replay onto main `bae41bf22c` (random_events #1082, #1060): the merge probe read 5 conflicting semDT files at head `112c015e2f`.
- Combine both sides of each conflict by hand, rerun the catch-up for main's new code, run the static checks, then a full `CI=true` test run.
- Ask before force-pushing to `work`; LucaKro's approval of `112c015e2f` will likely go stale. Keep #1084 a draft.

## Done
- Stack pass: success, nothing to do (`work/main` at `bae41bf22c`). `/upstream-reviews`: #1084 approved by LucaKro, no threads; CI tests skipped on the draft (only `find-api-breaks` and `label` ran, both passed).
- Fields commit applied cleanly; methods commit: 5 conflicts combined by hand (random_events keywords from main + semDT keywords from the branch). 0 positional lines in semDT; no fields call edits needed.

- Replayed: fields `60167ecf0e`, methods `29f598dbda`; catch-up found nothing new; mypy `call-arg` diff empty; import sweep as main; ORM regenerated.

## Next
- Read `runs/rebased7*` (CI=true), then ask before force-pushing.
- Labels: `api-break` is set; `maintenance` still missing (the user).
