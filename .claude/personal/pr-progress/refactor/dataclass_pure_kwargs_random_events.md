## Plan
- Make random_events' 16 dataclasses `kw_only=True` and its 58 methods keyword-only, with the callers in every package, doc and test (item random-events of plan keyword-only-migration).
- Plan and helpers: `~/.claude/plans/keyword-only-random-events.md`, `.files/` next to it.

## Done
- Branch `refactor/dataclass_pure_kwargs_random_events` from main `e2f28cbe15`, bootstrap `8721e88a37` pushed to `work`; item recorded in progress.
- Baseline at `e2f28cbe15`: 16 dataclasses, 58 methods; about 1000 positional call sites in about 250 files.

## Next
- Commit 1 (fields), then commit 2 (methods), with codemods retargeted to random_events; jedi-resolved calls only.
- Verify: checkers, mypy, tests per package, docs cells. Ask before each commit and push.
- The user opens the draft PR from `.files/pr_body.md` and adds `api-break`.
