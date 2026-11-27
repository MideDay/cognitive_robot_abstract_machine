## Plan
- Make random_events' 16 dataclasses `kw_only=True` and its 58 methods keyword-only, with the callers in every package, doc and test (item random-events of plan keyword-only-migration).
- Plan, status and helpers: `~/.claude/plans/keyword-only-random-events.md`, `.files/` next to it.

## Done
- Bootstrap `8721e88a37` pushed to `work`. Commit 1 (fields) `7d12c7ee93`, local.
- 2026-11-27: main `913a92c231` merged locally as `8759e30cf1`; 6 positional `Continuous`/`Integer` calls from #1069 keywordized (uncommitted, goes into the methods commit's catch-up). 58 methods confirmed (`CPPWrapper._from_cpp` was only missed by the earlier check run).

## Next
- Commit 2 (methods): `kwonly_methods.py` with `CHECK_FILE=check_merged.txt` over the merged file list, then `md_methods.py` for the docs.
- Verify: checkers, mypy, tests per package, docs cells. Ask before each commit and push.
- The user opens the draft PR from `.files/pr_body.md` and adds `api-break`.
