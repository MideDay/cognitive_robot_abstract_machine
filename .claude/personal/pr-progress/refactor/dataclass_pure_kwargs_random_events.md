## Plan
- Make random_events' 16 dataclasses `kw_only=True` and its 58 methods keyword-only, with the callers in every package, doc and test (item random-events of plan keyword-only-migration).
- Plan, status and helpers: `~/.claude/plans/keyword-only-random-events.md`, `.files/` next to it.

## Done
- Bootstrap `8721e88a37` pushed to `work`; item recorded in progress.
- Commit 1 (fields) `7d12c7ee93`, local, not pushed: 82 files; random_events tests pass.

## Next
- Commit 2 (methods): check why 57 method violations show, not 58; run `kwonly_methods.py --dry`, then the docs.
- Verify: checkers, mypy, tests per package, docs cells. Ask before each commit and push.
- The user opens the draft PR from `.files/pr_body.md` and adds `api-break`.
