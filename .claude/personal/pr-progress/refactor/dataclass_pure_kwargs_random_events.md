## Plan
- Make random_events' 16 dataclasses `kw_only=True` and its 58 methods keyword-only, with the callers in every package, doc and test (item random-events of plan keyword-only-migration).
- Plan, status and helpers: `~/.claude/plans/keyword-only-random-events.md`, `.files/` next to it.

## Done
- Bootstrap `8721e88a37` pushed to `work`. Commit 1 (fields) `7d12c7ee93` and the merge of main `913a92c231` (`8759e30cf1`), local.
- Methods change in the working tree: 58 methods, 988 codemod + 25 hand call edits, 53 doc edits, plus 6 constructions commit 1 missed (runtime classes) - the user wants those in commit 2.
- Read 2026-11-27 on the working tree: every suite green except 6 gazebo warehouse errors (ROS package missing locally); CI=true MuJoCo, live session and coraplex green; 12 of 14 docs run, the 2 krrood ones fail on main too.

## Next
- Ask before commit 2 and before pushing with `maintenance push`.
- The user opens the draft PR from `.files/pr_body.md` and adds `api-break`.
