## Plan
- Make random_events' 16 dataclasses `kw_only=True` and its 58 methods keyword-only, with the callers in every package, doc and test (item random-events of plan keyword-only-migration).
- Plan, status and helpers: `~/.claude/plans/keyword-only-random-events.md`, `.files/` next to it.

## Done
- Pushed to `work` 2026-11-27, head `72e3a2174b`: bootstrap `8721e88a37`, fields `7d12c7ee93`, merge of main `913a92c231` as `8759e30cf1`, methods `72e3a2174b` (58 methods, 988 codemod + 25 hand call edits, 53 doc edits, and 6 calls commit 1 missed).
- Read 2026-11-27 on the tree of `72e3a2174b`: every suite green except 6 gazebo warehouse errors (ROS package missing locally); CI=true MuJoCo, live session and coraplex green; 12 of 14 docs run, the 2 krrood ones fail on main too.

## Next
- The user opens the draft PR from `.files/pr_body.md` and adds `api-break`.
