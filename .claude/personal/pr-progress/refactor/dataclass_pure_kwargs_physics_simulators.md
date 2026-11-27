## Plan
- Make physics_simulators' 7 dataclasses `kw_only=True` and its 47 methods (plus 23 test methods) keyword-only, with the callers in semDT, cramph, giskardpy, coraplex, experiments and the semDT book (item physics-simulators of plan keyword-only-migration).
- `simulator_callback` is a decorator and gets `@called_positionally`; `start` hands `run` its constraints by `kwargs=`.
- Plan, decisions and helpers: `~/.claude/plans/keyword-only-physics-simulators.md`, `.files/` next to it.

## Done
- Bootstrap `bd70eed92e` on main `bae41bf22c` pushed to `work`; item recorded in progress (no PR number yet).

## Next
- Commit 1 (fields): codemod, checker, tests; ask before committing.
- Commit 2 (methods), then verify: checkers, mypy, tests per package, the book page's cells.
- The user opens the draft PR from `.files/pr_body.md` and adds `api-break`.
