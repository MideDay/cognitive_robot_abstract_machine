## Plan
- Make physics_simulators' 7 dataclasses `kw_only=True` and its 47 methods (plus 23 test methods) keyword-only, with the callers in semDT, cramph, giskardpy, coraplex and experiments (item physics-simulators of plan keyword-only-migration).
- Plan, decisions and helpers: `~/.claude/plans/keyword-only-physics-simulators.md`, `.files/` next to it.

## Done
- Local, not pushed: fields `2034c6480f`, methods `61774ad5fc` (`simulator_callback` marked `@called_positionally`; physics_simulators now depends on `krrood>=26.10.0`, user approved), check fixes `41057ec89c` (CI's `krrood check --since HEAD^1`).
- Checker: 0 positional left; mypy same as main. Tests read 2026-11-27 with CI=true on the tree of `41057ec89c` (it changed mid-run only by docstrings and type hints): physics_simulators 25, cramph pacer 12, semDT adapters 54, multi_sim 41, robots 53, giskardpy 1+2, coraplex 1, experiments live session 15 - all pass.

## Next
- Ask before pushing to `work`; the user opens the draft PR from `.files/pr_body.md` and adds `api-break`.
- Ask whether the never-constructed `MujocoRenderer` can be removed (separately).
