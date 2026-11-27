## Plan
- Make physics_simulators' 7 dataclasses `kw_only=True` and its 47 methods (plus 23 test methods) keyword-only, with the callers in semDT, cramph, giskardpy, coraplex and experiments (item physics-simulators of plan keyword-only-migration).
- Plan, decisions and helpers: `~/.claude/plans/keyword-only-physics-simulators.md`, `.files/` next to it.

## Done
- Local, not pushed: fields `2034c6480f`, methods `61774ad5fc` (`simulator_callback` marked `@called_positionally`; physics_simulators now depends on `krrood>=26.10.0`, user approved), check fixes `41057ec89c` (CI's `krrood check --since HEAD^1`).
- Checker 0 positional left; mypy same as main. Tests with CI=true pass for physics_simulators, cramph pacer, semDT MuJoCo adapters, multi_sim and robots, giskardpy MuJoCo and coraplex mobile robot.

## Next
- Experiments live-session tests are still running; then ask before pushing to `work`.
- The user opens the draft PR from `.files/pr_body.md` and adds `api-break`.
