# refactor/dataclass_pure_kwargs_semantic_digital_twin

## State (2026-11-26)
- Two local commits on cram2 main `81bf3ace24`: fields `1afa0bb6a5`, methods `ace404badc`. Not pushed, no pull request. The details of each replay are in `~/.claude/plans/i-am-currently-working-peppy-bumblebee.md`.
- Stack pass: success, nothing to restack. Upstream reviews: no upstream pull request exists for the branch.

## Resolve (2026-11-26)
- Found: with `CI=true`, 5 tests in `test/experiments_test/live_session_test/test_session.py` fail with `TypeError: MujocoSim.step_simulation() takes 1 positional argument but 2 were given`. The cell script `cell_scripts/swing_the_hinge.py` is run with `exec`, so neither `krrood check`, mypy nor the import sweep sees it. No earlier local run set `CI=true`, so the 17 CI-only MuJoCo test files had never run against this branch.
- Fix (working tree, for the methods commit): `step_simulation(duration=…)` twice in `swing_the_hinge.py`. Also `*` on the two stand-ins for `MultiSim.step_simulation`: `SimulationThatCountsItsSteps` in `test/coraplex_test/conftest.py`, and `SimulationOfSettlingJoints` in `test/cramph_test/test_executor/test_pacer.py`, where `*` replaces `@called_positionally`. The standing rule says a stand-in follows its original once that package converts.
- Verification: the CI-gated batches `runs/ci_gated_{sdt,rest,coraplex}` (script `run_ci_gated.sh`), then a rerun of the `runs/rebased5` batches that died during `sdt_robots`.
