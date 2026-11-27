# refactor/dataclass_pure_kwargs_semantic_digital_twin

## Plan
- #1084 is ready for review; the full CI on cf65425f18 is watched and each failing job reported with its log (no local test runs for this PR, the user's call).
- When main moves: replay with the catch-up codemods and the static checks, then ask before force-pushing.

## Done
- Replayed onto main `5c65530b6d` (#1081 fruit_material_qualities): fields `866c713941` (1 conflict: `HasRootBody` combines main's `HasMaterialElasticity` base with `kw_only=True`), methods `cf65425f18` (applied cleanly; 2 catch-up edits `create_with_root_body(root_body_name=…)` in main's new tests).
- Checks: 0 positional in semDT; mypy `call-arg` diff empty over main's changed files; files main left alone equal the previous push.
- Force-pushed with a lease (approved) on 2026-11-27; merge probe at `cf65425f18` against `5c65530b6d`: clean. Labels `api-break` and `maintenance` set.

- CI read at `cf65425f18`: semDT, giskardpy, coraplex, robokudo, cramph, segmind, probabilistic_model, examples/demos, find-api-breaks and 6 docs builds passed. experiments and krrood failed in `uv sync`: lxml 6.2.0 had only its sdist on PyPI at 15:02:46 UTC, wheels came 15:04 (no test ran). Rerunning them needs cram2 admin rights (403); the user's account has triage only.
- small_packages fails its `krrood check --since HEAD^1` step (769 pre-existing violations on touched lines, 0 positional), so its tests do not run; the user chose to leave it as is.

## Next
- experiments and krrood need a rerun by a cram2 admin, or the next push.
- The semDT docs build was still running when last read.
