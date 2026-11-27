# refactor/dataclass_pure_kwargs_semantic_digital_twin

## Plan
- #1084 is ready for review; the full CI on cf65425f18 is watched and each failing job reported with its log (no local test runs for this PR, the user's call).
- When main moves: replay with the catch-up codemods and the static checks, then ask before force-pushing.

## Done
- Replayed onto main `5c65530b6d` (#1081 fruit_material_qualities): fields `866c713941` (1 conflict: `HasRootBody` combines main's `HasMaterialElasticity` base with `kw_only=True`), methods `cf65425f18` (applied cleanly; 2 catch-up edits `create_with_root_body(root_body_name=…)` in main's new tests).
- Checks: 0 positional in semDT; mypy `call-arg` diff empty over main's changed files; files main left alone equal the previous push.
- Force-pushed with a lease (approved) on 2026-11-27; merge probe at `cf65425f18` against `5c65530b6d`: clean. Labels `api-break` and `maintenance` set.

## Next
- Read the CI result on `cf65425f18`.
