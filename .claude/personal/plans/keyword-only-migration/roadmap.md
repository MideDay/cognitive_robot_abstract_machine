# Keyword-only migration: standing rules

- One PR per package: commit 1 makes the dataclass fields keyword-only, commit 2 the method parameters,
  and an optional commit 3 fixes the rule violations on the lines they touch.
- The change is mechanical: add kw_only=True, add *, and pass keywords at every call site, in every package, doc and test.
- Keep every parameter name; a rename needs the user's approval first.
- A foreign caller that passes arguments positionally gets @called_positionally instead of *.
- A stand-in for a method of a package not yet converted stays positional until that package converts.
- Run the positional checkers with PYTHONPATH=<repo>:$PYTHONPATH and check each demo in a process of its own;
  migrate modules that cannot be imported by hand, since krrood check skips them.
- When two migration branches touch the same callers, whichever merges second reruns the catch-up codemods.
- The codemods live in ~/.claude/plans/i-am-working-on-compiled-possum.files/; copy and retarget them per package.
- Ask before every commit and push; push only to work, and the user opens each PR as a draft with api-break.
