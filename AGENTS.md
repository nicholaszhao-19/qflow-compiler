# Repository hygiene

- Add files only when they implement the requested change, test it, or preserve evidence needed to reproduce a published result. Update the existing source, script, or document when it already serves that purpose.
- Keep drafts, scratch notes, caches, temporary logs, and local benchmark reruns outside the tracked tree. Delete disposable files created for the task before committing. Keep private reports and credentials out of Git.
- Before every commit, review `git status --short`, `git diff --cached --stat`, and `git diff --cached`. Unstage or remove duplicates, unrelated edits, and regenerated outputs that add no evidence. Preserve required inputs, benchmark records, tests, and license notices.
- If tracked files change, regenerate `release_manifest.json` with `python scripts/package_release.py` and review the staged changes again before committing or pushing.
