Manage worktrees: list them, copy changed gitignored files from one into the main worktree, then delete it.

Input: optional worktree path as $ARGUMENTS. If omitted, just list all worktrees.

## Step 1 — List all worktrees

Run:
```
git worktree list
```

If no $ARGUMENTS were provided, stop here and show the list to the user.

## Step 2 — Identify the main worktree

The main worktree is the first entry returned by:
```
git worktree list --porcelain
```

## Step 3 — Find gitignored files in the target worktree

Run from the target worktree path:
```
git -C <target-path> ls-files --others --ignored --exclude-standard
```

Skip these noise patterns (never copy them):
- `recordings/` — runtime audio captures
- `logs/` — runtime log files
- `models/` — large binary ONNX models
- `.gitnexus/` — local code-intelligence index
- `agora-pipeline`, `gemini-llm-server`, `basic-pipeline-test` — compiled binaries
- `scripts/.venv/` — python virtual env
- `.DS_Store`

## Step 4 — Diff each remaining file against the main worktree

For each file not matching the skip patterns:
- If the file does not exist in the main worktree: flag it as **new**.
- If it exists but differs: show a unified diff (`diff <target-file> <main-file>`), flag as **changed**.
- If identical: note it as identical, no action needed.

## Step 5 — Decide what to copy

For each **new** or **changed** file, show the user the diff and ask whether to copy it from the target worktree into the main worktree. Copy only files the user confirms.

If a file requires creating a parent directory in the main worktree, create it first.

## Step 6 — Delete the worktree

After all copy decisions are made, confirm with the user before deleting. Then run:
```
git worktree remove <target-path> --force
```

Confirm deletion succeeded by running `git worktree list` again.
