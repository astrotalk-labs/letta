Plan and start a new task with proper workspace setup.

## Steps

1. **Understand the task**: If $ARGUMENTS is empty, ask what to work on. Otherwise use $ARGUMENTS.

2. **Assess scope**: Ask if this is BIG (architecture change, new pipeline mode, new provider) or SMALL (bug fix, config change, prompt tweak).

3. **Gather context**: Read relevant files, search for existing patterns. Use GitNexus if indexed. Identify files that will change.

4. **Branch and worktree**: Create a feature branch from the current branch. If using a worktree, create it under `/Users/astrotalk/work/` (not `.claude/`):
   - `feat/` for new features
   - `fix/` for bug fixes
   - `chore/` for cleanup/config

   Name the worktree directory as `{repo_name}_{base_branch}_{feature_name}` where:
   - `repo_name` = the git repo name (e.g. `voice-agent`)
   - `base_branch` = the branch you are branching from (e.g. `main`)
   - `feature_name` = short snake_case description of the task (e.g. `silence_nudges`)

   ```
   git worktree add -b <branch-name> /Users/astrotalk/work/<repo_name>_<base_branch>_<feature_name> <base-branch>
   ```

   Example: `git worktree add -b feat/silence-nudges /Users/astrotalk/work/voice-agent_main_silence_nudges main`

5. **Copy gitignored assets to worktree**: These files are gitignored and must be copied from the main tree:
   ```
   MAIN_TREE="$(git worktree list | head -1 | awk '{print $1}')"
   cp -r "$MAIN_TREE/.claude" .
   cp "$MAIN_TREE/.env.local" .
   mkdir -p models
   cp "$MAIN_TREE/models/silero_vad.onnx" models/
   ```

6. **Plan**: For BIG tasks, enter plan mode and design the approach. For SMALL tasks, proceed directly.

7. **Verify build**: Before starting work, ensure the project builds:
   ```
   CGO_CFLAGS="-I/opt/homebrew/include/onnxruntime" CGO_LDFLAGS="-L/opt/homebrew/lib -lonnxruntime" go build ./...
   ```

## Rules
- Always check existing code for similar patterns before writing new code
- Read CLAUDE.md for project conventions
- Use haiku for exploration subagents, sonnet for implementation
