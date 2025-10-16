# Agent Guidelines

## 🚨 CRITICAL: NEVER COMMIT DIRECTLY TO MAIN 🚨

**ALWAYS create a feature branch first. No exceptions. No shortcuts.**

### Before EVERY commit, follow this workflow:

1. **Check current branch**: Run `git branch --show-current`
2. **If on main**: STOP! Create a feature branch:
   ```bash
   git checkout -b feature/descriptive-name
   ```
3. **Make your changes and commit**
4. **Push and create a pull request**
5. **Wait for approval before merging**

If you notice the current branch has already been merged, stop and switch to an up-to-date `main` (and warn the user to do the same) before starting new work.

---

## General Guidelines

- Keep `README.md` synchronized with the command-line options defined in `generate_cache.py` and prompts in `generate_cache_cli.py` whenever those flags change.
- Update the "File Outputs" section of `README.md` whenever new artifacts are produced or existing ones are renamed.
- Prefer small, focused commits with clear messages describing the intent of the change.
- Remember the site is called "Who Should Lose." Rooting guidance must use that voice, e.g., "Root against the Lions" instead of "Root for their opponent."
- Know the project structure:
  - Core pipeline scripts live in the root directory
  - `scripts/` – Utility and helper scripts (one-off tools, backfill scripts, etc.)
  - `docs/` – Technical documentation and guides (not user-facing like README.md)
  - `data/` – All data artifacts and cached outputs
  - Root-level `.md` files are limited to `README.md`, `AGENTS.md`, `CONTRIBUTING.md`, etc.
- Know the purpose of the key data artifacts:
  - `data/analysis_cache.json` – Canonical simulation outputs: playoff odds, chaos metrics, significant games, and other sim-derived stats (AI text lives elsewhere).
  - `data/team_analyses.json` – Per-team AI write-ups generated for the rooting guide.
  - `data/game_analyses.json` – Game-level previews and recaps.
  - `data/dashboard_content.json` – Headlines and copy blocks assembled for the public dashboard.
  - `data/standings_cache.json` – Expanded standings with tiebreakers.
  - `data/power_rankings_history.json` – Historical power rankings by week with movement tracking.

## Working with GitHub PR Feedback

GitHub distinguishes between **issue comments** (top-level on the PR) and **review comments** (inline on a diff). `gh pr view --comments` only shows the former. To make sure you see *all* feedback:

1. Identify the repo/name combo once per shell session:
   ```bash
   REPO=$(gh repo view --json nameWithOwner --jq .nameWithOwner)
   ```
2. List issue comments (general discussion):
   ```bash
   gh pr view <pr-number> --comments
   ```
3. List review comments (inline notes Codex or humans leave on specific hunks):
   ```bash
   gh api repos/$REPO/pulls/<pr-number>/comments --jq '.[].body'
   ```
   Add `--paginate` if the list is long. For extra context (file/line), drop the `--jq` filter to see the full JSON payload.

Always scan both outputs before declaring a PR “clean.” If Codex or a reviewer flags an issue, respond in-place after pushing the fix so the conversation stays in context.

## Creating Pull Requests with Descriptions

`gh pr create --body-file -` often fails in this environment because the GitHub CLI tries to launch an editor (“cannot start document portal”) and ends up publishing a PR with an empty description. To guarantee a review-ready summary:

1. Use a heredoc with `--body` so the description is baked in from the start:
   ```bash
   gh pr create --title "Your PR Title" --body "$(cat <<'EOF'
   ## Summary

   Your multi-line overview…

   ## Changes
   - Bullet 1
   - Bullet 2

   ## Testing
   - [x] Command
   EOF
   )"
   ```
   The `cat <<'EOF'` form preserves Markdown (and `$`/`` ` `` characters) without unintended expansion.
2. Immediately verify the description landed:
   ```bash
   gh pr view <pr-number> --json body --jq .body
   ```

Never leave a PR without a description—the review bots (and humans!) rely on it.
