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
