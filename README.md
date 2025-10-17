# Who Should Lose Data Pipeline

Automation pipeline for the Who Should Lose project. The tooling ingests live NFL data, runs Monte Carlo simulations to quantify playoff scenarios, and generates AI-written summaries for teams, games, and the public dashboard.

## Overview

The pipeline is orchestrated by [`generate_cache.py`](./generate_cache.py) and is designed to be run end-to-end or in targeted slices depending on what needs to be updated.

High-level phases:

1. **Data prerequisites** – Refresh schedule, scores, standings, rosters, coordinator information, Sagarin ratings, and derived statistics in the `data/` directory.
2. **Season simulations** – Run Monte Carlo simulations of remaining games to produce playoff odds and seeding projections.
3. **AI content** – Generate team summaries, game recaps/previews, and dashboard copy using Anthropic or OpenAI models (or skip pieces of this phase via flags).
4. **Deployment helpers** – Optionally copy artifacts, commit them, or push updates to downstream dashboards.

Key supporting modules include:

- [`update_scores.py`](./update_scores.py) – Pulls score and scheduling changes from ESPN.
- [`team_stats.py`](./team_stats.py) & [`team_starters.py`](./team_starters.py) – Build CSVs used as simulation inputs.
- [`simulate_season.py`](./simulate_season.py) & [`playoff_analysis.py`](./playoff_analysis.py) – Core Monte Carlo logic and "most impactful games" calculations.
- [`generate_analyses.py`](./generate_analyses.py) & [`generate_dashboard.py`](./generate_dashboard.py) – Create AI-authored content consumed by the dashboard.
- [`generate_cache_cli.py`](./generate_cache_cli.py) – Interactive wizard for assembling repeatable command invocations.

## Getting Started

### Requirements

- Python 3.10 or newer.
- `pip install -r requirements.txt`
- Optional: `pip install questionary` for the interactive CLI wizard.

### Environment variables

Create a `.env` file (or export variables in your shell) with at least one AI provider key:

```env
# Anthropic Claude (preferred by default)
CLAUDE_API_KEY=your-key

# Optional: OpenAI key if you want to use GPT models
OPENAI_API_KEY=your-key
```

Optional environment variables:

- `AI_ANALYSIS_WORKERS` – Parallelism for team AI generation (default `3`).
- `GAME_ANALYSIS_WORKERS` – Parallelism for game analysis generation (default `3`).

### Prepare runtime directories

Create a logs directory before running the pipeline (the AI service will also write under `logs/`):

```bash
mkdir -p logs
```

The repository already includes a populated `data/` directory with cached inputs. The scripts will refresh the contents in-place.

## Project Structure

```
who-should-lose-data/
├── README.md              # This file - project overview and usage
├── AGENTS.md             # Guidelines for AI agents working on this project
├── generate_cache.py     # Main pipeline orchestrator
├── generate_cache_cli.py # Interactive command builder
├── simulate_season.py    # Monte Carlo simulation engine
├── power_rankings.py     # R1+SOV power ranking algorithm
├── scripts/              # Utility and helper scripts
│   ├── compare_power_rankings.py
│   └── backfill_playoff_probs.py
├── docs/                 # Technical documentation
│   └── POWER_RANKINGS.md # Power rankings algorithm documentation
├── data/                 # All data artifacts (schedule, stats, analyses)
└── logs/                 # Runtime logs (create before first run)
```

For detailed information about the power rankings algorithm, see [`docs/POWER_RANKINGS.md`](./docs/POWER_RANKINGS.md).

## Quick Start

```bash
# Full run with 10,000 simulations
python generate_cache.py --simulations 10000

# Quick smoke test with limited simulations and no live AI calls
python generate_cache.py --simulations 100 --test-mode

# Update only the prerequisite data files
python generate_cache.py --data-only
```

## Raw Data Snapshots

Raw API responses and league datasets can be captured ahead of a run for reproducibility. The collector stores everything under `data/raw/` and writes a manifest that the main pipeline can reuse.

```bash
# Gather the latest scoreboard, ESPN game data, and nflreadpy tables
python collect_raw_data.py --season 2025 --week 5

# Use the captured manifest (auto-detected as data/raw/manifest/latest.json)
python generate_cache.py --raw-manifest data/raw/manifest/latest.json --simulations 1000
```

Artifacts saved in `data/raw/` include:

- ESPN scoreboard, per-game summaries, box scores, team leaders, injuries, depth charts, and news
- nflreadpy tables (schedules, team stats, **play-by-play**, player stats, snap counts, depth charts, rosters, etc.) filtered to the requested week
- Source HTML for the Sagarin ratings page (reused by the cache builder when present)
- A manifest linking each dataset to its on-disk path

> **Tip:** Run `collect_raw_data.py` for each week you need in the aggregate metrics (e.g., `--week 1` up through the current week). The pipeline now reads the season-to-date play-by-play and player stats from `data/raw/`, so backfilling prior weeks ensures conversion and red-zone rates stay accurate. `generate_cache.py` will log a warning if the manifest is missing any required raw datasets. ESPN game and team endpoints are harvested in parallel, so a full weekly snapshot completes significantly faster than the earlier serial downloader.

`generate_cache.py` automatically loads `data/raw/manifest/latest.json` when available, falling back to live API calls only if the manifest is missing.

## Command-Line Options

All flags are optional and can be combined to tailor the workflow. Run `python generate_cache.py --help` for the authoritative list.

### Data generation

- `--skip-data` – Skip ALL prerequisite data updates (schedule, standings, stats, starters, team metadata, etc.).
- `--data-only` – Stop after data prerequisites (no simulations or AI stages).
- `--deploy-only` – Skip all generation and only run deployment/commit logic (implies skipping data, simulations, and every AI phase).
- `--force-sagarin` – Force a fresh Sagarin scrape instead of using cached rankings.
- `--raw-manifest PATH` – Load pre-collected raw data from the given manifest (defaults to `data/raw/manifest/latest.json` when present).

### Simulation controls

- `--simulations N` – Number of Monte Carlo simulations to run (default `1000`).
- `--skip-sims` – Reuse existing simulation outputs instead of running new simulations.
- `--seed N` – Random seed for reproducible simulations.

### AI generation

- `--skip-team-ai` – Skip team-level AI summaries.
- `--skip-game-ai` – Skip game recaps and previews.
- `--skip-dashboard-ai` – Skip dashboard headline/section generation.
- `--regenerate-team-ai VALUE` – Regenerate team AI for comma-separated abbreviations (e.g., `"DET,MIN"`) or `"all"`.
- `--regenerate-game-ai VALUE` – Regenerate specific ESPN IDs, upcoming `preview`, completed `analysis`, combined `all`, or games matching `team:DET,MIN`.
- `--ai-model MODEL` – Override the AI model alias (see [`ai_service.py`](./ai_service.py) for supported values).
- `--test-mode` – Disable outbound AI calls; useful for local testing.

### Deployment and workflow helpers

- `--commit` – Commit modified artifacts to git when the run succeeds.
- `--deploy-netlify` – Copy artifacts to the Netlify repository.
- `--copy-to PATH` – Copy the refreshed `data/` directory to another location when finished.

## Interactive command builder

For repeatable workflows, run the wizard:

```bash
python generate_cache_cli.py
```

The script walks through the available operations, saves presets in `cache_cli_presets.json`, and prints the assembled `generate_cache.py` command without executing it automatically.

## Outputs

Artifacts are written to the `data/` directory. Key files include:

| File | Description |
|------|-------------|
| `analysis_cache.json` | Simulation results, team AI, playoff odds, Super Bowl projections, power rankings. |
| `game_analyses.json` | Game recaps and upcoming previews. |
| `dashboard_content.json` | AI narrative for the public dashboard (headlines, stat blurbs, etc.). |
| `pre_game_impacts.json` | Cached impact calculations for unplayed games. |
| `standings_cache.json` | Standings with tiebreakers, division/conference records, and win percentages. |
| `power_rankings_history.json` | Historical power rankings by week with movement tracking. |
| `team_stats.json` | Aggregated team-level statistics used as simulation inputs. |
| `team_starters.json` | Starter depth charts and season-to-date production. |
| `teams.json` | Team metadata (colors, head coach, coordinators, stadium, ESPN IDs). |
| `schedule.json` | Season schedule with live score updates. |
| `sagarin.json` | Latest Sagarin ratings, prior-week comparison, and scrape history (used for simulations). |

## Pipeline details

### Phase 1 – Data prerequisites

Triggered unless `--skip-data` is set (or `--deploy-only` is used). Updates schedule/scores, team stats, starters, consolidated team metadata, standings, and Sagarin ratings. Outputs land in `data/`.

### Phase 2 – Simulations

Runs unless `--skip-sims` or `--data-only` is set. Uses [`simulate_season.py`](./simulate_season.py) for Monte Carlo results and [`playoff_analysis.py`](./playoff_analysis.py) to identify impactful games.

### Phase 3 – AI generation

Conditionally generates:

- **Game AI** (`generate_analyses.py`) – Post-game recaps and upcoming previews.
- **Team AI** (`prompt_builder.py` + `ai_service.py`) – Summary blurbs, playoff chances, and key matchups.
- **Dashboard AI** (`generate_dashboard.py`) – Narrative copy assembled from deterministic stat selection.

### Phase 4 – Deployment hooks

If requested, copy artifacts, push to Netlify, and/or commit changes. Use `--deploy-only` to re-run this phase without regenerating data.

## Troubleshooting

- **"No API key found"** – Populate `CLAUDE_API_KEY` or `OPENAI_API_KEY` in `.env`.
- **Simulations feel slow** – Reduce `--simulations` during local runs or limit to specific AI regeneration flags.
- **Game analysis errors** – Confirm ESPN IDs exist in `data/schedule.json` and that the ESPN API is reachable.
- **Team AI requires simulations** – Regeneration depends on fresh simulations. Run with `--simulations ...` before `--regenerate-team-ai`.
- **Logging** – Check `logs/` for detailed traces (e.g., `logs/generate_cache.log`, `logs/ai_service.log`). Create the directory ahead of time if it does not exist.

## Development tips

- Use `python generate_cache.py --test-mode --simulations 10` for a fast end-to-end smoke test.
- When adding new flags or outputs, update both this README and the interactive CLI wizard.
- Cached prompt templates live under `data/prompts/`; adjust them alongside code changes for consistent AI tone.
