# Who Should Lose Data Pipeline

NFL playoff scenario analysis tool that generates simulation data, standings, and AI-powered game/team analysis.

## Overview

This system generates playoff probability data through a 4-phase pipeline:

1. **Data Prerequisites** - Fetches/updates core data files (schedule, stats, standings)
2. **Game AI Analysis** - Generates post-game analyses and pre-game previews
3. **Simulations** - Runs Monte Carlo simulations for playoff odds
4. **Team AI Analysis** - Generates AI-powered team summaries and insights

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your API keys
```

## Quick Start

```bash
# Full run with 10,000 simulations
python generate_cache.py --simulations 10000

# Quick test with 100 simulations
python generate_cache.py --simulations 100 --test-mode

# Update data files only (no simulations or AI)
python generate_cache.py --data-only
```

## Command-Line Options

### Data Generation Options

**`--skip-data`**
Skip ALL data file generation (schedule updates, stats, standings, etc.). Use this when you only want to run simulations or AI analysis with existing data.

```bash
python generate_cache.py --skip-data --simulations 5000
```

**`--data-only`**
Generate data files only, then exit. Skips simulations and AI analysis. Useful for updating game scores and stats without running time-consuming operations.

```bash
python generate_cache.py --data-only
```

### Simulation Options

**`--simulations N`** (default: 1000)
Number of Monte Carlo simulations to run for playoff probability calculations.

```bash
python generate_cache.py --simulations 10000
```

**`--skip-sims`**
Skip running new simulations, use existing simulation data from `analysis_cache.json`. Useful when you only want to regenerate AI analysis.

```bash
python generate_cache.py --skip-sims --regenerate-team-ai "all"
```

**`--seed N`**
Random seed for reproducible simulation results. Same seed + same number of simulations = identical results.

```bash
python generate_cache.py --simulations 1000 --seed 42
```

### AI Analysis Options

**`--skip-team-ai`**
Skip team AI analysis generation. Team analysis uses simulation data and generates summaries for each team.

```bash
python generate_cache.py --skip-team-ai
```

**`--skip-game-ai`**
Skip game AI analysis generation. Game analysis includes both post-game recaps and pre-game previews.

```bash
python generate_cache.py --skip-game-ai
```

**`--regenerate-team-ai "TEAM1,TEAM2"` or `"all"`**
Regenerate team AI analysis for specific teams (by abbreviation) or all teams. Useful when you want to update team summaries without re-running simulations.

```bash
# Regenerate specific teams
python generate_cache.py --skip-data --skip-sims --regenerate-team-ai "DET,MIN,ARI"

# Regenerate all teams
python generate_cache.py --skip-data --skip-sims --regenerate-team-ai "all"
```

**`--regenerate-game-ai "ID1,ID2"` or `"analysis"` or `"preview"` or `"all"`**
Regenerate game AI analysis. Supports multiple modes:

- **ESPN Game IDs**: Regenerate specific games
- **`"analysis"`**: Regenerate all completed game analyses
- **`"preview"`**: Regenerate all upcoming game previews
- **`"all"`**: Regenerate everything

```bash
# Regenerate specific games by ESPN ID
python generate_cache.py --skip-data --regenerate-game-ai "401772856,401772855"

# Regenerate all completed game analyses
python generate_cache.py --skip-data --regenerate-game-ai "analysis"

# Regenerate all upcoming game previews
python generate_cache.py --skip-data --regenerate-game-ai "preview"

# Regenerate all game AI
python generate_cache.py --skip-data --regenerate-game-ai "all"
```

### Deployment Options

**`--commit`**
Automatically commit changes to git if generation succeeds. Includes cache files and data updates.

```bash
python generate_cache.py --simulations 10000 --commit
```

**`--deploy-netlify`**
Deploy generated data files to the Netlify repository for frontend hosting.

```bash
python generate_cache.py --deploy-netlify
```

**`--deploy-render`**
Deploy data files to Render web host via SSH.

```bash
python generate_cache.py --deploy-render
```

**`--test-mode`**
Run in test mode (disables actual AI API calls). Useful for testing the pipeline without incurring API costs.

```bash
python generate_cache.py --test-mode --simulations 100
```

### Deprecated Options

These options still work but will show warnings. Use the new alternatives instead:

- `--skip-ai` → Use `--skip-team-ai` and `--skip-game-ai`
- `--regenerate-ai` → Use `--regenerate-team-ai`
- `--no-copy-data` → No longer used

## Common Usage Patterns

### Update Everything (Weekly Run)
```bash
python generate_cache.py --simulations 10000 --commit --deploy-netlify
```

### Quick Data Refresh (After Games)
```bash
# Update scores and stats only
python generate_cache.py --data-only
```

### Regenerate Game Previews Only
```bash
# When new games are scheduled
python generate_cache.py --skip-data --regenerate-game-ai "preview"
```

### Regenerate Game Analyses After Scores Updated
```bash
# After updating scores, regenerate analyses for completed games
python generate_cache.py --data-only
python generate_cache.py --skip-data --regenerate-game-ai "analysis"
```

### Test Run Before Production
```bash
# Quick test with fewer simulations
python generate_cache.py --simulations 100 --test-mode --skip-game-ai
```

### Regenerate AI for Specific Teams
```bash
# If team summaries need updates (injury news, etc.)
python generate_cache.py --skip-data --skip-sims --regenerate-team-ai "DET,GB,MIN"
```

### Full Run with Reproducible Results
```bash
# For comparing changes across code versions
python generate_cache.py --simulations 10000 --seed 12345
```

## File Outputs

| File | Size | Description |
|------|------|-------------|
| `analysis_cache.json` | ~885KB | Simulation results + team AI analysis |
| `game_analyses.json` | ~359KB | Game recaps and previews |
| `standings_cache.json` | ~80KB | Current NFL standings and tiebreakers |
| `team_stats.csv` | ~23KB | Team statistics (offense/defense rankings) |
| `team_starters.csv` | ~192KB | Starting lineups and player status |
| `schedule.csv` | ~25KB | Game schedule with scores |
| `coordinators.csv` | ~1.4KB | Offensive/defensive coordinators |

## Pipeline Phases

### Phase 1: Data Prerequisites
Runs when `--skip-data` is NOT used, or always if neither `--skip-data` nor `--data-only` specified.

1. Update `schedule.csv` (fetch latest scores/dates from ESPN)
2. Generate `team_stats.csv` (calculate team rankings)
3. Generate `team_starters.csv` (fetch rosters from ESPN)
4. Fetch `coordinators.csv` (scrape coordinator data)
5. Generate `standings_cache.json` (calculate standings/tiebreakers)

### Phase 2: Game AI Analysis
Runs when `--skip-game-ai` and `--data-only` are NOT used.

- Generates post-game analyses for completed games
- Generates pre-game previews for upcoming games (within next 7 days)
- Saves to `game_analyses.json`
- Can be filtered with `--regenerate-game-ai`

### Phase 3: Simulations
Runs when `--skip-sims` and `--data-only` are NOT used.

- Runs Monte Carlo simulations of remaining NFL schedule
- Calculates playoff probabilities, division winners, seeding
- Determines "significant games" (games that impact playoff odds)
- Saves simulation data to `analysis_cache.json`

### Phase 4: Team AI Analysis
Runs when `--skip-team-ai` and `--data-only` are NOT used.

- Generates AI-powered team summaries
- Analyzes playoff scenarios and key games
- Uses simulation data + stats + standings
- Saves AI analysis to `analysis_cache.json`
- Can be filtered with `--regenerate-team-ai`

## Troubleshooting

### "No API key found" errors
Make sure you have `.env` file with:
```
CLAUDE_API_KEY=your-key-here
# or
OPENAI_API_KEY=your-key-here
```

### Simulations taking too long
Use fewer simulations for testing:
```bash
python generate_cache.py --simulations 100
```

### Git push fails to main
The script tries to push to `main` branch. Make sure:
1. You have push permissions
2. You're on the correct branch
3. Remote is properly configured

### Game analysis fails
Check if ESPN API is accessible and game IDs are correct in `schedule.csv`.

### Team AI analysis errors
Ensure simulation data exists first. You can't generate team AI without simulation data:
```bash
# This will fail:
python generate_cache.py --skip-sims --regenerate-team-ai "all"

# Do this instead:
python generate_cache.py --simulations 1000 --skip-game-ai
```

## Development

### Running Tests
```bash
python generate_cache.py --test-mode --simulations 10
```

### Environment Variables

**`AI_ANALYSIS_WORKERS`** (default: 3)
Number of parallel workers for team AI analysis generation. Higher values speed up processing but increase API load.

```bash
# Run with 5 parallel workers
AI_ANALYSIS_WORKERS=5 python generate_cache.py --simulations 1000
```

**`GAME_ANALYSIS_WORKERS`** (default: 3)
Number of parallel workers for game AI analysis (recaps and previews). Higher values speed up processing but increase API/ESPN load.

```bash
# Run with 5 parallel workers
GAME_ANALYSIS_WORKERS=5 python generate_cache.py --regenerate-game-ai "all"
```

### Adding New Options
1. Add argument to parser in `generate_cache.py` main()
2. Update this README
3. Add tests in test suite

### Code Structure
- `generate_cache.py` - Main pipeline orchestration
- `generate_analyses.py` - Game AI analysis logic (parallelized with ThreadPoolExecutor)
- `simulate_season.py` - Monte Carlo simulation engine
- `ai_service.py` - AI model API wrapper
- `calculate_standings_cache.py` - NFL standings calculations

## License

[Your License Here]
