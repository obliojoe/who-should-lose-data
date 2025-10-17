# NFL Power Rankings System - R1+SOV Algorithm

## Overview

This project implements an algorithmic NFL power ranking system using the **R1+SOV (Record First + Strength of Victory)** algorithm. This algorithm was selected after comprehensive testing of 27 different approaches, achieving a **0.943 correlation** with external consensus rankings (NFL.com, ESPN, NBC, FOX, Yahoo, USA Today).

**Current Status:** Weeks 1-6 historical rankings complete, using accurate point-in-time data

## The Algorithm: R1+SOV

### Component Weights

| Component | Weight | Description |
|-----------|--------|-------------|
| Win Percentage | 60% | Primary factor - team's current win% |
| Playoff Probability | 15% | Monte Carlo simulation of playoff chances |
| Current Playoff Seed | 10% | Team's current seeding position (1-32) |
| Strength of Victory (SOV) | 15% | Average win% of teams defeated |

### Strength of Victory (SOV)

SOV measures the quality of a team's wins by calculating the average win percentage of all teams they've beaten. This provides context to a team's record:

- Team with 5 wins over .800 teams: SOV = 0.800 (strong schedule)
- Team with 5 wins over .200 teams: SOV = 0.200 (weak schedule)

### Why R1+SOV?

After testing 27 different algorithms, R1+SOV emerged as the best match to expert consensus:

- **Highest correlation:** 0.943 vs external rankings
- **Record-first philosophy:** Prioritizes actual wins (60%)
- **Quality of wins matters:** SOV (15%) differentiates between similar records
- **Playoff context:** Incorporates playoff probability (15%) and seeding (10%)
- **Balance:** Doesn't over-weight any single metric

## Research Process

### Testing Methodology

We compared 27 algorithmic approaches against 6 major external sources:

**External Sources:**
- NFL.com Power Rankings
- ESPN Power Rankings
- NBC (Pro Football Talk)
- FOX Sports
- Yahoo Sports
- USA Today

**Algorithm Categories Tested:**
1. **Composite approaches:** Weighted blends with/without Sagarin
2. **Core algorithms:** Elo, Multi-factor, Seed-priority, Quality, Momentum, Clutch
3. **Specialized:** QB Excellence, Defensive Dominance, Red Zone, 3rd Down, Division
4. **Record-first variants:** Testing different 15% components (Momentum, Point Diff, Turnover, SOS, Clutch, Division, SOV)

### Top 5 Correlation Results

| Rank | Algorithm | Correlation | Notes |
|------|-----------|-------------|-------|
| 1 | **R1+SOV** | **0.943** | **Selected for implementation** |
| 2 | Record1st | 0.940 | Pure record focus |
| 3 | Composite | 0.933 | Complex weighted blend |
| 4 | R1+SOV+Mom | 0.932 | SOV + momentum hybrid |
| 5 | R1+Mom | 0.926 | Record + recent form |

R1+SOV was chosen because it:
- Had the highest correlation
- Uses intuitive, explainable metrics
- Balances wins with quality of wins
- Performed consistently across different weeks

### Key Finding: Tampa Bay Test Case

During Week 7 testing, Tampa Bay was **unanimously ranked #1** by all 6 external sources. Our algorithm comparison showed:

- R1+SOV: Ranked TB #1 ✓
- Record1st: Ranked TB #1 ✓
- Composite: Ranked TB #3
- Point-diff heavy algorithms: Ranked TB #7-14

TB's +14 point differential was relatively low, causing algorithms that heavily weight point differential to undervalue them despite their 5-1 record and #1 NFC seed position.

## Technical Implementation

### Core Files

**power_rankings.py** (233 lines)
- `PowerRankings` class implementing R1+SOV algorithm
- Loads team stats, schedule, playoff probabilities
- Calculates SOV for each team
- Generates ranked list with composite scores

**power_rankings_tracker.py** (271 lines)
- `PowerRankingsTracker` class for historical tracking
- Week-by-week ranking storage
- Movement tracking (week-to-week changes)
- Supports backfilling historical weeks
- Point-in-time accuracy (only uses data available at that week)

**Data Storage:**
- `data/power_rankings_history.json` (92KB)
  - Complete rankings for weeks 1-6
  - Includes: rank, record, scores, movement, breakdown
  - Last updated: 2024-10-14

### Point-in-Time Accuracy

Historical rankings use accurate "slice of time" data:

✓ **Records:** Recalculated using only games through target week
✓ **Strength of Victory:** Calculated using opponent records as of that week
✓ **Playoff Seeds:** Recalculated based on standings at that point
✓ **Playoff Probabilities:** Set to 0 for historical weeks (no historical simulation data)

**Verification Example (Week 3):**
- Tampa Bay: 3-0 (beat ATL, HOU, NYJ)
- Opponents' records: ATL (1-2), HOU (0-3), NYJ (0-3)
- Calculated SOV: 0.111
- History file SOV: 0.111 ✓

### Historical Weights

For weeks 1-6, rankings are effectively weighted as:
- Win%: 60%
- Playoff probability: 0% (no historical data)
- Current playoff seed: 10%
- Strength of victory: 15%
- *Remaining 15% unused in historical mode*

## Usage

### Calculate Current Week Rankings

```python
from power_rankings import PowerRankings

pr = PowerRankings()
rankings = pr.r1_sov_rankings()

# Display top 10
for team in rankings[:10]:
    print(f"#{team['rank']} {team['team']} ({team['record']}) - Score: {team['composite_score']:.1f}")
```

### Track Rankings Over Time

```python
from power_rankings_tracker import PowerRankingsTracker

tracker = PowerRankingsTracker()

# Update current week
data = tracker.update_current_week(week_num=7, force=True)

# Backfill historical weeks
tracker.backfill_historical_weeks(start_week=1, end_week=6)

# Get team history
history = tracker.get_team_history('TB')
for week in history:
    print(f"Week {week['week']}: #{week['rank']} ({week['record']}) - Move: {week['movement']:+d}")
```

### Command Line Usage

```bash
# Calculate current week rankings
python3 power_rankings.py

# Update current week tracking
python3 power_rankings_tracker.py

# Backfill historical weeks
python3 power_rankings_tracker.py backfill

# View team history
python3 power_rankings_tracker.py team TB
```

## Data Requirements

### Input Files (data/)

- **team_stats.json** - Current records, point differentials
- **schedule.json** - All games with scores (completed games only)
- **analysis_cache.json** - Playoff probabilities from Monte Carlo simulation

### Output Files (data/)

- **power_rankings_history.json** - Week-by-week rankings with movement tracking

## Integration with Dashboard

The power ranking system integrates with the main dashboard generation:

1. **generate_dashboard.py** - Uses `PowerRankings` to calculate current rankings
2. **Dashboard display** - Shows top teams with movement indicators
3. **Team analyses** - Each team's analysis includes their current rank
4. **Historical context** - Tracks how teams move week to week

Rankings are displayed in:
- Main dashboard header
- Team-by-team analysis sections
- Power rankings comparison tables

## Future Enhancements

### Potential Improvements

1. **Historical playoff probabilities** - Backfill Monte Carlo simulations for past weeks
2. **Confidence intervals** - Add uncertainty metrics to rankings
3. **Trending analysis** - Identify teams moving up/down over multi-week periods
4. **Predictive validation** - Test how well rankings predict future game outcomes
5. **Real-time updates** - Auto-update rankings as games complete

### Alternative Algorithms

The research phase identified several strong alternatives:

- **R1+SOV+Mom (r=0.932)** - Adds recent form momentum
- **Record1st (r=0.940)** - Simpler, pure record focus
- **Composite (r=0.933)** - More complex multi-factor blend

These are preserved in `compare_power_rankings.py` for future experimentation.

## Development History

**Branch:** `feature/r1-sov-power-rankings`

**Recent Commits:**
- `71ba1a3` - Add power rankings to team analyses (team headers)
- `8adaa14` - Replace Sagarin with R1+SOV power rankings in dashboard
- `2472d1e` - Add R1+SOV power rankings algorithm
- `3f7b7d2` - Remove critical warnings
- `f7156f4` - Generate cache update (local)

**Related Branch:** `feature/power-rankings`
- Contains original research: `data/comprehensive_rankings_analysis.md`
- Algorithm comparison: `data/ultimate_rankings_comparison.md`
- Testing scripts: `compare_power_rankings.py`

## Performance

**Algorithm Execution:** ~50ms to calculate all 32 team rankings

**Historical Backfill:** ~1 second per week (weeks 1-6 in ~6 seconds)

**Data Storage:** 92KB for 6 weeks of complete rankings data

## Conclusion

The R1+SOV power ranking system provides:
- **Objective, data-driven rankings** based on actual performance
- **Strong correlation** (0.943) with expert consensus
- **Explainable metrics** that make intuitive sense
- **Historical accuracy** with proper point-in-time data
- **Quality differentiation** through Strength of Victory
- **Easy integration** with existing dashboard system

The system balances simplicity with sophistication, avoiding over-fitting while capturing the nuances that separate similarly-performing teams.
