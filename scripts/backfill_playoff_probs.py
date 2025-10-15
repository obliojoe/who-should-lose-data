#!/usr/bin/env python3
"""
Backfill Power Rankings with Historical Playoff Probabilities

This script recalculates power rankings for weeks 4-6 using the correct
playoff probabilities extracted from git history.

Weeks 1-3 remain with 0% playoff probabilities (no historical data available).
"""

import json
from pathlib import Path
from power_rankings_tracker import PowerRankingsTracker
from power_rankings import PowerRankings


def load_historical_playoff_probs(week_num: int) -> dict:
    """Load historical playoff probabilities for a specific week"""
    prob_file = Path(f'/tmp/week{week_num}_playoff_probs.json')

    if not prob_file.exists():
        print(f"Warning: No playoff probability data for week {week_num}")
        return {}

    with open(prob_file, 'r') as f:
        return json.load(f)


def backfill_week_with_playoff_probs(tracker: PowerRankingsTracker, week_num: int):
    """
    Recalculate rankings for a specific week using historical playoff probabilities

    Args:
        tracker: PowerRankingsTracker instance
        week_num: Week number to recalculate
    """
    print(f"Recalculating Week {week_num} with historical playoff probabilities...")

    # Load historical playoff probabilities
    playoff_probs = load_historical_playoff_probs(week_num)

    if not playoff_probs:
        print(f"  No historical playoff probs found, skipping week {week_num}")
        return

    # Create a PowerRankings instance
    pr = PowerRankings(data_dir='data')

    # Override playoff probabilities with historical data
    pr.playoff_probs = playoff_probs

    # Filter schedule to only include games through the target week
    pr.schedule = [
        game for game in pr.schedule
        if game['week'] <= week_num
    ]

    # Recalculate team records based on filtered schedule
    for team in pr.teams:
        pr.teams[team]['wins'] = 0
        pr.teams[team]['losses'] = 0
        pr.teams[team]['ties'] = 0

    # Count wins/losses from filtered schedule
    for game in pr.schedule:
        away = game['away_team']
        home = game['home_team']

        if game['away_score'] > game['home_score']:
            pr.teams[away]['wins'] += 1
            pr.teams[home]['losses'] += 1
        elif game['home_score'] > game['away_score']:
            pr.teams[home]['wins'] += 1
            pr.teams[away]['losses'] += 1
        else:
            pr.teams[away]['ties'] += 1
            pr.teams[home]['ties'] += 1

    # Recalculate win percentages
    for team in pr.teams:
        wins = pr.teams[team]['wins']
        losses = pr.teams[team]['losses']
        ties = pr.teams[team]['ties']
        games = wins + losses + ties

        if games > 0:
            pr.teams[team]['win_pct'] = (wins + 0.5 * ties) / games
        else:
            pr.teams[team]['win_pct'] = 0.0

        pr.teams[team]['record'] = f"{wins}-{losses}" + (f"-{ties}" if ties > 0 else "")

    # Recalculate playoff seeds based on filtered records
    pr._calculate_playoff_seeds()

    # Get rankings with historical playoff probabilities
    rankings = pr.r1_sov_rankings()

    # Get previous week rankings for movement calculation
    week_key = str(week_num)
    prev_week_key = str(week_num - 1)
    previous_rankings = {}

    if prev_week_key in tracker.history['weeks']:
        previous_rankings = {
            r['team']: r['rank']
            for r in tracker.history['weeks'][prev_week_key]['rankings']
        }

    # Add movement and previous rank to current rankings
    for team_rank in rankings:
        team = team_rank['team']
        prev_rank = previous_rankings.get(team)

        team_rank['previous_rank'] = prev_rank

        if prev_rank is not None:
            team_rank['movement'] = prev_rank - team_rank['rank']
        else:
            team_rank['movement'] = 0

    # Store in history
    from datetime import datetime
    tracker.history['weeks'][week_key] = {
        'week': week_num,
        'rankings': rankings,
        'calculated_at': datetime.now().isoformat(),
        'backfilled': True,
        'playoff_probs_source': 'git_history'
    }

    print(f"  ✓ Week {week_num} recalculated with playoff probabilities")
    print(f"    Top 3: ", end='')
    for i in range(3):
        team = rankings[i]['team']
        playoff_prob = rankings[i]['playoff_prob']
        print(f"#{i+1} {team} ({playoff_prob:.1f}%)", end='  ')
    print()


def main():
    """Main backfill process"""
    print("="*70)
    print("BACKFILL POWER RANKINGS WITH HISTORICAL PLAYOFF PROBABILITIES")
    print("="*70)
    print()

    # Load existing tracker
    tracker = PowerRankingsTracker(data_dir='data')

    print("Current rankings status:")
    for week in sorted([int(k) for k in tracker.history['weeks'].keys()]):
        week_data = tracker.history['weeks'][str(week)]
        backfilled = week_data.get('backfilled', False)
        status = "✓ backfilled" if backfilled else "○ original"
        print(f"  Week {week}: {status}")
    print()

    # Backfill weeks 4-6
    print("Backfilling weeks 4-6 with historical playoff probabilities...")
    print()

    for week in [4, 5, 6]:
        backfill_week_with_playoff_probs(tracker, week)
        print()

    # Save updated history
    tracker._save_history()

    print("="*70)
    print("BACKFILL COMPLETE")
    print("="*70)
    print()
    print("Rankings for weeks 4-6 now include historical playoff probabilities.")
    print("Weeks 1-3 remain with 0% playoff probs (no historical data).")
    print()

    # Show summary
    print("Summary of changes:")
    for week in [4, 5, 6]:
        rankings = tracker.history['weeks'][str(week)]['rankings']
        print(f"\nWeek {week} Top 5:")
        for i in range(5):
            team = rankings[i]['team']
            record = rankings[i]['record']
            playoff_prob = rankings[i]['playoff_prob']
            composite = rankings[i]['composite_score']
            print(f"  #{i+1}: {team} {record} - Playoff: {playoff_prob:.1f}% - Score: {composite:.1f}")


if __name__ == "__main__":
    main()
