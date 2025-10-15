#!/usr/bin/env python3
"""
Power Rankings Tracker

Calculates and stores weekly power rankings using the R1+SOV algorithm.
Supports backfilling historical weeks and tracking movement over time.
"""

import csv
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from power_rankings import PowerRankings


class PowerRankingsTracker:
    """Track power rankings over time"""

    def __init__(self, data_dir: str = 'data'):
        self.data_dir = Path(data_dir)
        self.history_file = self.data_dir / 'power_rankings_history.json'
        self.history = self._load_history()

    def _load_history(self) -> Dict:
        """Load existing rankings history"""
        if self.history_file.exists():
            with open(self.history_file, 'r') as f:
                return json.load(f)
        return {'weeks': {}, 'last_updated': None}

    def _save_history(self):
        """Save rankings history to file"""
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2)

    def calculate_week_rankings(self, week_num: int, max_week: Optional[int] = None,
                               use_historical_mode: bool = False) -> List[Dict]:
        """
        Calculate rankings for a specific week

        Args:
            week_num: Week number to calculate rankings for
            max_week: Maximum week to include in schedule (filters games)
            use_historical_mode: If True, sets playoff_prob to 0 (for backfilling)

        Returns:
            List of team rankings
        """
        # Use max_week if provided, otherwise use week_num
        filter_week = max_week if max_week is not None else week_num

        # Create a PowerRankings instance
        pr = PowerRankings(data_dir=str(self.data_dir))

        # For historical weeks, zero out playoff probabilities (we don't have them)
        if use_historical_mode:
            pr.playoff_probs = {team: 0.0 for team in pr.teams.keys()}

        # Filter schedule to only include games through the target week
        pr.schedule = [
            game for game in pr.schedule
            if game['week'] <= filter_week
        ]

        # Recalculate team records based on filtered schedule
        # Reset all teams to 0-0
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

        # Get rankings
        rankings = pr.r1_sov_rankings()

        return rankings

    def update_current_week(self, week_num: int, force: bool = False,
                           is_historical: bool = False) -> Dict:
        """
        Update rankings for the current week

        Args:
            week_num: Current week number
            force: Force update even if week already exists
            is_historical: If True, uses historical mode (no playoff prob)

        Returns:
            Rankings data with movement information
        """
        week_key = str(week_num)

        # Check if already calculated
        if week_key in self.history['weeks'] and not force:
            print(f"Week {week_num} already calculated. Use force=True to recalculate.")
            return self.history['weeks'][week_key]

        # Calculate current rankings
        rankings = self.calculate_week_rankings(week_num, use_historical_mode=is_historical)

        # Get previous week rankings for movement calculation
        prev_week_key = str(week_num - 1)
        previous_rankings = {}
        if prev_week_key in self.history['weeks']:
            previous_rankings = {
                r['team']: r['rank']
                for r in self.history['weeks'][prev_week_key]['rankings']
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
        self.history['weeks'][week_key] = {
            'week': week_num,
            'rankings': rankings,
            'calculated_at': datetime.now().isoformat()
        }
        self.history['last_updated'] = datetime.now().isoformat()

        self._save_history()

        return self.history['weeks'][week_key]

    def backfill_historical_weeks(self, start_week: int, end_week: int):
        """
        Backfill rankings for historical weeks

        Note: Historical weeks use 0 for playoff probability since we don't
        have Monte Carlo simulation data for past weeks. Rankings will be
        based on: Win% (60%), Seed (10%), and SOV (15%).

        Args:
            start_week: First week to backfill
            end_week: Last week to backfill
        """
        print(f"Backfilling weeks {start_week} to {end_week}...")
        print("Note: Historical weeks use 0 for playoff prob (no sim data)")
        print()

        for week in range(start_week, end_week + 1):
            print(f"  Calculating week {week}...", end=' ')
            self.update_current_week(week, force=True, is_historical=True)
            print(f"✓")

        print(f"\nBackfilled {end_week - start_week + 1} weeks")

    def get_current_rankings(self, week_num: int) -> List[Dict]:
        """Get rankings for a specific week"""
        week_key = str(week_num)

        if week_key not in self.history['weeks']:
            # Calculate if not exists
            self.update_current_week(week_num)

        return self.history['weeks'][week_key]['rankings']

    def get_team_history(self, team: str) -> List[Dict]:
        """
        Get ranking history for a specific team

        Returns list of: {week, rank, movement, score}
        """
        history = []

        for week_key in sorted(self.history['weeks'].keys(), key=int):
            week_data = self.history['weeks'][week_key]

            # Find team in this week's rankings
            for team_rank in week_data['rankings']:
                if team_rank['team'] == team:
                    history.append({
                        'week': week_data['week'],
                        'rank': team_rank['rank'],
                        'movement': team_rank.get('movement', 0),
                        'score': team_rank['composite_score'],
                        'record': team_rank['record']
                    })
                    break

        return history


if __name__ == "__main__":
    import sys

    tracker = PowerRankingsTracker()

    if len(sys.argv) > 1 and sys.argv[1] == 'backfill':
        # Backfill weeks 1-6
        tracker.backfill_historical_weeks(1, 6)

        # Show current week 6 rankings
        print("\nWeek 6 Rankings:")
        rankings = tracker.get_current_rankings(6)
        print(f"{'Rank':<6} {'Team':<6} {'Record':<10} {'Prev':<6} {'Move':<6}")
        print("-" * 40)
        for r in rankings[:10]:
            prev = f"#{r['previous_rank']}" if r.get('previous_rank') else "NEW"
            move = f"{r['movement']:+d}" if r.get('movement') else "-"
            print(f"{r['rank']:<6} {r['team']:<6} {r['record']:<10} {prev:<6} {move:<6}")

    elif len(sys.argv) > 1 and sys.argv[1] == 'team':
        # Show team history
        team = sys.argv[2] if len(sys.argv) > 2 else 'TB'
        history = tracker.get_team_history(team)

        print(f"\n{team} Power Rankings History:")
        print(f"{'Week':<6} {'Rank':<6} {'Record':<10} {'Move':<6}")
        print("-" * 30)
        for h in history:
            move = f"{h['movement']:+d}" if h['movement'] else "-"
            print(f"{h['week']:<6} #{h['rank']:<5} {h['record']:<10} {move:<6}")

    else:
        # Update current week (7)
        current_week = 7
        print(f"Calculating Week {current_week} rankings...")
        data = tracker.update_current_week(current_week, force=True)

        rankings = data['rankings']
        print(f"\nWeek {current_week} Power Rankings:")
        print(f"{'Rank':<6} {'Team':<6} {'Record':<10} {'Score':<8} {'Prev':<6} {'Move':<6}")
        print("-" * 50)

        for r in rankings[:15]:
            prev = f"#{r['previous_rank']}" if r.get('previous_rank') else "NEW"
            move = f"{r['movement']:+d}" if r.get('movement') else "-"
            print(f"{r['rank']:<6} {r['team']:<6} {r['record']:<10} {r['composite_score']:<8.1f} {prev:<6} {move:<6}")
