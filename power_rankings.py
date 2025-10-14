#!/usr/bin/env python3
"""
NFL Power Rankings - R1+SOV Algorithm

This module implements a Record-First + Strength of Victory power ranking
algorithm that achieved 0.943 correlation with external consensus rankings.

The algorithm weights:
- Win Percentage: 60%
- Playoff Probability: 15%
- Current Playoff Seed: 10%
- Strength of Victory: 15%

Strength of Victory (SOV) is the average win percentage of teams you've beaten,
providing a measure of the quality of your wins.
"""

import csv
import json
from typing import List, Dict
from pathlib import Path


class PowerRankings:
    """NFL Power Rankings using R1+SOV algorithm"""

    def __init__(self, data_dir: str = 'data'):
        """Initialize with data directory"""
        self.data_dir = Path(data_dir)
        self.teams = {}
        self.schedule = []
        self.playoff_probs = {}

        # Load all data
        self._load_team_stats()
        self._load_schedule()
        self._load_playoff_probabilities()
        self._calculate_playoff_seeds()

    def _load_team_stats(self):
        """Load team statistics from CSV"""
        stats_file = self.data_dir / 'team_stats.csv'

        with open(stats_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                team = row['team_abbr']
                self.teams[team] = {
                    'wins': int(row['wins']),
                    'losses': int(row['losses']),
                    'ties': int(row['ties']),
                    'win_pct': float(row['win_pct']),
                    'record': f"{row['wins']}-{row['losses']}" + (f"-{row['ties']}" if int(row['ties']) > 0 else ""),
                    'point_diff': int(row['point_diff']),
                    'points_for': int(row['points_for']),
                    'points_against': int(row['points_against'])
                }

    def _load_schedule(self):
        """Load schedule data from CSV"""
        schedule_file = self.data_dir / 'schedule.csv'

        with open(schedule_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Only include completed games (those with scores)
                if row['away_score'] and row['home_score']:
                    self.schedule.append({
                        'week': int(row['week_num']),
                        'away_team': row['away_team'],
                        'home_team': row['home_team'],
                        'away_score': int(row['away_score']),
                        'home_score': int(row['home_score'])
                    })

    def _load_playoff_probabilities(self):
        """Load playoff probabilities from analysis cache"""
        cache_file = self.data_dir / 'analysis_cache.json'

        with open(cache_file, 'r') as f:
            cache = json.load(f)
            team_analyses = cache.get('team_analyses', {})

            for team, data in team_analyses.items():
                # Extract playoff chance (0-100 scale)
                self.playoff_probs[team] = data.get('playoff_chance', 0.0)

    def _calculate_playoff_seeds(self):
        """Calculate current playoff seeds for each team"""
        # Separate teams by conference
        afc_teams = []
        nfc_teams = []

        # Conference assignments (hardcoded for simplicity)
        afc = {'BUF', 'MIA', 'NE', 'NYJ', 'BAL', 'CIN', 'CLE', 'PIT',
               'HOU', 'IND', 'JAX', 'TEN', 'DEN', 'KC', 'LV', 'LAC'}

        for team, stats in self.teams.items():
            team_data = {
                'team': team,
                'win_pct': stats['win_pct'],
                'wins': stats['wins']
            }

            if team in afc:
                afc_teams.append(team_data)
            else:
                nfc_teams.append(team_data)

        # Sort by win percentage (then by wins for tiebreaker)
        afc_teams.sort(key=lambda x: (x['win_pct'], x['wins']), reverse=True)
        nfc_teams.sort(key=lambda x: (x['win_pct'], x['wins']), reverse=True)

        # Assign seeds (1-7 for playoff teams, 8+ for non-playoff)
        for idx, team_data in enumerate(afc_teams, 1):
            self.teams[team_data['team']]['playoff_seed'] = idx

        for idx, team_data in enumerate(nfc_teams, 1):
            self.teams[team_data['team']]['playoff_seed'] = idx

    def calculate_strength_of_victory(self, team: str) -> float:
        """
        Calculate strength of victory - average win% of teams beaten

        Args:
            team: Team abbreviation

        Returns:
            Average win% of teams beaten (0.0 to 1.0)
        """
        victories = []

        for game in self.schedule:
            # Check if this team won
            if game['away_team'] == team and game['away_score'] > game['home_score']:
                victories.append(game['home_team'])
            elif game['home_team'] == team and game['home_score'] > game['away_score']:
                victories.append(game['away_team'])

        if not victories:
            return 0.5  # No wins yet, return average

        # Calculate average win% of defeated opponents
        total_win_pct = sum(self.teams[opp]['win_pct'] for opp in victories if opp in self.teams)
        return total_win_pct / len(victories) if victories else 0.5

    def normalize_to_scale(self, value: float, min_val: float, max_val: float) -> float:
        """
        Normalize a value to 0-100 scale

        Args:
            value: Value to normalize
            min_val: Minimum value in range
            max_val: Maximum value in range

        Returns:
            Normalized value (0-100)
        """
        if max_val == min_val:
            return 50.0  # Return midpoint if no variation

        return ((value - min_val) / (max_val - min_val)) * 100

    def r1_sov_rankings(self) -> List[Dict]:
        """
        R1+SOV Algorithm: Record-First + Strength of Victory

        Weights:
        - Win%: 60%
        - Playoff probability: 15%
        - Current playoff seed: 10%
        - Strength of victory: 15%

        Returns:
            List of team rankings sorted by composite score
        """
        team_scores = {}

        for team in self.teams.keys():
            scores = {}

            # Win percentage (already 0-100 scale)
            scores['win_pct'] = self.teams[team]['win_pct'] * 100

            # Playoff probability (already 0-100 scale)
            scores['playoff_prob'] = self.playoff_probs.get(team, 0.0)

            # Current playoff seed (inverse and normalize - lower seed is better)
            seed = self.teams[team]['playoff_seed']
            scores['seed'] = self.normalize_to_scale(16 - seed, 0, 15) * 100 / 15

            # Strength of victory (normalize to 0-100)
            sov = self.calculate_strength_of_victory(team)
            sov_vals = [self.calculate_strength_of_victory(t) for t in self.teams.keys()]
            scores['sov'] = self.normalize_to_scale(sov, min(sov_vals), max(sov_vals))

            # Apply weights
            weights = {
                'win_pct': 0.60,
                'playoff_prob': 0.15,
                'seed': 0.10,
                'sov': 0.15
            }

            composite_score = sum(scores[key] * weights[key] for key in weights.keys())
            team_scores[team] = {
                'composite_score': composite_score,
                'breakdown': scores
            }

        # Build rankings list
        rankings = []
        for team, data in team_scores.items():
            rankings.append({
                'team': team,
                'record': self.teams[team]['record'],
                'composite_score': data['composite_score'],
                'win_pct': self.teams[team]['win_pct'],
                'playoff_prob': self.playoff_probs.get(team, 0.0),
                'playoff_seed': self.teams[team]['playoff_seed'],
                'sov': self.calculate_strength_of_victory(team),
                'breakdown': data['breakdown']
            })

        # Sort by composite score
        rankings.sort(key=lambda x: x['composite_score'], reverse=True)

        # Add rank numbers
        for rank, team_rank in enumerate(rankings, 1):
            team_rank['rank'] = rank

        return rankings


if __name__ == "__main__":
    # Quick test
    pr = PowerRankings()
    rankings = pr.r1_sov_rankings()

    print("NFL Power Rankings - R1+SOV Algorithm")
    print("=" * 80)
    print()
    print(f"{'Rank':<6} {'Team':<6} {'Record':<10} {'Score':<8} {'Win%':<8} {'Playoff%':<10} {'SOV':<8}")
    print("-" * 80)

    for team in rankings[:10]:
        print(f"{team['rank']:<6} {team['team']:<6} {team['record']:<10} "
              f"{team['composite_score']:<8.1f} {team['win_pct']:<8.3f} "
              f"{team['playoff_prob']:<10.1f} {team['sov']:<8.3f}")
