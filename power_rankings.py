#!/usr/bin/env python3
"""
Power Rankings Module

Implements multiple algorithms for calculating NFL power rankings:
1. Weighted Composite - Blend of multiple metrics with tuned weights
2. Elo Dynamic - Elo-style rating system updated after each game
3. Multi-Factor Score - Independent scoring across multiple dimensions
4. AI-Enhanced - Uses base algorithm + AI adjustments for intangibles

Author: Claude Code
Date: 2025-10-14
"""

import json
import csv
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class TeamData:
    """Container for all team data used in rankings"""
    abbr: str
    record: str
    wins: int
    losses: int
    ties: int
    win_pct: float
    points_for: float
    points_against: float
    point_diff: float
    epa_per_game: float
    turnover_margin: int
    third_down_pct: float
    red_zone_pct: float
    sagarin_rating: float
    sagarin_prev_rating: float
    playoff_prob: float
    division_prob: float
    super_bowl_prob: float
    current_seed: Optional[int]


class PowerRankings:
    """Main class for calculating power rankings"""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.teams = {}
        self.schedule = []
        self.load_all_data()

    def load_all_data(self):
        """Load all necessary data from files"""
        self._load_team_stats()
        self._load_sagarin()
        self._load_simulation_results()
        self._load_schedule()

    def _load_team_stats(self):
        """Load team statistics from CSV"""
        stats_file = f"{self.data_dir}/team_stats.csv"

        with open(stats_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                team = row['team_abbr']

                # Initialize team data
                self.teams[team] = {
                    'abbr': team,
                    'record': f"{row['wins']}-{row['losses']}" + (f"-{row['ties']}" if int(row['ties']) > 0 else ""),
                    'wins': int(row['wins']),
                    'losses': int(row['losses']),
                    'ties': int(row['ties']),
                    'win_pct': float(row['win_pct']),
                    'points_for': float(row['points_for']),
                    'points_against': float(row['points_against']),
                    'point_diff': float(row['point_diff']),
                    'epa_per_game': float(row['epa_per_game']),
                    'turnover_margin': int(row['turnover_margin']),
                    'third_down_pct': float(row['third_down_pct']),
                    'red_zone_pct': float(row['red_zone_pct']),
                }

    def _load_sagarin(self):
        """Load Sagarin ratings from CSV"""
        sagarin_file = f"{self.data_dir}/sagarin.csv"

        with open(sagarin_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                team = row['team_abbr']
                if team in self.teams:
                    self.teams[team]['sagarin_rating'] = float(row['rating'])
                    self.teams[team]['sagarin_prev_rating'] = float(row['previous_rating'])

    def _load_simulation_results(self):
        """Load simulation results (playoff probabilities, etc.)"""
        cache_file = f"{self.data_dir}/analysis_cache.json"

        with open(cache_file, 'r') as f:
            data = json.load(f)

        team_analyses = data.get('team_analyses', {})

        for team, analysis in team_analyses.items():
            if team in self.teams:
                self.teams[team]['playoff_prob'] = analysis.get('playoff_chance', 0.0)
                self.teams[team]['division_prob'] = analysis.get('division_chance', 0.0)
                self.teams[team]['super_bowl_prob'] = analysis.get('super_bowl_win_chance', 0.0)

                # Get current seed from standings
                # For now, we'll determine from playoff probability ranking
                self.teams[team]['current_seed'] = None  # Will calculate later

    def _load_schedule(self):
        """Load schedule for calculating recent form and SOS"""
        schedule_file = f"{self.data_dir}/schedule.csv"

        with open(schedule_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Only include completed games (has scores)
                if row['away_score'] and row['home_score']:
                    self.schedule.append({
                        'week': int(row['week_num']),
                        'date': row['game_date'],
                        'away_team': row['away_team'],
                        'home_team': row['home_team'],
                        'away_score': int(row['away_score']),
                        'home_score': int(row['home_score']),
                    })

    def calculate_recent_form(self, team: str, num_games: int = 3) -> float:
        """
        Calculate win percentage in last N games
        Returns: win% in last N games (0.0 to 1.0)
        """
        team_games = []

        for game in sorted(self.schedule, key=lambda x: x['week'], reverse=True):
            if game['away_team'] == team or game['home_team'] == team:
                team_games.append(game)
                if len(team_games) >= num_games:
                    break

        if not team_games:
            return self.teams[team]['win_pct']  # Fall back to overall win%

        wins = 0
        for game in team_games:
            if game['away_team'] == team:
                if game['away_score'] > game['home_score']:
                    wins += 1
            else:  # home team
                if game['home_score'] > game['away_score']:
                    wins += 1

        return wins / len(team_games)

    def calculate_strength_of_schedule(self, team: str) -> float:
        """
        Calculate strength of schedule based on opponent Sagarin ratings
        Returns: average Sagarin rating of opponents faced
        """
        opponents = []

        for game in self.schedule:
            if game['away_team'] == team:
                opponents.append(game['home_team'])
            elif game['home_team'] == team:
                opponents.append(game['away_team'])

        if not opponents:
            return 20.0  # Average Sagarin rating

        total_rating = sum(self.teams[opp]['sagarin_rating'] for opp in opponents if opp in self.teams)
        return total_rating / len(opponents) if opponents else 20.0

    @staticmethod
    def normalize_to_percentile(values: List[float]) -> List[float]:
        """
        Convert list of values to percentiles (0-100)
        Higher is better
        """
        if not values or len(values) == 1:
            return [50.0] * len(values)

        sorted_vals = sorted(values)
        percentiles = []

        for val in values:
            rank = sorted_vals.index(val) + 1
            percentile = (rank / len(values)) * 100
            percentiles.append(percentile)

        return percentiles

    @staticmethod
    def normalize_to_scale(value: float, min_val: float, max_val: float, target_min: float = 0, target_max: float = 100) -> float:
        """Normalize a value to a target scale"""
        if max_val == min_val:
            return (target_min + target_max) / 2

        normalized = (value - min_val) / (max_val - min_val)
        scaled = normalized * (target_max - target_min) + target_min
        return max(target_min, min(target_max, scaled))

    # ========== ALGORITHM 1: WEIGHTED COMPOSITE ==========

    def weighted_composite_rankings(self, weights: Optional[Dict[str, float]] = None) -> List[Dict]:
        """
        Calculate power rankings using weighted composite of multiple metrics

        Default weights:
        - Sagarin: 15%
        - Playoff probability: 25%
        - Point differential: 20%
        - Recent form (last 3): 15%
        - Record quality: 10%
        - EPA/game: 10%
        - Turnover margin: 5%
        """
        if weights is None:
            weights = {
                'sagarin': 0.15,
                'playoff_prob': 0.25,
                'point_diff': 0.20,
                'recent_form': 0.15,
                'record': 0.10,
                'epa': 0.10,
                'turnovers': 0.05,
            }

        # Calculate each component score (0-100 scale)
        team_scores = {}

        for team in self.teams.keys():
            scores = {}

            # Sagarin (normalize to 0-100)
            sagarin_vals = [t['sagarin_rating'] for t in self.teams.values()]
            sagarin_min, sagarin_max = min(sagarin_vals), max(sagarin_vals)
            scores['sagarin'] = self.normalize_to_scale(
                self.teams[team]['sagarin_rating'], sagarin_min, sagarin_max
            )

            # Playoff probability (already 0-100)
            scores['playoff_prob'] = self.teams[team]['playoff_prob']

            # Point differential per game (normalize, handle negatives)
            point_diff_vals = [t['point_diff'] / max(t['wins'] + t['losses'] + t['ties'], 1)
                             for t in self.teams.values()]
            pd_min, pd_max = min(point_diff_vals), max(point_diff_vals)
            team_pd_per_game = self.teams[team]['point_diff'] / max(
                self.teams[team]['wins'] + self.teams[team]['losses'] + self.teams[team]['ties'], 1
            )
            scores['point_diff'] = self.normalize_to_scale(team_pd_per_game, pd_min, pd_max)

            # Recent form (last 3 games win%, already 0-1, convert to 0-100)
            scores['recent_form'] = self.calculate_recent_form(team) * 100

            # Record quality (win% adjusted by SOS)
            sos = self.calculate_strength_of_schedule(team)
            sos_adjusted_win_pct = self.teams[team]['win_pct'] * (sos / 20.0)  # 20 is avg Sagarin
            scores['record'] = sos_adjusted_win_pct * 100

            # EPA per game
            epa_vals = [t['epa_per_game'] for t in self.teams.values()]
            epa_min, epa_max = min(epa_vals), max(epa_vals)
            scores['epa'] = self.normalize_to_scale(
                self.teams[team]['epa_per_game'], epa_min, epa_max
            )

            # Turnover margin
            to_vals = [t['turnover_margin'] for t in self.teams.values()]
            to_min, to_max = min(to_vals), max(to_vals)
            scores['turnovers'] = self.normalize_to_scale(
                self.teams[team]['turnover_margin'], to_min, to_max
            )

            # Calculate weighted composite
            composite_score = sum(scores[key] * weights[key] for key in weights.keys())

            team_scores[team] = {
                'team': team,
                'record': self.teams[team]['record'],
                'composite_score': composite_score,
                'breakdown': scores,
            }

        # Sort by composite score descending
        ranked = sorted(team_scores.values(), key=lambda x: x['composite_score'], reverse=True)

        # Add rank
        for i, team_data in enumerate(ranked, 1):
            team_data['rank'] = i

        return ranked

    # ========== ALGORITHM 2: ELO DYNAMIC ==========

    def elo_dynamic_rankings(self, k_factor: float = 25, home_advantage: float = 3.0) -> List[Dict]:
        """
        Calculate power rankings using Elo rating system

        Args:
            k_factor: How much ratings change per game (higher = more reactive)
            home_advantage: Points added to home team's effective rating
        """
        # Initialize Elo ratings from Sagarin (scaled to ~1000-2000 range)
        elo_ratings = {}
        sagarin_vals = [t['sagarin_rating'] for t in self.teams.values()]
        sagarin_min, sagarin_max = min(sagarin_vals), max(sagarin_vals)

        for team in self.teams.keys():
            # Scale Sagarin (12-26) to Elo (1000-2000)
            normalized = (self.teams[team]['sagarin_rating'] - sagarin_min) / (sagarin_max - sagarin_min)
            elo_ratings[team] = 1000 + (normalized * 1000)

        initial_elo = dict(elo_ratings)  # Store initial for comparison

        # Process games in chronological order
        for game in sorted(self.schedule, key=lambda x: x['week']):
            away = game['away_team']
            home = game['home_team']

            if away not in elo_ratings or home not in elo_ratings:
                continue

            # Calculate expected outcome
            home_elo_adjusted = elo_ratings[home] + home_advantage
            elo_diff = home_elo_adjusted - elo_ratings[away]
            expected_home_win = 1 / (1 + 10 ** (-elo_diff / 400))

            # Actual outcome (1 = home win, 0 = away win)
            actual_home_win = 1 if game['home_score'] > game['away_score'] else 0

            # Margin of victory multiplier (diminishing returns)
            margin = abs(game['home_score'] - game['away_score'])
            mov_multiplier = math.log(margin + 1) / math.log(20)  # Max ~1.3 for 20-point game
            mov_multiplier = min(mov_multiplier, 2.0)  # Cap at 2x

            # Update ratings
            home_change = k_factor * mov_multiplier * (actual_home_win - expected_home_win)
            elo_ratings[home] += home_change
            elo_ratings[away] -= home_change

        # Create ranked list
        team_data = []
        for team in self.teams.keys():
            team_data.append({
                'team': team,
                'record': self.teams[team]['record'],
                'elo_rating': elo_ratings[team],
                'initial_elo': initial_elo[team],
                'elo_change': elo_ratings[team] - initial_elo[team],
            })

        # Sort by Elo descending
        ranked = sorted(team_data, key=lambda x: x['elo_rating'], reverse=True)

        # Add rank
        for i, team_info in enumerate(ranked, 1):
            team_info['rank'] = i

        return ranked

    # ========== ALGORITHM 3: MULTI-FACTOR SCORE ==========

    def multi_factor_rankings(self) -> List[Dict]:
        """
        Calculate power rankings by scoring across multiple independent dimensions

        Dimensions:
        1. Record Score - Win% adjusted for strength of schedule
        2. Performance Score - Points scored vs allowed
        3. Efficiency Score - EPA, 3rd down%, red zone%
        4. Momentum Score - Recent form (last 3 games)
        5. Predictive Score - Playoff probability and SB probability
        6. Market Score - Sagarin rating (proxy for market/expert opinion)
        """
        team_scores = {}

        for team in self.teams.keys():
            scores = {}

            # 1. Record Score (0-100)
            sos = self.calculate_strength_of_schedule(team)
            sos_factor = sos / 20.0  # Normalize around league average
            record_score = self.teams[team]['win_pct'] * sos_factor * 100
            scores['record'] = min(100, record_score)

            # 2. Performance Score (0-100) - based on point differential
            ppg = self.teams[team]['points_for'] / max(self.teams[team]['wins'] + self.teams[team]['losses'] + self.teams[team]['ties'], 1)
            opp_ppg = self.teams[team]['points_against'] / max(self.teams[team]['wins'] + self.teams[team]['losses'] + self.teams[team]['ties'], 1)
            # Scale: -20 to +20 point diff → 0 to 100
            net_ppg = ppg - opp_ppg
            performance_score = self.normalize_to_scale(net_ppg, -20, 20, 0, 100)
            scores['performance'] = performance_score

            # 3. Efficiency Score (0-100) - average of EPA, 3rd down, red zone
            epa_vals = [t['epa_per_game'] for t in self.teams.values()]
            epa_pct = self.normalize_to_scale(
                self.teams[team]['epa_per_game'],
                min(epa_vals), max(epa_vals), 0, 100
            )
            third_down_score = self.teams[team]['third_down_pct']  # Already in %
            red_zone_score = self.teams[team]['red_zone_pct']  # Already in %
            scores['efficiency'] = (epa_pct + third_down_score + red_zone_score) / 3

            # 4. Momentum Score (0-100) - last 3 games
            recent_form = self.calculate_recent_form(team, 3)
            scores['momentum'] = recent_form * 100

            # 5. Predictive Score (0-100) - playoff and SB probability
            playoff_pct = self.teams[team]['playoff_prob']
            sb_pct = self.teams[team]['super_bowl_prob'] * 2  # Weight SB higher
            predictive_score = (playoff_pct + sb_pct) / 3
            scores['predictive'] = min(100, predictive_score)

            # 6. Market Score (0-100) - Sagarin
            sagarin_vals = [t['sagarin_rating'] for t in self.teams.values()]
            market_score = self.normalize_to_scale(
                self.teams[team]['sagarin_rating'],
                min(sagarin_vals), max(sagarin_vals), 0, 100
            )
            scores['market'] = market_score

            # Average all dimensions
            multi_factor_score = sum(scores.values()) / len(scores)

            team_scores[team] = {
                'team': team,
                'record': self.teams[team]['record'],
                'multi_factor_score': multi_factor_score,
                'breakdown': scores,
            }

        # Sort by multi-factor score descending
        ranked = sorted(team_scores.values(), key=lambda x: x['multi_factor_score'], reverse=True)

        # Add rank
        for i, team_data in enumerate(ranked, 1):
            team_data['rank'] = i

        return ranked

    # ========== ALGORITHM 4: AI-ENHANCED ==========

    def ai_enhanced_rankings(self, base_algorithm: str = 'composite', ai_model: str = 'sonnet-3.7') -> List[Dict]:
        """
        Calculate power rankings using base algorithm + AI adjustments

        Args:
            base_algorithm: Which base algorithm to use ('composite', 'elo', 'multifactor')
            ai_model: Which AI model to use for enhancement
        """
        from ai_service import AIService, resolve_model_name

        # Get base rankings
        if base_algorithm == 'elo':
            base_rankings = self.elo_dynamic_rankings()
        elif base_algorithm == 'multifactor':
            base_rankings = self.multi_factor_rankings()
        else:  # default to composite
            base_rankings = self.weighted_composite_rankings()

        # Prepare context for AI
        context_data = []
        for i, team_rank in enumerate(base_rankings[:15], 1):  # Top 15 teams
            team = team_rank['team']
            team_info = self.teams[team]

            # Get recent games
            recent_games = []
            for game in sorted(self.schedule, key=lambda x: x['week'], reverse=True)[:3]:
                if game['away_team'] == team:
                    result = "W" if game['away_score'] > game['home_score'] else "L"
                    score = f"{game['away_score']}-{game['home_score']}"
                    recent_games.append(f"{result} @ {game['home_team']} ({score})")
                elif game['home_team'] == team:
                    result = "W" if game['home_score'] > game['away_score'] else "L"
                    score = f"{game['home_score']}-{game['away_score']}"
                    recent_games.append(f"{result} vs {game['away_team']} ({score})")

                if len(recent_games) >= 3:
                    break

            context_data.append({
                'rank': i,
                'team': team,
                'record': team_info['record'],
                'point_diff': f"+{team_info['point_diff']:.1f}" if team_info['point_diff'] > 0 else f"{team_info['point_diff']:.1f}",
                'playoff_prob': f"{team_info['playoff_prob']:.1f}%",
                'recent_games': recent_games[:3],
            })

        # Build AI prompt
        prompt = f"""You are an NFL analyst reviewing power rankings. A data-driven algorithm has produced these rankings, but you can adjust them based on your expert judgment.

**Base Rankings (Top 15)**:

"""

        for team_data in context_data:
            recent_str = ", ".join(team_data['recent_games']) if team_data['recent_games'] else "No recent games"
            prompt += f"{team_data['rank']}. **{team_data['team']}** ({team_data['record']}) - Point Diff: {team_data['point_diff']}, Playoff Prob: {team_data['playoff_prob']}\n"
            prompt += f"   Recent: {recent_str}\n\n"

        prompt += """
**Your Task**:
Review these rankings and make adjustments (±3 spots maximum per team) based on:
- Quality of wins/losses (did they beat good teams or bad teams?)
- Recent momentum and trends
- Key injuries or roster changes you're aware of
- "Eye test" - does this team pass the eye test for their ranking?
- Strength of schedule remaining

**Important Rules**:
- You can ONLY adjust teams by ±3 spots from their base rank
- You must explain your reasoning for each adjustment
- Consider the full body of work, not just 1-2 games

**Return Format** (JSON):
```json
{
  "adjustments": [
    {
      "team": "TEAM_ABBR",
      "base_rank": X,
      "adjusted_rank": Y,
      "reasoning": "Brief explanation of why you adjusted this team"
    }
  ],
  "summary": "Overall assessment of the rankings and key themes"
}
```

Only include teams you want to adjust. Return valid JSON only.
"""

        # Call AI
        ai = AIService(model_override=resolve_model_name(ai_model))
        response, status = ai.generate_analysis(
            prompt,
            system_message="You are an expert NFL analyst providing nuanced power ranking adjustments based on context and recent performance."
        )

        if status != 'success':
            # If AI fails, return base rankings with note
            for team_rank in base_rankings:
                team_rank['ai_adjusted_rank'] = team_rank['rank']
                team_rank['ai_reasoning'] = "AI enhancement unavailable"
                team_rank['ai_summary'] = "AI service error"
            return base_rankings

        # Parse AI response
        try:
            ai_data = json.loads(response)
            adjustments = {adj['team']: adj for adj in ai_data.get('adjustments', [])}
            summary = ai_data.get('summary', '')

            # Apply adjustments
            adjusted_rankings = []
            for team_rank in base_rankings:
                team = team_rank['team']
                base_rank = team_rank['rank']

                if team in adjustments:
                    adj = adjustments[team]
                    # Validate adjustment is within ±3
                    proposed_rank = adj['adjusted_rank']
                    if abs(proposed_rank - base_rank) <= 3:
                        team_rank['ai_adjusted_rank'] = proposed_rank
                        team_rank['ai_reasoning'] = adj['reasoning']
                    else:
                        # Invalid adjustment, keep base
                        team_rank['ai_adjusted_rank'] = base_rank
                        team_rank['ai_reasoning'] = f"Adjustment rejected (>±3 spots)"
                else:
                    team_rank['ai_adjusted_rank'] = base_rank
                    team_rank['ai_reasoning'] = "No adjustment needed"

                adjusted_rankings.append(team_rank)

            # Re-sort by adjusted rank
            adjusted_rankings.sort(key=lambda x: x['ai_adjusted_rank'])

            # Add summary
            for team_rank in adjusted_rankings:
                team_rank['ai_summary'] = summary

            return adjusted_rankings

        except json.JSONDecodeError:
            # If can't parse, return base rankings
            for team_rank in base_rankings:
                team_rank['ai_adjusted_rank'] = team_rank['rank']
                team_rank['ai_reasoning'] = "AI response parsing failed"
                team_rank['ai_summary'] = "Could not parse AI response"
            return base_rankings


    def get_playoff_seed_rank(self, team: str) -> int:
        """Get team's current playoff seed (1-7 in each conference, or 8+ if out)"""
        # Get conference
        afc_teams = ['BUF', 'MIA', 'NE', 'NYJ', 'BAL', 'CIN', 'CLE', 'PIT', 'HOU', 'IND', 'JAX', 'TEN', 'DEN', 'KC', 'LV', 'LAC']
        conference = 'AFC' if team in afc_teams else 'NFC'

        # Sort conference teams by win%, then by point diff
        conf_teams = []
        for t in self.teams.keys():
            is_afc = t in afc_teams
            if (conference == 'AFC' and is_afc) or (conference == 'NFC' and not is_afc):
                conf_teams.append({
                    'team': t,
                    'win_pct': self.teams[t]['win_pct'],
                    'point_diff': self.teams[t]['point_diff']
                })

        conf_teams.sort(key=lambda x: (x['win_pct'], x['point_diff']), reverse=True)

        # Find team's seed
        for seed, t in enumerate(conf_teams, 1):
            if t['team'] == team:
                return seed

        return 8  # Default if not found

    def calculate_win_streak(self, team: str) -> int:
        """Calculate current win streak (negative for loss streak)"""
        # Get games for team (as home or away)
        team_games = []
        for g in self.schedule:
            if g['away_team'] == team:
                is_win = g['away_score'] > g['home_score']
                team_games.append((g['week'], is_win))
            elif g['home_team'] == team:
                is_win = g['home_score'] > g['away_score']
                team_games.append((g['week'], is_win))

        if not team_games:
            return 0

        team_games.sort(key=lambda x: x[0])

        streak = 0
        last_result = None

        for week, is_win in reversed(team_games):
            if last_result is None:
                last_result = is_win
                streak = 1 if is_win else -1
            elif is_win == last_result:
                if is_win:
                    streak += 1
                else:
                    streak -= 1
            else:
                break

        return streak

    def calculate_close_game_win_pct(self, team: str, margin: int = 8) -> float:
        """Calculate win% in close games (within margin points)"""
        close_games = []

        for g in self.schedule:
            score_diff = abs(g['away_score'] - g['home_score'])

            if score_diff <= margin:
                if g['away_team'] == team:
                    is_win = g['away_score'] > g['home_score']
                    close_games.append(is_win)
                elif g['home_team'] == team:
                    is_win = g['home_score'] > g['away_score']
                    close_games.append(is_win)

        if not close_games:
            return 0.5  # Neutral if no close games

        wins = sum(1 for is_win in close_games if is_win)
        return wins / len(close_games)

    def seeding_priority_rankings(self) -> List[Dict]:
        """
        Algorithm 5: Seeding-Priority
        Weights current playoff position heavily
        - Playoff seed rank: 35%
        - Playoff probability: 30%
        - Record: 20%
        - Point differential: 10%
        - Recent form: 5%
        """
        team_scores = {}

        for team in self.teams.keys():
            scores = {}

            # Playoff seed (lower is better, convert to 0-100)
            seed = self.get_playoff_seed_rank(team)
            scores['seed'] = self.normalize_to_scale(16 - seed, 0, 15) * 100 / 15

            # Playoff probability
            scores['playoff_prob'] = self.teams[team]['playoff_prob']

            # Record (win%)
            scores['record'] = self.teams[team]['win_pct'] * 100

            # Point differential per game
            games = self.teams[team]['wins'] + self.teams[team]['losses'] + self.teams[team]['ties']
            pd_per_game = self.teams[team]['point_diff'] / max(games, 1)
            pd_vals = [t['point_diff'] / max(t['wins'] + t['losses'] + t['ties'], 1) for t in self.teams.values()]
            scores['point_diff'] = self.normalize_to_scale(pd_per_game, min(pd_vals), max(pd_vals)) * 100

            # Recent form
            scores['recent_form'] = self.calculate_recent_form(team) * 100

            # Calculate weighted score
            weights = {
                'seed': 0.35,
                'playoff_prob': 0.30,
                'record': 0.20,
                'point_diff': 0.10,
                'recent_form': 0.05
            }

            seeding_score = sum(scores[key] * weights[key] for key in weights.keys())

            team_scores[team] = {
                'seeding_score': seeding_score,
                'breakdown': scores
            }

        # Create rankings
        rankings = []
        for team, data in team_scores.items():
            rankings.append({
                'team': team,
                'record': self.teams[team]['record'],
                'seeding_score': data['seeding_score'],
                'breakdown': data['breakdown']
            })

        rankings.sort(key=lambda x: x['seeding_score'], reverse=True)

        for rank, team_rank in enumerate(rankings, 1):
            team_rank['rank'] = rank

        return rankings

    def win_quality_rankings(self) -> List[Dict]:
        """
        Algorithm 6: Win-Quality
        Emphasizes quality wins over margin
        - Win percentage: 25%
        - Playoff probability: 30%
        - Strength of schedule: 20%
        - Point differential: 15%
        - EPA: 10%
        """
        team_scores = {}

        for team in self.teams.keys():
            scores = {}

            # Win percentage
            scores['win_pct'] = self.teams[team]['win_pct'] * 100

            # Playoff probability
            scores['playoff_prob'] = self.teams[team]['playoff_prob']

            # Strength of schedule
            sos = self.calculate_strength_of_schedule(team)
            sos_vals = [self.calculate_strength_of_schedule(t) for t in self.teams.keys()]
            scores['sos'] = self.normalize_to_scale(sos, min(sos_vals), max(sos_vals)) * 100

            # Point differential per game
            games = self.teams[team]['wins'] + self.teams[team]['losses'] + self.teams[team]['ties']
            pd_per_game = self.teams[team]['point_diff'] / max(games, 1)
            pd_vals = [t['point_diff'] / max(t['wins'] + t['losses'] + t['ties'], 1) for t in self.teams.values()]
            scores['point_diff'] = self.normalize_to_scale(pd_per_game, min(pd_vals), max(pd_vals)) * 100

            # EPA per game
            epa_per_game = self.teams[team]['epa_per_game']
            epa_vals = [t['epa_per_game'] for t in self.teams.values()]
            scores['epa'] = self.normalize_to_scale(epa_per_game, min(epa_vals), max(epa_vals)) * 100

            # Calculate weighted score
            weights = {
                'win_pct': 0.25,
                'playoff_prob': 0.30,
                'sos': 0.20,
                'point_diff': 0.15,
                'epa': 0.10
            }

            quality_score = sum(scores[key] * weights[key] for key in weights.keys())

            team_scores[team] = {
                'quality_score': quality_score,
                'breakdown': scores
            }

        # Create rankings
        rankings = []
        for team, data in team_scores.items():
            rankings.append({
                'team': team,
                'record': self.teams[team]['record'],
                'quality_score': data['quality_score'],
                'breakdown': data['breakdown']
            })

        rankings.sort(key=lambda x: x['quality_score'], reverse=True)

        for rank, team_rank in enumerate(rankings, 1):
            team_rank['rank'] = rank

        return rankings

    def momentum_weighted_rankings(self) -> List[Dict]:
        """
        Algorithm 7: Momentum-Weighted
        Recent performance matters more
        - Recent form (last 2 games weighted 3x): 25%
        - Playoff probability: 30%
        - Win streak: 15%
        - Record: 15%
        - Point differential: 15%
        """
        team_scores = {}

        for team in self.teams.keys():
            scores = {}

            # Recent form (last 2 games)
            team_games = []
            for g in self.schedule:
                if g['away_team'] == team:
                    is_win = g['away_score'] > g['home_score']
                    team_games.append((g['week'], is_win))
                elif g['home_team'] == team:
                    is_win = g['home_score'] > g['away_score']
                    team_games.append((g['week'], is_win))

            team_games.sort(key=lambda x: x[0])
            last_2 = team_games[-2:] if len(team_games) >= 2 else team_games
            wins_last_2 = sum(1 for week, is_win in last_2 if is_win)
            scores['recent_form'] = (wins_last_2 / len(last_2) * 100) if last_2 else 50.0

            # Playoff probability
            scores['playoff_prob'] = self.teams[team]['playoff_prob']

            # Win streak (normalize -6 to +6)
            streak = self.calculate_win_streak(team)
            scores['win_streak'] = self.normalize_to_scale(streak, -6, 6) * 100

            # Record
            scores['record'] = self.teams[team]['win_pct'] * 100

            # Point differential per game
            games = self.teams[team]['wins'] + self.teams[team]['losses'] + self.teams[team]['ties']
            pd_per_game = self.teams[team]['point_diff'] / max(games, 1)
            pd_vals = [t['point_diff'] / max(t['wins'] + t['losses'] + t['ties'], 1) for t in self.teams.values()]
            scores['point_diff'] = self.normalize_to_scale(pd_per_game, min(pd_vals), max(pd_vals)) * 100

            # Calculate weighted score
            weights = {
                'recent_form': 0.25,
                'playoff_prob': 0.30,
                'win_streak': 0.15,
                'record': 0.15,
                'point_diff': 0.15
            }

            momentum_score = sum(scores[key] * weights[key] for key in weights.keys())

            team_scores[team] = {
                'momentum_score': momentum_score,
                'breakdown': scores
            }

        # Create rankings
        rankings = []
        for team, data in team_scores.items():
            rankings.append({
                'team': team,
                'record': self.teams[team]['record'],
                'momentum_score': data['momentum_score'],
                'breakdown': data['breakdown']
            })

        rankings.sort(key=lambda x: x['momentum_score'], reverse=True)

        for rank, team_rank in enumerate(rankings, 1):
            team_rank['rank'] = rank

        return rankings

    def clutch_performance_rankings(self) -> List[Dict]:
        """
        Algorithm 8: Clutch Performance
        Rewards winning close games
        - Playoff probability: 30%
        - Close game win%: 20%
        - Record: 20%
        - Win streak: 15%
        - Point differential: 15%
        """
        team_scores = {}

        for team in self.teams.keys():
            scores = {}

            # Playoff probability
            scores['playoff_prob'] = self.teams[team]['playoff_prob']

            # Close game win%
            scores['close_game_pct'] = self.calculate_close_game_win_pct(team) * 100

            # Record
            scores['record'] = self.teams[team]['win_pct'] * 100

            # Win streak
            streak = self.calculate_win_streak(team)
            scores['win_streak'] = self.normalize_to_scale(streak, -6, 6) * 100

            # Point differential per game
            games = self.teams[team]['wins'] + self.teams[team]['losses'] + self.teams[team]['ties']
            pd_per_game = self.teams[team]['point_diff'] / max(games, 1)
            pd_vals = [t['point_diff'] / max(t['wins'] + t['losses'] + t['ties'], 1) for t in self.teams.values()]
            scores['point_diff'] = self.normalize_to_scale(pd_per_game, min(pd_vals), max(pd_vals)) * 100

            # Calculate weighted score
            weights = {
                'playoff_prob': 0.30,
                'close_game_pct': 0.20,
                'record': 0.20,
                'win_streak': 0.15,
                'point_diff': 0.15
            }

            clutch_score = sum(scores[key] * weights[key] for key in weights.keys())

            team_scores[team] = {
                'clutch_score': clutch_score,
                'breakdown': scores
            }

        # Create rankings
        rankings = []
        for team, data in team_scores.items():
            rankings.append({
                'team': team,
                'record': self.teams[team]['record'],
                'clutch_score': data['clutch_score'],
                'breakdown': data['breakdown']
            })

        rankings.sort(key=lambda x: x['clutch_score'], reverse=True)

        for rank, team_rank in enumerate(rankings, 1):
            team_rank['rank'] = rank

        return rankings

    def simple_predictive_rankings(self) -> List[Dict]:
        """
        Algorithm 9: Simple Predictive
        Focus on future success predictors
        - Playoff probability: 50%
        - Record: 30%
        - EPA: 20%
        """
        team_scores = {}

        for team in self.teams.keys():
            scores = {}

            # Playoff probability (already 0-100)
            scores['playoff_prob'] = self.teams[team]['playoff_prob']

            # Record
            scores['record'] = self.teams[team]['win_pct'] * 100

            # EPA per game
            epa_per_game = self.teams[team]['epa_per_game']
            epa_vals = [t['epa_per_game'] for t in self.teams.values()]
            scores['epa'] = self.normalize_to_scale(epa_per_game, min(epa_vals), max(epa_vals)) * 100

            # Calculate weighted score
            weights = {
                'playoff_prob': 0.50,
                'record': 0.30,
                'epa': 0.20
            }

            predictive_score = sum(scores[key] * weights[key] for key in weights.keys())

            team_scores[team] = {
                'predictive_score': predictive_score,
                'breakdown': scores
            }

        # Create rankings
        rankings = []
        for team, data in team_scores.items():
            rankings.append({
                'team': team,
                'record': self.teams[team]['record'],
                'predictive_score': data['predictive_score'],
                'breakdown': data['breakdown']
            })

        rankings.sort(key=lambda x: x['predictive_score'], reverse=True)

        for rank, team_rank in enumerate(rankings, 1):
            team_rank['rank'] = rank

        return rankings


    def qb_excellence_rankings(self) -> List[Dict]:
        """
        Algorithm 10: QB Excellence
        Rewards elite quarterback play
        - Passer rating: 30%
        - Completion %: 15%
        - Yards per attempt: 15%
        - TD/INT ratio: 15%
        - Win%: 25%
        """
        team_scores = {}

        for team in self.teams.keys():
            scores = {}

            # Get QB stats from team_stats
            passer_rating = 0
            completion_pct = 0
            yards_per_att = 0
            td_int_ratio = 1.0

            # Read team_stats.csv for this team
            import csv
            with open(f'{self.data_dir}/team_stats.csv', 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row['team_abbr'] == team:
                        passer_rating = float(row['passer_rating']) if row['passer_rating'] else 0
                        completion_pct = float(row['completion_pct']) if row['completion_pct'] else 0
                        yards_per_att = float(row['yards_per_attempt']) if row['yards_per_attempt'] else 0
                        passing_tds = int(row['passing_tds']) if row['passing_tds'] else 0
                        passing_ints = int(row['passing_interceptions']) if row['passing_interceptions'] else 1
                        td_int_ratio = passing_tds / max(passing_ints, 1)
                        break

            # Normalize each metric
            scores['passer_rating'] = self.normalize_to_scale(passer_rating, 0, 140) * 100
            scores['completion_pct'] = completion_pct  # Already 0-100
            scores['yards_per_att'] = self.normalize_to_scale(yards_per_att, 4, 10) * 100
            scores['td_int_ratio'] = self.normalize_to_scale(min(td_int_ratio, 6), 0, 6) * 100
            scores['win_pct'] = self.teams[team]['win_pct'] * 100

            weights = {
                'passer_rating': 0.30,
                'completion_pct': 0.15,
                'yards_per_att': 0.15,
                'td_int_ratio': 0.15,
                'win_pct': 0.25
            }

            qb_score = sum(scores[key] * weights[key] for key in weights.keys())
            team_scores[team] = {'qb_score': qb_score, 'breakdown': scores}

        rankings = []
        for team, data in team_scores.items():
            rankings.append({
                'team': team,
                'record': self.teams[team]['record'],
                'qb_score': data['qb_score'],
                'breakdown': data['breakdown']
            })

        rankings.sort(key=lambda x: x['qb_score'], reverse=True)
        for rank, team_rank in enumerate(rankings, 1):
            team_rank['rank'] = rank

        return rankings

    def defensive_dominance_rankings(self) -> List[Dict]:
        """
        Algorithm 11: Defensive Dominance
        Rewards elite defenses
        - Points against per game: 30%
        - Def sacks: 20%
        - Def interceptions: 20%
        - Third down def %: 15%
        - Win%: 15%
        """
        team_scores = {}

        for team in self.teams.keys():
            scores = {}

            # Get defensive stats
            pts_against = self.teams[team]['points_against'] / max(self.teams[team]['wins'] + self.teams[team]['losses'] + self.teams[team]['ties'], 1)

            def_sacks = 0
            def_ints = 0
            third_down_def = 50.0

            import csv
            with open(f'{self.data_dir}/team_stats.csv', 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row['team_abbr'] == team:
                        def_sacks = float(row['def_sacks']) if row['def_sacks'] else 0
                        def_ints = float(row['def_interceptions']) if row['def_interceptions'] else 0
                        third_down_def = float(row['third_down_pct_against']) if row['third_down_pct_against'] else 50.0
                        break

            # Normalize (lower pts against is better)
            pts_vals = [t['points_against'] / max(t['wins'] + t['losses'] + t['ties'], 1) for t in self.teams.values()]
            scores['pts_against'] = self.normalize_to_scale(40 - pts_against, 0, 40) * 100

            scores['def_sacks'] = self.normalize_to_scale(def_sacks, 0, 30) * 100
            scores['def_ints'] = self.normalize_to_scale(def_ints, 0, 10) * 100
            scores['third_down_def'] = 100 - third_down_def  # Lower is better
            scores['win_pct'] = self.teams[team]['win_pct'] * 100

            weights = {
                'pts_against': 0.30,
                'def_sacks': 0.20,
                'def_ints': 0.20,
                'third_down_def': 0.15,
                'win_pct': 0.15
            }

            def_score = sum(scores[key] * weights[key] for key in weights.keys())
            team_scores[team] = {'def_score': def_score, 'breakdown': scores}

        rankings = []
        for team, data in team_scores.items():
            rankings.append({
                'team': team,
                'record': self.teams[team]['record'],
                'def_score': data['def_score'],
                'breakdown': data['breakdown']
            })

        rankings.sort(key=lambda x: x['def_score'], reverse=True)
        for rank, team_rank in enumerate(rankings, 1):
            team_rank['rank'] = rank

        return rankings

    def red_zone_mastery_rankings(self) -> List[Dict]:
        """
        Algorithm 12: Red Zone Mastery
        Rewards red zone excellence
        - Red zone TD %: 40%
        - Red zone def % (lower): 30%
        - Win%: 30%
        """
        team_scores = {}

        for team in self.teams.keys():
            scores = {}

            rz_pct = self.teams[team]['red_zone_pct']

            rz_def_pct = 50.0
            import csv
            with open(f'{self.data_dir}/team_stats.csv', 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row['team_abbr'] == team:
                        rz_def_pct = float(row['red_zone_pct_against']) if row['red_zone_pct_against'] else 50.0
                        break

            scores['rz_offense'] = rz_pct
            scores['rz_defense'] = 100 - rz_def_pct  # Lower is better for defense
            scores['win_pct'] = self.teams[team]['win_pct'] * 100

            weights = {
                'rz_offense': 0.40,
                'rz_defense': 0.30,
                'win_pct': 0.30
            }

            rz_score = sum(scores[key] * weights[key] for key in weights.keys())
            team_scores[team] = {'rz_score': rz_score, 'breakdown': scores}

        rankings = []
        for team, data in team_scores.items():
            rankings.append({
                'team': team,
                'record': self.teams[team]['record'],
                'rz_score': data['rz_score'],
                'breakdown': data['breakdown']
            })

        rankings.sort(key=lambda x: x['rz_score'], reverse=True)
        for rank, team_rank in enumerate(rankings, 1):
            team_rank['rank'] = rank

        return rankings

    def third_down_excellence_rankings(self) -> List[Dict]:
        """
        Algorithm 13: Third Down Excellence
        Rewards converting and stopping third downs
        - Third down %: 35%
        - Third down def %: 35%
        - Win%: 30%
        """
        team_scores = {}

        for team in self.teams.keys():
            scores = {}

            third_down_pct = self.teams[team]['third_down_pct']

            third_down_def = 50.0
            import csv
            with open(f'{self.data_dir}/team_stats.csv', 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row['team_abbr'] == team:
                        third_down_def = float(row['third_down_pct_against']) if row['third_down_pct_against'] else 50.0
                        break

            scores['third_down_off'] = third_down_pct
            scores['third_down_def'] = 100 - third_down_def
            scores['win_pct'] = self.teams[team]['win_pct'] * 100

            weights = {
                'third_down_off': 0.35,
                'third_down_def': 0.35,
                'win_pct': 0.30
            }

            td_score = sum(scores[key] * weights[key] for key in weights.keys())
            team_scores[team] = {'td_score': td_score, 'breakdown': scores}

        rankings = []
        for team, data in team_scores.items():
            rankings.append({
                'team': team,
                'record': self.teams[team]['record'],
                'td_score': data['td_score'],
                'breakdown': data['breakdown']
            })

        rankings.sort(key=lambda x: x['td_score'], reverse=True)
        for rank, team_rank in enumerate(rankings, 1):
            team_rank['rank'] = rank

        return rankings

    def division_dominance_rankings(self) -> List[Dict]:
        """
        Algorithm 14: Division Dominance
        Rewards division/conference success
        - Division win %: 40%
        - Conference record: 30%
        - Overall win%: 30%
        """
        team_scores = {}

        for team in self.teams.keys():
            scores = {}

            div_win_pct = 0
            conf_wins = 0
            conf_games = 0

            import csv
            with open(f'{self.data_dir}/team_stats.csv', 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row['team_abbr'] == team:
                        div_win_pct = float(row['div_win_pct']) if row['div_win_pct'] else 0
                        conf_record = row['conf_record'] if row['conf_record'] else '0-0'
                        parts = conf_record.split('-')
                        if len(parts) >= 2:
                            conf_wins = int(parts[0])
                            conf_losses = int(parts[1])
                            conf_games = conf_wins + conf_losses
                        break

            conf_win_pct = (conf_wins / conf_games * 100) if conf_games > 0 else 50.0

            scores['div_win_pct'] = div_win_pct * 100
            scores['conf_win_pct'] = conf_win_pct
            scores['overall_win_pct'] = self.teams[team]['win_pct'] * 100

            weights = {
                'div_win_pct': 0.40,
                'conf_win_pct': 0.30,
                'overall_win_pct': 0.30
            }

            div_score = sum(scores[key] * weights[key] for key in weights.keys())
            team_scores[team] = {'div_score': div_score, 'breakdown': scores}

        rankings = []
        for team, data in team_scores.items():
            rankings.append({
                'team': team,
                'record': self.teams[team]['record'],
                'div_score': data['div_score'],
                'breakdown': data['breakdown']
            })

        rankings.sort(key=lambda x: x['div_score'], reverse=True)
        for rank, team_rank in enumerate(rankings, 1):
            team_rank['rank'] = rank

        return rankings

    def record_first_rankings(self) -> List[Dict]:
        """
        Algorithm 15: Record First
        Pure win% with minimal other factors
        - Win%: 70%
        - Playoff prob: 20%
        - Current seed: 10%
        """
        team_scores = {}

        for team in self.teams.keys():
            scores = {}

            scores['win_pct'] = self.teams[team]['win_pct'] * 100
            scores['playoff_prob'] = self.teams[team]['playoff_prob']

            seed = self.get_playoff_seed_rank(team)
            scores['seed'] = self.normalize_to_scale(16 - seed, 0, 15) * 100 / 15

            weights = {
                'win_pct': 0.70,
                'playoff_prob': 0.20,
                'seed': 0.10
            }

            record_score = sum(scores[key] * weights[key] for key in weights.keys())
            team_scores[team] = {'record_score': record_score, 'breakdown': scores}

        rankings = []
        for team, data in team_scores.items():
            rankings.append({
                'team': team,
                'record': self.teams[team]['record'],
                'record_score': data['record_score'],
                'breakdown': data['breakdown']
            })

        rankings.sort(key=lambda x: x['record_score'], reverse=True)
        for rank, team_rank in enumerate(rankings, 1):
            team_rank['rank'] = rank

        return rankings

    def conference_leader_rankings(self) -> List[Dict]:
        """
        Algorithm 16: Conference Leader
        Heavily weights conference position
        - Current playoff seed: 50%
        - Conference record: 30%
        - Playoff probability: 20%
        """
        team_scores = {}

        for team in self.teams.keys():
            scores = {}

            seed = self.get_playoff_seed_rank(team)
            scores['seed'] = self.normalize_to_scale(16 - seed, 0, 15) * 100 / 15

            conf_wins = 0
            conf_games = 0

            import csv
            with open(f'{self.data_dir}/team_stats.csv', 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row['team_abbr'] == team:
                        conf_record = row['conf_record'] if row['conf_record'] else '0-0'
                        parts = conf_record.split('-')
                        if len(parts) >= 2:
                            conf_wins = int(parts[0])
                            conf_losses = int(parts[1])
                            conf_games = conf_wins + conf_losses
                        break

            scores['conf_win_pct'] = (conf_wins / conf_games * 100) if conf_games > 0 else 50.0
            scores['playoff_prob'] = self.teams[team]['playoff_prob']

            weights = {
                'seed': 0.50,
                'conf_win_pct': 0.30,
                'playoff_prob': 0.20
            }

            conf_score = sum(scores[key] * weights[key] for key in weights.keys())
            team_scores[team] = {'conf_score': conf_score, 'breakdown': scores}

        rankings = []
        for team, data in team_scores.items():
            rankings.append({
                'team': team,
                'record': self.teams[team]['record'],
                'conf_score': data['conf_score'],
                'breakdown': data['breakdown']
            })

        rankings.sort(key=lambda x: x['conf_score'], reverse=True)
        for rank, team_rank in enumerate(rankings, 1):
            team_rank['rank'] = rank

        return rankings

    def turnover_battle_rankings(self) -> List[Dict]:
        """
        Algorithm 17: Turnover Battle
        Heavy emphasis on turnovers
        - Turnover margin: 50%
        - Win%: 30%
        - Playoff prob: 20%
        """
        team_scores = {}

        for team in self.teams.keys():
            scores = {}

            to_margin = self.teams[team]['turnover_margin']
            to_vals = [t['turnover_margin'] for t in self.teams.values()]
            scores['to_margin'] = self.normalize_to_scale(to_margin, min(to_vals), max(to_vals)) * 100

            scores['win_pct'] = self.teams[team]['win_pct'] * 100
            scores['playoff_prob'] = self.teams[team]['playoff_prob']

            weights = {
                'to_margin': 0.50,
                'win_pct': 0.30,
                'playoff_prob': 0.20
            }

            to_score = sum(scores[key] * weights[key] for key in weights.keys())
            team_scores[team] = {'to_score': to_score, 'breakdown': scores}

        rankings = []
        for team, data in team_scores.items():
            rankings.append({
                'team': team,
                'record': self.teams[team]['record'],
                'to_score': data['to_score'],
                'breakdown': data['breakdown']
            })

        rankings.sort(key=lambda x: x['to_score'], reverse=True)
        for rank, team_rank in enumerate(rankings, 1):
            team_rank['rank'] = rank

        return rankings

    def record1st_momentum_rankings(self) -> List[Dict]:
        """
        Algorithm 18: Record1st + Momentum
        Win% with recent form boost
        - Win%: 60%
        - Playoff prob: 15%
        - Current seed: 10%
        - Recent form (last 3 games): 15%
        """
        team_scores = {}

        for team in self.teams.keys():
            scores = {}

            scores['win_pct'] = self.teams[team]['win_pct'] * 100
            scores['playoff_prob'] = self.teams[team]['playoff_prob']

            seed = self.get_playoff_seed_rank(team)
            scores['seed'] = self.normalize_to_scale(16 - seed, 0, 15) * 100 / 15

            scores['recent_form'] = self.calculate_recent_form(team, 3) * 100

            weights = {
                'win_pct': 0.60,
                'playoff_prob': 0.15,
                'seed': 0.10,
                'recent_form': 0.15
            }

            record_score = sum(scores[key] * weights[key] for key in weights.keys())
            team_scores[team] = {'record_score': record_score, 'breakdown': scores}

        rankings = []
        for team, data in team_scores.items():
            rankings.append({
                'team': team,
                'record': self.teams[team]['record'],
                'record_score': data['record_score'],
                'breakdown': data['breakdown']
            })

        rankings.sort(key=lambda x: x['record_score'], reverse=True)
        for rank, team_rank in enumerate(rankings, 1):
            team_rank['rank'] = rank

        return rankings

    def record1st_point_diff_rankings(self) -> List[Dict]:
        """
        Algorithm 19: Record1st + Point Differential
        Win% with point differential boost
        - Win%: 60%
        - Playoff prob: 15%
        - Current seed: 10%
        - Point differential: 15%
        """
        team_scores = {}

        for team in self.teams.keys():
            scores = {}

            scores['win_pct'] = self.teams[team]['win_pct'] * 100
            scores['playoff_prob'] = self.teams[team]['playoff_prob']

            seed = self.get_playoff_seed_rank(team)
            scores['seed'] = self.normalize_to_scale(16 - seed, 0, 15) * 100 / 15

            games = self.teams[team]['wins'] + self.teams[team]['losses'] + self.teams[team]['ties']
            pd_per_game = self.teams[team]['point_diff'] / max(games, 1)
            pd_vals = [t['point_diff'] / max(t['wins'] + t['losses'] + t['ties'], 1) for t in self.teams.values()]
            scores['point_diff'] = self.normalize_to_scale(pd_per_game, min(pd_vals), max(pd_vals)) * 100

            weights = {
                'win_pct': 0.60,
                'playoff_prob': 0.15,
                'seed': 0.10,
                'point_diff': 0.15
            }

            record_score = sum(scores[key] * weights[key] for key in weights.keys())
            team_scores[team] = {'record_score': record_score, 'breakdown': scores}

        rankings = []
        for team, data in team_scores.items():
            rankings.append({
                'team': team,
                'record': self.teams[team]['record'],
                'record_score': data['record_score'],
                'breakdown': data['breakdown']
            })

        rankings.sort(key=lambda x: x['record_score'], reverse=True)
        for rank, team_rank in enumerate(rankings, 1):
            team_rank['rank'] = rank

        return rankings

    def record1st_turnover_rankings(self) -> List[Dict]:
        """
        Algorithm 20: Record1st + Turnover Edge
        Win% with turnover margin boost
        - Win%: 60%
        - Playoff prob: 15%
        - Current seed: 10%
        - Turnover margin: 15%
        """
        team_scores = {}

        for team in self.teams.keys():
            scores = {}

            scores['win_pct'] = self.teams[team]['win_pct'] * 100
            scores['playoff_prob'] = self.teams[team]['playoff_prob']

            seed = self.get_playoff_seed_rank(team)
            scores['seed'] = self.normalize_to_scale(16 - seed, 0, 15) * 100 / 15

            to_margin = self.teams[team]['turnover_margin']
            to_vals = [t['turnover_margin'] for t in self.teams.values()]
            scores['to_margin'] = self.normalize_to_scale(to_margin, min(to_vals), max(to_vals)) * 100

            weights = {
                'win_pct': 0.60,
                'playoff_prob': 0.15,
                'seed': 0.10,
                'to_margin': 0.15
            }

            record_score = sum(scores[key] * weights[key] for key in weights.keys())
            team_scores[team] = {'record_score': record_score, 'breakdown': scores}

        rankings = []
        for team, data in team_scores.items():
            rankings.append({
                'team': team,
                'record': self.teams[team]['record'],
                'record_score': data['record_score'],
                'breakdown': data['breakdown']
            })

        rankings.sort(key=lambda x: x['record_score'], reverse=True)
        for rank, team_rank in enumerate(rankings, 1):
            team_rank['rank'] = rank

        return rankings

    def record1st_sos_rankings(self) -> List[Dict]:
        """
        Algorithm 21: Record1st + Strength of Schedule
        Win% with SOS boost
        - Win%: 60%
        - Playoff prob: 15%
        - Current seed: 10%
        - Strength of schedule: 15%
        """
        team_scores = {}

        for team in self.teams.keys():
            scores = {}

            scores['win_pct'] = self.teams[team]['win_pct'] * 100
            scores['playoff_prob'] = self.teams[team]['playoff_prob']

            seed = self.get_playoff_seed_rank(team)
            scores['seed'] = self.normalize_to_scale(16 - seed, 0, 15) * 100 / 15

            sos = self.calculate_strength_of_schedule(team)
            sos_vals = [self.calculate_strength_of_schedule(t) for t in self.teams.keys()]
            scores['sos'] = self.normalize_to_scale(sos, min(sos_vals), max(sos_vals)) * 100

            weights = {
                'win_pct': 0.60,
                'playoff_prob': 0.15,
                'seed': 0.10,
                'sos': 0.15
            }

            record_score = sum(scores[key] * weights[key] for key in weights.keys())
            team_scores[team] = {'record_score': record_score, 'breakdown': scores}

        rankings = []
        for team, data in team_scores.items():
            rankings.append({
                'team': team,
                'record': self.teams[team]['record'],
                'record_score': data['record_score'],
                'breakdown': data['breakdown']
            })

        rankings.sort(key=lambda x: x['record_score'], reverse=True)
        for rank, team_rank in enumerate(rankings, 1):
            team_rank['rank'] = rank

        return rankings

    def record1st_clutch_rankings(self) -> List[Dict]:
        """
        Algorithm 22: Record1st + Clutch
        Win% with close game performance boost
        - Win%: 60%
        - Playoff prob: 15%
        - Current seed: 10%
        - Close game win% (≤8 pts): 15%
        """
        team_scores = {}

        for team in self.teams.keys():
            scores = {}

            scores['win_pct'] = self.teams[team]['win_pct'] * 100
            scores['playoff_prob'] = self.teams[team]['playoff_prob']

            seed = self.get_playoff_seed_rank(team)
            scores['seed'] = self.normalize_to_scale(16 - seed, 0, 15) * 100 / 15

            scores['close_game_pct'] = self.calculate_close_game_win_pct(team, 8) * 100

            weights = {
                'win_pct': 0.60,
                'playoff_prob': 0.15,
                'seed': 0.10,
                'close_game_pct': 0.15
            }

            record_score = sum(scores[key] * weights[key] for key in weights.keys())
            team_scores[team] = {'record_score': record_score, 'breakdown': scores}

        rankings = []
        for team, data in team_scores.items():
            rankings.append({
                'team': team,
                'record': self.teams[team]['record'],
                'record_score': data['record_score'],
                'breakdown': data['breakdown']
            })

        rankings.sort(key=lambda x: x['record_score'], reverse=True)
        for rank, team_rank in enumerate(rankings, 1):
            team_rank['rank'] = rank

        return rankings

    def record1st_division_rankings(self) -> List[Dict]:
        """
        Algorithm 23: Record1st + Division Focus
        Win% with division dominance boost
        - Win%: 55%
        - Playoff prob: 15%
        - Current seed: 10%
        - Division win%: 20%
        """
        team_scores = {}

        for team in self.teams.keys():
            scores = {}

            scores['win_pct'] = self.teams[team]['win_pct'] * 100
            scores['playoff_prob'] = self.teams[team]['playoff_prob']

            seed = self.get_playoff_seed_rank(team)
            scores['seed'] = self.normalize_to_scale(16 - seed, 0, 15) * 100 / 15

            # Read division win% from CSV
            div_win_pct = 0
            import csv
            with open(f'{self.data_dir}/team_stats.csv', 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row['team_abbr'] == team:
                        div_win_pct = float(row['div_win_pct']) if row['div_win_pct'] else 0
                        break

            scores['div_win_pct'] = div_win_pct * 100

            weights = {
                'win_pct': 0.55,
                'playoff_prob': 0.15,
                'seed': 0.10,
                'div_win_pct': 0.20
            }

            record_score = sum(scores[key] * weights[key] for key in weights.keys())
            team_scores[team] = {'record_score': record_score, 'breakdown': scores}

        rankings = []
        for team, data in team_scores.items():
            rankings.append({
                'team': team,
                'record': self.teams[team]['record'],
                'record_score': data['record_score'],
                'breakdown': data['breakdown']
            })

        rankings.sort(key=lambda x: x['record_score'], reverse=True)
        for rank, team_rank in enumerate(rankings, 1):
            team_rank['rank'] = rank

        return rankings


if __name__ == "__main__":
    # Quick test
    pr = PowerRankings()

    print("Testing Weighted Composite Rankings...")
    composite = pr.weighted_composite_rankings()
    print(f"Top 5: {[(r['team'], r['rank'], round(r['composite_score'], 2)) for r in composite[:5]]}")

    print("\nTesting Elo Dynamic Rankings...")
    elo = pr.elo_dynamic_rankings()
    print(f"Top 5: {[(r['team'], r['rank'], round(r['elo_rating'], 2)) for r in elo[:5]]}")

    print("\nTesting Multi-Factor Rankings...")
    multi = pr.multi_factor_rankings()
    print(f"Top 5: {[(r['team'], r['rank'], round(r['multi_factor_score'], 2)) for r in multi[:5]]}")
