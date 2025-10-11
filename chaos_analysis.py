"""
Chaos Analysis - Calculate volatility and multi-game scenario impacts

This module provides:
1. Team chaos scores - how volatile is each team's situation this week
2. Week chaos index - overall volatility across the league
3. Multi-game scenario analysis - "if PHI, SF, KC all lose, DET moves to #1 seed"
"""

import json
import logging
from collections import defaultdict
from itertools import combinations
from typing import Dict, List, Tuple, Any

logger = logging.getLogger(__name__)

# Seed values for expected value calculation (from generate_cache.py)
SEED_VALUES = {
    1: 100,
    2: 75,
    3: 55,
    4: 50,
    5: 40,
    6: 35,
    7: 30,
    0: 0  # Missed playoffs
}


def calculate_team_chaos_score(
    team_abbr: str,
    significant_games: List[Dict],
    current_playoff_pct: float,
    current_division_pct: float,
    current_seed: int
) -> Tuple[float, Dict[str, Any]]:
    """
    Calculate chaos score for a single team based on possible outcomes this week.

    Returns:
        tuple: (chaos_score, details_dict)
            - chaos_score: 0-100 rating of volatility
            - details_dict: breakdown of best/worst cases
    """
    if not significant_games:
        return 0.0, {
            'best_case': 'No significant games',
            'worst_case': 'No significant games',
            'explanation': 'Stable week'
        }

    # Track best and worst possible outcomes
    best_playoff_pct = current_playoff_pct
    worst_playoff_pct = current_playoff_pct
    best_division_pct = current_division_pct
    worst_division_pct = current_division_pct
    best_seed_value = SEED_VALUES.get(current_seed, 0)
    worst_seed_value = SEED_VALUES.get(current_seed, 0)

    # Analyze each significant game
    for game in significant_games:
        debug = game.get('debug_stats', {})

        # Get outcome probabilities
        home_playoff = debug.get('home_playoff_pct', current_playoff_pct)
        away_playoff = debug.get('away_playoff_pct', current_playoff_pct)
        home_division = debug.get('home_division_pct', current_division_pct)
        away_division = debug.get('away_division_pct', current_division_pct)

        # Get seed distributions
        home_seeds = debug.get('home_seeds', {})
        away_seeds = debug.get('away_seeds', {})

        # Calculate expected seed values for each outcome
        home_seed_value = sum(
            SEED_VALUES.get(int(seed), 0) * (pct / 100)
            for seed, pct in home_seeds.items()
        ) if home_seeds else best_seed_value

        away_seed_value = sum(
            SEED_VALUES.get(int(seed), 0) * (pct / 100)
            for seed, pct in away_seeds.items()
        ) if away_seeds else best_seed_value

        # Track extremes
        best_playoff_pct = max(best_playoff_pct, home_playoff, away_playoff)
        worst_playoff_pct = min(worst_playoff_pct, home_playoff, away_playoff)
        best_division_pct = max(best_division_pct, home_division, away_division)
        worst_division_pct = min(worst_division_pct, home_division, away_division)
        best_seed_value = max(best_seed_value, home_seed_value, away_seed_value)
        worst_seed_value = min(worst_seed_value, home_seed_value, away_seed_value)

    # Calculate chaos components
    playoff_swing = best_playoff_pct - worst_playoff_pct
    division_swing = best_division_pct - worst_division_pct
    seed_swing = best_seed_value - worst_seed_value

    # Weighted chaos score (0-100)
    chaos_score = min(100, (
        playoff_swing * 0.6 +           # Playoff odds most important
        division_swing * 0.3 +          # Division race matters
        (seed_swing / 100) * 50         # Seed volatility (normalized)
    ))

    # Generate human-readable descriptions
    details = {
        'chaos_score': round(chaos_score, 1),
        'playoff_swing': round(playoff_swing, 1),
        'division_swing': round(division_swing, 1),
        'seed_swing': round(seed_swing, 1),
        'best_case': {
            'playoff_pct': round(best_playoff_pct, 1),
            'division_pct': round(best_division_pct, 1),
            'seed_value': round(best_seed_value, 1)
        },
        'worst_case': {
            'playoff_pct': round(worst_playoff_pct, 1),
            'division_pct': round(worst_division_pct, 1),
            'seed_value': round(worst_seed_value, 1)
        }
    }

    # Add explanation based on chaos level
    if chaos_score >= 85:
        details['explanation'] = 'Make-or-break week'
    elif chaos_score >= 75:
        details['explanation'] = 'High stakes week'
    elif chaos_score >= 60:
        details['explanation'] = 'Notable volatility'
    else:
        details['explanation'] = 'Relatively stable'

    return round(chaos_score, 1), details


def calculate_week_chaos_index(team_chaos_scores: Dict[str, float]) -> Dict[str, Any]:
    """
    Calculate overall league chaos index for the week.

    Args:
        team_chaos_scores: Dict mapping team_abbr to chaos score

    Returns:
        dict with chaos index and highlights
    """
    if not team_chaos_scores:
        return {
            'score': 0.0,
            'description': 'No chaos data available',
            'highest_chaos_teams': []
        }

    scores = list(team_chaos_scores.values())
    avg_chaos = sum(scores) / len(scores)

    # Count high-chaos teams
    high_chaos_teams = [
        (team, score) for team, score in team_chaos_scores.items()
        if score >= 60
    ]
    high_chaos_teams.sort(key=lambda x: x[1], reverse=True)

    # Generate description
    if avg_chaos >= 65:
        description = f'Extremely volatile week - {len(high_chaos_teams)} teams face high-stakes scenarios'
    elif avg_chaos >= 50:
        description = f'Above average volatility - {len(high_chaos_teams)} teams in pivotal situations'
    elif avg_chaos >= 35:
        description = 'Moderate volatility across the league'
    else:
        description = 'Relatively calm week for most teams'

    return {
        'score': round(avg_chaos, 1),
        'description': description,
        'num_high_chaos_teams': len(high_chaos_teams),
        'highest_chaos_teams': [
            {'team': team, 'chaos_score': score}
            for team, score in high_chaos_teams[:5]
        ]
    }


def analyze_multi_game_scenarios(
    team_abbr: str,
    significant_games: List[Dict],
    teams_dict: Dict,
    schedule: List[Dict],
    current_week: int,
    standings_cache: Dict,
    current_seed: int,
    max_scenarios: int = 10,
    min_joint_probability: float = 2.0
) -> List[Dict[str, Any]]:
    """
    Analyze combinations of game outcomes and their impact on team's seed/playoff position.

    This uses a hybrid approach:
    1. Deterministic: Calculate immediate seed changes based on game outcomes
    2. Simulation: Use existing sim data to show playoff probability impacts

    Args:
        team_abbr: Team to analyze
        significant_games: Team's significant games with debug_stats
        teams_dict: All teams data
        schedule: Full schedule
        current_week: Current week number
        standings_cache: Current standings/records for all teams
        current_seed: Team's current playoff seed (0 if not in playoffs)
        max_scenarios: Maximum scenarios to return
        min_joint_probability: Minimum % probability to consider (filters rare scenarios)

    Returns:
        List of scenario dicts with impact details
    """
    if not significant_games or len(significant_games) < 2:
        logger.info(f"Not enough significant games for multi-game analysis for {team_abbr}")
        return []

    # Get this team's conference
    team_conference = teams_dict[team_abbr]['conference']

    # Filter to games that affect seeds (conference games + team's own games)
    relevant_games = []
    for game in significant_games:
        game_teams = [game['away_team'], game['home_team']]

        # Include if: team is playing, or both teams in same conference as target team
        if team_abbr in game_teams:
            relevant_games.append(game)
        elif (teams_dict[game['away_team']]['conference'] == team_conference and
              teams_dict[game['home_team']]['conference'] == team_conference):
            relevant_games.append(game)

    if len(relevant_games) < 2:
        logger.info(f"Not enough conference-relevant games for {team_abbr}")
        return []

    # Limit to top 5 most impactful games to keep computation manageable
    relevant_games = relevant_games[:5]

    scenarios = []

    # Analyze 2-game combinations
    for game1, game2 in combinations(relevant_games, 2):
        # For each combination, try all 4 outcome possibilities
        outcomes = [
            ('home', 'home'),
            ('home', 'away'),
            ('away', 'home'),
            ('away', 'away')
        ]

        for outcome1, outcome2 in outcomes:
            # Calculate joint probability
            prob1 = game1.get('home_prob', 50) if outcome1 == 'home' else (100 - game1.get('home_prob', 50))
            prob2 = game2.get('home_prob', 50) if outcome2 == 'home' else (100 - game2.get('home_prob', 50))
            joint_prob = (prob1 * prob2) / 100

            if joint_prob < min_joint_probability:
                continue

            # Determine which seed this team would have after these outcomes
            # This is complex - for now, use the simulation data as proxy
            # We'll look at the debug_stats to estimate seed changes

            # Get expected playoff % and seed distribution for each outcome
            if outcome1 == 'home':
                playoff_pct_1 = game1.get('debug_stats', {}).get('home_playoff_pct', 0)
                seeds_1 = game1.get('debug_stats', {}).get('home_seeds', {})
            else:
                playoff_pct_1 = game1.get('debug_stats', {}).get('away_playoff_pct', 0)
                seeds_1 = game1.get('debug_stats', {}).get('away_seeds', {})

            if outcome2 == 'home':
                playoff_pct_2 = game2.get('debug_stats', {}).get('home_playoff_pct', 0)
                seeds_2 = game2.get('debug_stats', {}).get('home_seeds', {})
            else:
                playoff_pct_2 = game2.get('debug_stats', {}).get('away_playoff_pct', 0)
                seeds_2 = game2.get('debug_stats', {}).get('away_seeds', {})

            # Average the two predictions (rough approximation)
            avg_playoff_pct = (playoff_pct_1 + playoff_pct_2) / 2

            # For seed, take the most likely seed from combined distributions
            # This is a simplification - proper implementation would run deterministic calculation
            combined_seeds = defaultdict(float)
            for seed, pct in seeds_1.items():
                combined_seeds[seed] += pct / 2
            for seed, pct in seeds_2.items():
                combined_seeds[seed] += pct / 2

            likely_seed = max(combined_seeds.items(), key=lambda x: x[1])[0] if combined_seeds else 0
            likely_seed = int(likely_seed)

            # Check if this creates an interesting seed change
            if likely_seed != current_seed and likely_seed > 0:
                # Build scenario description
                outcomes_desc = []
                winner1 = game1['home_team'] if outcome1 == 'home' else game1['away_team']
                winner2 = game2['home_team'] if outcome2 == 'home' else game2['away_team']

                outcomes_desc.append(f"{winner1} wins")
                outcomes_desc.append(f"{winner2} wins")

                scenarios.append({
                    'games': [
                        f"{game1['away_team']} @ {game1['home_team']}",
                        f"{game2['away_team']} @ {game2['home_team']}"
                    ],
                    'outcomes': outcomes_desc,
                    'joint_probability': round(joint_prob, 1),
                    'current_seed': current_seed,
                    'resulting_seed': likely_seed,
                    'seed_change': likely_seed - current_seed if current_seed > 0 else None,
                    'playoff_probability': round(avg_playoff_pct, 1),
                    'description': f"If {' AND '.join(outcomes_desc)}, {team_abbr} moves to #{likely_seed} seed" +
                                 (f" (from #{current_seed})" if current_seed > 0 else " (into playoffs)")
                })

    # Also check 3-game combinations if we have enough games
    if len(relevant_games) >= 3:
        for game1, game2, game3 in list(combinations(relevant_games, 3))[:10]:  # Limit combinations
            # Only check specific high-impact scenarios (all favorites win, all underdogs win)
            interesting_outcome_sets = [
                ('home', 'home', 'home'),  # All home teams win
                ('away', 'away', 'away'),  # All away teams win
            ]

            for outcome1, outcome2, outcome3 in interesting_outcome_sets:
                prob1 = game1.get('home_prob', 50) if outcome1 == 'home' else (100 - game1.get('home_prob', 50))
                prob2 = game2.get('home_prob', 50) if outcome2 == 'home' else (100 - game2.get('home_prob', 50))
                prob3 = game3.get('home_prob', 50) if outcome3 == 'home' else (100 - game3.get('home_prob', 50))
                joint_prob = (prob1 * prob2 * prob3) / 10000

                if joint_prob < min_joint_probability:
                    continue

                # Similar analysis as 2-game
                outcomes_desc = []
                winner1 = game1['home_team'] if outcome1 == 'home' else game1['away_team']
                winner2 = game2['home_team'] if outcome2 == 'home' else game2['away_team']
                winner3 = game3['home_team'] if outcome3 == 'home' else game3['away_team']

                outcomes_desc.append(f"{winner1} wins")
                outcomes_desc.append(f"{winner2} wins")
                outcomes_desc.append(f"{winner3} wins")

                # For 3-game combos, use a more conservative estimate
                # Just note this is a high-volatility scenario
                scenarios.append({
                    'games': [
                        f"{game1['away_team']} @ {game1['home_team']}",
                        f"{game2['away_team']} @ {game2['home_team']}",
                        f"{game3['away_team']} @ {game3['home_team']}"
                    ],
                    'outcomes': outcomes_desc,
                    'joint_probability': round(joint_prob, 1),
                    'current_seed': current_seed,
                    'resulting_seed': None,  # Don't estimate for 3-game combos
                    'seed_change': None,
                    'playoff_probability': None,
                    'description': f"If {' AND '.join(outcomes_desc)}, {team_abbr}'s playoff picture could shift significantly"
                })

    # Sort by joint probability * impact
    scenarios.sort(key=lambda x: (
        abs(x.get('seed_change', 0) or 0) * x['joint_probability']
    ), reverse=True)

    # Return top scenarios
    return scenarios[:max_scenarios]


def add_chaos_context_to_prompt(
    base_prompt: str,
    chaos_score: float,
    chaos_details: Dict[str, Any],
    team_abbr: str
) -> str:
    """
    Conditionally add chaos context to team analysis prompt.
    Only adds context if chaos_score >= 60.

    Args:
        base_prompt: Existing prompt text
        chaos_score: Team's chaos score (0-100)
        chaos_details: Details dict from calculate_team_chaos_score
        team_abbr: Team abbreviation

    Returns:
        Modified prompt with chaos context added (if applicable)
    """
    if chaos_score < 60:
        return base_prompt

    chaos_context = f"""
{'=' * 70}
CHAOS ALERT - HIGH STAKES WEEK
{'=' * 70}
This is an unusually volatile week for {team_abbr}. Their playoff situation could swing dramatically.

Chaos Score: {chaos_score}/100 ({chaos_details.get('explanation', 'High volatility')})

Best Case Scenario:
- Playoff odds: {chaos_details['best_case']['playoff_pct']}%
- Division odds: {chaos_details['best_case']['division_pct']}%

Worst Case Scenario:
- Playoff odds: {chaos_details['worst_case']['playoff_pct']}%
- Division odds: {chaos_details['worst_case']['division_pct']}%

Potential Swing: {chaos_details.get('playoff_swing', 0):.1f}% playoff odds

GUIDANCE: You may reference these high stakes in your analysis, but don't overdo it.
Let the drama emerge naturally from the stats and context. Consider mentioning:
- The urgency/importance of this week (if chaos >= 85)
- What's at stake in upcoming games (if chaos >= 75)
- The volatility of their situation (if chaos >= 60)

{'=' * 70}

"""

    # Insert chaos context after the role/voice section
    # Find a good insertion point (after CRITICAL RULES or similar)
    insertion_markers = [
        "CRITICAL RULES",
        "OUTPUT REQUIREMENTS",
        "TEAM STATISTICS"
    ]

    for marker in insertion_markers:
        if marker in base_prompt:
            parts = base_prompt.split(marker, 1)
            return parts[0] + chaos_context + marker + parts[1]

    # If no marker found, append at end
    return base_prompt + "\n" + chaos_context
