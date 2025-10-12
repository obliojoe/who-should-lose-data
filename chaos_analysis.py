"""
Chaos Analysis - Calculate volatility for teams and the league

This module provides:
1. Team chaos scores - how volatile is each team's situation this week
2. Week chaos index - overall volatility across the league
"""

import json
import logging
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
