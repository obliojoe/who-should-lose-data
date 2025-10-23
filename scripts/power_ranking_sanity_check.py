#!/usr/bin/env python3
"""Lightweight power-ranking comparison tool.

Runs a handful of simple ranking approaches side-by-side with the
production R1+SOV algorithm so we can occasionally sanity-check how
alternate methods behave as the season evolves.

The script intentionally keeps everything in one place and avoids any
external dependencies – run it whenever fresh data lands in `data/`.

Example:
    python scripts/power_ranking_sanity_check.py --top 5
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import sys
import json

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from power_rankings import PowerRankings


def _normalize(value: float, min_val: float, max_val: float) -> float:
    if max_val == min_val:
        return 50.0
    return ((value - min_val) / (max_val - min_val)) * 100


def _load_latest_sagarin() -> Dict[str, float]:
    sagarin_path = ROOT / 'data' / 'sagarin.json'
    if not sagarin_path.exists():
        return {}
    with sagarin_path.open('r', encoding='utf-8') as fh:
        payload = json.load(fh)
    ratings_payload = payload.get('ratings')
    if isinstance(ratings_payload, dict):
        return {team: float(info.get('rating', 0.0)) for team, info in ratings_payload.items()}

    ratings_history = payload.get('history')
    if isinstance(ratings_history, list) and ratings_history:
        latest = max(ratings_history, key=lambda entry: entry.get('week', 0))
        return {abbr: float(score) for abbr, score in latest.get('ratings', [])}

    return {}


def _team_game_results(pr: PowerRankings) -> Dict[str, List[float]]:
    results: Dict[str, List[float]] = {team: [] for team in pr.teams}
    games = sorted(pr.schedule, key=lambda g: (g['week'], g['home_team'], g['away_team']))
    for game in games:
        home = game['home_team']
        away = game['away_team']
        hs = game['home_score']
        as_ = game['away_score']
        if hs > as_:
            results[home].append(1.0)
            results[away].append(0.0)
        elif hs < as_:
            results[away].append(1.0)
            results[home].append(0.0)
        else:
            results[home].append(0.5)
            results[away].append(0.5)
    return results


def _recent_form(team: str, results_map: Dict[str, List[float]], window: int = 3) -> float:
    games = results_map.get(team, [])
    if not games:
        return 50.0
    recent = games[-window:]
    return sum(recent) / len(recent) * 100


def _spearman_rank_correlation(a: Dict[str, int], b: Dict[str, int]) -> float:
    """Return Spearman rank correlation between two rank maps.

    We assume each team has a unique rank. Implementation sticks to the
    closed-form formula to avoid pulling in SciPy.
    """

    common = [team for team in a.keys() if team in b]
    n = len(common)
    if n < 2:
        return 0.0

    diff_sq_sum = sum((a[team] - b[team]) ** 2 for team in common)
    return 1 - (6 * diff_sq_sum) / (n * (n**2 - 1))


def _rankings_to_map(rows: Iterable[Dict[str, int]]) -> Dict[str, int]:
    """Convert rankings list to {team: rank}."""

    return {row['team']: row['rank'] for row in rows}


def _attach_ranks(rows: List[Dict]) -> List[Dict]:
    """Assign 1-based ranks in-place based on existing ordering."""

    for idx, row in enumerate(rows, 1):
        row['rank'] = idx
    return rows


def _rank_by_record(pr: PowerRankings) -> List[Dict]:
    rows = []
    for team, stats in pr.teams.items():
        rows.append(
            {
                'team': team,
                'record': stats['record'],
                'score': stats['win_pct'] * 100,
                'point_diff': stats['point_diff'],
                'playoff_prob': pr.playoff_probs.get(team, 0.0),
            }
        )
    rows.sort(key=lambda r: (r['score'], r['point_diff'], r['playoff_prob']), reverse=True)
    return _attach_ranks(rows)


def _rank_by_point_diff(pr: PowerRankings) -> List[Dict]:
    rows = []
    for team, stats in pr.teams.items():
        rows.append(
            {
                'team': team,
                'record': stats['record'],
                'score': stats['point_diff'],
                'win_pct': stats['win_pct'] * 100,
            }
        )
    rows.sort(key=lambda r: (r['score'], r['win_pct']), reverse=True)
    return _attach_ranks(rows)


def _rank_by_strength_of_victory(pr: PowerRankings) -> List[Dict]:
    sov_values = [pr.calculate_strength_of_victory(team) for team in pr.teams]
    min_sov = min(sov_values)
    max_sov = max(sov_values)

    rows = []
    for team, stats in pr.teams.items():
        sov = pr.calculate_strength_of_victory(team)
        sov_score = _normalize(sov, min_sov, max_sov)
        rows.append(
            {
                'team': team,
                'record': stats['record'],
                'score': sov_score,
                'raw_sov': sov,
            }
        )

    rows.sort(key=lambda r: (r['score'], r['raw_sov']), reverse=True)
    return _attach_ranks(rows)


def _rank_by_blended(pr: PowerRankings) -> List[Dict]:
    """Simple alternate blend (record + point diff + SOV + playoff odds)."""

    diff_values = [stats['point_diff'] for stats in pr.teams.values()]
    min_diff = min(diff_values)
    max_diff = max(diff_values)

    sov_values = [pr.calculate_strength_of_victory(team) for team in pr.teams]
    min_sov = min(sov_values)
    max_sov = max(sov_values)

    rows = []
    for team, stats in pr.teams.items():
        win_score = stats['win_pct'] * 100
        diff_score = _normalize(stats['point_diff'], min_diff, max_diff)
        sov_score = _normalize(pr.calculate_strength_of_victory(team), min_sov, max_sov)
        playoff_score = pr.playoff_probs.get(team, 0.0)

        composite = (
            win_score * 0.50
            + diff_score * 0.20
            + sov_score * 0.20
            + playoff_score * 0.10
        )

        rows.append(
            {
                'team': team,
                'record': stats['record'],
                'score': composite,
                'win_pct': win_score,
                'point_diff_score': diff_score,
                'sov_score': sov_score,
                'playoff_prob': playoff_score,
            }
        )

    rows.sort(key=lambda r: r['score'], reverse=True)
    return _attach_ranks(rows)


def _rank_by_blended_with_sagarin(pr: PowerRankings) -> List[Dict]:
    """Blend that adds a 10% Sagarin component (others scaled by 0.9)."""

    diff_values = [stats['point_diff'] for stats in pr.teams.values()]
    min_diff = min(diff_values)
    max_diff = max(diff_values)

    sov_values = [pr.calculate_strength_of_victory(team) for team in pr.teams]
    min_sov = min(sov_values)
    max_sov = max(sov_values)

    sagarin_map = _load_latest_sagarin()
    sag_values = list(sagarin_map.values())
    if sag_values:
        min_sag = min(sag_values)
        max_sag = max(sag_values)
    else:
        min_sag = 0.0
        max_sag = 1.0

    rows = []
    for team, stats in pr.teams.items():
        win_score = stats['win_pct'] * 100
        diff_score = _normalize(stats['point_diff'], min_diff, max_diff)
        sov_score = _normalize(pr.calculate_strength_of_victory(team), min_sov, max_sov)
        playoff_score = pr.playoff_probs.get(team, 0.0)
        sag_score = _normalize(sagarin_map.get(team, min_sag), min_sag, max_sag)

        composite = (
            win_score * 0.45
            + diff_score * 0.18
            + sov_score * 0.18
            + playoff_score * 0.09
            + sag_score * 0.10
        )

        rows.append(
            {
                'team': team,
                'record': stats['record'],
                'score': composite,
                'win_pct': win_score,
                'point_diff_score': diff_score,
                'sov_score': sov_score,
                'playoff_prob': playoff_score,
                'sagarin_score': sag_score,
            }
        )

    rows.sort(key=lambda r: r['score'], reverse=True)
    return _attach_ranks(rows)


def _print_top(method: str, rows: List[Dict], top_n: int) -> None:
    print(f"\n{method} — Top {top_n}")
    print("-" * 40)
    for row in rows[:top_n]:
        score = row.get('score', row.get('composite_score', 0.0))
        print(f"#{row['rank']:>2} {row['team']:<3} {row['record']:<7} | score={score:6.2f}")


def _rank_weighted_composite(pr: PowerRankings, weights: Dict[str, float]) -> List[Dict]:
    sagarin_map = _load_latest_sagarin()
    diff_values = [stats['point_diff'] for stats in pr.teams.values()]
    min_diff, max_diff = min(diff_values), max(diff_values)

    epa_values = [stats.get('total_epa', 0.0) for stats in pr.teams.values()]
    min_epa, max_epa = min(epa_values), max(epa_values)

    to_margin = [stats.get('turnover_margin', 0) for stats in pr.teams.values()]
    min_to, max_to = min(to_margin), max(to_margin)

    results_map = _team_game_results(pr)

    sag_values = sagarin_map.values()
    min_sag = min(sag_values) if sag_values else 0.0
    max_sag = max(sag_values) if sag_values else 1.0

    rows: List[Dict] = []
    for team, stats in pr.teams.items():
        win_score = stats['win_pct'] * 100
        point_diff_score = _normalize(stats['point_diff'], min_diff, max_diff)
        epa_score = _normalize(stats.get('total_epa', 0.0), min_epa, max_epa)
        turnover_score = _normalize(stats.get('turnover_margin', 0.0), min_to, max_to)
        recent_score = _recent_form(team, results_map)
        playoff_score = pr.playoff_probs.get(team, 0.0)
        sagarin_score = _normalize(sagarin_map.get(team, min_sag), min_sag, max_sag)

        components = {
            'sagarin': sagarin_score,
            'playoff_prob': playoff_score,
            'point_diff': point_diff_score,
            'recent_form': recent_score,
            'record': win_score,
            'epa': epa_score,
            'turnovers': turnover_score,
        }

        composite = sum(components[key] * weights.get(key, 0.0) for key in components)

        rows.append(
            {
                'team': team,
                'record': stats['record'],
                'score': composite,
                'components': components,
            }
        )

    rows.sort(key=lambda r: r['score'], reverse=True)
    return _attach_ranks(rows)


def _elo_expected(rating_a: float, rating_b: float) -> float:
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def _rank_elo(pr: PowerRankings) -> List[Dict]:
    ratings: Dict[str, float] = {team: 1500.0 for team in pr.teams}
    initial = ratings.copy()
    games = sorted(pr.schedule, key=lambda g: (g['week'], g['home_team'], g['away_team']))

    for game in games:
        home, away = game['home_team'], game['away_team']
        hs, as_ = game['home_score'], game['away_score']
        margin = abs(hs - as_)
        if margin == 0:
            outcome_home = 0.5
        elif hs > as_:
            outcome_home = 1.0
        else:
            outcome_home = 0.0

        home_adv = 55.0
        expected_home = _elo_expected(ratings[home] + home_adv, ratings[away])
        expected_away = 1 - expected_home

        k = 25.0 * (1 + 0.1 * max(0, margin - 1))

        ratings[home] += k * (outcome_home - expected_home)
        ratings[away] += k * ((1 - outcome_home) - expected_away)

    rows = []
    for team, rating in ratings.items():
        rows.append(
            {
                'team': team,
                'record': pr.teams[team]['record'],
                'score': rating,
                'elo_change': rating - initial[team],
            }
        )

    rows.sort(key=lambda r: r['score'], reverse=True)
    return _attach_ranks(rows)


def _rank_multi_factor(pr: PowerRankings) -> List[Dict]:
    sagarin_map = _load_latest_sagarin()
    results_map = _team_game_results(pr)

    def collect(metric: str) -> Tuple[float, float]:
        values = [pr.teams[team].get(metric, 0.0) for team in pr.teams]
        return min(values), max(values)

    min_diff, max_diff = collect('point_diff')
    min_epa, max_epa = collect('total_epa')
    min_pf, max_pf = collect('points_for')
    min_sov = min(pr.calculate_strength_of_victory(team) for team in pr.teams)
    max_sov = max(pr.calculate_strength_of_victory(team) for team in pr.teams)
    sag_values = sagarin_map.values()
    min_sag = min(sag_values) if sag_values else 0.0
    max_sag = max(sag_values) if sag_values else 1.0

    rows = []
    for team, stats in pr.teams.items():
        record_score = stats['win_pct'] * 100
        performance_score = _normalize(stats['point_diff'], min_diff, max_diff)
        efficiency_score = _normalize(stats.get('total_epa', 0.0), min_epa, max_epa)
        momentum_score = _recent_form(team, results_map)
        predictive_score = pr.playoff_probs.get(team, 0.0)
        market_score = _normalize(sagarin_map.get(team, min_sag), min_sag, max_sag)
        sov_score = _normalize(pr.calculate_strength_of_victory(team), min_sov, max_sov)
        explosive_score = _normalize(stats['points_for'], min_pf, max_pf)

        components = {
            'record': record_score,
            'performance': performance_score,
            'efficiency': efficiency_score,
            'momentum': momentum_score,
            'predictive': predictive_score,
            'market': market_score,
            'sov': sov_score,
            'scoring': explosive_score,
        }

        overall = sum(components.values()) / len(components)

        rows.append(
            {
                'team': team,
                'record': stats['record'],
                'score': overall,
                'components': components,
            }
        )

    rows.sort(key=lambda r: r['score'], reverse=True)
    return _attach_ranks(rows)
def main() -> None:
    parser = argparse.ArgumentParser(description="Compare power ranking heuristics against R1+SOV")
    parser.add_argument('--top', type=int, default=8, help='How many teams to print for each method (default: 8)')
    args = parser.parse_args()

    pr = PowerRankings()

    baseline = pr.r1_sov_rankings()
    baseline_map = _rankings_to_map(baseline)

    base_weights = {
        'sagarin': 0.15,
        'playoff_prob': 0.25,
        'point_diff': 0.20,
        'recent_form': 0.15,
        'record': 0.10,
        'epa': 0.10,
        'turnovers': 0.05,
    }

    low_sagarin_weights = {
        **base_weights,
        'sagarin': 0.05,
        'point_diff': 0.25,
        'recent_form': 0.20,
    }

    no_sagarin_weights = {
        **base_weights,
        'sagarin': 0.0,
        'playoff_prob': 0.30,
        'point_diff': 0.25,
        'recent_form': 0.20,
    }

    candidates = [
        ('R1+SOV + point diff', pr.r1_sov_pointdiff_rankings()),
        ('Record (win pct)', _rank_by_record(pr)),
        ('Point differential', _rank_by_point_diff(pr)),
        ('Strength of victory', _rank_by_strength_of_victory(pr)),
        ('Blended (50/20/20/10)', _rank_by_blended(pr)),
        ('Blended (+10% Sagarin)', _rank_by_blended_with_sagarin(pr)),
        ('Weighted composite (15% Sag)', _rank_weighted_composite(pr, base_weights)),
        ('Weighted composite (5% Sag)', _rank_weighted_composite(pr, low_sagarin_weights)),
        ('Weighted composite (0% Sag)', _rank_weighted_composite(pr, no_sagarin_weights)),
        ('Elo dynamic', _rank_elo(pr)),
        ('Multi-factor snapshot', _rank_multi_factor(pr)),
    ]

    print("Baseline: R1+SOV algorithm\n")
    _print_top('R1+SOV', baseline, args.top)

    print("\nComparison summary vs R1+SOV:")
    print("Method                             | Spearman | Avg Δ rank")
    print("-----------------------------------+----------+-----------")

    for label, rankings in candidates:
        rank_map = _rankings_to_map(rankings)
        corr = _spearman_rank_correlation(baseline_map, rank_map)
        avg_delta = sum(abs(baseline_map[team] - rank_map[team]) for team in baseline_map) / len(baseline_map)
        print(f"{label:35s} | {corr:8.3f} | {avg_delta:9.2f}")
        _print_top(label, rankings, args.top)


if __name__ == "__main__":
    main()
