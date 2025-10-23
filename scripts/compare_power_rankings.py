#!/usr/bin/env python3
"""
Power Rankings Comparison Script

Generates all four power ranking algorithms and creates a comprehensive
comparison report showing how different methods rank teams.

Usage:
    python compare_power_rankings.py [--include-ai]

Options:
    --include-ai: Include AI-enhanced rankings (slower due to API call)
"""

import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from power_rankings import PowerRankings
from scipy import stats


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

    history = payload.get('history')
    if isinstance(history, list) and history:
        latest = max(history, key=lambda entry: entry.get('week', 0))
        return {abbr: float(score) for abbr, score in latest.get('ratings', [])}

    return {}


def _team_game_results(pr: PowerRankings) -> Dict[str, List[float]]:
    results: Dict[str, List[float]] = {team: [] for team in pr.teams}
    games = sorted(pr.schedule, key=lambda g: (g['week'], g['home_team'], g['away_team']))
    for game in games:
        home, away = game['home_team'], game['away_team']
        hs, as_ = game['home_score'], game['away_score']
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


def _recent_form(team: str, result_map: Dict[str, List[float]], window: int = 3) -> float:
    games = result_map.get(team, [])
    if not games:
        return 50.0
    recent = games[-window:]
    return sum(recent) / len(recent) * 100


def calculate_correlation(ranks1, ranks2):
    """Calculate Spearman correlation between two ranking lists"""
    # Create mapping of team to rank for each list
    team_to_rank1 = {r['team']: r['rank'] for r in ranks1}
    team_to_rank2 = {r['team']: r['rank'] for r in ranks2}

    # Get common teams
    common_teams = set(team_to_rank1.keys()) & set(team_to_rank2.keys())

    if len(common_teams) < 2:
        return 0.0

    # Extract ranks for common teams
    ranks_a = [team_to_rank1[team] for team in common_teams]
    ranks_b = [team_to_rank2[team] for team in common_teams]

    # Calculate Spearman correlation
    correlation, _ = stats.spearmanr(ranks_a, ranks_b)

    return correlation


def _rank_weighted_composite(pr: PowerRankings, weights: Dict[str, float]) -> List[Dict]:
    sagarin_map = _load_latest_sagarin()
    diff_values = [stats['point_diff'] for stats in pr.teams.values()]
    min_diff, max_diff = min(diff_values), max(diff_values)

    epa_values = [stats.get('total_epa', 0.0) for stats in pr.teams.values()]
    min_epa, max_epa = min(epa_values), max(epa_values)

    to_values = [stats.get('turnover_margin', 0.0) for stats in pr.teams.values()]
    min_to, max_to = min(to_values), max(to_values)

    results_map = _team_game_results(pr)

    sagarin_values = list(sagarin_map.values())
    min_sag = min(sagarin_values) if sagarin_values else 0.0
    max_sag = max(sagarin_values) if sagarin_values else 1.0

    rows = []
    for team, stats in pr.teams.items():
        win_score = stats['win_pct'] * 100
        point_diff_score = _normalize(stats['point_diff'], min_diff, max_diff)
        recent_score = _recent_form(team, results_map)
        playoff_score = pr.playoff_probs.get(team, 0.0)
        epa_score = _normalize(stats.get('total_epa', 0.0), min_epa, max_epa)
        turnover_score = _normalize(stats.get('turnover_margin', 0.0), min_to, max_to)
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

        composite_score = sum(components[key] * weights.get(key, 0.0) for key in components)

        rows.append(
            {
                'team': team,
                'record': stats['record'],
                'composite_score': composite_score,
                'breakdown': components,
            }
        )

    rows.sort(key=lambda r: r['composite_score'], reverse=True)
    for idx, row in enumerate(rows, 1):
        row['rank'] = idx
    return rows


def _elo_expected(rating_a: float, rating_b: float) -> float:
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def _rank_elo(pr: PowerRankings) -> List[Dict]:
    ratings = {team: 1500.0 for team in pr.teams}
    starting = ratings.copy()
    games = sorted(pr.schedule, key=lambda g: (g['week'], g['home_team'], g['away_team']))

    for game in games:
        home, away = game['home_team'], game['away_team']
        hs, as_ = game['home_score'], game['away_score']
        if hs == as_:
            outcome_home = 0.5
        elif hs > as_:
            outcome_home = 1.0
        else:
            outcome_home = 0.0

        home_adv = 55.0
        expected_home = _elo_expected(ratings[home] + home_adv, ratings[away])
        expected_away = 1 - expected_home

        margin = abs(hs - as_)
        k = 25.0 * (1 + 0.1 * max(0, margin - 1))

        ratings[home] += k * (outcome_home - expected_home)
        ratings[away] += k * ((1 - outcome_home) - expected_away)

    rows = []
    for team, rating in ratings.items():
        rows.append(
            {
                'team': team,
                'record': pr.teams[team]['record'],
                'elo_rating': rating,
                'elo_change': rating - starting[team],
            }
        )

    rows.sort(key=lambda r: r['elo_rating'], reverse=True)
    for idx, row in enumerate(rows, 1):
        row['rank'] = idx
    return rows


def _rank_multi_factor(pr: PowerRankings) -> List[Dict]:
    sagarin_map = _load_latest_sagarin()
    results_map = _team_game_results(pr)

    def _collect(metric: str) -> Tuple[float, float]:
        values = [pr.teams[team].get(metric, 0.0) for team in pr.teams]
        return min(values), max(values)

    min_diff, max_diff = _collect('point_diff')
    min_epa, max_epa = _collect('total_epa')
    min_pf, max_pf = _collect('points_for')
    sov_values = [pr.calculate_strength_of_victory(team) for team in pr.teams]
    min_sov, max_sov = min(sov_values), max(sov_values)
    sagarin_values = list(sagarin_map.values())
    min_sag = min(sagarin_values) if sagarin_values else 0.0
    max_sag = max(sagarin_values) if sagarin_values else 1.0

    rows = []
    for team, stats in pr.teams.items():
        components = {
            'record': stats['win_pct'] * 100,
            'performance': _normalize(stats['point_diff'], min_diff, max_diff),
            'efficiency': _normalize(stats.get('total_epa', 0.0), min_epa, max_epa),
            'momentum': _recent_form(team, results_map),
            'predictive': pr.playoff_probs.get(team, 0.0),
            'market': _normalize(sagarin_map.get(team, min_sag), min_sag, max_sag),
            'sov': _normalize(pr.calculate_strength_of_victory(team), min_sov, max_sov),
            'scoring': _normalize(stats['points_for'], min_pf, max_pf),
        }

        overall = sum(components.values()) / len(components)

        rows.append(
            {
                'team': team,
                'record': stats['record'],
                'multi_factor_score': overall,
                'breakdown': components,
            }
        )

    rows.sort(key=lambda r: r['multi_factor_score'], reverse=True)
    for idx, row in enumerate(rows, 1):
        row['rank'] = idx
    return rows


def _ai_enhanced_rankings(base_rankings: List[Dict], model_alias: str = 'sonnet-3.7') -> List[Dict]:
    try:
        from ai_service import AIService, resolve_model_name
    except ImportError as exc:
        raise RuntimeError('AI service module not available') from exc

    service = AIService(model_override=resolve_model_name(model_alias))
    if not service.client:
        raise RuntimeError('AI client unavailable - configure CLAUDE_API_KEY or OPENAI_API_KEY')

    context_rows = []
    for row in base_rankings[:20]:
        breakdown = row.get('breakdown', {})
        context_rows.append({
            'team': row['team'],
            'rank': row['rank'],
            'record': row['record'],
            'composite_score': round(row.get('composite_score', 0.0), 2),
            'win_pct_score': round(breakdown.get('record', 0.0), 2),
            'point_diff_score': round(breakdown.get('point_diff', 0.0), 2),
            'playoff_prob_score': round(breakdown.get('playoff_prob', 0.0), 2),
            'recent_form_score': round(breakdown.get('recent_form', 0.0), 2),
            'sagarin_score': round(breakdown.get('sagarin', 0.0), 2),
        })

    prompt_payload = {
        'instructions': (
            "You are reviewing NFL power rankings. You may adjust a small number of teams "
            "by at most ±3 positions to account for qualitative factors (injuries, eye test, "
            "coaching, momentum)."
        ),
        'constraints': [
            'Never move a team more than 3 spots from its current rank.',
            'Only adjust if you have a concrete reason.',
            'Prefer leaving rankings unchanged when uncertain.',
            'Return JSON with keys "summary" (string) and "adjustments" (list).',
            'Each adjustment must include team abbreviation, new_rank (1-32), and reason.',
        ],
        'base_rankings': context_rows,
    }

    prompt_text = json.dumps(prompt_payload, indent=2)
    response, status = service.generate_analysis(prompt_text)
    if status != 'success':
        raise RuntimeError(response)

    try:
        parsed = json.loads(response)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f'AI response was not valid JSON: {exc}') from exc

    adjustments = parsed.get('adjustments', []) or []
    summary = parsed.get('summary')

    base_rank_map = {row['team']: row['rank'] for row in base_rankings}
    reason_map: Dict[str, str] = {row['team']: 'No adjustment' for row in base_rankings}
    target_rank_map: Dict[str, int] = {row['team']: row['rank'] for row in base_rankings}

    for adj in adjustments:
        team = adj.get('team') or adj.get('team_abbr')
        if team not in base_rank_map:
            continue
        try:
            new_rank = int(adj.get('new_rank'))
        except (TypeError, ValueError):
            continue
        base_rank = base_rank_map[team]
        bounded = max(1, min(32, new_rank))
        if abs(bounded - base_rank) > 3:
            bounded = max(base_rank - 3, min(base_rank + 3, bounded))
        target_rank_map[team] = bounded
        reason = adj.get('reason') or adj.get('rationale') or 'Adjustment applied'
        reason_map[team] = reason

    enriched: List[Dict] = []
    for row in base_rankings:
        team = row['team']
        enriched.append({
            'team': team,
            'record': row['record'],
            'composite_score': row.get('composite_score'),
            'rank': row['rank'],
            'ai_adjusted_rank': target_rank_map[team],
            'ai_reasoning': reason_map[team],
        })

    enriched.sort(key=lambda r: (r['ai_adjusted_rank'], r['rank']))
    for idx, row in enumerate(enriched, 1):
        row['ai_adjusted_rank'] = idx
    if summary and enriched:
        enriched[0]['ai_summary'] = summary

    return enriched


BASE_COMPOSITE_WEIGHTS = {
    'sagarin': 0.15,
    'playoff_prob': 0.25,
    'point_diff': 0.20,
    'recent_form': 0.15,
    'record': 0.10,
    'epa': 0.10,
    'turnovers': 0.05,
}

LOW_SAG_WEIGHTS = {
    'sagarin': 0.05,
    'playoff_prob': 0.25,
    'point_diff': 0.25,
    'recent_form': 0.20,
    'record': 0.10,
    'epa': 0.10,
    'turnovers': 0.05,
}

NO_SAG_WEIGHTS = {
    'sagarin': 0.00,
    'playoff_prob': 0.30,
    'point_diff': 0.25,
    'recent_form': 0.20,
    'record': 0.10,
    'epa': 0.10,
    'turnovers': 0.05,
}

def generate_comparison_report(pr, include_ai=False):
    """Generate comprehensive comparison report"""

    print("Calculating power rankings using multiple algorithms...")
    print()

    # Run all algorithms
    print("1. Running baseline R1+SOV algorithm...")
    r1_sov = pr.r1_sov_rankings()

    print("2. Running R1+SOV with point differential (experimental)...")
    r1_sov_pd = pr.r1_sov_pointdiff_rankings()

    print("3. Running Weighted Composite algorithm (15% Sagarin)...")
    composite = _rank_weighted_composite(pr, BASE_COMPOSITE_WEIGHTS)

    print("4. Running Weighted Composite - Low Sagarin (5%)...")
    composite_low_sag = _rank_weighted_composite(pr, LOW_SAG_WEIGHTS)

    print("5. Running Weighted Composite - No Sagarin (0%)...")
    composite_no_sag = _rank_weighted_composite(pr, NO_SAG_WEIGHTS)

    print("6. Running Elo Dynamic algorithm...")
    elo = _rank_elo(pr)

    print("7. Running Multi-Factor algorithm...")
    multifactor = _rank_multi_factor(pr)

    ai_enhanced = None
    if include_ai:
        print("8. Running AI-Enhanced algorithm (this may take a minute)...")
        try:
            ai_enhanced = _ai_enhanced_rankings(composite, model_alias='sonnet-3.7')
        except Exception as e:
            print(f"   WARNING: AI enhancement failed: {e}")
            print("   Continuing without AI rankings...")

    print()
    print("Generating comparison report...")

    # Build markdown report
    report_lines = []

    # Header
    report_lines.append("# NFL Power Rankings Comparison - Week 7, 2025")
    report_lines.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Algorithm 1: R1+SOV
    report_lines.append("## Algorithm 1: R1+SOV (baseline)")
    report_lines.append("\n**Methodology:** Record-first + strength of victory")
    report_lines.append("- Win%: 60%, Playoff Prob: 15%, Seed: 10%, Strength of Victory: 15%\n")

    report_lines.append("| Rank | Team | Record | Score | Win% | Playoff% | Seed | SOV |")
    report_lines.append("|------|------|--------|-------|------|----------|------|-----|")

    for team_data in r1_sov[:15]:
        bd = team_data['breakdown']
        report_lines.append(
            f"| {team_data['rank']:2d} | {team_data['team']:3s} | "
            f"{team_data['record']:5s} | {team_data['composite_score']:5.1f} | "
            f"{bd['win_pct']:5.1f} | {bd['playoff_prob']:5.1f} | "
            f"{bd['seed']:5.1f} | {bd['sov']:5.1f} |"
        )

    report_lines.append("\n---\n")

    # Algorithm 2: R1+SOV + Point Differential
    report_lines.append("## Algorithm 2: R1+SOV + Point Differential (experimental)")
    report_lines.append("\n**Methodology:** Baseline R1+SOV with 10% point differential contribution")
    report_lines.append("- Win%: 50%, Playoff Prob: 15%, Seed: 10%, SOV: 15%, Point Diff: 10%\n")

    report_lines.append("| Rank | Team | Record | Score | Win% | Playoff% | Seed | SOV | Pt Diff |")
    report_lines.append("|------|------|--------|-------|------|----------|------|-----|---------|")

    for team_data in r1_sov_pd[:15]:
        bd = team_data['breakdown']
        report_lines.append(
            f"| {team_data['rank']:2d} | {team_data['team']:3s} | "
            f"{team_data['record']:5s} | {team_data['composite_score']:5.1f} | "
            f"{bd['win_pct']:5.1f} | {bd['playoff_prob']:5.1f} | "
            f"{bd['seed']:5.1f} | {bd['sov']:5.1f} | {bd['point_diff']:7.1f} |"
        )

    report_lines.append("\n---\n")

    # Algorithm 3: Weighted Composite
    report_lines.append("## Algorithm 3: Weighted Composite")
    report_lines.append("\n**Methodology:** Weighted blend of multiple metrics")
    report_lines.append("- Sagarin: 15%, Playoff Prob: 25%, Point Diff: 20%, Recent Form: 15%,")
    report_lines.append("  Record Quality: 10%, EPA: 10%, Turnovers: 5%\n")

    report_lines.append("| Rank | Team | Record | Score | Sagarin | Playoff% | Pt Diff | Form |")
    report_lines.append("|------|------|--------|-------|---------|----------|---------|------|")

    for team_data in composite[:15]:
        bd = team_data['breakdown']
        report_lines.append(
            f"| {team_data['rank']:2d} | {team_data['team']:3s} | "
            f"{team_data['record']:5s} | {team_data['composite_score']:5.1f} | "
            f"{bd['sagarin']:4.1f} | {bd['playoff_prob']:5.1f} | "
            f"{bd['point_diff']:5.1f} | {bd['recent_form']:5.1f} |"
        )

    report_lines.append("\n---\n")

    # Algorithm 2: Weighted Composite - Low Sagarin
    report_lines.append("## Algorithm 2: Weighted Composite - Low Sagarin (5%)")
    report_lines.append("\n**Methodology:** Weighted blend with reduced Sagarin influence")
    report_lines.append("- Sagarin: 5% (-10%), Playoff Prob: 25%, Point Diff: 25% (+5%), Recent Form: 20% (+5%),")
    report_lines.append("  Record Quality: 10%, EPA: 10%, Turnovers: 5%\n")

    report_lines.append("| Rank | Team | Record | Score | Sagarin | Playoff% | Pt Diff | Form |")
    report_lines.append("|------|------|--------|-------|---------|----------|---------|------|")

    for team_data in composite_low_sag[:15]:
        bd = team_data['breakdown']
        report_lines.append(
            f"| {team_data['rank']:2d} | {team_data['team']:3s} | "
            f"{team_data['record']:5s} | {team_data['composite_score']:5.1f} | "
            f"{bd['sagarin']:4.1f} | {bd['playoff_prob']:5.1f} | "
            f"{bd['point_diff']:5.1f} | {bd['recent_form']:5.1f} |"
        )

    report_lines.append("\n---\n")

    # Algorithm 3: Weighted Composite - No Sagarin
    report_lines.append("## Algorithm 3: Weighted Composite - No Sagarin (0%)")
    report_lines.append("\n**Methodology:** Weighted blend without Sagarin ratings")
    report_lines.append("- Sagarin: 0%, Playoff Prob: 30% (+5%), Point Diff: 25% (+5%), Recent Form: 20% (+5%),")
    report_lines.append("  Record Quality: 10%, EPA: 10%, Turnovers: 5%\n")

    report_lines.append("| Rank | Team | Record | Score | Playoff% | Pt Diff | Form | EPA |")
    report_lines.append("|------|------|--------|-------|----------|---------|------|-----|")

    for team_data in composite_no_sag[:15]:
        bd = team_data['breakdown']
        report_lines.append(
            f"| {team_data['rank']:2d} | {team_data['team']:3s} | "
            f"{team_data['record']:5s} | {team_data['composite_score']:5.1f} | "
            f"{bd['playoff_prob']:5.1f} | {bd['point_diff']:5.1f} | "
            f"{bd['recent_form']:5.1f} | {bd['epa']:4.1f} |"
        )

    report_lines.append("\n---\n")

    # Algorithm 5: Elo Dynamic
    report_lines.append("## Algorithm 6: Elo Dynamic")
    report_lines.append("\n**Methodology:** Elo rating system updated after each game")
    report_lines.append("- K-factor: 25, Home advantage: 3 points, Margin of victory multiplier\n")

    report_lines.append("| Rank | Team | Record | Elo Rating | Change from Start |")
    report_lines.append("|------|------|--------|------------|-------------------|")

    for team_data in elo[:15]:
        change_str = f"+{team_data['elo_change']:.1f}" if team_data['elo_change'] > 0 else f"{team_data['elo_change']:.1f}"
        report_lines.append(
            f"| {team_data['rank']:2d} | {team_data['team']:3s} | "
            f"{team_data['record']:5s} | {team_data['elo_rating']:7.1f} | {change_str:>11s} |"
        )

    report_lines.append("\n---\n")

    # Algorithm 6: Multi-Factor
    report_lines.append("## Algorithm 7: Multi-Factor Score")
    report_lines.append("\n**Methodology:** Independent scoring across 6 dimensions (averaged)")
    report_lines.append("- Record, Performance, Efficiency, Momentum, Predictive, Market\n")

    report_lines.append("| Rank | Team | Record | Score | Record | Perform | Effic | Moment | Predict | Market |")
    report_lines.append("|------|------|--------|-------|--------|---------|-------|--------|---------|--------|")

    for team_data in multifactor[:15]:
        bd = team_data['breakdown']
        report_lines.append(
            f"| {team_data['rank']:2d} | {team_data['team']:3s} | "
            f"{team_data['record']:5s} | {team_data['multi_factor_score']:5.1f} | "
            f"{bd['record']:5.1f} | {bd['performance']:6.1f} | {bd['efficiency']:5.1f} | "
            f"{bd['momentum']:6.1f} | {bd['predictive']:7.1f} | {bd['market']:6.1f} |"
        )

    report_lines.append("\n---\n")

    # Algorithm 6: AI-Enhanced (if available)
    if ai_enhanced:
        report_lines.append("## Algorithm 6: AI-Enhanced Rankings")
        report_lines.append("\n**Methodology:** Weighted Composite + AI adjustments (±3 spots max)")
        report_lines.append("- AI Model: Sonnet 3.7\n")

        report_lines.append("| Rank | Team | Record | Base | AI Adj | Reasoning |")
        report_lines.append("|------|------|--------|------|--------|-----------|")

        for team_data in ai_enhanced[:15]:
            base_rank = team_data.get('rank', team_data.get('ai_adjusted_rank'))
            ai_rank = team_data['ai_adjusted_rank']
            change = "" if base_rank == ai_rank else f"({base_rank}→{ai_rank})"
            reasoning = team_data.get('ai_reasoning', 'No adjustment')[:50]

            report_lines.append(
                f"| {ai_rank:2d} | {team_data['team']:3s} | "
                f"{team_data['record']:5s} | {base_rank:4d} | {change:8s} | {reasoning} |"
            )

        # Add AI summary
        if ai_enhanced[0].get('ai_summary'):
            report_lines.append(f"\n**AI Analysis:** {ai_enhanced[0]['ai_summary']}\n")

        report_lines.append("\n---\n")

    # Side-by-Side Comparison
    report_lines.append("## Side-by-Side Comparison (Top 20)")
    report_lines.append("\n| Team | R1 | R1+PD | Comp | Low-Sag | No-Sag | Elo | Multi | AI | Variance | Record |")
    report_lines.append("|------|----|-------|------|---------|--------|-----|-------|-----|----------|--------|")

    # Create team lookup maps
    r1_map = {r['team']: r['rank'] for r in r1_sov}
    r1_pd_map = {r['team']: r['rank'] for r in r1_sov_pd}
    comp_map = {r['team']: r['rank'] for r in composite}
    comp_low_sag_map = {r['team']: r['rank'] for r in composite_low_sag}
    comp_no_sag_map = {r['team']: r['rank'] for r in composite_no_sag}
    elo_map = {r['team']: r['rank'] for r in elo}
    multi_map = {r['team']: r['rank'] for r in multifactor}
    ai_map = {r['team']: r['ai_adjusted_rank'] for r in ai_enhanced} if ai_enhanced else {}

    # Get all teams in top 20 of any ranking
    top_teams = set()
    for ranking in [r1_sov[:20], r1_sov_pd[:20], composite[:20], composite_low_sag[:20], composite_no_sag[:20], elo[:20], multifactor[:20]]:
        top_teams.update(r['team'] for r in ranking)

    if ai_enhanced:
        top_teams.update(r['team'] for r in ai_enhanced[:20])

    # Sort by average rank
    team_avg_ranks = []
    for team in top_teams:
        ranks = [
            r1_map.get(team, 99),
            r1_pd_map.get(team, 99),
            comp_map.get(team, 99),
            comp_low_sag_map.get(team, 99),
            comp_no_sag_map.get(team, 99),
            elo_map.get(team, 99),
            multi_map.get(team, 99)
        ]
        if ai_map:
            ranks.append(ai_map.get(team, 99))
        avg_rank = sum(ranks) / len(ranks)
        team_avg_ranks.append((team, avg_rank, ranks))

    team_avg_ranks.sort(key=lambda x: x[1])

    for team, avg_rank, ranks in team_avg_ranks[:20]:
        team_record = pr.teams[team]['record']

        r1_rank = r1_map.get(team, '-')
        r1_pd_rank = r1_pd_map.get(team, '-')
        comp_rank = comp_map.get(team, '-')
        comp_low_sag_rank = comp_low_sag_map.get(team, '-')
        comp_no_sag_rank = comp_no_sag_map.get(team, '-')
        elo_rank = elo_map.get(team, '-')
        multi_rank = multi_map.get(team, '-')
        ai_rank = ai_map.get(team, '-') if ai_map else '-'

        # Calculate variance
        valid_ranks = [r for r in ranks if r != 99]
        if len(valid_ranks) > 1:
            variance = max(valid_ranks) - min(valid_ranks)
        else:
            variance = 0

        report_lines.append(
            f"| {team:3s} | {r1_rank:>2} | {r1_pd_rank:>5} | {comp_rank:4} | {comp_low_sag_rank:7} | {comp_no_sag_rank:6} | "
            f"{elo_rank:3} | {multi_rank:5} | {ai_rank:5} | {variance:8d} | {team_record:6s} |"
        )

    report_lines.append("\n---\n")

    # Correlation Analysis
    report_lines.append("## Algorithm Correlation Matrix")
    report_lines.append("\n**Spearman Correlation** (1.0 = perfect agreement)\n")

    correlations = {
        'R1 vs R1+PointDiff': calculate_correlation(r1_sov, r1_sov_pd),
        'R1 vs Composite (15% Sag)': calculate_correlation(r1_sov, composite),
        'R1 vs Elo': calculate_correlation(r1_sov, elo),
        'R1 vs Multi-Factor': calculate_correlation(r1_sov, multifactor),
        'Composite (15% Sag) vs Low Sag (5%)': calculate_correlation(composite, composite_low_sag),
        'Composite (15% Sag) vs No Sag (0%)': calculate_correlation(composite, composite_no_sag),
        'Low Sag (5%) vs No Sag (0%)': calculate_correlation(composite_low_sag, composite_no_sag),
        'Composite vs Elo': calculate_correlation(composite, elo),
        'Composite vs Multi-Factor': calculate_correlation(composite, multifactor),
        'No Sagarin vs Elo': calculate_correlation(composite_no_sag, elo),
        'No Sagarin vs Multi-Factor': calculate_correlation(composite_no_sag, multifactor),
    }

    if ai_enhanced:
        correlations['R1 vs AI'] = calculate_correlation(r1_sov, ai_enhanced)
        correlations['Composite vs AI'] = calculate_correlation(composite, ai_enhanced)
        correlations['No Sagarin vs AI'] = calculate_correlation(composite_no_sag, ai_enhanced)

    report_lines.append("| Comparison | Correlation |")
    report_lines.append("|------------|-------------|")

    for comp_name, corr in correlations.items():
        report_lines.append(f"| {comp_name:30s} | {corr:11.3f} |")

    report_lines.append("\n---\n")

    # Key Findings
    report_lines.append("## Key Findings\n")

    # Biggest disagreements
    report_lines.append("### Biggest Disagreements (Variance > 8)\n")

    big_disagreements = [(team, avg, ranks) for team, avg, ranks in team_avg_ranks if max(r for r in ranks if r != 99) - min(r for r in ranks if r != 99) > 8]

    if big_disagreements:
        for team, _, ranks in big_disagreements[:5]:
            valid_ranks = [r for r in ranks if r != 99]
            variance = max(valid_ranks) - min(valid_ranks)
            report_lines.append(f"- **{team}**: Ranks range from {min(valid_ranks)} to {max(valid_ranks)} (variance: {variance})")
    else:
        report_lines.append("No major disagreements - all algorithms are relatively aligned!")

    report_lines.append("")

    # IND placement (user's specific concern)
    ind_ranks = {
        'R1+SOV (baseline)': r1_map.get('IND', 'N/A'),
        'R1+SOV + PointDiff': r1_pd_map.get('IND', 'N/A'),
        'Composite (15% Sagarin)': comp_map.get('IND', 'N/A'),
        'Low Sagarin (5%)': comp_low_sag_map.get('IND', 'N/A'),
        'No Sagarin (0%)': comp_no_sag_map.get('IND', 'N/A'),
        'Elo': elo_map.get('IND', 'N/A'),
        'Multi-Factor': multi_map.get('IND', 'N/A'),
    }
    if ai_map:
        ind_ranks['AI-Enhanced'] = ai_map.get('IND', 'N/A')

    report_lines.append("### Indianapolis Colts (5-1, #1 AFC Seed) Placement\n")
    for algo, rank in ind_ranks.items():
        report_lines.append(f"- {algo}: **#{rank}**")

    avg_ind_rank = sum(r for r in ind_ranks.values() if isinstance(r, int)) / sum(1 for r in ind_ranks.values() if isinstance(r, int))
    report_lines.append(f"\nAverage rank: **#{avg_ind_rank:.1f}**")
    report_lines.append(f"Sagarin rank: **#{[r['rank'] for r in composite if r['team'] == 'IND'][0]}** (using Composite as reference)")

    report_lines.append("\n---\n")

    # Recommendations
    report_lines.append("## Recommendations\n")

    avg_corr = sum(correlations.values()) / len(correlations)

    if avg_corr > 0.85:
        report_lines.append(f"✅ **High Agreement** (avg correlation: {avg_corr:.3f})")
        report_lines.append("- All algorithms show strong agreement, suggesting robust rankings")
        report_lines.append("- Any algorithm would be reliable for production use")
    elif avg_corr > 0.70:
        report_lines.append(f"⚠️ **Moderate Agreement** (avg correlation: {avg_corr:.3f})")
        report_lines.append("- Algorithms show reasonable agreement with some divergence")
        report_lines.append("- Consider using a composite approach or the algorithm that best matches your priorities")
    else:
        report_lines.append(f"❌ **Low Agreement** (avg correlation: {avg_corr:.3f})")
        report_lines.append("- Significant disagreement between algorithms")
        report_lines.append("- Further tuning or investigation recommended")

    report_lines.append("")

    # Algorithm-specific recommendations
    report_lines.append("### Algorithm-Specific Notes\n")
    report_lines.append("**Weighted Composite:**")
    report_lines.append("- Best for: Balanced view emphasizing playoff probability and point differential")
    report_lines.append("- Strengths: Highly customizable weights, captures multiple dimensions")
    report_lines.append("")

    report_lines.append("**Elo Dynamic:**")
    report_lines.append("- Best for: Game-by-game performance tracking, recency bias")
    report_lines.append("- Strengths: Self-updating, mathematically principled, no arbitrary weights")
    report_lines.append("")

    report_lines.append("**Multi-Factor:**")
    report_lines.append("- Best for: Comprehensive evaluation across independent dimensions")
    report_lines.append("- Strengths: Considers record, performance, efficiency, momentum, prediction, market")
    report_lines.append("")

    if ai_enhanced:
        report_lines.append("**AI-Enhanced:**")
        report_lines.append("- Best for: Incorporating subjective 'eye test' and recent context")
        report_lines.append("- Strengths: Adds human-like judgment, adapts to injuries and trends")
        report_lines.append("")

    # Join all lines
    report = "\n".join(report_lines)

    return report


def main():
    include_ai = '--include-ai' in sys.argv

    print("="*80)
    print("NFL POWER RANKINGS COMPARISON")
    print("="*80)
    print()

    if include_ai:
        print("AI-enhanced rankings: ENABLED (this will take longer)")
    else:
        print("AI-enhanced rankings: DISABLED (use --include-ai to enable)")

    print()

    # Initialize power rankings
    pr = PowerRankings()

    # Generate report
    report = generate_comparison_report(pr, include_ai=include_ai)

    # Save to file
    output_file = "data/power_rankings_comparison.md"

    with open(output_file, 'w') as f:
        f.write(report)

    print()
    print("="*80)
    print(f"Report saved to: {output_file}")
    print("="*80)
    print()

    # Print quick summary
    print("Quick Preview - Top 10 Teams (Weighted Composite):")
    print()

    composite_preview = _rank_weighted_composite(pr, BASE_COMPOSITE_WEIGHTS)
    for team_data in composite_preview[:10]:
        print(f"  {team_data['rank']:2d}. {team_data['team']:3s} ({team_data['record']:5s}) - Score: {team_data['composite_score']:5.1f}")

    print()


if __name__ == "__main__":
    main()
