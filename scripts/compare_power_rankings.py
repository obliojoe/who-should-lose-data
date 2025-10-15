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
from power_rankings import PowerRankings
from scipy import stats


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


def generate_comparison_report(pr, include_ai=False):
    """Generate comprehensive comparison report"""

    print("Calculating power rankings using multiple algorithms...")
    print()

    # Run all algorithms
    print("1. Running Weighted Composite algorithm (15% Sagarin)...")
    composite = pr.weighted_composite_rankings()

    print("2. Running Weighted Composite - Low Sagarin (5%)...")
    composite_low_sag = pr.weighted_composite_rankings(weights={
        'sagarin': 0.05,
        'playoff_prob': 0.25,
        'point_diff': 0.25,  # +5%
        'recent_form': 0.20,  # +5%
        'record': 0.10,
        'epa': 0.10,
        'turnovers': 0.05,
    })

    print("3. Running Weighted Composite - No Sagarin (0%)...")
    composite_no_sag = pr.weighted_composite_rankings(weights={
        'sagarin': 0.00,
        'playoff_prob': 0.30,  # +5%
        'point_diff': 0.25,    # +5%
        'recent_form': 0.20,   # +5%
        'record': 0.10,
        'epa': 0.10,
        'turnovers': 0.05,
    })

    print("4. Running Elo Dynamic algorithm...")
    elo = pr.elo_dynamic_rankings()

    print("5. Running Multi-Factor algorithm...")
    multifactor = pr.multi_factor_rankings()

    ai_enhanced = None
    if include_ai:
        print("6. Running AI-Enhanced algorithm (this may take a minute)...")
        try:
            ai_enhanced = pr.ai_enhanced_rankings(base_algorithm='composite', ai_model='sonnet-3.7')
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

    # Algorithm 1: Weighted Composite
    report_lines.append("## Algorithm 1: Weighted Composite")
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

    # Algorithm 4: Elo Dynamic
    report_lines.append("## Algorithm 4: Elo Dynamic")
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

    # Algorithm 5: Multi-Factor
    report_lines.append("## Algorithm 5: Multi-Factor Score")
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
    report_lines.append("\n| Team | Comp | Low-Sag | No-Sag | Elo | Multi | AI | Variance | Record |")
    report_lines.append("|------|------|---------|--------|-----|-------|-------|----------|--------|")

    # Create team lookup maps
    comp_map = {r['team']: r['rank'] for r in composite}
    comp_low_sag_map = {r['team']: r['rank'] for r in composite_low_sag}
    comp_no_sag_map = {r['team']: r['rank'] for r in composite_no_sag}
    elo_map = {r['team']: r['rank'] for r in elo}
    multi_map = {r['team']: r['rank'] for r in multifactor}
    ai_map = {r['team']: r['ai_adjusted_rank'] for r in ai_enhanced} if ai_enhanced else {}

    # Get all teams in top 20 of any ranking
    top_teams = set()
    for ranking in [composite[:20], composite_low_sag[:20], composite_no_sag[:20], elo[:20], multifactor[:20]]:
        top_teams.update(r['team'] for r in ranking)

    if ai_enhanced:
        top_teams.update(r['team'] for r in ai_enhanced[:20])

    # Sort by average rank
    team_avg_ranks = []
    for team in top_teams:
        ranks = [
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
            f"| {team:3s} | {comp_rank:4} | {comp_low_sag_rank:7} | {comp_no_sag_rank:6} | "
            f"{elo_rank:3} | {multi_rank:5} | {ai_rank:5} | {variance:8d} | {team_record:6s} |"
        )

    report_lines.append("\n---\n")

    # Correlation Analysis
    report_lines.append("## Algorithm Correlation Matrix")
    report_lines.append("\n**Spearman Correlation** (1.0 = perfect agreement)\n")

    correlations = {
        'Composite (15% Sag) vs Low Sag (5%)': calculate_correlation(composite, composite_low_sag),
        'Composite (15% Sag) vs No Sag (0%)': calculate_correlation(composite, composite_no_sag),
        'Low Sag (5%) vs No Sag (0%)': calculate_correlation(composite_low_sag, composite_no_sag),
        'Composite vs Elo': calculate_correlation(composite, elo),
        'Composite vs Multi-Factor': calculate_correlation(composite, multifactor),
        'No Sagarin vs Elo': calculate_correlation(composite_no_sag, elo),
        'No Sagarin vs Multi-Factor': calculate_correlation(composite_no_sag, multifactor),
    }

    if ai_enhanced:
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

    composite = pr.weighted_composite_rankings()
    for team_data in composite[:10]:
        print(f"  {team_data['rank']:2d}. {team_data['team']:3s} ({team_data['record']:5s}) - Score: {team_data['composite_score']:5.1f}")

    print()


if __name__ == "__main__":
    main()
