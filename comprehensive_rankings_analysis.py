#!/usr/bin/env python3
"""
Comprehensive Power Rankings Analysis
Runs all algorithms and compares with external sources
"""

import sys
import json
import csv
import statistics
from datetime import datetime
from power_rankings import PowerRankings
from scipy import stats


def load_external_rankings():
    """Load external power rankings from CSV"""
    external = {}

    with open('data/external_power_rankings_week7.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            team = row['Team']
            ranks = []
            for source in ['NFL.com', 'ESPN', 'NBC_PFT', 'FOX', 'Yahoo', 'USA_Today']:
                if row[source] and row[source].strip():
                    ranks.append(int(row[source]))

            if ranks:
                external[team] = {
                    'ranks': ranks,
                    'avg': statistics.mean(ranks),
                    'min': min(ranks),
                    'max': max(ranks)
                }

    return external


def calculate_correlation(ranks1, ranks2):
    """Calculate Spearman correlation between two ranking lists"""
    team_to_rank1 = {r['team']: r['rank'] for r in ranks1}
    team_to_rank2 = {r['team']: r['rank'] for r in ranks2}

    common_teams = set(team_to_rank1.keys()) & set(team_to_rank2.keys())

    if len(common_teams) < 2:
        return 0.0

    ranks_a = [team_to_rank1[team] for team in common_teams]
    ranks_b = [team_to_rank2[team] for team in common_teams]

    correlation, _ = stats.spearmanr(ranks_a, ranks_b)

    return correlation


def calculate_external_correlation(algorithm_ranks, external_rankings):
    """Calculate correlation between algorithm and external average"""
    algo_dict = {r['team']: r['rank'] for r in algorithm_ranks}

    common_teams = set(algo_dict.keys()) & set(external_rankings.keys())

    if len(common_teams) < 2:
        return 0.0

    algo_ranks = [algo_dict[team] for team in common_teams]
    ext_ranks = [external_rankings[team]['avg'] for team in common_teams]

    correlation, _ = stats.spearmanr(algo_ranks, ext_ranks)

    return correlation


def main():
    include_ai = '--include-ai' in sys.argv

    print("=" * 80)
    print("COMPREHENSIVE NFL POWER RANKINGS ANALYSIS - WEEK 7, 2025")
    print("=" * 80)
    print()

    if include_ai:
        print("AI-enhanced rankings: ENABLED")
    else:
        print("AI-enhanced rankings: DISABLED (use --include-ai to enable)")

    print()
    print("Running all algorithms...")
    print()

    # Initialize
    pr = PowerRankings()
    external = load_external_rankings()

    # Run all algorithms
    print("1. Weighted Composite (15% Sagarin)...")
    composite = pr.weighted_composite_rankings()

    print("2. Weighted Composite - Low Sagarin (5%)...")
    composite_low_sag = pr.weighted_composite_rankings(weights={
        'sagarin': 0.05,
        'playoff_prob': 0.25,
        'point_diff': 0.25,
        'recent_form': 0.20,
        'record': 0.10,
        'epa': 0.10,
        'turnovers': 0.05,
    })

    print("3. Weighted Composite - No Sagarin (0%)...")
    composite_no_sag = pr.weighted_composite_rankings(weights={
        'sagarin': 0.00,
        'playoff_prob': 0.30,
        'point_diff': 0.25,
        'recent_form': 0.20,
        'record': 0.10,
        'epa': 0.10,
        'turnovers': 0.05,
    })

    print("4. Elo Dynamic...")
    elo = pr.elo_dynamic_rankings()

    print("5. Multi-Factor...")
    multifactor = pr.multi_factor_rankings()

    print("6. Seeding-Priority...")
    seeding = pr.seeding_priority_rankings()

    print("7. Win-Quality...")
    win_quality = pr.win_quality_rankings()

    print("8. Momentum-Weighted...")
    momentum = pr.momentum_weighted_rankings()

    print("9. Clutch Performance...")
    clutch = pr.clutch_performance_rankings()

    print("10. Simple Predictive...")
    simple_pred = pr.simple_predictive_rankings()

    print("11. QB Excellence...")
    qb_excel = pr.qb_excellence_rankings()

    print("12. Defensive Dominance...")
    def_dom = pr.defensive_dominance_rankings()

    print("13. Red Zone Mastery...")
    rz_mastery = pr.red_zone_mastery_rankings()

    print("14. Third Down Excellence...")
    third_down = pr.third_down_excellence_rankings()

    print("15. Division Dominance...")
    div_dom = pr.division_dominance_rankings()

    print("16. Record First...")
    record_first = pr.record_first_rankings()

    print("17. Conference Leader...")
    conf_leader = pr.conference_leader_rankings()

    print("18. Turnover Battle...")
    to_battle = pr.turnover_battle_rankings()

    ai_enhanced = None
    if include_ai:
        print("19. AI-Enhanced (this may take a minute)...")
        try:
            ai_enhanced = pr.ai_enhanced_rankings(base_algorithm='composite', ai_model='sonnet-3.7')
        except Exception as e:
            print(f"   WARNING: AI enhancement failed: {e}")

    print()
    print("Generating comprehensive analysis report...")
    print()

    # Build report
    report_lines = []

    # Header
    report_lines.append("# COMPREHENSIVE POWER RANKINGS ANALYSIS - WEEK 7, 2025")
    report_lines.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report_lines.append("---\n")

    # Executive Summary
    report_lines.append("## Executive Summary\n")
    report_lines.append("This report compares **18 different algorithmic approaches** to NFL power rankings,")
    report_lines.append("plus AI-enhanced rankings, against **6 major external sources** (NFL.com, ESPN, NBC,")
    report_lines.append("FOX, Yahoo, USA Today).\n")

    # External consensus
    ext_sorted = sorted(external.items(), key=lambda x: x[1]['avg'])
    report_lines.append("### External Rankings Consensus (Top 10)\n")
    report_lines.append("| Rank | Team | Avg | Range | Sources |")
    report_lines.append("|------|------|-----|-------|---------|")
    for i, (team, data) in enumerate(ext_sorted[:10], 1):
        report_lines.append(f"| {i} | {team} | {data['avg']:.1f} | {data['min']}-{data['max']} | 5-6 |")
    report_lines.append("\n**Key Finding:** TB (Tampa Bay) is **unanimous #1** across all external sources.\n")
    report_lines.append("---\n")

    # Algorithm Comparison Table
    report_lines.append("## Top 10 Comparison Across All Algorithms\n")

    all_algos = {
        'Composite': composite,
        'Low-Sag': composite_low_sag,
        'No-Sag': composite_no_sag,
        'Elo': elo,
        'Multi': multifactor,
        'Seed': seeding,
        'Quality': win_quality,
        'Momentum': momentum,
        'Clutch': clutch,
        'Predict': simple_pred,
        'QB-Excel': qb_excel,
        'Def-Dom': def_dom,
        'RedZone': rz_mastery,
        '3rdDown': third_down,
        'Div-Dom': div_dom,
        'Record1st': record_first,
        'Conf-Lead': conf_leader,
        'Turnover': to_battle
    }

    if ai_enhanced:
        all_algos['AI'] = ai_enhanced

    # Create team lookup maps
    algo_maps = {}
    for name, rankings in all_algos.items():
        if name == 'AI':
            algo_maps[name] = {r['team']: r.get('ai_adjusted_rank', r['rank']) for r in rankings}
        else:
            algo_maps[name] = {r['team']: r['rank'] for r in rankings}

    # Get all teams in top 15 of any algorithm
    top_teams = set()
    for rankings in all_algos.values():
        if rankings:
            for team_data in rankings[:15]:
                top_teams.add(team_data['team'])

    # Calculate average rank for each team across all algorithms
    team_avg_ranks = []
    for team in top_teams:
        ranks = []
        for name, rank_map in algo_maps.items():
            ranks.append(rank_map.get(team, 99))

        # Add external average
        ext_rank = external.get(team, {}).get('avg', 99)

        avg_rank = statistics.mean(ranks)
        team_avg_ranks.append((team, avg_rank, ranks, ext_rank))

    team_avg_ranks.sort(key=lambda x: x[1])

    # Build table header
    header = "| Team | Ext | " + " | ".join(all_algos.keys()) + " | Avg | Variance |"
    separator = "|------|-----|" + "|".join(["-" * (len(name) + 2) for name in all_algos.keys()]) + "|-----|----------|"

    report_lines.append(header)
    report_lines.append(separator)

    for team, avg, ranks, ext_rank in team_avg_ranks[:15]:
        ext_str = f"{ext_rank:.1f}" if ext_rank != 99 else "-"
        record = pr.teams[team]['record']

        rank_strs = []
        for name in all_algos.keys():
            rank = algo_maps[name].get(team, '-')
            rank_strs.append(str(rank) if rank != '-' else '-')

        valid_ranks = [r for r in ranks if r != 99]
        variance = max(valid_ranks) - min(valid_ranks) if len(valid_ranks) > 1 else 0

        row = f"| {team} ({record}) | {ext_str} | " + " | ".join(f"{r:>3}" for r in rank_strs) + f" | {avg:.1f} | {variance} |"
        report_lines.append(row)

    report_lines.append("\n---\n")

    # TB Analysis
    report_lines.append("## Tampa Bay Buccaneers (5-1) - #1 NFC Seed Analysis\n")
    report_lines.append("**External Consensus:** Unanimous #1 (6/6 sources)\n")
    report_lines.append("**Our Algorithms:**\n")

    tb_ranks = {}
    for name, rank_map in algo_maps.items():
        tb_ranks[name] = rank_map.get('TB', '-')

    tb_rank_list = [(name, rank) for name, rank in tb_ranks.items()]
    tb_rank_list.sort(key=lambda x: x[1] if x[1] != '-' else 99)

    for name, rank in tb_rank_list:
        report_lines.append(f"- **{name}**: #{rank}")

    report_lines.append(f"\n**Best TB Ranking:** {tb_rank_list[0][0]} (#{tb_rank_list[0][1]})")
    report_lines.append(f"**Average Rank:** #{statistics.mean([r for n, r in tb_rank_list if r != '-']):.1f}\n")

    report_lines.append("**Why TB ranks lower in some algorithms:**")
    report_lines.append("- Point differential: +14 (relatively low compared to IND +78, DET +49)")
    report_lines.append("- Algorithms weighting point diff heavily penalize TB")
    report_lines.append("- TB excels in: Win%, Playoff probability (93%), Current seeding (#1 NFC)\n")

    report_lines.append("---\n")

    # Correlation Analysis
    report_lines.append("## Algorithm Correlation with External Rankings\n")
    report_lines.append("**Spearman Correlation** (1.0 = perfect match with external consensus)\n")
    report_lines.append("| Algorithm | Correlation | Notes |")
    report_lines.append("|-----------|-------------|-------|")

    correlations = []
    for name, rankings in all_algos.items():
        corr = calculate_external_correlation(rankings, external)
        correlations.append((name, corr))

        # Add interpretation
        if corr > 0.85:
            note = "Excellent match"
        elif corr > 0.75:
            note = "Good match"
        elif corr > 0.65:
            note = "Moderate match"
        else:
            note = "Significant differences"

        report_lines.append(f"| {name:15s} | {corr:11.3f} | {note} |")

    correlations.sort(key=lambda x: x[1], reverse=True)
    report_lines.append(f"\n**Best Match:** {correlations[0][0]} (r={correlations[0][1]:.3f})")
    report_lines.append(f"**Weakest Match:** {correlations[-1][0]} (r={correlations[-1][1]:.3f})\n")

    report_lines.append("---\n")

    # Key Disagreements
    report_lines.append("## Notable Disagreements (Variance > 8)\n")

    big_disagreements = [(team, avg, ranks, ext_rank) for team, avg, ranks, ext_rank in team_avg_ranks
                        if len([r for r in ranks if r != 99]) > 1 and
                        (max([r for r in ranks if r != 99]) - min([r for r in ranks if r != 99])) > 8]

    if big_disagreements:
        for team, avg, ranks, ext_rank in big_disagreements[:10]:
            valid_ranks = [r for r in ranks if r != 99]
            variance = max(valid_ranks) - min(valid_ranks)
            ext_str = f"{ext_rank:.1f}" if ext_rank != 99 else "N/A"
            report_lines.append(f"- **{team}**: Range {min(valid_ranks)}-{max(valid_ranks)} (variance: {variance}), External avg: {ext_str}")
    else:
        report_lines.append("No major disagreements found.")

    report_lines.append("\n---\n")

    # Detailed Algorithm Descriptions
    report_lines.append("## Algorithm Descriptions\n")

    algo_descriptions = {
        'Composite': "Weighted blend: Sagarin 15%, Playoff Prob 25%, Point Diff 20%, Recent Form 15%, Record 10%, EPA 10%, Turnovers 5%",
        'Low-Sag': "Same as Composite but Sagarin reduced to 5%, extra weight to Point Diff and Recent Form",
        'No-Sag': "Composite without Sagarin: Playoff Prob 30%, Point Diff 25%, Recent Form 20%, Record 10%, EPA 10%, Turnovers 5%",
        'Elo': "Dynamic Elo rating (K=25) updated game-by-game with margin of victory multiplier",
        'Multi': "Average of 6 independent dimensions: Record, Performance, Efficiency, Momentum, Predictive, Market",
        'Seed': "**Seeding-Priority:** Current playoff seed 35%, Playoff prob 30%, Record 20%, Point diff 10%, Recent form 5%",
        'Quality': "**Win-Quality:** Win% 25%, Playoff prob 30%, Strength of schedule 20%, Point diff 15%, EPA 10%",
        'Momentum': "**Momentum-Weighted:** Recent form (last 2) 25%, Playoff prob 30%, Win streak 15%, Record 15%, Point diff 15%",
        'Clutch': "**Clutch Performance:** Playoff prob 30%, Close game win% 20%, Record 20%, Win streak 15%, Point diff 15%",
        'Predict': "**Simple Predictive:** Playoff probability 50%, Record 30%, EPA 20%",
        'QB-Excel': "**QB Excellence:** Passer rating 30%, Completion% 15%, Yards/att 15%, TD/INT ratio 15%, Win% 25%",
        'Def-Dom': "**Defensive Dominance:** Pts against 30%, Def sacks 20%, Def INTs 20%, 3rd down def 15%, Win% 15%",
        'RedZone': "**Red Zone Mastery:** Red zone TD% 40%, Red zone def% 30%, Win% 30%",
        '3rdDown': "**Third Down Excellence:** 3rd down% 35%, 3rd down def% 35%, Win% 30%",
        'Div-Dom': "**Division Dominance:** Division win% 40%, Conference record 30%, Overall win% 30%",
        'Record1st': "**Record First:** Win% 70%, Playoff prob 20%, Current seed 10%",
        'Conf-Lead': "**Conference Leader:** Current seed 50%, Conference record 30%, Playoff prob 20%",
        'Turnover': "**Turnover Battle:** Turnover margin 50%, Win% 30%, Playoff prob 20%"
    }

    for name, desc in algo_descriptions.items():
        report_lines.append(f"**{name}:** {desc}\n")

    report_lines.append("---\n")

    # Recommendations
    report_lines.append("## Recommendations\n")
    report_lines.append(f"Based on correlation analysis, the algorithm that **best matches external consensus** is:")
    report_lines.append(f"### **{correlations[0][0]}** (r={correlations[0][1]:.3f})\n")

    # Find which algorithm ranks TB highest
    tb_best = tb_rank_list[0]
    report_lines.append(f"For **maximizing TB's ranking** while staying algorithmic:")
    report_lines.append(f"### **{tb_best[0]}** (ranks TB at #{tb_best[1]})\n")

    report_lines.append("**Strategic Recommendations:**\n")
    report_lines.append("1. If goal is to **match external consensus**: Use algorithm with highest correlation")
    report_lines.append("2. If goal is to **value current standings**: Use Seeding-Priority or Win-Quality")
    report_lines.append("3. If goal is to **predict future success**: Use Simple Predictive")
    report_lines.append("4. If goal is to **balance multiple factors**: Use Weighted Composite (with or without Sagarin)\n")

    # Save report
    output_file = "data/comprehensive_rankings_analysis.md"
    with open(output_file, 'w') as f:
        f.write("\n".join(report_lines))

    print("=" * 80)
    print(f"Report saved to: {output_file}")
    print("=" * 80)
    print()

    # Print summary
    print("QUICK SUMMARY:")
    print()
    print("External Consensus Top 5:")
    for i, (team, data) in enumerate(ext_sorted[:5], 1):
        print(f"  {i}. {team} (avg: {data['avg']:.1f})")

    print()
    print(f"Best Algorithm Match: {correlations[0][0]} (r={correlations[0][1]:.3f})")
    print(f"TB's Best Ranking: {tb_best[0]} (#{tb_best[1]})")
    print()
    print(f"Full analysis available in: {output_file}")
    print()


if __name__ == "__main__":
    main()
