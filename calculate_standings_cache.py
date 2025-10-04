#!/usr/bin/env python3
"""
Calculate current standings and save to cache file.
This runs the same logic as app.py but saves results to avoid repeated calculation.
"""

import sys
import os
import json
from collections import defaultdict
from datetime import datetime

# Import from local modules (data/ directory aware)
from playoff_utils import load_teams, load_schedule
from tiebreakers import apply_tiebreakers, apply_wildcard_tiebreakers, calculate_win_pct


def calculate_current_standings(teams, schedule):
    """
    Calculate current standings exactly as app.py does.
    Returns dict with divisional, conference, and playoff views.
    """
    # First calculate base standings
    standings = {}
    for team in teams:
        standings[team] = {
            'wins': 0, 'losses': 0, 'ties': 0,
            'division_wins': 0, 'division_losses': 0, 'division_ties': 0,
            'conference_wins': 0, 'conference_losses': 0, 'conference_ties': 0,
            'points_for': 0, 'points_against': 0,
            'opponents': set(),
            'defeated_opponents': set(),
            'strength_of_schedule': 0.0,
            'strength_of_victory': 0.0
        }

    # Build head-to-head record
    head_to_head = defaultdict(lambda: defaultdict(int))

    # Process each completed game
    for game in schedule:
        if not (game['away_score'] and game['home_score']):
            continue

        away_team = game['away_team']
        home_team = game['home_team']
        away_score = int(game['away_score'])
        home_score = int(game['home_score'])

        # Record head-to-head
        if away_score > home_score:
            head_to_head[away_team][home_team] += 1
        elif home_score > away_score:
            head_to_head[home_team][away_team] += 1

        # Update points
        standings[away_team]['points_for'] += away_score
        standings[away_team]['points_against'] += home_score
        standings[home_team]['points_for'] += home_score
        standings[home_team]['points_against'] += away_score

        standings[away_team]['opponents'].add(home_team)
        standings[home_team]['opponents'].add(away_team)

        # Check divisions/conferences
        same_division = teams[away_team]['division'] == teams[home_team]['division']
        same_conference = teams[away_team]['conference'] == teams[home_team]['conference']

        # Update records
        if away_score > home_score:
            standings[away_team]['wins'] += 1
            standings[home_team]['losses'] += 1
            standings[away_team]['defeated_opponents'].add(home_team)

            if same_division:
                standings[away_team]['division_wins'] += 1
                standings[home_team]['division_losses'] += 1
            if same_conference:
                standings[away_team]['conference_wins'] += 1
                standings[home_team]['conference_losses'] += 1

        elif home_score > away_score:
            standings[home_team]['wins'] += 1
            standings[away_team]['losses'] += 1
            standings[home_team]['defeated_opponents'].add(away_team)

            if same_division:
                standings[home_team]['division_wins'] += 1
                standings[away_team]['division_losses'] += 1
            if same_conference:
                standings[home_team]['conference_wins'] += 1
                standings[away_team]['conference_losses'] += 1
        else:  # Tie
            standings[away_team]['ties'] += 1
            standings[home_team]['ties'] += 1

            if same_division:
                standings[away_team]['division_ties'] += 1
                standings[home_team]['division_ties'] += 1
            if same_conference:
                standings[away_team]['conference_ties'] += 1
                standings[home_team]['conference_ties'] += 1

    # Calculate strength of schedule and victory
    for team in standings:
        # Strength of schedule
        total_opp_games = 0
        total_opp_wins = 0
        total_opp_ties = 0

        for opp in standings[team]['opponents']:
            opp_total_games = standings[opp]['wins'] + standings[opp]['losses'] + standings[opp]['ties']
            if opp_total_games > 0:
                total_opp_games += opp_total_games
                total_opp_wins += standings[opp]['wins']
                total_opp_ties += standings[opp]['ties']

        if total_opp_games > 0:
            standings[team]['strength_of_schedule'] = (total_opp_wins + 0.5 * total_opp_ties) / total_opp_games

        # Strength of victory
        total_defeated_games = 0
        total_defeated_wins = 0
        total_defeated_ties = 0

        for opp in standings[team]['defeated_opponents']:
            opp_total_games = standings[opp]['wins'] + standings[opp]['losses'] + standings[opp]['ties']
            if opp_total_games > 0:
                total_defeated_games += opp_total_games
                total_defeated_wins += standings[opp]['wins']
                total_defeated_ties += standings[opp]['ties']

        if total_defeated_games > 0:
            standings[team]['strength_of_victory'] = (total_defeated_wins + 0.5 * total_defeated_ties) / total_defeated_games

    # Convert sets to lists for JSON serialization
    for team in standings:
        standings[team]['opponents'] = list(standings[team]['opponents'])
        standings[team]['defeated_opponents'] = list(standings[team]['defeated_opponents'])

    # Build formatted standings by division
    divisional = {}
    for conference in ['AFC', 'NFC']:
        divisional[conference] = {}
        for division in [f'{conference} North', f'{conference} South', f'{conference} East', f'{conference} West']:
            division_teams = [team for team, info in teams.items() if info['division'] == division]

            # Sort by record
            sorted_teams = sorted(division_teams,
                                key=lambda t: (standings[t]['wins'], -standings[t]['losses']),
                                reverse=True)

            # Apply tiebreakers if needed
            record_groups = defaultdict(list)
            for team in sorted_teams:
                win_pct = calculate_win_pct(standings[team])
                record_groups[win_pct].append(team)

            final_order = []
            for win_pct in sorted(record_groups.keys(), reverse=True):
                teams_in_group = record_groups[win_pct]
                if len(teams_in_group) > 1:
                    # Get tiebreaker explanations for division standings
                    sorted_group = apply_tiebreakers(teams_in_group, standings, teams,
                                                    head_to_head=head_to_head,
                                                    division=division,
                                                    return_explanations=True)
                    for team, explanation in sorted_group:
                        standings[team]['tiebreaker_explanation'] = explanation
                        final_order.append(team)
                else:
                    team = teams_in_group[0]
                    standings[team]['tiebreaker_explanation'] = None
                    final_order.append(team)

            # Store as objects with rank
            divisional[conference][division] = [
                {
                    'team': team,
                    'rank': rank + 1,
                    **standings[team]
                }
                for rank, team in enumerate(final_order)
            ]

    # Build conference standings (uses wildcard tiebreaker rules)
    conference = {}
    for conf in ['AFC', 'NFC']:
        conf_teams = [team for team, info in teams.items() if info['conference'] == conf]

        # Sort by record
        sorted_conf_teams = sorted(conf_teams,
                                  key=lambda t: calculate_win_pct(standings[t]),
                                  reverse=True)

        # Apply wildcard tiebreakers (conference record prioritized)
        record_groups = defaultdict(list)
        for team in sorted_conf_teams:
            win_pct = calculate_win_pct(standings[team])
            record_groups[win_pct].append(team)

        final_conf_order = []
        for win_pct in sorted(record_groups.keys(), reverse=True):
            teams_in_group = record_groups[win_pct]
            if len(teams_in_group) > 1:
                sorted_group = apply_wildcard_tiebreakers(teams_in_group, standings, teams,
                                                         head_to_head=head_to_head,
                                                         return_explanations=True)
                for team, explanation in sorted_group:
                    # Store explanation in standings for conference view
                    if 'conference_tiebreaker_explanation' not in standings[team]:
                        standings[team]['conference_tiebreaker_explanation'] = explanation
                    final_conf_order.append(team)
            else:
                team = teams_in_group[0]
                standings[team]['conference_tiebreaker_explanation'] = None
                final_conf_order.append(team)

        # Store as objects with rank
        conference[conf] = [
            {
                'team': team,
                'rank': rank + 1,
                **standings[team]
            }
            for rank, team in enumerate(final_conf_order)
        ]

    # Build playoff standings
    playoff = {}
    for conf in ['AFC', 'NFC']:
        # Get division winners
        division_winners = []
        remaining_teams = []

        divisions = {}
        for team, info in teams.items():
            if info['conference'] == conf:
                div = info['division']
                if div not in divisions:
                    divisions[div] = []
                divisions[div].append((team, standings[team]))

        # Get winner from each division
        for division, div_teams in divisions.items():
            sorted_div = sorted(div_teams, key=lambda x: calculate_win_pct(x[1]), reverse=True)

            # Apply tiebreakers
            record_groups = defaultdict(list)
            for team, info in sorted_div:
                win_pct = calculate_win_pct(info)
                record_groups[win_pct].append(team)

            final_div_order = []
            for win_pct in sorted(record_groups.keys(), reverse=True):
                teams_in_group = record_groups[win_pct]
                if len(teams_in_group) > 1:
                    sorted_group = apply_tiebreakers(teams_in_group, standings, teams,
                                                    head_to_head=head_to_head,
                                                    division=division)
                    final_div_order.extend(sorted_group)
                else:
                    final_div_order.extend(teams_in_group)

            division_winners.append((final_div_order[0], standings[final_div_order[0]]))
            for team in final_div_order[1:]:
                remaining_teams.append((team, standings[team]))

        # Sort division winners for seeding
        div_winner_groups = defaultdict(list)
        for team, info in division_winners:
            win_pct = calculate_win_pct(info)
            div_winner_groups[win_pct].append(team)

        sorted_division_winners = []
        for win_pct in sorted(div_winner_groups.keys(), reverse=True):
            teams_in_group = div_winner_groups[win_pct]
            if len(teams_in_group) > 1:
                sorted_group = apply_tiebreakers(teams_in_group, standings, teams,
                                                head_to_head=head_to_head,
                                                return_explanations=True)
                for team, explanation in sorted_group:
                    standings[team]['tiebreaker_explanation'] = explanation
                    sorted_division_winners.append((team, standings[team]))
            else:
                team = teams_in_group[0]
                standings[team]['tiebreaker_explanation'] = None
                sorted_division_winners.append((team, standings[team]))

        # Seed division winners (1-4) - store as objects
        seeded_division_winners = [
            {
                'team': team,
                'seed': seed,
                'division': teams[team]['division'],
                **info
            }
            for seed, (team, info) in enumerate(sorted_division_winners, 1)
        ]

        # Sort remaining teams for wild card
        sorted_remaining = sorted(remaining_teams, key=lambda x: calculate_win_pct(x[1]), reverse=True)

        # Apply wild card tiebreakers
        record_groups = defaultdict(list)
        for team, info in sorted_remaining:
            win_pct = calculate_win_pct(info)
            record_groups[win_pct].append(team)

        final_wildcard_order = []
        for win_pct in sorted(record_groups.keys(), reverse=True):
            teams_in_group = record_groups[win_pct]
            if len(teams_in_group) > 1:
                sorted_group = apply_wildcard_tiebreakers(teams_in_group, standings, teams,
                                                         head_to_head=head_to_head,
                                                         return_explanations=True)
                for team, explanation in sorted_group:
                    standings[team]['tiebreaker_explanation'] = explanation
                    final_wildcard_order.append(team)
            else:
                team = teams_in_group[0]
                standings[team]['tiebreaker_explanation'] = None
                final_wildcard_order.append(team)

        # Seed wild cards (5-7) - store as objects
        wild_cards = [
            {
                'team': team,
                'seed': seed,
                'division': teams[team]['division'],
                **standings[team]
            }
            for seed, team in enumerate(final_wildcard_order[:3], 5)
        ]

        # Remaining teams (8+) - store as objects
        eliminated = [
            {
                'team': team,
                'seed': seed,
                'division': teams[team]['division'],
                **standings[team]
            }
            for seed, team in enumerate(final_wildcard_order[3:], 8)
        ]

        playoff[conf] = {
            'division_winners': seeded_division_winners,
            'wild_cards': wild_cards,
            'eliminated': eliminated
        }

    return {
        'divisional': divisional,
        'conference': conference,
        'playoff': playoff
    }


def main():
    """Generate and save standings cache."""
    print("Loading teams and schedule...")
    teams = load_teams()
    schedule = load_schedule()

    print("Calculating current standings...")
    standings_data = calculate_current_standings(teams, schedule)

    # Add timestamp
    cache = {
        'timestamp': datetime.now().isoformat(),
        'standings': standings_data
    }

    # Save to cache file (in data/ directory, will be copied to persist/ later)
    output_path = 'data/standings_cache.json'
    print(f"Saving to {output_path}...")

    with open(output_path, 'w') as f:
        json.dump(cache, f, indent=2)

    print(f"✓ Standings cache generated successfully")
    print(f"  - Divisional standings: {sum(len(divs) for conf in standings_data['divisional'].values() for divs in conf.values())} teams across 8 divisions")
    print(f"  - Conference standings: {len(standings_data['conference']['AFC']) + len(standings_data['conference']['NFC'])} teams")
    print(f"  - Playoff picture calculated for both conferences")


if __name__ == '__main__':
    main()
