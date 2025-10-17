from collections import defaultdict
import datetime
import logging
import os
import sys

from tqdm import tqdm
from tiebreakers import apply_tiebreakers, apply_wildcard_tiebreakers
from playoff_utils import format_percentage, load_teams

from datetime import datetime as dt
import json
from playoff_status import PLAYOFF_STATUS

# Set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create logs directory if it doesn't exist
if not os.path.exists('logs'):
    os.makedirs('logs')

# Set up file handler
file_handler = logging.FileHandler('logs/playoff_analysis.log')
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add a console handler to see logs in real-time
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)
logger.propagate = False

def determine_division_winners(standings, teams):
    division_winners = []
    
    # Group teams by division
    division_teams = defaultdict(list)
    for team in standings:
        division_teams[teams[team]['division']].append(team)
    
    # For each division, apply tiebreakers
    for division, div_teams in division_teams.items():
        winner = apply_tiebreakers(div_teams, standings, division=division)[0]
        division_winners.append(winner)
    
    return division_winners

def determine_wild_cards(standings, teams, division_winners):
    """
    Determine wild card teams based on standings and division winners.
    Selects the top three teams per conference that did not win their division.
    """
    wild_cards = {}  # Changed to return a dict with conference keys
    conferences = set(team_info['conference'] for team_info in teams.values())
    
    for conference in conferences:
        # Get division winners for this conference
        conf_div_winners = [team for team in division_winners if teams[team]['conference'] == conference]
        
        # Exclude division winners from wild card consideration
        non_division_teams = {team: record for team, record in standings.items()
                              if team not in division_winners and teams[team]['conference'] == conference}
        
        # Sort the remaining teams by win percentage, then by other tiebreakers as needed
        sorted_non_division = sorted(non_division_teams.items(), key=lambda item: (
            item[1]['wins'] / (item[1]['wins'] + item[1]['losses'] + item[1].get('ties', 0)),
            item[1]['wins'],
            -item[1]['losses']
        ), reverse=True)
        
        # Select the top three teams as wild cards per conference
        wild_cards[conference] = [team for team, _ in sorted_non_division[:3]]
    
    return wild_cards

def made_playoffs(team, division_winners, wild_cards, teams):
    conference = teams[team]['conference']
    return team in division_winners or team in wild_cards[conference]

def load_schedule():
    with open('data/schedule.json', 'r', encoding='utf-8') as f:
        schedule = json.load(f)
    return schedule

def get_game_context(team, game, teams):
    context = []
    away_team, home_team = game['away_team'], game['home_team']
    
    if team == away_team or team == home_team:
        return "This is your team's game."
    
    if teams[away_team]['division'] == teams[team]['division']:
        context.append(f"{away_team} is in your division.")
    if teams[home_team]['division'] == teams[team]['division']:
        context.append(f"{home_team} is in your division.")
    
    if teams[away_team]['conference'] == teams[team]['conference']:
        context.append(f"{away_team} is competing for the same conference playoff spots.")
    if teams[home_team]['conference'] == teams[team]['conference']:
        context.append(f"{home_team} is competing for the same conference playoff spots.")
    
    if not context:
        context.append("This game affects overall conference standings and tiebreakers.")
    
    return " ".join(context)

def get_relevant_games(schedule):
    """Get all games from the current week (played and unplayed) plus unplayed games from next week"""
    current_week_games = []
    next_week_games = []

    # Find the first unplayed game to determine current week
    first_unplayed_game = None
    for game in schedule:
        if not game['away_score'] and not game['home_score']:
            first_unplayed_game = game
            break

    if first_unplayed_game:
        current_week_num = int(first_unplayed_game['week_num'])
        next_week_num = current_week_num + 1

        # Get ALL games from current week (played and unplayed)
        # Get only UNPLAYED games from next week
        for game in schedule:
            if int(game['week_num']) == current_week_num:
                current_week_games.append(game)
            elif int(game['week_num']) == next_week_num:
                if not game['away_score'] and not game['home_score']:
                    next_week_games.append(game)

    # Start with current week's games (all of them)
    relevant_games = current_week_games

    # Count only unplayed games in current week for the threshold check
    current_week_unplayed = [g for g in current_week_games if not g['away_score'] and not g['home_score']]

    # If there are 2 or fewer unplayed games remaining in current week, include next week's games
    if len(current_week_unplayed) <= 2:
        relevant_games.extend(next_week_games)
    
    return relevant_games

def determine_wild_cards_with_tiebreakers(standings, teams, division_winners, simulation_result=None):
    """Determine wild card teams using full NFL tiebreaker rules."""
    # Load actual game results for points
    schedule = load_schedule()
    actual_points = defaultdict(lambda: {'points_for': 0, 'points_against': 0})
    
    # First get actual points from played games
    for game in schedule:
        if game['away_score'] and game['home_score']:  # If game has been played
            away_score = int(game['away_score'])
            home_score = int(game['home_score'])
            away_team = game['away_team']
            home_team = game['home_team']
            
            actual_points[away_team]['points_for'] += away_score
            actual_points[away_team]['points_against'] += home_score
            actual_points[home_team]['points_for'] += home_score
            actual_points[home_team]['points_against'] += away_score

    # Initialize standings with actual points
    for team in standings:
        if 'opponents' not in standings[team]:
            standings[team]['opponents'] = set()
        if 'common_games' not in standings[team]:
            standings[team]['common_games'] = defaultdict(lambda: {'wins': 0, 'losses': 0, 'ties': 0})
        if 'defeated_opponents' not in standings[team]:
            standings[team]['defeated_opponents'] = set()
        standings[team]['points_for'] = actual_points[team]['points_for']
        standings[team]['points_against'] = actual_points[team]['points_against']

    # Add simulated game points
    if simulation_result and 'game_results' in simulation_result:
        for game in simulation_result['game_results']:
            home_team = game['home_team']
            away_team = game['away_team']
            home_score = game['home_score']
            away_score = game['away_score']
            
            standings[home_team]['points_for'] += home_score
            standings[home_team]['points_against'] += away_score
            standings[away_team]['points_for'] += away_score
            standings[away_team]['points_against'] += home_score

    # Process game results to populate opponents, common_games, and add estimated points for simulated games
    for team in standings:
        for opp, h2h in standings[team].get('head_to_head', {}).items():
            standings[team]['opponents'].add(opp)
            if h2h.get('wins', 0) > h2h.get('losses', 0):
                standings[team]['defeated_opponents'].add(opp)
            
            # Add common games data
            standings[team]['common_games'][opp] = {
                'wins': h2h.get('wins', 0),
                'losses': h2h.get('losses', 0),
                'ties': h2h.get('ties', 0)
            }

    wild_cards = {}
    conferences = set(team_info['conference'] for team_info in teams.values())
    
    # Build head-to-head data from standings
    head_to_head = {}
    for team in standings:
        if 'head_to_head' in standings[team]:
            head_to_head[team] = standings[team]['head_to_head']
    
    for conference in conferences:
        # Get non-division winners for this conference
        non_division_teams = [team for team, record in standings.items()
                            if team not in division_winners and 
                            teams[team]['conference'] == conference]
        
        # Group teams by win percentage first
        win_pct_groups = {}
        for team in non_division_teams:
            win_pct = round(standings[team]['wins'] / 
                          (standings[team]['wins'] + standings[team]['losses'] + 
                           standings[team].get('ties', 0)), 3)
            if win_pct not in win_pct_groups:
                win_pct_groups[win_pct] = []
            win_pct_groups[win_pct].append(team)
        
        # Process groups in win percentage order
        sorted_teams = []
        for win_pct in sorted(win_pct_groups.keys(), reverse=True):
            teams_in_group = win_pct_groups[win_pct]
            if len(teams_in_group) > 1:
                # Pass head_to_head data to tiebreaker function
                sorted_group = apply_wildcard_tiebreakers(teams_in_group, standings, teams, head_to_head)
                sorted_teams.extend(sorted_group)
            else:
                sorted_teams.extend(teams_in_group)
        
        # Take top 3 teams as wild cards
        wild_cards[conference] = sorted_teams[:3]
    
    return wild_cards

def get_impact_explanation(playoff_impact, division_impact, top_seed_impact):
    """Helper function to determine which impact is most significant"""
    impacts = [
        ("Playoff impact", playoff_impact),
        ("Division impact", division_impact * 1.2),
        ("Top seed impact", top_seed_impact * 1.5)
    ]
    return max(impacts, key=lambda x: x[1])[0]

def calculate_max_possible_wins(team, schedule):
    """Calculate maximum possible wins for a team based on remaining schedule"""
    # Get current wins
    current_wins = get_current_standings(schedule)[team]['wins']
    
    # Count remaining games
    remaining_games = 0
    for game in schedule:
        if game['away_score'] == '' and game['home_score'] == '':  # Unplayed game
            if game['away_team'] == team or game['home_team'] == team:
                remaining_games += 1
    
    # Max possible wins is current wins plus all remaining games
    return current_wins + remaining_games

def calculate_win_pct(wins, losses, ties=0):
    """Calculate win percentage including ties"""
    total_games = wins + losses + ties
    if total_games == 0:
        return 0.0
    return (wins + 0.5 * ties) / total_games

def has_clinched_playoffs(team_abbr):
    """Simple lookup for playoff clinch status"""
    return PLAYOFF_STATUS.get(team_abbr, {}).get('playoffs', False)

def has_clinched_division(team_abbr):
    """Simple lookup for division clinch status"""
    return PLAYOFF_STATUS.get(team_abbr, {}).get('division', False)

def has_clinched_top_seed(team_abbr):
    """Simple lookup for top seed clinch status"""
    return PLAYOFF_STATUS.get(team_abbr, {}).get('top_seed', False)

def get_current_standings(schedule):
    """Get current standings from completed games"""
    standings = defaultdict(lambda: {
        'wins': 0, 
        'losses': 0, 
        'ties': 0
    })
    
    def _parse_score(value):
        if value in (None, '', 'nan'):
            return None
        try:
            return int(value)
        except (ValueError, TypeError):
            try:
                return int(float(value))
            except (ValueError, TypeError):
                return None

    for game in schedule:
        away_score = _parse_score(game.get('away_score'))
        home_score = _parse_score(game.get('home_score'))

        if away_score is not None and home_score is not None:  # If game has been played
            
            if away_score > home_score:
                standings[game['away_team']]['wins'] += 1
                standings[game['home_team']]['losses'] += 1
            elif home_score > away_score:
                standings[game['home_team']]['wins'] += 1
                standings[game['away_team']]['losses'] += 1
            else:
                standings[game['away_team']]['ties'] += 1
                standings[game['home_team']]['ties'] += 1
                
    return standings
