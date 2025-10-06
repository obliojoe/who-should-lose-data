#!/usr/bin/env python3

import base64
import contextlib
from io import StringIO
import sys
import os
import json
import argparse
import logging
import hashlib
import csv
from datetime import datetime
import traceback
from tqdm import tqdm
from pathlib import Path
import random
import numpy as np
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from anthropic import Anthropic
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
import pandas as pd
from ai_service import AIService
from prompt_builder import build_team_analysis_prompt
from espn_api import ESPNAPIService
from simulate_season import simulate_season
from playoff_analysis import get_relevant_games
from team_starters import save_team_starters
from tiebreakers import (
    apply_tiebreakers, 
    calculate_win_pct
)
from playoff_utils import format_percentage, load_teams, load_schedule, get_data_dir, generate_sagarin_hash
from team_stats import save_team_stats
from scrape_sagarin import scrape_sagarin
import shutil
from update_scores import update_scores_and_dates
from generate_analyses import batch_analyze_games
import subprocess

is_ci = os.environ.get('CI') == 'true'

def filter_team_starters(data, team_abbr):
    """Filter team starters data for a specific team."""
    lines = data.split('\n')
    header = lines[0]
    filtered_lines = [line for line in lines[1:] if line.startswith(team_abbr)]
    return header + '\n' + '\n'.join(filtered_lines)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/generate_cache.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# use a time cvalue to determine how long parts of the script are taking
script_start_time = datetime.now()

# Pre-compute valid scores and their weights once
VALID_SCORES = np.array([
    0, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37,
    38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54
])

COMMON_SCORES = {0, 3, 6, 7, 10, 13, 14, 17, 20, 21, 24, 27, 28, 31, 34, 35}
SCORE_WEIGHTS = np.array([3 if score in COMMON_SCORES else 1 for score in VALID_SCORES])

# Seed value weights for impact calculation
# Higher seeds are significantly more valuable due to playoff advantages
# Values informed by historical playoff performance data (1975-2024)
SEED_VALUES = {
    1: 100,  # Bye week + home field advantage throughout playoffs
    2: 80,   # Bye week + likely home championship game
    3: 50,   # Home wild card game
    4: 45,   # Home wild card game (slightly harder path)
    5: 30,   # Road wild card game
    6: 35,   # Road wild card game (harder path)
    7: 20,   # Road wild card game (hardest path)
    0: 0     # Missed playoffs
}


def get_seed_multiplier_mapping():
    """Return the seed multiplier mapping used for impact calculations."""
    multipliers = {
        str(seed): value
        for seed, value in sorted(SEED_VALUES.items())
        if seed != 0
    }
    multipliers['Miss'] = SEED_VALUES.get(0, 0)
    return multipliers

def validate_cache(cache_data):
    """Validate the structure of the cache data"""
    required_keys = {'timestamp', 'num_simulations', 'team_analyses', 'playoff_odds', 'super_bowl'}
    if not all(key in cache_data for key in required_keys):
        return False
        
    # Validate team analyses
    for team_analysis in cache_data['team_analyses'].values():
        if not all(key in team_analysis for key in ['playoff_chance', 'division_chance', 'significant_games']):
            return False
    
    # Validate playoff odds
    for conference in ['AFC', 'NFC']:
        if conference not in cache_data['playoff_odds']:
            return False
    
    # Validate Super Bowl data
    if not all(key in cache_data['super_bowl'] for key in ['appearances', 'wins']):
        return False
    
    return True

def calculate_standings(teams, schedule):
    # Initialize standings dictionary with all stats
    standings = {team: {
        'wins': 0, 'losses': 0, 'ties': 0,
        'division_wins': 0, 'division_losses': 0, 'division_ties': 0,
        'conference_wins': 0, 'conference_losses': 0, 'conference_ties': 0,
        'points_for': 0, 'points_against': 0,
        'opponents': set(),
        'defeated_opponents': set(),
        'common_games': defaultdict(lambda: {'wins': 0, 'losses': 0, 'ties': 0}),
        'strength_of_schedule': 0.0,
        'strength_of_victory': 0.0
    } for team in teams}

    # Initialize head-to-head records
    head_to_head = defaultdict(lambda: defaultdict(int))

    # Process all games to populate standings and head-to-head records
    for game in schedule:
        # Only process games that have been played
        if game['away_score'] == '' or game['home_score'] == '':
            continue
            
        away_score = int(game['away_score'])
        home_score = int(game['home_score'])
        away_team = game['away_team']
        home_team = game['home_team']

        # Update head-to-head records
        if away_score > home_score:
            head_to_head[away_team][home_team] += 1
        elif home_score > away_score:
            head_to_head[home_team][away_team] += 1

        # Update points and opponents regardless of outcome
        standings[away_team]['points_for'] += away_score
        standings[away_team]['points_against'] += home_score
        standings[home_team]['points_for'] += home_score
        standings[home_team]['points_against'] += away_score

        standings[away_team]['opponents'].add(home_team)
        standings[home_team]['opponents'].add(away_team)

        # Check if teams are in same division/conference
        same_division = teams[away_team]['division'] == teams[home_team]['division']
        same_conference = teams[away_team]['conference'] == teams[home_team]['conference']

        # Update records based on game result
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

    # Calculate strength of schedule and victory for each team
    for team in standings:
        # Calculate strength of schedule
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
        
        # Calculate strength of victory
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

    # Format standings by conference and division with proper sorting
    formatted_standings = {}
    for conference in ['AFC', 'NFC']:
        formatted_standings[conference] = {}
        for division in [f'{conference} North', f'{conference} South', f'{conference} East', f'{conference} West']:
            # Get teams in this division and verify count
            division_teams = [team for team, info in teams.items() 
                            if info['division'] == division]
            
            # logger.info(f"\nProcessing {division}")
            # logger.info(f"Initial division teams: {division_teams}")
            # logger.info(f"Number of teams in division: {len(division_teams)}")
            
            # Verify each team has standings data
            for team in division_teams:
                if team not in standings:
                    logger.error(f"Missing standings data for {team}")
                    standings[team] = {'wins': 0, 'losses': 0, 'ties': 0}
                
                win_pct = calculate_win_pct(standings[team])
                record = f"{standings[team]['wins']}-{standings[team]['losses']}"
                if standings[team].get('ties', 0) > 0:
                    record += f"-{standings[team]['ties']}"
                # logger.info(f"{team}: {record} ({win_pct:.3f})")
            
            # First sort by win percentage
            sorted_teams = sorted(division_teams,
                                key=lambda t: calculate_win_pct(standings[t]),
                                reverse=True)
            
            # logger.info(f"Teams after initial win pct sort: {sorted_teams}")
            
            # Group teams by win percentage (using rounded values to handle floating point)
            win_pct_groups = {}
            for team in sorted_teams:
                win_pct = round(calculate_win_pct(standings[team]), 3)
                if win_pct not in win_pct_groups:
                    win_pct_groups[win_pct] = []
                win_pct_groups[win_pct].append(team)
            
            # Process groups in win percentage order
            final_order = []
            for win_pct in sorted(win_pct_groups.keys(), reverse=True):
                teams_in_group = win_pct_groups[win_pct]
                # logger.info(f"Processing win pct group {win_pct}: {teams_in_group}")
                
                if len(teams_in_group) > 1:
                    # Apply tiebreakers for teams with same win percentage
                    sorted_group = apply_tiebreakers(
                        teams_in_group,
                        standings,
                        head_to_head=head_to_head,
                        division=division,
                        return_explanations=True
                    )
                    # logger.info(f"After tiebreakers: {[t[0] for t in sorted_group]}")
                    
                    # Verify we got back the same number of teams we sent in
                    if len(sorted_group) != len(teams_in_group):
                        logger.error(f"Tiebreaker returned wrong number of teams! Expected {len(teams_in_group)}, got {len(sorted_group)}")
                        # Try again with division-specific tiebreakers
                        sorted_group = apply_tiebreakers(
                            teams_in_group,
                            standings,
                            head_to_head=head_to_head,
                            division=division,
                            return_explanations=True,
                            force_division_rules=True  # New parameter to ensure division tiebreakers are used
                        )
                        
                        if len(sorted_group) == len(teams_in_group):
                            # Use the proper tiebreaker results if we got them
                            for team, explanation in sorted_group:
                                standings[team]['tiebreaker_explanation'] = explanation
                                final_order.append(team)
                        else:
                            # Only fall back to secondary criteria if tiebreakers still fail
                            logger.error("Division tiebreakers also failed, falling back to secondary criteria")
                            sorted_teams = sorted(teams_in_group,
                                               key=lambda t: (
                                                   standings[t]['division_wins'],
                                                   standings[t]['conference_wins'],
                                                   standings[t]['points_for'] - standings[t]['points_against']
                                               ),
                                               reverse=True)
                            for team in sorted_teams:
                                explanation = []
                                if standings[team]['division_wins'] > 0:
                                    explanation.append(f"Division wins: {standings[team]['division_wins']}")
                                if standings[team]['conference_wins'] > 0:
                                    explanation.append(f"Conference wins: {standings[team]['conference_wins']}")
                                net_points = standings[team]['points_for'] - standings[team]['points_against']
                                explanation.append(f"Net points: {net_points}")
                                
                                standings[team]['tiebreaker_explanation'] = "; ".join(explanation)
                                final_order.append(team)
                    else:
                        # Use tiebreaker results
                        for team, explanation in sorted_group:
                            standings[team]['tiebreaker_explanation'] = explanation
                            final_order.append(team)
                else:
                    team = teams_in_group[0]
                    standings[team]['tiebreaker_explanation'] = None
                    final_order.append(team)
            
            # logger.info(f"Final order before verification: {final_order}")
            # logger.info(f"Length of final order: {len(final_order)}")
            # logger.info(f"Length of division teams: {len(division_teams)}")
            
            # Verify all teams are included
            if len(final_order) != len(division_teams):
                logger.error(f"MISSING TEAMS IN {division}!")
                logger.error(f"Division teams: {division_teams}")
                logger.error(f"Final order: {final_order}")
                missing_teams = set(division_teams) - set(final_order)
                logger.error(f"Missing teams: {missing_teams}")
                # Add missing teams
                for team in missing_teams:
                    # logger.info(f"Adding missing team {team} to final order")
                    final_order.append(team)
            
            # Create final sorted list with records
            formatted_standings[conference][division] = [
                (team, standings[team]) for team in final_order
            ]
            
            # Final verification
            final_teams = [t for t, _ in formatted_standings[conference][division]]
            # logger.info(f"Final teams in {division}: {final_teams}")
            # logger.info(f"Number of teams in final standings: {len(final_teams)}")

    return formatted_standings

def get_division_standings(standings_data):
    division_standings = {}
    for conference in standings_data:
        for division in standings_data[conference]:
            division_standings[division] = []
            for team, info in standings_data[conference][division]:
                record = f"{info['wins']}-{info['losses']}"
                if info['ties'] > 0:
                    record += f"-{info['ties']}"
                division_standings[division].append({
                    'team': team,
                    'record': record,
                    'win_pct': (info['wins'] + 0.5 * info['ties']) / (info['wins'] + info['losses'] + info['ties'])
                })
    return division_standings


def build_prompt_context(schedule=None):
    """Load heavy prompt inputs once so they can be reused across teams."""
    data_dir = get_data_dir()

    # Load shared CSV data once
    team_starters = pd.read_csv(os.path.join(data_dir, 'team_starters.csv'))
    schedule_df = pd.read_csv(os.path.join(data_dir, 'schedule.csv'))
    team_stats_df = pd.read_csv(os.path.join(data_dir, 'team_stats.csv'))
    teams_df = pd.read_csv(os.path.join(data_dir, 'teams.csv'))

    from prompt_builder import calculate_league_rankings

    context = {
        'data_dir': data_dir,
        'team_starters': team_starters,
        'schedule_df': schedule_df,
        'team_stats_df': team_stats_df,
        'league_rankings': calculate_league_rankings(team_stats_df.copy()),
        'team_notes_df': pd.read_csv(os.path.join(data_dir, 'team_notes.csv')),
        'schedule_csv': Path(data_dir, 'schedule.csv').read_text(),
        'teams_df': teams_df,
        'schedule_list': schedule if schedule is not None else load_schedule(),
    }

    coordinators_path = os.path.join(data_dir, 'coordinators.csv')
    if os.path.exists(coordinators_path):
        context['coordinators_df'] = pd.read_csv(coordinators_path)
    else:
        logger.warning(f"Coordinators file not found at {coordinators_path} - coordinator info will be omitted")
        context['coordinators_df'] = None

    return context


def generate_team_analysis_prompt(team_abbr, team_info, team_record, teams, cache_data, standings_data, prompt_context=None):
    """Generate the AI analysis prompt for a team using the same format as cache generation"""

    prompt_context = prompt_context or build_prompt_context()

    data_dir = prompt_context['data_dir']
    schedule = prompt_context['schedule_list']
    schedule_pd = prompt_context['schedule_df']
    teamstats_pd = prompt_context['team_stats_df']
    league_rankings = prompt_context['league_rankings']
    team_notes_pd = prompt_context['team_notes_df']
    team_starters = prompt_context['team_starters'].copy()
    schedule_data = prompt_context['schedule_csv']
    coordinators_pd = prompt_context.get('coordinators_df')
    teams_df = prompt_context['teams_df']

    def get_team_notes(team_abbrs):
        """Get team notes from team_notes.csv"""
        # return the notes for the team_abbrs, including the team_abbrs themselves
        return team_notes_pd[team_notes_pd['team_abbr'].isin(team_abbrs)]['notes'].tolist()

    def get_team_info_from_schedule(team_abbr):
        """Get head coach and stadium from most recent game for a team."""
        team_games = schedule_pd[
            ((schedule_pd['home_team'] == team_abbr) | (schedule_pd['away_team'] == team_abbr)) &
            (schedule_pd['home_score'].astype(str).str.len() > 0) & (schedule_pd['away_score'].astype(str).str.len() > 0)
        ]
        if not team_games.empty:
            most_recent_game = team_games.iloc[-1]
            head_coach = most_recent_game['home_coach'] if most_recent_game['home_team'] == team_abbr else most_recent_game['away_coach']
            return head_coach, most_recent_game['stadium']
        return None, None

    def get_next_game_stadium(team_abbr):
        """Get the location of the next game for a given team."""
        # next game for this team is the first game where the team is either home or away and does not have scores
        next_game_stadium = schedule_pd[
            (schedule_pd['home_team'] == team_abbr) | (schedule_pd['away_team'] == team_abbr) &
            (schedule_pd['home_score'].astype(str).str.len() == 0) & (schedule_pd['away_score'].astype(str).str.len() == 0)
        ]['stadium'].iloc[0]
        return next_game_stadium

    def get_starting_qb_from_team_starters(team_abbr):
        """Get the starting QB for a given team, preferring active players."""
        qb_data = team_starters[(team_starters['team_abbr'] == team_abbr) & (team_starters['position'] == 'QB')]
        if not qb_data.empty:
            # Sort by status: ACT first, then others, then get name and status
            # Prefer ACT (Active) over INA (Inactive), RES (Reserve), etc.
            qb_data = qb_data.sort_values(by='status', key=lambda x: x.map({'ACT': 0, 'RES': 1, 'INA': 2, 'CUT': 3}))
            qb_name = qb_data['player_name'].iloc[0]
            qb_status = qb_data['status'].iloc[0]

            # Return name with status if not active
            if qb_status == 'ACT':
                return qb_name
            else:
                status_text = {
                    'INA': 'Inactive',
                    'RES': 'Reserve',
                    'CUT': 'Cut',
                    'OUT': 'Out',
                    'DOUBTFUL': 'Doubtful',
                    'QUESTIONABLE': 'Questionable'
                }.get(qb_status, qb_status)
                return f"{qb_name} ({status_text})"
        logger.warning(f"No starting QB found for {team_abbr}!")
        return None

    # get head coach and stadium from most recent game in schedule data (the last game where team_abbr is either home or away team and has scores)
    head_coach, stadium = get_team_info_from_schedule(team_abbr)

    # Add head coach to teams dict for this team
    if head_coach:
        teams[team_abbr]['head_coach'] = head_coach

    # filter out the schedule data for the team in question, include the csv header row
    schedule_df = pd.read_csv(StringIO(schedule_data))
    schedule_df = schedule_df[
        ((schedule_df['home_team'] == team_abbr) | (schedule_df['away_team'] == team_abbr))
    ]

    def format_score(score):
        """Convert score to proper format, handling empty strings and NaN values"""
        if pd.isna(score) or score == '' or score == 'nan':
            return ''
        try:
            return str(int(float(score)))
        except (ValueError, TypeError):
            return ''

    # Apply formatting to all scores to clean up nan values
    schedule_df['home_score'] = schedule_df['home_score'].apply(format_score)
    schedule_df['away_score'] = schedule_df['away_score'].apply(format_score)

    schedule_data = schedule_df.to_csv(index=False)

    # get this team's previous played game
    previous_game = None
    for game in schedule:
        if (game['home_score'] != '' and game['away_score'] != '') and (game['home_team'] == team_abbr or game['away_team'] == team_abbr):  
            previous_game = game

    # get this team's next unplayed game
    next_game = None
    for game in schedule:
        if (game['home_score'] == '' and game['away_score'] == '') and (game['home_team'] == team_abbr or game['away_team'] == team_abbr):
            next_game = game
            break

    # Handle case where team has no remaining games
    if next_game is None:
        next_opponent = "NONE"
        next_opponent_qb = "NONE"
        next_opponent_coach = "NONE"
        next_game_stadium = "NONE"
        next_opponent_clincher_text = ""
        next_opponent_streak = "NONE"
        next_opponent_teamstats_encoded = ""
        next_game = {
            'game_date': 'NO REMAINING GAMES',
            'away_team': 'NONE',
            'home_team': 'NONE'
        }
    else:
        next_opponent = next_game['away_team'] if next_game['home_team'] == team_abbr else next_game['home_team']
        next_opponent_qb = get_starting_qb_from_team_starters(next_opponent)
        next_opponent_coach = get_team_info_from_schedule(next_opponent)[0]
        next_game_stadium = get_next_game_stadium(team_abbr)

        # Add head coach to teams dict for opponent
        if next_opponent_coach and next_opponent != "NONE":
            teams[next_opponent]['head_coach'] = next_opponent_coach
        
        # Get next opponent stats
        next_opponent_teamstats_data = teamstats_pd[teamstats_pd['team_abbr'] == next_opponent]
        next_opponent_teamstats_text = next_opponent_teamstats_data.to_csv(index=False)
        next_opponent_teamstats_encoded = base64.b64encode(next_opponent_teamstats_text.encode()).decode()
        
        next_opponent_clincher = next_opponent_teamstats_data['clincher'].iloc[0]
        next_opponent_clincher_text = "ELIMINATED" if next_opponent_clincher == "e" else "CLINCHED PLAYOFFS" if next_opponent_clincher == "x" else "CLINCHED THE DIVISION" if next_opponent_clincher == "z" else "#1 SEED" if next_opponent_clincher == "*" else "NOT CLINCHED / NOT ELIMINATED"
        next_opponent_clincher_text = f"- PLAYOFF STATUS: {next_opponent_clincher_text}" if next_opponent_clincher_text != "" else ""
        
        # Get next opponent streak
        next_opponent_streak = next_opponent_teamstats_data['streak_display'].iloc[0]
        streak_type = "wins" if next_opponent_streak.startswith("W") else "losses"
        streak_num = next_opponent_streak[1:]
        next_opponent_streak = f"{streak_num} {streak_type}"

    # filter the team stats for the team in question, include the csv header row
    teamstats_data = teamstats_pd[teamstats_pd['team_abbr'] == team_abbr]
    teamstats_text = teamstats_data.to_csv(index=False)
    teamstats_encoded = base64.b64encode(teamstats_text.encode()).decode()

    # get current team starters AND next opponent starters
    team_starters = team_starters[team_starters['team_abbr'].isin([team_abbr, next_opponent])]

    # Only include essential columns to reduce prompt size
    essential_columns = [
        'team_abbr', 'position', 'player_name', 'number', 'formation', 'status',
        'height', 'weight', 'college', 'experience',
        'passing_yards_season', 'passing_tds_season', 'interceptions_season',
        'rushing_yards_season', 'rushing_tds_season',
        'receiving_yards_season', 'receiving_tds_season', 'games_played'
    ]
    # Only use columns that exist
    available_essential = [col for col in essential_columns if col in team_starters.columns]
    team_starters_filtered = team_starters[available_essential]

    team_starters_text = team_starters_filtered.to_csv(index=False)
    team_starters_encoded = base64.b64encode(team_starters_text.encode()).decode()

    # Extract key injuries for both teams from ESPN event summary API
    def get_injuries_from_espn(espn_event_id, team_abbr_filter):
        """Get injuries from ESPN event summary API"""
        if not espn_event_id:
            return []

        try:
            url = f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary?event={espn_event_id}"
            response = requests.get(url, timeout=5)

            if response.status_code == 200:
                data = response.json()
                injury_data = data.get('injuries', [])

                # Find the team's injury data
                for team_injuries in injury_data:
                    if team_injuries.get('team', {}).get('abbreviation') == team_abbr_filter:
                        injuries = []
                        for injury in team_injuries.get('injuries', []):
                            player_name = injury.get('athlete', {}).get('displayName', 'Unknown')
                            position = injury.get('athlete', {}).get('position', {}).get('abbreviation', 'N/A')
                            status = injury.get('status', 'Unknown')
                            injury_type = injury.get('details', {}).get('type', '')

                            # Only include significant injury statuses
                            if status in ['Out', 'Doubtful', 'Questionable', 'Injured Reserve']:
                                injuries.append({
                                    'player': player_name,
                                    'position': position,
                                    'status': status,
                                    'type': injury_type
                                })
                        return injuries
                return []
        except Exception as e:
            logger.warning(f"Failed to fetch injuries from ESPN for {team_abbr_filter}: {e}")
            return []

    # Get injuries from ESPN event summary if we have an ESPN ID, otherwise fall back to team_starters
    team_injuries = []
    opponent_injuries = []

    if next_game and next_game.get('espn_id'):
        team_injuries = get_injuries_from_espn(next_game['espn_id'], team_abbr)
        if next_opponent != "NONE":
            opponent_injuries = get_injuries_from_espn(next_game['espn_id'], next_opponent)

    # Fallback to team_starters if ESPN data is not available
    if not team_injuries or (next_opponent != "NONE" and not opponent_injuries):
        def get_key_injuries_fallback(team_abbr_filter):
            """Fallback: Get list of notable injured/inactive players from team_starters"""
            injured = team_starters[
                (team_starters['team_abbr'] == team_abbr_filter) &
                (team_starters['status'].isin(['INA', 'OUT', 'DOUBTFUL', 'QUESTIONABLE']))
            ]
            if injured.empty:
                return []
            # Focus on skill positions
            key_positions = ['QB', 'RB', 'WR', 'TE', 'LT', 'RT', 'LDE', 'RDE', 'MLB', 'CB', 'SS', 'FS']
            injured_key = injured[injured['position'].isin(key_positions)]
            return [{'player': row['player_name'], 'position': row['position'], 'status': row['status'], 'type': ''}
                    for _, row in injured_key.iterrows()]

        if not team_injuries:
            team_injuries = get_key_injuries_fallback(team_abbr)
        if next_opponent != "NONE" and not opponent_injuries:
            opponent_injuries = get_key_injuries_fallback(next_opponent)

    # Fetch recent news headlines for both teams
    def get_team_news(team_abbr_filter):
        """Get recent ESPN headlines for a team (only team-specific, not generic NFL news)"""
        try:
            team_row = teams_df[teams_df['team_abbr'] == team_abbr_filter]

            if team_row.empty or 'espn_api_id' not in team_row.columns:
                return []

            espn_team_id = team_row['espn_api_id'].iloc[0]
            team_name = team_row['mascot'].iloc[0]
            team_city = team_row['city'].iloc[0]

            url = f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/news?team={espn_team_id}&limit=20"

            response = requests.get(url, timeout=3)
            if response.status_code == 200:
                data = response.json()
                articles = data.get('articles', [])

                # Only include headlines that mention the team name, city, or are clearly team-specific
                team_specific = []
                for a in articles:
                    headline = a.get('headline', '')
                    description = a.get('description', '')

                    # Check if headline or description mentions the team
                    contains_team = (team_name in headline or team_city in headline or
                                   team_name in description or team_city in description or
                                   team_abbr_filter in headline or team_abbr_filter in description)

                    # Exclude generic multi-team articles unless they mention this specific team
                    generic_multi_team = any(pattern in headline for pattern in [
                        'all 32 teams', 'Every team', 'every team', 'NFL Week', 'Week 5 predictions',
                        'Week 5 uniforms', 'Latest Madden ratings'
                    ])

                    if contains_team and not generic_multi_team:
                        team_specific.append({'headline': headline, 'published': a.get('published', '')})
                        if len(team_specific) >= 3:  # Stop at 3
                            break

                return team_specific
            return []
        except Exception as e:
            logger.warning(f"Failed to fetch news for {team_abbr_filter}: {e}")
            return []

    team_news = get_team_news(team_abbr)
    opponent_news = get_team_news(next_opponent) if next_opponent != "NONE" else []

    # get this team's starting qb from team_starters
    starting_qb = get_starting_qb_from_team_starters(team_abbr)

    # Generate division standings
    division_standings = get_division_standings(standings_data)
    
    # Format division standings string
    standings_text = ""
    for division, teams_list in division_standings.items():
        standings_text += f"\n{division}\n"
        for team in sorted(teams_list, key=lambda x: x['win_pct'], reverse=True):
            standings_text += f"{team['team']}: {team['record']}\n"

    # Format record string
    record_str = f"{team_record['wins']}-{team_record['losses']}"
    if team_record.get('ties', 0) > 0:
        record_str += f"-{team_record['ties']}"

    # Get points data
    points_for = team_record.get('points_for', 0)
    points_against = team_record.get('points_against', 0)
    point_differential = points_for - points_against

    try:
        team_data = cache_data['team_analyses'][team_abbr]
    except Exception as e:
        print(f"Error getting team data: {e}")
        team_data = None
        exit()

    # get current week from last played game in the schedule data
    current_week = 0
    for game in schedule:
        if game['home_score'] == '' and game['away_score'] == '':
            current_week = int(game['week_num'])
            break

    playoff_chance = format_percentage(team_data['playoff_chance'])
    division_chance = format_percentage(team_data['division_chance'])
    top_seed_chance = format_percentage(team_data['top_seed_chance'])
    sb_appearance_chance = format_percentage(team_data['super_bowl_appearance_chance'])
    sb_win_chance = format_percentage(team_data['super_bowl_win_chance'])

    current_playoff_seed = teamstats_data['playoff_seed'].iloc[0]
    current_streak = teamstats_data['streak_display'].iloc[0]
    # Extract number and type from streak (e.g. "W9" -> "9 wins")
    streak_type = "wins" if current_streak.startswith("W") else "losses"
    streak_num = current_streak[1:] # Get everything after first character
    current_streak = f"{streak_num} {streak_type}"
    
    division = teams[team_abbr]['division']
    conference = teams[team_abbr]['conference']

    significant_games = team_data['significant_games'][:3]            
    
    team_notes = get_team_notes([team_abbr, next_opponent])
    special_notes = f""
    for note in team_notes:
        if note and not pd.isna(note):  # Only add non-empty and non-nan notes
            special_notes += f"- {note}\n"

    if special_notes != "":
        special_notes = f"\nSPECIAL NOTES:\n{special_notes}"
    
    # get next opponent's streak
    next_opponent_streak = next_opponent_streak

    sports_analysts_list = "Pat McAfee, Stephen A. Smith, Rich Eisen, Mitch Albom, Peter King, Booger McFarland, Adam Rank, Chris Simms, Laura Okmin, Tony Kornheiser"
    roast_comedians_list = "Jeff Ross, Anthony Jeselnik, Dave Attell, Lisa Lampanelli, Jim Norton, Daniel Tosh, Bill Burr, Nikki Glaser, Greg Giraldo"
    other_comedians_list = "Norm Macdonald, John Mulaney, Aziz Ansari, Hannibal Buress, Whitney Cummings, Maria Bamford, Tig Notaro, Sarah Silverman, Amy Schumer, Jim Jefferies, Louis C.K., Jim Gaffigan, Seth Macfarlane, George Carlin, Steven Wright, Mitch Hedberg, Bill Hicks"
    # pick 3 random sports analysts and 3 random roast comedians
    sports_analysts = " and ".join(random.sample(sports_analysts_list.split(", "), 3))
    roast_comedians = " and ".join(random.sample(roast_comedians_list.split(", "), 3))
    other_comedians = " and ".join(random.sample(other_comedians_list.split(", "), 3))
    
    ai_fun_rule = ""
    
    # flip a coin to decide if we're writing a pure comedy piece or fake news story
    if random.choice([True, False]):
        ai_fun_rule = f"- Important: Write in the style of these comedians and writers: {other_comedians}"
    else:
        ai_fun_rule = f"- Important: Write a fake news story in the style of The Onion, or the style of SNL's Weekend Update"

    team_notes = get_team_notes([team_abbr, next_opponent])

    # Format significant games with impact scores
    games_text = "Most important games in the upcoming week:"
    for game in significant_games:
        date = game['date']

        away = f"{teams[game['away_team']]['city']} {teams[game['away_team']]['mascot']}"
        home = f"{teams[game['home_team']]['city']} {teams[game['home_team']]['mascot']}"
        
        # Get impact details from debug_stats
        debug = game.get('debug_stats', {})
        playoff_impact = debug.get('playoff_impact', 0)
        division_impact = debug.get('division_impact', 0)
        top_seed_impact = debug.get('top_seed_impact', 0)
        # Calculate probability changes
        baseline_playoff = team_data['playoff_chance']
        baseline_division = team_data['division_chance']
        baseline_top_seed = team_data.get('top_seed_chance', 0)
        baseline_seed = team_data.get('seed_chance', 0)
        away_playoff_pct = format_percentage(debug.get('away_playoff_pct', 0))
        home_playoff_pct = format_percentage(debug.get('home_playoff_pct', 0))
        away_division_pct = format_percentage(debug.get('away_division_pct', 0))
        home_division_pct = format_percentage(debug.get('home_division_pct', 0))
        away_top_seed_pct = format_percentage(debug.get('away_top_seed_pct', 0))
        home_top_seed_pct = format_percentage(debug.get('home_top_seed_pct', 0))
        
        seed_impact = debug.get('seed_impact', 0)
        seed_impact_pct = format_percentage(seed_impact)
        
        # Calculate changes using the formatted values
        away_playoff_change = away_playoff_pct - baseline_playoff
        home_playoff_change = home_playoff_pct - baseline_playoff
        away_division_change = away_division_pct - baseline_division
        home_division_change = home_division_pct - baseline_division
        away_top_seed_change = away_top_seed_pct - baseline_top_seed
        home_top_seed_change = home_top_seed_pct - baseline_top_seed
        seed_change = seed_impact_pct - baseline_seed
        games_text += f"""

{away} @ {home} ({date})
--------------------------------
If {away} wins:
- Playoff odds: {away_playoff_pct:.1f}% ({away_playoff_change:+.1f}%)
- Division odds: {away_division_pct:.1f}% ({away_division_change:+.1f}%)
- #1 Seed odds: {away_top_seed_pct:.1f}% ({away_top_seed_change:+.1f}%)
- Seed improvement odds: {seed_change:.1f}% ({seed_impact_pct:+.1f}%)

If {home} wins:
- Playoff odds: {home_playoff_pct:.1f}% ({home_playoff_change:+.1f}%)
- Division odds: {home_division_pct:.1f}% ({home_division_change:+.1f}%)
- #1 Seed odds: {home_top_seed_pct:.1f}% ({home_top_seed_change:+.1f}%)
- Seed improvement odds: {seed_change:.1f}% ({seed_impact_pct:+.1f}%)
--------------------------------
""" 

    # Get team and opponent stats rows
    team_stats_row = teamstats_pd[teamstats_pd['team_abbr'] == team_abbr].iloc[0]
    opponent_stats_row = teamstats_pd[teamstats_pd['team_abbr'] == next_opponent].iloc[0] if next_opponent != "NONE" else None

    # Get coordinator data if available
    team_coordinators = None
    opponent_coordinators = None
    if coordinators_pd is not None:
        team_coord_row = coordinators_pd[coordinators_pd['team_abbr'] == team_abbr]
        if not team_coord_row.empty:
            team_coordinators = team_coord_row.iloc[0]

        if next_opponent != "NONE":
            opponent_coord_row = coordinators_pd[coordinators_pd['team_abbr'] == next_opponent]
            if not opponent_coord_row.empty:
                opponent_coordinators = opponent_coord_row.iloc[0]

    # Fetch ESPN context (betting lines, weather) if available
    espn_context = None
    if next_game and next_game.get('espn_id'):
        espn_service = ESPNAPIService()
        espn_event_id = next_game['espn_id']
        is_home = next_game['home_team'] == team_abbr
        espn_context = espn_service.get_game_context(
            espn_event_id,
            next_game['home_team'],
            next_game['away_team']
        )

    # Use prompt builder with 4-section approach
    prompt = build_team_analysis_prompt(
        team_abbr=team_abbr,
        team_stats_row=team_stats_row,
        opponent_abbr=next_opponent,
        opponent_stats_row=opponent_stats_row,
        teams_dict=teams,
        team_injuries=team_injuries,
        opponent_injuries=opponent_injuries,
        team_news=team_news,
        opponent_news=opponent_news,
        team_schedule=schedule,
        playoff_chance=playoff_chance,
        division_chance=division_chance,
        top_seed_chance=top_seed_chance,
        sb_appearance_chance=sb_appearance_chance,
        sb_win_chance=sb_win_chance,
        current_week=current_week,
        standings_data=standings_data,
        team_coordinators=team_coordinators,
        opponent_coordinators=opponent_coordinators,
        espn_context=espn_context,
        league_rankings=league_rankings
    )

        
    # if (team_abbr == 'ARI' or team_abbr == 'MIN' or team_abbr == 'DET'):
    #     logger.info("\n=== AI Prompt ===")
    #     logger.info(f"Team: {team_info['city']} {team_info['mascot']}")
    #     logger.info("\nPrompt being sent to AI:")
    #     logger.info(prompt)
    #     logger.info("===================================\n")
    
    return prompt


def get_team_record(standings_data, team_abbr):
    """Extract team record from standings data"""
    for conference in standings_data:
        for division in standings_data[conference]:
            for team, record in standings_data[conference][division]:
                if team == team_abbr:
                    return record
                
    print(f"Team record not found for {team_abbr}")
    return None

def analyze_batch_game_impacts(batch_results, game_impacts, teams):
    """
    Analyze a batch of simulation results to update game impacts.
    """
    team_items = list(teams.items())

    # For each simulation in the batch
    for result in batch_results:
        # Get the playoff teams for quick lookup
        division_winners = set(result.get('division_winners', []))
        wild_cards = {team for conf in result.get('wild_cards', {}).values() 
                     for team in conf}
        playoff_teams = division_winners | wild_cards
        
        # Get playoff seeding for each conference
        playoff_seeds = {}
        for conf, order in result.get('playoff_order', {}).items():
            for seed, team in enumerate(order, 1):
                playoff_seeds[team] = seed
        
        # Get top seeds for each conference
        top_seeds = {}
        for conf, order in result.get('playoff_order', {}).items():
            if order:  # If there's a playoff order
                top_seeds[conf] = order[0]  # First team is top seed
        
        # Analyze each game's impact
        for game in result.get('game_results', []):
            game_id = f"{game['away_team']}@{game['home_team']}"
            home_team = game['home_team']
            away_team = game['away_team']
            home_win = game['home_score'] > game['away_score']
            
            # Initialize impact data if needed
            if game_id not in game_impacts:
                game_impacts[game_id] = defaultdict(lambda: {
                    'home_wins_playoff': 0,
                    'away_wins_playoff': 0,
                    'home_wins_division': 0,
                    'away_wins_division': 0,
                    'home_wins_top_seed': 0,
                    'away_wins_top_seed': 0,
                    'home_wins_sb_appearance': 0,
                    'away_wins_sb_appearance': 0,
                    'home_wins_sb_win': 0,
                    'away_wins_sb_win': 0,
                    'home_wins_count': 0,
                    'away_wins_count': 0,
                    'total_sims': 0,
                    'home_wins_seeds': defaultdict(int),  # Track seed frequencies
                    'away_wins_seeds': defaultdict(int)   # Track seed frequencies
                })
            
            # Update counts for each team
            for team, team_info in team_items:
                impact = game_impacts[game_id][team]
                conf = team_info['conference']
                
                if home_win:
                    impact['home_wins_count'] += 1
                    if team in division_winners:
                        impact['home_wins_division'] += 1
                    if team in playoff_teams:
                        impact['home_wins_playoff'] += 1
                        if team in playoff_seeds:
                            impact['home_wins_seeds'][playoff_seeds[team]] += 1
                    if team == top_seeds.get(conf):
                        impact['home_wins_top_seed'] += 1
                    if team in result['super_bowl'].get('teams', []):
                        impact['home_wins_sb_appearance'] += 1
                    if team == result['super_bowl'].get('winner'):
                        impact['home_wins_sb_win'] += 1
                else:
                    impact['away_wins_count'] += 1
                    if team in division_winners:
                        impact['away_wins_division'] += 1
                    if team in playoff_teams:
                        impact['away_wins_playoff'] += 1
                        if team in playoff_seeds:
                            impact['away_wins_seeds'][playoff_seeds[team]] += 1
                    if team == top_seeds.get(conf):
                        impact['away_wins_top_seed'] += 1
                    if team in result['super_bowl'].get('teams', []):
                        impact['away_wins_sb_appearance'] += 1
                    if team == result['super_bowl'].get('winner'):
                        impact['away_wins_sb_win'] += 1
                        
                impact['total_sims'] += 1

def calculate_game_impact(game_id, team_abbr, game_impacts):
    """Calculate the impact of a game on a team's playoff chances using expected seed value"""
    impact = game_impacts[game_id][team_abbr]
    total_sims = impact['total_sims']
    if total_sims == 0:
        return 0, {}

    # Split game_id once
    away_team, home_team = game_id.split('@')

    # Calculate seed percentages for each outcome
    home_seeds = defaultdict(float)
    away_seeds = defaultdict(float)

    if impact['home_wins_count'] > 0:
        for seed, count in impact['home_wins_seeds'].items():
            home_seeds[seed] = round((count / impact['home_wins_count']) * 100, 2)

    if impact['away_wins_count'] > 0:
        for seed, count in impact['away_wins_seeds'].items():
            away_seeds[seed] = round((count / impact['away_wins_count']) * 100, 2)

    # Calculate percentages for each outcome
    home_playoff_pct = (impact['home_wins_playoff'] / impact['home_wins_count']) * 100 if impact['home_wins_count'] > 0 else 0
    away_playoff_pct = (impact['away_wins_playoff'] / impact['away_wins_count']) * 100 if impact['away_wins_count'] > 0 else 0

    home_division_pct = (impact['home_wins_division'] / impact['home_wins_count']) * 100 if impact['home_wins_count'] > 0 else 0
    away_division_pct = (impact['away_wins_division'] / impact['away_wins_count']) * 100 if impact['away_wins_count'] > 0 else 0

    home_top_seed_pct = (impact['home_wins_top_seed'] / impact['home_wins_count']) * 100 if impact['home_wins_count'] > 0 else 0
    away_top_seed_pct = (impact['away_wins_top_seed'] / impact['away_wins_count']) * 100 if impact['away_wins_count'] > 0 else 0

    # Super Bowl percentage calculations
    home_sb_appearance_pct = (impact['home_wins_sb_appearance'] / impact['home_wins_count']) * 100 if impact['home_wins_count'] > 0 else 0
    away_sb_appearance_pct = (impact['away_wins_sb_appearance'] / impact['away_wins_count']) * 100 if impact['away_wins_count'] > 0 else 0
    home_sb_win_pct = (impact['home_wins_sb_win'] / impact['home_wins_count']) * 100 if impact['home_wins_count'] > 0 else 0
    away_sb_win_pct = (impact['away_wins_sb_win'] / impact['away_wins_count']) * 100 if impact['away_wins_count'] > 0 else 0

    # Calculate expected seed value for each outcome
    # Account for missed playoffs (no seed = 0 value)
    home_expected_value = 0
    away_expected_value = 0

    # Calculate expected value when home wins
    for seed, pct in home_seeds.items():
        home_expected_value += SEED_VALUES.get(seed, 0) * (pct / 100)
    # Account for missed playoffs
    home_miss_playoffs_pct = 100 - home_playoff_pct
    home_expected_value += SEED_VALUES[0] * (home_miss_playoffs_pct / 100)

    # Calculate expected value when away wins
    for seed, pct in away_seeds.items():
        away_expected_value += SEED_VALUES.get(seed, 0) * (pct / 100)
    # Account for missed playoffs
    away_miss_playoffs_pct = 100 - away_playoff_pct
    away_expected_value += SEED_VALUES[0] * (away_miss_playoffs_pct / 100)

    # Total impact is the absolute difference in expected seed values
    total_impact = abs(home_expected_value - away_expected_value)

    # Determine which outcome is better for this team
    if home_expected_value > away_expected_value:
        root_against = away_team
    else:
        root_against = home_team

    # Create clean debug stats dictionary
    debug_stats = {
        # Simulation metadata
        'total_sims': total_sims,
        'home_wins': impact['home_wins_count'],
        'away_wins': impact['away_wins_count'],

        # Outcome percentages (useful for UI display)
        'home_playoff_pct': round(home_playoff_pct, 1),
        'away_playoff_pct': round(away_playoff_pct, 1),
        'home_division_pct': round(home_division_pct, 1),
        'away_division_pct': round(away_division_pct, 1),
        'home_top_seed_pct': round(home_top_seed_pct, 1),
        'away_top_seed_pct': round(away_top_seed_pct, 1),
        'home_sb_appearance_pct': round(home_sb_appearance_pct, 1),
        'away_sb_appearance_pct': round(away_sb_appearance_pct, 1),
        'home_sb_win_pct': round(home_sb_win_pct, 1),
        'away_sb_win_pct': round(away_sb_win_pct, 1),

        # Seed distributions (needed for calculation transparency)
        'home_seeds': dict(home_seeds),
        'away_seeds': dict(away_seeds),

        # Expected seed value metrics (the new core calculation)
        'home_expected_seed_value': round(home_expected_value, 1),
        'away_expected_seed_value': round(away_expected_value, 1),
        'seed_value_impact': round(total_impact, 1),

        # Root against (useful for UI)
        'root_against': root_against
    }

    return total_impact, debug_stats

def generate_cache(num_simulations=1000, skip_sims=False, skip_team_ai=False, output_path='data/analysis_cache.json', copy_data=True, test_mode=False, regenerate_team_ai=None, seed=None):
    """Generate the analysis cache file"""
    is_ci = os.environ.get('CI') == 'true'  # Check for CI environment
    output_path = Path(output_path)

    # Generate or use provided seed for reproducibility
    if seed is None:
        # Use a random seed but make sure it's a simple integer
        seed = np.random.randint(0, 2**31 - 1)
    logger.info(f"Using master seed: {seed}")

    logger.info(f"Starting cache generation with {num_simulations} simulations")
    logger.info(f"Simulations are {'skipped' if skip_sims else 'enabled'}")
    logger.info(f"Team AI analysis is {'skipped' if skip_team_ai else 'enabled'}")

    # Initialize AI service if needed
    ai_service = None
    if not skip_team_ai:
        ai_service = AIService()
        ai_service.test_mode = test_mode
        success, message = ai_service.test_connection()
        if not success:
            logger.warning(f"AI service connection failed: {message}")
            logger.warning("Continuing without team AI analysis")
            ai_service = None

    # Load required data
    teams = load_teams()
    schedule = load_schedule()
    home_field_advantage = float(scrape_sagarin())
    sagarin_hash = generate_sagarin_hash()

    # Initialize cache structure with empty team analyses
    cache_data = {
        'timestamp': datetime.now().isoformat(),
        'num_simulations': 0,  # Initialize to 0, will be set properly later
        'sagarin_hash': sagarin_hash,
        'seed_multipliers': get_seed_multiplier_mapping(),
        'team_analyses': {},
        'playoff_odds': {'AFC': {}, 'NFC': {}},
        'super_bowl': {
            'appearances': {},
            'wins': {}
        }
    }

    # Initialize team_analyses with basic structure for each team
    for team_abbr, team_info in teams.items():
        cache_data['team_analyses'][team_abbr] = {
            'team': team_abbr,
            'city': team_info['city'],
            'mascot': team_info['mascot'],
            'conference': team_info['conference'],
            'division': team_info['division'],
            'division_position': 0,
            'playoff_chance': 0,
            'division_chance': 0,
            'top_seed_chance': 0,
            'super_bowl_appearance_chance': 0,
            'super_bowl_win_chance': 0,
            'num_simulations': num_simulations,
            'significant_games': []
        }

    # Try to load existing cache
    try:
        with open('data/analysis_cache.json', 'r') as f:
            existing_cache = json.load(f)
            logger.info("Loaded existing cache file")
            
            # Update cache data with existing values if skipping sims
            if skip_sims:
                # Preserve the existing number of simulations and timestamp
                cache_data['num_simulations'] = existing_cache.get('num_simulations', 0)
                cache_data['timestamp'] = existing_cache.get('timestamp', datetime.now().isoformat())
                logger.info(f"Preserving existing simulation count: {cache_data['num_simulations']}")
                cache_data['seed_multipliers'] = existing_cache.get(
                    'seed_multipliers',
                    cache_data['seed_multipliers']
                )
            else:
                # Use the new simulation count if running sims (timestamp already set above)
                cache_data['num_simulations'] = num_simulations
                logger.info(f"Using new simulation count: {cache_data['num_simulations']}")
                cache_data['seed_multipliers'] = get_seed_multiplier_mapping()

            for team_abbr in teams:
                if team_abbr in existing_cache.get('team_analyses', {}):
                    # Preserve all probability values from existing cache
                    existing_team = existing_cache['team_analyses'][team_abbr]
                    cache_data['team_analyses'][team_abbr].update({
                        'playoff_chance': existing_team.get('playoff_chance', 0),
                        'division_chance': existing_team.get('division_chance', 0),
                        'top_seed_chance': existing_team.get('top_seed_chance', 0),
                        'super_bowl_appearance_chance': existing_team.get('super_bowl_appearance_chance', 0),
                        'super_bowl_win_chance': existing_team.get('super_bowl_win_chance', 0),
                        'significant_games': existing_team.get('significant_games', []),
                        'num_simulations': existing_team.get('num_simulations', num_simulations)
                    })
            cache_data['playoff_odds'] = existing_cache.get('playoff_odds', {'AFC': {}, 'NFC': {}})
            cache_data['super_bowl'] = existing_cache.get('super_bowl', {'appearances': {}, 'wins': {}})
    except (FileNotFoundError, json.JSONDecodeError):
        logger.info("No existing cache file found or invalid format")
        existing_cache = {}
        # Set simulation count based on skip_sims flag
        cache_data['num_simulations'] = 0 if skip_sims else num_simulations
        logger.info(f"Setting initial simulation count: {cache_data['num_simulations']}")

    # Initialize all counters regardless of skip_sims
    playoff_appearances = defaultdict(int)
    division_wins = defaultdict(int)
    super_bowl_appearances = defaultdict(int)
    super_bowl_wins = defaultdict(int)
    top_seed_wins = defaultdict(int)
    game_impacts = defaultdict(lambda: defaultdict(lambda: {
        'home_wins_playoff': 0,
        'away_wins_playoff': 0,
        'home_wins_division': 0,
        'away_wins_division': 0,
        'home_wins_top_seed': 0,
        'away_wins_top_seed': 0,
        'home_wins_sb_appearance': 0,  # New field
        'away_wins_sb_appearance': 0,  # New field
        'home_wins_sb_win': 0,         # New field
        'away_wins_sb_win': 0,         # New field
        'home_wins_count': 0,
        'away_wins_count': 0,
        'total_sims': 0
    }))

    if skip_sims:
        # Load values from existing cache
        for team_abbr, team_data in existing_cache.get('team_analyses', {}).items():
            playoff_pct = team_data.get('playoff_chance', 0)
            division_pct = team_data.get('division_chance', 0)
            sb_appearance_pct = team_data.get('super_bowl_appearance_chance', 0)
            sb_win_pct = team_data.get('super_bowl_win_chance', 0)
            
            # Convert percentages back to counts
            playoff_appearances[team_abbr] = int((playoff_pct / 100) * num_simulations)
            division_wins[team_abbr] = int((division_pct / 100) * num_simulations)
            super_bowl_appearances[team_abbr] = int((sb_appearance_pct / 100) * num_simulations)
            super_bowl_wins[team_abbr] = int((sb_win_pct / 100) * num_simulations)

    if not skip_sims:
        # Initialize game_impacts before batch processing
        game_impacts = defaultdict(lambda: defaultdict(lambda: {
            'home_wins_playoff': 0,
            'away_wins_playoff': 0,
            'home_wins_division': 0,
            'away_wins_division': 0,
            'home_wins_top_seed': 0,
            'away_wins_top_seed': 0,
            'home_wins_sb_appearance': 0,  # New field
            'away_wins_sb_appearance': 0,  # New field
            'home_wins_sb_win': 0,         # New field
            'away_wins_sb_win': 0,         # New field
            'home_wins_count': 0,
            'away_wins_count': 0,
            'total_sims': 0
        }))
        
        # Process in batches
        BATCH_SIZE = 1000
        for batch_start in tqdm(range(0, num_simulations, BATCH_SIZE)):
            batch_size = min(BATCH_SIZE, num_simulations - batch_start)

            # Run one batch of simulations at a time to reuse loaded data inside simulate_season
            # Use seed + batch_start to ensure each batch has different but reproducible results
            batch_results = simulate_season(
                num_simulations=batch_size,
                home_field_advantage=home_field_advantage,
                random_seed=seed + batch_start
            )

            for result in batch_results:
                # Update counters immediately
                for team in result.get('division_winners', []):
                    playoff_appearances[team] += 1
                    division_wins[team] += 1

                for conf, teams_list in result.get('wild_cards', {}).items():
                    for team in teams_list:
                        playoff_appearances[team] += 1

                # Add top seed tracking - FIX: Track for each conference separately
                for conf, order in result.get('playoff_order', {}).items():
                    if order and len(order) > 0:  # Make sure we have a valid order
                        top_seed = order[0]  # First team is top seed
                        top_seed_wins[top_seed] += 1  # Count this as a top seed appearance

                if 'super_bowl' in result:
                    for team in result['super_bowl'].get('teams', []):
                        super_bowl_appearances[team] += 1
                    if 'winner' in result['super_bowl']:
                        super_bowl_wins[result['super_bowl']['winner']] += 1

            # Process game impacts for this batch
            analyze_batch_game_impacts(batch_results, game_impacts, teams)

        # After processing all batches, update the cache data with playoff odds
        cache_data['playoff_odds'] = {
            'AFC': {},
            'NFC': {}
        }
        
        # Calculate playoff odds for each team
        for team_abbr, team_info in teams.items():
            conf = team_info['conference']
            if conf not in cache_data['playoff_odds']:
                cache_data['playoff_odds'][conf] = {}
                
            cache_data['playoff_odds'][conf][team_abbr] = {
                'playoff': round((playoff_appearances[team_abbr] / num_simulations) * 100, 1),
                'division': round((division_wins[team_abbr] / num_simulations) * 100, 1)
            }
            
        # Update Super Bowl data
        cache_data['super_bowl'] = {
            'appearances': {},
            'wins': {}
        }
        
        for team_abbr in teams:
            cache_data['super_bowl']['appearances'][team_abbr] = round((super_bowl_appearances[team_abbr] / num_simulations) * 100, 1)
            cache_data['super_bowl']['wins'][team_abbr] = round((super_bowl_wins[team_abbr] / num_simulations) * 100, 1)
            # clear team's significant games
            cache_data['team_analyses'][team_abbr]['significant_games'] = []

        # After all batches, process game impacts for significant games
        # Get only relevant upcoming games
        relevant_games = get_relevant_games(schedule)
        logger.info(f"Found {len(relevant_games)} relevant games")

        # Clear each team's significant games before adding new ones
        for team_abbr in teams:
            if team_abbr in cache_data['team_analyses']:
                cache_data['team_analyses'][team_abbr]['significant_games'] = []
            if team_abbr not in cache_data['team_analyses']:
                cache_data['team_analyses'][team_abbr] = {
                    'playoff_chance': (playoff_appearances[team_abbr] / num_simulations) * 100,
                    'division_chance': (division_wins[team_abbr] / num_simulations) * 100,
                    'top_seed_chance': (top_seed_wins[team_abbr] / num_simulations) * 100,
                    'significant_games': []
                }
        for game in relevant_games:
            if game['home_score'] == '' and game['away_score'] == '':  # Unplayed game
                game_id = f"{game['away_team']}@{game['home_team']}"
                
                for team_abbr in teams:
                    total_impact, debug_stats = calculate_game_impact(
                        game_id, team_abbr, game_impacts
                    )

                    if total_impact > 0:
                        cache_data['team_analyses'][team_abbr]['significant_games'].append({
                            'date': game['game_date'],
                            'away_team': game['away_team'],
                            'home_team': game['home_team'],
                            'impact': round(total_impact, 1),  # Now includes seed impact
                            'gametime': game['gametime'],
                            'stadium': game['stadium'],
                            'week': int(game['week_num']),
                            'day': game.get('day_of_week', 'Sunday'),
                            'root_against': debug_stats['root_against'],
                            'espn_id': game['espn_id'],
                            'debug_stats': debug_stats
                        })

        # Sort significant games for each team by date/week then espn_id for consistent ordering across runs
        for team_analysis in cache_data['team_analyses'].values():
            if 'significant_games' in team_analysis:
                team_analysis['significant_games'].sort(key=lambda x: (x['week'], x['espn_id']))

    # After all batches are processed, but before saving
    standings_data = calculate_standings(teams, schedule)
    
    for team_abbr, team_info in teams.items():
        # Get division position
        conference = team_info['conference']
        division = team_info['division']
        division_teams = standings_data[conference][division]
        division_position = next(i for i, (t, _) in enumerate(division_teams, 1) if t == team_abbr)
        
        update_values = {
            'team': team_abbr,
            'city': team_info['city'],
            'mascot': team_info['mascot'],
            'conference': conference,
            'division': division,
            'division_position': division_position,
            'num_simulations': cache_data['num_simulations']  # Use the preserved simulation count
        }

        # Only update probability values if not skipping sims
        if not skip_sims:
            update_values.update({
                'playoff_chance': round((playoff_appearances[team_abbr] / num_simulations) * 100, 1),
                'division_chance': round((division_wins[team_abbr] / num_simulations) * 100, 1),
                'top_seed_chance': round((top_seed_wins[team_abbr] / num_simulations) * 100, 1),
                'super_bowl_appearance_chance': round((super_bowl_appearances[team_abbr] / num_simulations) * 100, 1),
                'super_bowl_win_chance': round((super_bowl_wins[team_abbr] / num_simulations) * 100, 1)
            })
        
        cache_data['team_analyses'][team_abbr].update(update_values)

    # PHASE 3: Generate AI analysis
    # Determine which teams need AI analysis
    teams_to_analyze = None
    if regenerate_team_ai:
        if regenerate_team_ai.lower() == 'all':
            # Regenerate all teams
            teams_to_analyze = list(teams.keys())
            logger.info(f"Regenerating AI analysis for all teams")
        else:
            # Parse comma-separated list of teams
            teams_to_analyze = [t.strip().upper() for t in regenerate_team_ai.split(',')]
            logger.info(f"Regenerating AI analysis for specific teams: {', '.join(teams_to_analyze)}")
            # Validate team codes
            invalid_teams = [t for t in teams_to_analyze if t not in teams]
            if invalid_teams:
                logger.error(f"Invalid team codes: {', '.join(invalid_teams)}")
                sys.exit(1)

    # First, preserve all existing AI analysis
    for team_abbr in teams.keys():
        if team_abbr in existing_cache.get('team_analyses', {}):
            if 'ai_analysis' in existing_cache['team_analyses'][team_abbr]:
                cache_data['team_analyses'][team_abbr]['ai_analysis'] = existing_cache['team_analyses'][team_abbr]['ai_analysis']
            if 'ai_status' in existing_cache['team_analyses'][team_abbr]:
                cache_data['team_analyses'][team_abbr]['ai_status'] = existing_cache['team_analyses'][team_abbr]['ai_status']

    if (not skip_team_ai or teams_to_analyze) and ai_service and ai_service.client:
        if teams_to_analyze:
            logger.info(f"Regenerating AI analysis for {len(teams_to_analyze)} team(s)...")
        else:
            logger.info("Generating AI analysis for all teams...")

        prompt_context = build_prompt_context(schedule)

        # Determine which teams to process
        if teams_to_analyze:
            teams_to_process = {k: v for k, v in teams.items() if k in teams_to_analyze}
        else:
            teams_to_process = teams

        # Generate AI analysis for each team
        total_teams = len(teams_to_process)
        is_ci = os.environ.get('CI') == 'true'

        # Log CI status
        logger.info(f"CI environment detected: {is_ci}")
        logger.info(f"CI env var value: '{os.environ.get('CI', 'not set')}'")
        
        # In CI mode, update every 25%, otherwise use default tqdm updates
        mininterval = 15.0 if is_ci else 0.1  # Seconds between updates
        miniters = max(1, total_teams // 4) if is_ci else 1  # Minimum iterations between updates
        
        logger.info(f"Progress update settings: mininterval={mininterval}s, miniters={miniters}")
        
        def run_team_analysis(team_abbr, team_info):
            team_record = get_team_record(standings_data, team_abbr)
            prompt = generate_team_analysis_prompt(
                team_abbr,
                team_info,
                team_record,
                teams,
                cache_data,
                standings_data,
                prompt_context=prompt_context
            )
            return ai_service.generate_analysis(prompt)

        max_workers = int(os.environ.get('AI_ANALYSIS_WORKERS', '3'))
        if max_workers < 1:
            max_workers = 1
        max_workers = min(max_workers, total_teams) if total_teams else 1
        logger.info(f"AI analysis concurrency: up to {max_workers} worker(s)")

        executor = ThreadPoolExecutor(max_workers=max_workers)
        try:
            future_to_team = {
                executor.submit(run_team_analysis, team_abbr, team_info): team_abbr
                for team_abbr, team_info in teams_to_process.items()
            }

            with tqdm(total=total_teams,
                      desc="Generating AI analysis",
                      mininterval=mininterval,
                      miniters=miniters) as pbar:
                for future in as_completed(future_to_team):
                    team_abbr = future_to_team[future]
                    try:
                        ai_analysis, status = future.result()
                        cache_data['team_analyses'][team_abbr]['ai_analysis'] = ai_analysis
                        cache_data['team_analyses'][team_abbr]['ai_status'] = status
                    except Exception as e:
                        logger.error(f"AI analysis failed for {team_abbr}: {e}")
                        cache_data['team_analyses'][team_abbr]['ai_analysis'] = None
                        cache_data['team_analyses'][team_abbr]['ai_status'] = 'error'
                    finally:
                        pbar.set_postfix(team=team_abbr)
                        pbar.update(1)
        except KeyboardInterrupt:
            logger.warning("\n\nKeyboardInterrupt received! Cancelling remaining tasks...")
            # Cancel all pending futures
            for future in future_to_team:
                future.cancel()
            # Shutdown executor immediately without waiting for running tasks
            executor.shutdown(wait=False, cancel_futures=True)
            logger.info("Shutdown complete. Exiting...")
            sys.exit(1)
        finally:
            # Ensure executor is always cleaned up
            executor.shutdown(wait=True)
    elif skip_team_ai and not teams_to_analyze:
        logger.info("Skipping team AI analysis generation (already preserved existing analysis above)")


    # Save cache file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(cache_data, f, indent=2)

    logger.info(f"Cache file generated successfully: {output_path}")
        
    return validate_cache(cache_data)

def deploy_to_netlify():
    """Deploy data files to who-should-lose-2 repo for Netlify deployment"""
    logger.info("Deploying data files to who-should-lose-2 repo...")

    # Temporary directory for the repo
    temp_dir = '/tmp/who-should-lose-2'
    repo_url = 'git@github.com:obliojoe/who-should-lose-2.git'

    try:
        # Clone or pull the repo
        if os.path.exists(temp_dir):
            logger.info("Pulling latest changes from who-should-lose-2...")
            subprocess.run(['git', '-C', temp_dir, 'pull'], check=True, capture_output=True)
        else:
            logger.info("Cloning who-should-lose-2 repo...")
            subprocess.run(['git', 'clone', repo_url, temp_dir], check=True, capture_output=True)

        # Create public/data directory if it doesn't exist
        data_dir = os.path.join(temp_dir, 'public', 'data')
        os.makedirs(data_dir, exist_ok=True)

        # Files to copy
        files_to_copy = [
            'data/analysis_cache.json',
            'data/team_stats.csv',
            'data/team_starters.csv',
            'data/schedule.csv',
            'data/teams.csv',
            'data/game_analyses.json',
            'data/team_notes.csv'
        ]

        # Optional files
        optional_files = [
            'data/standings_cache.json',
            'data/coordinators.csv',
            'data/sagarin.csv'
        ]

        for file in optional_files:
            if os.path.exists(file):
                files_to_copy.append(file)

        # Copy files to public/data
        for file in files_to_copy:
            if os.path.exists(file):
                dest = os.path.join(data_dir, os.path.basename(file))
                shutil.copy2(file, dest)
                logger.info(f"Copied {file} to {dest}")

        # Git operations
        os.chdir(temp_dir)

        # Check if there are changes
        result = subprocess.run(['git', 'status', '--porcelain'],
                              capture_output=True, text=True)

        if result.stdout.strip():
            logger.info("Changes detected, committing and pushing...")

            # Add all files in public/data
            subprocess.run(['git', 'add', 'public/data/'], check=True)

            # Commit
            commit_msg = f"Update data files - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            subprocess.run(['git', 'commit', '-m', commit_msg], check=True)

            # Push
            subprocess.run(['git', 'push'], check=True)

            logger.info("Successfully deployed to who-should-lose-2 repo. Netlify will auto-deploy.")
        else:
            logger.info("No changes to deploy.")

        return True

    except Exception as e:
        logger.error(f"Error deploying to Netlify: {e}")
        return False

def main():
    # Save the original directory at the start
    original_dir = os.getcwd()

    parser = argparse.ArgumentParser(description='Generate NFL analysis cache file')

    # Data Generation Options
    parser.add_argument('--skip-data', action='store_true',
                      help='Skip ALL data file generation (schedule, stats, standings, etc.)')
    parser.add_argument('--data-only', action='store_true',
                      help='Generate data files only, then exit (skip simulations and AI)')

    # Simulation Options
    parser.add_argument('--simulations', type=int, default=1000,
                      help='Number of simulations to run (default: 1000)')
    parser.add_argument('--skip-sims', action='store_true',
                      help='Skip running new simulations (use existing simulation data)')
    parser.add_argument('--seed', type=int, default=None,
                      help='Random seed for reproducible results')

    # AI Analysis Options
    parser.add_argument('--skip-team-ai', action='store_true',
                      help='Skip team AI analysis generation')
    parser.add_argument('--skip-game-ai', action='store_true',
                      help='Skip game AI analysis generation')
    parser.add_argument('--regenerate-team-ai', type=str,
                      help='Regenerate team AI for specific teams (comma-separated abbrs, e.g., "DET,MIN") or "all"')
    parser.add_argument('--regenerate-game-ai', type=str,
                      help='Regenerate game AI by ESPN IDs (e.g., "401772856,401772855"), or "analysis", "preview", or "all"')

    # Deprecated Options (kept for backward compatibility)
    parser.add_argument('--skip-ai', action='store_true',
                      help='DEPRECATED: Use --skip-team-ai and --skip-game-ai instead')
    parser.add_argument('--regenerate-ai', type=str,
                      help='DEPRECATED: Use --regenerate-team-ai instead')
    parser.add_argument('--no-copy-data', action='store_true',
                      help='DEPRECATED: No longer used')

    # Deployment Options
    parser.add_argument('--commit', action='store_true',
                      help='Commit changes to git if successful')
    parser.add_argument('--deploy-render', action='store_true',
                      help='Deploy data to Render web host via SSH')
    parser.add_argument('--deploy-netlify', action='store_true',
                      help='Deploy data files to who-should-lose-2 repo for Netlify deployment')
    parser.add_argument('--test-mode', action='store_true',
                      help='Run in test mode (disable AI calls)')

    args = parser.parse_args()
    logger.info(f"args: {args}")

    # Handle deprecated options with warnings
    if args.skip_ai:
        logger.warning("WARNING: --skip-ai is deprecated. Use --skip-team-ai and --skip-game-ai instead.")
        if not args.skip_team_ai:
            args.skip_team_ai = True
        if not args.skip_game_ai:
            args.skip_game_ai = True

    if args.regenerate_ai:
        logger.warning("WARNING: --regenerate-ai is deprecated. Use --regenerate-team-ai instead.")
        if not args.regenerate_team_ai:
            args.regenerate_team_ai = args.regenerate_ai

    if args.no_copy_data:
        logger.warning("WARNING: --no-copy-data is deprecated and no longer used.")

    copy_data = True  # Always copy data (no longer configurable)
    
    if not args.skip_data:
        logger.info("Generating data files")
        # Add score update at start
        logger.info("=> schedule.csv -- scores and dates")
        scores_updated = update_scores_and_dates()  # Capture return value
        logger.info(f"   Updated {scores_updated} game scores")  # Add log message

        # run team stats generation
        logger.info("=> Generating team_stats.csv")
        with contextlib.redirect_stdout(None):
            if not save_team_stats():
                logger.error("   Error: Failed to generate team stats")
                sys.exit(1)

        logger.info("=> Generating team_starters.csv")
        # run team starters generation
        with contextlib.redirect_stdout(None):
            if not save_team_starters("team_starters.csv"):
                logger.error("   Error: Failed to generate team starters")
                sys.exit(1)

        # Fetch coordinators data
        logger.info("=> Fetching coordinators.csv")
        try:
            from fetch_coordinators import save_coordinators_csv
            with contextlib.redirect_stdout(None):
                if not save_coordinators_csv():
                    logger.warning("   Warning: Failed to fetch coordinators - continuing without coordinator data")
        except Exception as e:
            logger.warning(f"Error fetching coordinators: {e}... continuing without coordinator data")

        # Generate standings cache (before AI analysis so it can potentially use this data)
        logger.info("=> Generating standings_cache.json")
        try:
            from calculate_standings_cache import main as generate_standings_cache
            with contextlib.redirect_stdout(None):
                generate_standings_cache()
        except Exception as e:
            logger.error(f"Error generating standings cache: {e}... continuing")

        # run game analyses generation (unless --skip-game-ai or --data-only)
        if not args.skip_game_ai and not args.data_only:
            logger.info("=> Generating game_analyses.json")
            try:
                # Determine regeneration parameters
                game_ids = None
                regenerate_type = None
                force_reanalyze = False

                if args.regenerate_game_ai:
                    if args.regenerate_game_ai.lower() in ['analysis', 'preview', 'all']:
                        regenerate_type = args.regenerate_game_ai.lower()
                        force_reanalyze = True
                    else:
                        # Treat as comma-separated ESPN IDs
                        game_ids = [gid.strip() for gid in args.regenerate_game_ai.split(',')]
                        force_reanalyze = True

                with contextlib.redirect_stdout(None):
                    batch_analyze_games(
                        force_reanalyze=force_reanalyze,
                        game_ids=game_ids,
                        regenerate_type=regenerate_type
                    )
            except Exception as e:
                logger.error(f"Error generating game analyses: {e}... continuing")
        elif args.skip_game_ai:
            logger.info("Skipping game AI analysis generation (--skip-game-ai)")
        elif args.data_only:
            logger.info("Skipping game AI analysis generation (--data-only)")

    else:
        logger.info("Skipping data files generation")

    # Handle game AI regeneration when --skip-data is used
    if args.skip_data and args.regenerate_game_ai and not args.skip_game_ai:
        logger.info("=> Generating game_analyses.json")
        try:
            # Determine regeneration parameters
            game_ids = None
            regenerate_type = None
            force_reanalyze = False

            if args.regenerate_game_ai.lower() in ['analysis', 'preview', 'all']:
                regenerate_type = args.regenerate_game_ai.lower()
                force_reanalyze = True
            else:
                # Treat as comma-separated ESPN IDs
                game_ids = [gid.strip() for gid in args.regenerate_game_ai.split(',')]
                force_reanalyze = True

            with contextlib.redirect_stdout(None):
                batch_analyze_games(
                    force_reanalyze=force_reanalyze,
                    game_ids=game_ids,
                    regenerate_type=regenerate_type
                )
        except Exception as e:
            logger.error(f"Error generating game analyses: {e}... continuing")

    # Exit early if --data-only
    if args.data_only:
        logger.info("Data generation complete (--data-only mode). Exiting.")
        return

    # Exit early if only regenerating game AI (and skipping sims)
    if args.regenerate_game_ai and args.skip_sims:
        logger.info("Game AI regeneration complete. Exiting (--skip-sims prevents further processing).")
        return

    success = generate_cache(
        num_simulations=args.simulations,
        skip_sims=args.skip_sims,
        skip_team_ai=args.skip_team_ai,
        copy_data=copy_data,
        test_mode=args.test_mode,
        regenerate_team_ai=args.regenerate_team_ai,
        seed=args.seed
    )

    # Removed persist directory copying - deployment now handled by --deploy-netlify

    if args.deploy_netlify and success:
        if not deploy_to_netlify():
            success = False

    if args.deploy_render:
        logger.info("Deploying files to Render web host via SSH...")
        try:
            # Remote server details
            remote_host = "srv-csmkmerv2p9s73bjo4eg@ssh.oregon.render.com"
            remote_path = "/opt/render/project/src/persist/"

            # Files to copy (required)
            files_to_copy = [
                'data/analysis_cache.json',
                'data/team_stats.csv',
                'data/team_starters.csv',
                'data/schedule.csv',
                'data/teams.csv',
                'data/game_analyses.json',
                'data/team_notes.csv'
            ]

            # Optional files to copy if they exist
            optional_files = [
                'data/coordinators.csv',
                'data/sagarin.csv',
                'data/standings_cache.json'
            ]
            for optional_file in optional_files:
                if os.path.exists(optional_file):
                    files_to_copy.append(optional_file)

            # Copy each file using scp
            for file in files_to_copy:
                scp_command = f"scp {file} {remote_host}:{remote_path}"
                logger.info(f"Copying {file}...")
                result = subprocess.run(scp_command, shell=True, capture_output=True, text=True)
                if result.returncode != 0:
                    raise Exception(f"SCP failed: {result.stderr}")

            logger.info("Successfully copied files to remote server")

        except Exception as e:
            logger.error(f"Error copying files via SSH: {e}")
            success = False

    # Git operations (only if files were copied successfully)
    if args.commit:
        try:
            logger.info("RUNNING GIT OPERATIONS...")
            # Change back to original directory
            os.chdir(original_dir)

            # Check for credentials in environment
            username = os.environ.get('GH_USERNAME')
            token = os.environ.get('GH_PAT')

            # Configure git user if in CI environment
            if os.environ.get('CI'):
                logger.info("configuring git user for CI")
                subprocess.run(['git', 'config', '--global', 'user.email', "github-actions@github.com"], check=True)
                subprocess.run(['git', 'config', '--global', 'user.name', "GitHub Actions"], check=True)

            # If we have credentials, configure git to use them for HTTPS
            if username and token:
                logger.info("Using credentials from environment for git authentication")
                # Set up credential helper with the token
                subprocess.run(['git', 'config', '--local', 'credential.helper',
                               f'!f() {{ echo "username={username}"; echo "password={token}"; }}; f'],
                               check=True)

            # Pull latest changes
            try:
                subprocess.run(['git', 'pull'], check=True, capture_output=True)
            except subprocess.CalledProcessError:
                logger.info("pull failed, continuing anyway...")
                pass

            # Add ALL data files (whether changed by script or not)
            subprocess.run(['git', 'add', 'data/'], check=True)
            logger.info("   Added all files in data/")

            # Check if there are staged changes
            result = subprocess.run(['git', 'diff', '--cached', '--quiet'], capture_output=True)
            if result.returncode != 0:  # Non-zero means there are changes
                commit_msg = f"generate_cache.py update ({'remote' if is_ci else 'local'}) - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                subprocess.run(['git', 'commit', '-m', commit_msg], check=True)
                logger.info(f"Committed changes to data files")
            else:
                logger.info("No changes to commit")

            # Push changes (use default git authentication)
            logger.info("pushing changes")
            subprocess.run(['git', 'push', '-u', 'origin', 'main'], check=True)
            logger.info("Successfully committed and pushed changes")

        except Exception as e:
            logger.error(f"Script completed, but error during git operations: {e}")
            success = False

    end_time = datetime.now()
    time_diff = end_time - script_start_time
    minutes = int(time_diff.total_seconds() // 60)
    seconds = int(time_diff.total_seconds() % 60)
    print(f"Total cache generation took {minutes} minutes and {seconds} seconds")
    
    # Exit with failure status if anything failed
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()