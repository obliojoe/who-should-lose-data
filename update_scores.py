import nflreadpy as nfl
import pandas as pd
import json
import numpy as np
from datetime import datetime
import requests
import pytz
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def normalize_team_name(team):
    """Convert team abbreviations to consistent format"""
    team_mapping = {
        'LA': 'LAR',  # Rams
        'LAR': 'LAR'  # Ensure LAR maps to itself
    }
    return team_mapping.get(team, team)

def get_game_date(game_id):
    """Fetch ESPN metadata and provide updated kickoff details for schedule.json."""
    espn_date = None
    game_date = None
    game_time = None
    game_day = None

    logger.info(f"Fetching game data for {game_id}...")
    url = f"https://cdn.espn.com/core/nfl/boxscore?xhr=1&gameId={game_id}"
    response = requests.get(url)
    if response.status_code == 200:
        try:
            data = response.json()
            espn_date = data['gamepackageJSON']['header']['competitions'][0]['date']
        except (KeyError, IndexError) as e:
            logger.error(f"Failed to parse game data for {game_id}: {e}")
            return None, None, None
    else:
        logger.error(f"Failed to fetch game data for {game_id}")
        return None, None, None

    if espn_date:
        # Parse the UTC datetime string
        utc_dt = datetime.strptime(espn_date, '%Y-%m-%dT%H:%M%z')
        utc_dt = utc_dt.replace(tzinfo=pytz.UTC)

        # Convert to Eastern Time
        eastern = pytz.timezone('America/New_York')
        eastern_dt = utc_dt.astimezone(eastern)

        # Format the date and time
        game_date = eastern_dt.strftime('%Y-%m-%d')
        game_time = eastern_dt.strftime('%H:%M')

        # if the time is 00:00, make it 13:00
        if game_time == '00:00':
            game_time = '13:00'

        # get the day of the week
        game_day = eastern_dt.strftime('%A')

    return game_date, game_time, game_day

def update_espn_ids(local_schedule: pd.DataFrame, latest_schedule: pd.DataFrame) -> pd.DataFrame:
    """Update ESPN IDs from latest schedule data."""

    espn_updates = []

    latest_schedule = latest_schedule.copy()
    latest_schedule['away_team'] = latest_schedule['away_team'].replace('LA', 'LAR')
    latest_schedule['home_team'] = latest_schedule['home_team'].replace('LA', 'LAR')

    for idx, game in local_schedule.iterrows():
        if pd.isna(game['away_team']) or pd.isna(game['home_team']) or pd.isna(game['week_num']):
            continue

        away_team = normalize_team_name(game['away_team'])
        home_team = normalize_team_name(game['home_team'])

        matching_game = latest_schedule[
            (latest_schedule['week'] == game['week_num']) &
            (latest_schedule['away_team'] == away_team) &
            (latest_schedule['home_team'] == home_team)
        ]

        if not matching_game.empty and not pd.isna(matching_game.iloc[0]['espn']):
            latest_espn_id = int(matching_game.iloc[0]['espn'])
            try:
                current_espn_id = int(game['espn_id']) if pd.notna(game['espn_id']) else None
            except (ValueError, TypeError):
                current_espn_id = None

            if current_espn_id != latest_espn_id:
                espn_updates.append((idx, latest_espn_id))
                logger.info(
                    "Will update ESPN ID for %s @ %s (Week %s): %s -> %s",
                    away_team,
                    home_team,
                    game['week_num'],
                    current_espn_id,
                    latest_espn_id,
                )

    for idx, espn_id in espn_updates:
        local_schedule.at[idx, 'espn_id'] = espn_id

    if espn_updates:
        logger.info("Updated ESPN IDs for %d games", len(espn_updates))
    else:
        logger.info("No ESPN IDs to update")

    return local_schedule

def update_scores_and_dates(year=2025, update_dates=False):
    """Update schedule.json with scores (and optionally dates) from completed games."""

    logger.info("Checking for score and date updates...")

    # Load current schedule
    with open('data/schedule.json', 'r', encoding='utf-8') as fh:
        local_schedule = pd.DataFrame(json.load(fh))
    logger.debug(f"Loaded {len(local_schedule)} games from local schedule")

    # Get latest schedule data from nfl_data_py
    logger.debug(f"Loading schedules from nfl_data_py...")
    latest_schedule = nfl.load_schedules([year]).to_pandas()
    logger.debug(f"Loaded {len(latest_schedule)} games from nfl_data_py")

    # First, update ESPN IDs
    logger.info("=== Updating ESPN IDs ===")
    local_schedule = update_espn_ids(local_schedule, latest_schedule)

    # Normalize team names in latest schedule for score matching (convert LA to LAR)
    latest_schedule = latest_schedule.copy()
    latest_schedule['away_team'] = latest_schedule['away_team'].replace('LA', 'LAR')
    latest_schedule['home_team'] = latest_schedule['home_team'].replace('LA', 'LAR')
    
    # Create a dictionary of completed games from latest data
    completed_games = {}
    for _, game in latest_schedule.iterrows():
        if not pd.isna(game['away_score']) and not pd.isna(game['home_score']):
            # Normalize team names
            away_team = normalize_team_name(game['away_team'])
            home_team = normalize_team_name(game['home_team'])
            key = f"{game['week']}_{away_team}_{home_team}"
            completed_games[key] = {
                'away_score': int(game['away_score']),
                'home_score': int(game['home_score'])
            }
    
    logger.debug(f"Found {len(completed_games)} completed games in latest data")
    
    # Store score updates
    score_updates = []
    date_updates = []
    
    # Find games that need updating
    for idx, game in local_schedule.iterrows():
        # Skip rows with missing team data
        if pd.isna(game['away_team']) or pd.isna(game['home_team']) or pd.isna(game['week_num']):
            continue

        # update the date and time
        if update_dates and not pd.isna(game['espn_id']):
            espn_id = int(float(game['espn_id']))  # Convert to int to remove .0
            espn_date, espn_time, espn_day = get_game_date(espn_id)
            if espn_date and espn_time and espn_day:  # Only update if we got valid data
                if (espn_date != game['game_date']) or (espn_time != game['gametime']) or (espn_day != game['weekday']):
                    date_updates.append((idx, espn_date, espn_time, espn_day))
                    logger.debug(f"Will update date for {game['away_team']} @ {game['home_team']} (Week {game['week_num']}): {espn_date} {espn_time} ({espn_day})")
        
        # Create key to match with completed games using normalized team names
        away_team = normalize_team_name(game['away_team'])
        home_team = normalize_team_name(game['home_team'])
        key = f"{game['week_num']}_{away_team}_{home_team}"
        
        if key in completed_games:
            latest_scores = completed_games[key]
            
            # Convert current scores to integers if they exist, otherwise use None
            try:
                current_away = int(float(game['away_score'])) if pd.notna(game['away_score']) else None
                current_home = int(float(game['home_score'])) if pd.notna(game['home_score']) else None
            except ValueError:
                current_away = None
                current_home = None
            
            # Update if either score is different or missing
            if current_away != latest_scores['away_score'] or current_home != latest_scores['home_score']:
                score_updates.append((idx, latest_scores['away_score'], latest_scores['home_score']))
                logger.debug(f"Will update score for {game['away_team']} @ {game['home_team']} (Week {game['week_num']})")
    
    for idx, away_score, home_score in score_updates:
        local_schedule.at[idx, 'away_score'] = away_score
        local_schedule.at[idx, 'home_score'] = home_score

    for idx, date, time, day in date_updates:
        local_schedule.at[idx, 'game_date'] = date
        local_schedule.at[idx, 'gametime'] = time
        local_schedule.at[idx, 'weekday'] = day

    local_schedule = local_schedule.sort_values(by=['game_date', 'gametime'])

    output_records = local_schedule.replace({np.nan: None}).to_dict(orient='records')
    with open('data/schedule.json', 'w', encoding='utf-8') as fh:
        json.dump(output_records, fh, indent=2)
    return len(score_updates)

if __name__ == "__main__":
    update_scores_and_dates()
