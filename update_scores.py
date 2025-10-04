import nflreadpy as nfl
import pandas as pd
import csv
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
    """Update schedule.csv with dates from completed games"""
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

def update_espn_ids(local_schedule, latest_schedule):
    """Update ESPN IDs from latest schedule data"""
    espn_updates = []

    # Normalize team names in latest schedule (convert LA to LAR)
    latest_schedule = latest_schedule.copy()
    latest_schedule['away_team'] = latest_schedule['away_team'].replace('LA', 'LAR')
    latest_schedule['home_team'] = latest_schedule['home_team'].replace('LA', 'LAR')

    for idx, game in local_schedule.iterrows():
        # Skip rows with missing team data
        if pd.isna(game['away_team']) or pd.isna(game['home_team']) or pd.isna(game['week_num']):
            continue

        # Normalize team names
        away_team = normalize_team_name(game['away_team'])
        home_team = normalize_team_name(game['home_team'])

        # Find matching game in latest schedule
        matching_game = latest_schedule[
            (latest_schedule['week'] == game['week_num']) &
            (latest_schedule['away_team'] == away_team) &
            (latest_schedule['home_team'] == home_team)
        ]

        if not matching_game.empty and not pd.isna(matching_game.iloc[0]['espn']):
            latest_espn_id = int(matching_game.iloc[0]['espn'])

            # Check if ESPN ID is different or missing
            try:
                current_espn_id = int(game['espn_id']) if pd.notna(game['espn_id']) else None
            except (ValueError, TypeError):
                current_espn_id = None

            if current_espn_id != latest_espn_id:
                espn_updates.append((idx, latest_espn_id))
                logger.info(f"Will update ESPN ID for {away_team} @ {home_team} (Week {game['week_num']}): {current_espn_id} -> {latest_espn_id}")

    if espn_updates:
        # Read the CSV file
        with open('data/schedule.csv', 'r', newline='') as f:
            rows = list(csv.reader(f))
            header = rows[0]
            espn_id_idx = header.index('espn_id')

            # Update ESPN ID cells
            for idx, espn_id in espn_updates:
                rows[idx + 1][espn_id_idx] = str(espn_id)

        # Write back to CSV
        with open('data/schedule.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(rows)

        logger.info(f"Updated ESPN IDs for {len(espn_updates)} games")

        # Reload the schedule after ESPN ID updates
        return pd.read_csv('data/schedule.csv')
    else:
        logger.info("No ESPN IDs to update")
        return local_schedule

def update_scores_and_dates(year=2025, update_dates=False):
    """Update schedule.csv with scores from completed games"""

    print("Checking for games to update...")

    # Load current schedule
    local_schedule = pd.read_csv('data/schedule.csv')
    print(f"Loaded {len(local_schedule)} games from local schedule")

    # Get latest schedule data from nfl_data_py
    logger.info(f"import_schedules([{year}])...")
    latest_schedule = nfl.load_schedules([year]).to_pandas()
    print(f"Loaded {len(latest_schedule)} games from nfl_data_py")

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
    
    print(f"Found {len(completed_games)} completed games in latest data")
    
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
                    print(f"==> Will update {game['away_team']} @ {game['home_team']} (Week {game['week_num']}): {espn_date} {espn_time} ({espn_day})")
        
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
                print(f" => Will update {game['away_team']} @ {game['home_team']} (Week {game['week_num']}):")
    
    if score_updates:
        # Read the CSV file
        with open('data/schedule.csv', 'r', newline='') as f:
            rows = list(csv.reader(f))
            header = rows[0]
            
            # Get column indices for scores
            away_score_idx = header.index('away_score')
            home_score_idx = header.index('home_score')
            
            # Update only the score cells
            for idx, away_score, home_score in score_updates:
                rows[idx + 1][away_score_idx] = str(away_score)
                rows[idx + 1][home_score_idx] = str(home_score)

        # Write back to CSV, preserving exact format
        with open('data/schedule.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(rows)
            
        # print(f"Updated scores for {len(score_updates)} games")
    else:
        # print("No new scores to update")
        pass
    
    if date_updates:
        # date field is game_date in the format yyyy-mm-dd 
        # time field is gametime in the format HH:MM
        with open('data/schedule.csv', 'r', newline='') as f:
            rows = list(csv.reader(f))
            header = rows[0]
            date_idx = header.index('game_date')
            time_idx = header.index('gametime')
            day_idx = header.index('weekday')
            for idx, date, time, day in date_updates:
                rows[idx + 1][date_idx] = date
                rows[idx + 1][time_idx] = time
                rows[idx + 1][day_idx] = day

        # Write back to CSV, preserving exact format
        with open('data/schedule.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(rows)
            
        # print(f"Updated dates for {len(date_updates)} games")
    else:
        # print("No new dates to update")
        pass
    
    # reorder the schedule.csv by date and time
    local_schedule = pd.read_csv('data/schedule.csv')

    # Convert score columns to integers, replacing NaN with empty string
    local_schedule['away_score'] = local_schedule['away_score'].apply(lambda x: str(int(x)) if pd.notna(x) else '')
    local_schedule['home_score'] = local_schedule['home_score'].apply(lambda x: str(int(x)) if pd.notna(x) else '')

    # Convert espn_id to integers, replacing NaN with empty string
    local_schedule['espn_id'] = local_schedule['espn_id'].apply(lambda x: str(int(float(x))) if pd.notna(x) else '')

    # Sort the schedule
    local_schedule = local_schedule.sort_values(by=['game_date', 'gametime'])

    # Write to CSV with scores as integers
    local_schedule.to_csv('data/schedule.csv', index=False)
    return len(score_updates)

if __name__ == "__main__":
    update_scores_and_dates()
