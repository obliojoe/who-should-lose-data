import nflreadpy as nfl
import pandas as pd
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def get_current_starters():
    current_year = 2025
    logger.info("load_rosters_weekly()...")
    # Get roster and depth chart data
    roster_data = nfl.load_rosters_weekly([current_year]).to_pandas()
    logger.info("load_depth_charts()...")
    depth_charts = nfl.load_depth_charts([current_year]).to_pandas()
    logger.info("load_player_stats()...")
    player_stats = nfl.load_player_stats([current_year]).to_pandas()
    logger.info("done!")

    # Get the most recent week's data
    latest_week = roster_data['week'].max()
    latest_roster = roster_data[roster_data['week'] == latest_week].copy()

    # Filter for starters (pos_rank == 1)
    # Include pos_rank 2 for RB to handle committees
    starters = depth_charts[
        (
            (depth_charts['pos_rank'] == 1) |
            ((depth_charts['pos_rank'] == 2) & (depth_charts['pos_abb'] == 'RB'))
        )
    ].copy()

    # Deduplicate - keep only one entry per player (prioritize non-special teams formations)
    # Sort so non-special teams formations come first
    starters['is_special_teams'] = starters['pos_grp'].str.contains('Special Teams', case=False, na=False)
    starters = starters.sort_values('is_special_teams')

    # Drop duplicates, keeping first (non-special teams when possible)
    starters = starters.drop_duplicates(subset=['gsis_id', 'team'], keep='first')
    starters = starters.drop(columns=['is_special_teams'])

    # Filter out players without valid teams
    starters = starters[starters['team'].notna() & (starters['team'] != '')]

    # Merge with roster data using GSIS ID
    starters = pd.merge(
        starters,
        latest_roster,
        left_on='gsis_id',
        right_on='gsis_id',
        how='left',
        suffixes=('_depth', '_roster')
    )

    # Merge with player stats for the most recent week
    latest_player_stats = player_stats[player_stats['week'] == latest_week]
    starters = pd.merge(
        starters,
        latest_player_stats,
        left_on='gsis_id',
        right_on='player_id',
        how='left',
        suffixes=('', '_stats')
    )

    # Aggregate season stats
    season_stats = player_stats.groupby('player_id').agg({
        'completions': 'sum',
        'attempts': 'sum',
        'passing_yards': 'sum',
        'passing_tds': 'sum',
        'passing_interceptions': 'sum',
        'carries': 'sum',
        'rushing_yards': 'sum',
        'rushing_tds': 'sum',
        'receptions': 'sum',
        'targets': 'sum',
        'receiving_yards': 'sum',
        'receiving_tds': 'sum'
    }).reset_index()

    starters = pd.merge(
        starters,
        season_stats,
        left_on='gsis_id',
        right_on='player_id',
        how='left',
        suffixes=('_week', '_season')
    )
    
    # Convert numeric fields to int
    int_columns = [
        'jersey_number', 'years_exp', 'entry_year', 'rookie_year', 'draft_number',
        'weight'
    ]
    for col in int_columns:
        if col in starters.columns:
            starters[col] = pd.to_numeric(starters[col], errors='coerce').fillna(0).astype(int)

    # Calculate age from birth_date if available
    if 'birth_date' in starters.columns:
        starters['age'] = pd.to_datetime('today').year - pd.to_datetime(starters['birth_date'], errors='coerce').dt.year

    # Count games played this season
    games_played = player_stats.groupby('player_id')['week'].nunique().reset_index()
    games_played.columns = ['player_id', 'games_played']
    starters = pd.merge(starters, games_played, left_on='gsis_id', right_on='player_id', how='left')

    # Standardize team codes AFTER all merges (use team_depth which is the original from depth chart)
    team_map = {'LA': 'LAR'}
    if 'team_depth' in starters.columns:
        starters['team'] = starters['team_depth'].replace(team_map)
    elif 'team' in starters.columns:
        starters['team'] = starters['team'].replace(team_map)

    # Select and rename columns for clarity
    base_columns = {
        'team': 'team_abbr',
        'pos_abb': 'position',
        'full_name': 'player_name',
        'jersey_number': 'number',
        'pos_grp': 'formation',
        'status': 'status',
        'height': 'height',
        'weight': 'weight',
        'college': 'college',
        'years_exp': 'experience',
        'age': 'age',
        'birth_date': 'birth_date',
        'entry_year': 'entry_year',
        'draft_number': 'draft_pick',
        # Basic stats - Weekly (from latest week)
        'completions_week': 'completions_week',
        'attempts_week': 'attempts_week',
        'passing_yards_week': 'passing_yards_week',
        'passing_tds_week': 'passing_tds_week',
        'passing_interceptions': 'interceptions_week',
        'carries_week': 'carries_week',
        'rushing_yards_week': 'rushing_yards_week',
        'rushing_tds_week': 'rushing_tds_week',
        'receptions_week': 'receptions_week',
        'targets_week': 'targets_week',
        'receiving_yards_week': 'receiving_yards_week',
        'receiving_tds_week': 'receiving_tds_week',
        # Basic stats - Season (aggregated)
        'completions_season': 'completions_season',
        'attempts_season': 'attempts_season',
        'passing_yards_season': 'passing_yards_season',
        'passing_tds_season': 'passing_tds_season',
        'passing_interceptions_season': 'interceptions_season',
        'carries_season': 'carries_season',
        'rushing_yards_season': 'rushing_yards_season',
        'rushing_tds_season': 'rushing_tds_season',
        'receptions_season': 'receptions_season',
        'targets_season': 'targets_season',
        'receiving_yards_season': 'receiving_yards_season',
        'receiving_tds_season': 'receiving_tds_season',
        'games_played': 'games_played'
    }

    # Only select columns that exist
    available_columns = {k: v for k, v in base_columns.items() if k in starters.columns}
    starters = starters[list(available_columns.keys())].rename(columns=available_columns)

    # Fill NaN values in stats columns with 0
    stat_columns = [col for col in starters.columns if any(x in col for x in ['_week', '_season', 'pct', 'per', 'rating'])]
    starters[stat_columns] = starters[stat_columns].fillna(0)
    
    # Save to CSV in data directory
    os.makedirs('data', exist_ok=True)
    csv_path = os.path.join('data', 'team_starters.csv')
    starters.to_csv(csv_path, index=False)
    print(f"\nSaved starters to {csv_path}")
    print(f"Data from Week {latest_week}")
    print(f"\nMerge success rate:")
    print(f"Total rows: {len(starters)}")
    print(f"Rows with roster data: {len(starters.dropna(subset=['status']))}")
    
    return starters

def save_team_starters(filename):
    try:
        team_starters_df = get_current_starters()
        
        # Save to CSV without index
        csv_path = os.path.join('data', filename)
        team_starters_df.to_csv(csv_path, index=False)
        return True
    except Exception as e:
        print(f"Error saving {filename}: {e}")
        return False

if __name__ == "__main__":
    starters = get_current_starters()
    save_team_starters('team_starters.csv')