import nflreadpy as nfl
import pandas as pd
import numpy as np
import requests
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def calculate_conversion_rates(pbp_data, team):
    """Calculate offensive and defensive conversion rates"""
    # Offensive conversions
    offense_plays = pbp_data[pbp_data['posteam'] == team]
    
    # Third downs - only run and pass plays
    third_downs_off = offense_plays[
        (offense_plays['down'] == 3) & 
        ((offense_plays['play_type'] == 'run') | (offense_plays['play_type'] == 'pass'))
    ]
    third_attempts_off = len(third_downs_off)
    third_conversions_off = len(third_downs_off[third_downs_off['yards_gained'] >= third_downs_off['ydstogo']])
    
    # Fourth downs
    fourth_downs_off = offense_plays[
        (offense_plays['down'] == 4) & 
        ((offense_plays['play_type'] == 'run') | (offense_plays['play_type'] == 'pass'))
    ]
    fourth_attempts_off = len(fourth_downs_off)
    fourth_conversions_off = len(fourth_downs_off[fourth_downs_off['yards_gained'] >= fourth_downs_off['ydstogo']])
    
    # Defensive conversions
    defense_plays = pbp_data[pbp_data['defteam'] == team]
    
    # Third downs against
    third_downs_def = defense_plays[
        (defense_plays['down'] == 3) & 
        ((defense_plays['play_type'] == 'run') | (defense_plays['play_type'] == 'pass'))
    ]
    third_attempts_def = len(third_downs_def)
    third_conversions_def = len(third_downs_def[third_downs_def['yards_gained'] >= third_downs_def['ydstogo']])
    
    # Fourth downs against
    fourth_downs_def = defense_plays[
        (defense_plays['down'] == 4) & 
        ((defense_plays['play_type'] == 'run') | (defense_plays['play_type'] == 'pass'))
    ]
    fourth_attempts_def = len(fourth_downs_def)
    fourth_conversions_def = len(fourth_downs_def[fourth_downs_def['yards_gained'] >= fourth_downs_def['ydstogo']])
    
    return {
        'third_down_attempts': third_attempts_off,
        'third_down_conversions': third_conversions_off,
        'third_down_pct': (third_conversions_off / third_attempts_off * 100) if third_attempts_off > 0 else 0,
        'fourth_down_attempts': fourth_attempts_off,
        'fourth_down_conversions': fourth_conversions_off,
        'fourth_down_pct': (fourth_conversions_off / fourth_attempts_off * 100) if fourth_attempts_off > 0 else 0,
        'third_down_attempts_against': third_attempts_def,
        'third_down_conversions_against': third_conversions_def,
        'third_down_pct_against': (third_conversions_def / third_attempts_def * 100) if third_attempts_def > 0 else 0,
        'fourth_down_attempts_against': fourth_attempts_def,
        'fourth_down_conversions_against': fourth_conversions_def,
        'fourth_down_pct_against': (fourth_conversions_def / fourth_attempts_def * 100) if fourth_attempts_def > 0 else 0
    }

def calculate_red_zone_stats(pbp_data, team):
    """Calculate red zone efficiency"""
    # Filter for red zone plays, excluding special teams
    red_zone_plays = pbp_data[
        (pbp_data['yardline_100'] < 20) & 
        (pbp_data['play_type'] != 'kickoff') &
        (pbp_data['play_type'] != 'extra_point') &
        (pbp_data['play_type_nfl'] != 'PAT2') &
        (pbp_data['two_point_attempt'] == 0) &
        (pd.isna(pbp_data['down']) == False)
    ]
    

    # Offensive red zone stats
    offense_rz = red_zone_plays[red_zone_plays['posteam'] == team]
    # Count unique drives that reached red zone
    rz_trips = len(offense_rz.groupby(['game_id', 'drive']).size())
    # Count touchdowns in red zone - ONLY INCLUDES TOUCHDOWNS FOR THIS TEAM
    rz_tds = len(offense_rz[
        (offense_rz['touchdown'] == 1) &
        (offense_rz['posteam'] == team) &
        (offense_rz['td_team'] == team)
    ].groupby(['game_id', 'drive']).size())
    
    # Defensive red zone stats
    defense_rz = red_zone_plays[red_zone_plays['defteam'] == team]
    # Count unique drives against that reached red zone
    rz_trips_against = len(defense_rz.groupby(['game_id', 'drive']).size())
    # Count touchdowns allowed in red zone
    rz_tds_against = len(defense_rz[
        (defense_rz['touchdown'] == 1) &
        (defense_rz['defteam'] == team) &
        (defense_rz['td_team'] != team)
    ].groupby(['game_id', 'drive']).size())
    
    return {
        'red_zone_trips': rz_trips,
        'red_zone_tds': rz_tds,
        'red_zone_pct': (rz_tds / rz_trips * 100) if rz_trips > 0 else 0,
        'red_zone_trips_against': rz_trips_against,
        'red_zone_tds_against': rz_tds_against,
        'red_zone_pct_against': (rz_tds_against / rz_trips_against * 100) if rz_trips_against > 0 else 0
    }

def get_espn_standings_data(teams_df):
    """Get additional standings data from ESPN API that isn't in nfl_data_py"""
    try:
        # Base URL for both conferences
        base_url = "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/seasons/2025/types/2/groups/{}/standings/0?lang=en&region=us"
        
        # Get AFC (group 8) and NFC (group 7) standings
        afc_response = requests.get(base_url.format(8))
        nfc_response = requests.get(base_url.format(7))
        
        # Combine the standings data
        afc_data = afc_response.json()
        nfc_data = nfc_response.json()
        combined_data = afc_data['standings'] + nfc_data['standings']
        
        # Extract only the fields we don't get from nfl_data_py
        espn_stats = {}
        for team_data in combined_data:
            # Get team details
            team_ref = team_data['team']['$ref']
            team_id = team_ref.split('/')[-1].split('?')[0]
            # Convert team_id to int for comparison since it comes as string from API
            team_id_int = int(team_id)
            team = teams_df[teams_df['espn_api_id'] == team_id_int]
            team_abbr = team['team_abbr'].values[0]
            logger.info(f"Fetching team details for {team_abbr}...")
            team_response = requests.get(team_ref)
            team_details = team_response.json()
            team_id = team_details['id']
            
            # Get overall record stats
            overall_record = next(r for r in team_data['records'] if r['name'] == 'overall')
            stats = {stat['name']: stat for stat in overall_record['stats']}
            
            # Get other record types
            div_record = next(r for r in team_data['records'] if r['name'] == 'vs. Div.')
            conf_record = next(r for r in team_data['records'] if r['name'] == 'vs. Conf.')
            road_record = next(r for r in team_data['records'] if r['name'] == 'Road')
            
            # Store only the fields we don't already have
            espn_stats[team_id] = {
                'espn_api_id': team_id,
                'league_win_pct': stats['leagueWinPercent']['value'],
                'div_win_pct': stats['divisionWinPercent']['value'],
                'games_behind': stats['gamesBehind']['value'],
                'ot_wins': stats['OTWins']['value'],
                'ot_losses': stats['OTLosses']['value'],
                'road_record': road_record['displayValue'],
                'conf_record': conf_record['displayValue'],
                'div_record': div_record['displayValue'],
                'playoff_seed': stats['playoffSeed']['value'],
                'clincher': stats['clincher']['displayValue'] if 'clincher' in stats else '',
                'streak_display': stats['streak']['displayValue']
            }
        
        return espn_stats
    except Exception as e:
        print(f"Error fetching ESPN standings data: {e}")
        return {}

def generate_team_stats():
    """Generate comprehensive team statistics"""
    # Import data
    logger.info("Loading nflreadpy team_stats (already aggregated)...")
    nfl_team_stats = nfl.load_team_stats([2025]).to_pandas()

    logger.info("schedule()...")
    schedule = nfl.load_schedules([2025]).to_pandas()
    logger.info("pbp_data()...")
    pbp_data = nfl.load_pbp([2025]).to_pandas()
    logger.info("teams_df()...")
    teams_df = pd.read_csv('data/teams.csv')
    logger.info("get_espn_standings_data()...")
    espn_stats = get_espn_standings_data(teams_df)
    logger.info("done!")

    # Create ESPN ID to team abbreviation mapping
    espn_id_to_abbr = dict(zip(teams_df['espn_api_id'].astype(str), teams_df['team_abbr']))
    
    # Get list of teams and convert LA to LAR
    teams = sorted(schedule['home_team'].unique())
    teams = ['LAR' if x == 'LA' else x for x in teams]

    # Aggregate nflreadpy team stats by team (sum across all weeks)
    nfl_team_stats['team_abbr'] = nfl_team_stats['team'].replace({'LA': 'LAR'})
    aggregated_stats = nfl_team_stats.groupby('team_abbr').sum(numeric_only=True).to_dict('index')

    # Initialize stats dictionary with nflreadpy data
    team_stats = {}

    for team in teams:
        print(f"Processing {team}...")

        # Start with nflreadpy aggregated stats (already has most of what we need!)
        team_stats[team] = aggregated_stats.get(team, {}).copy()

        # Convert LAR back to LA for data lookup in schedule/pbp
        lookup_team = 'LA' if team == 'LAR' else team

        # Get completed games for calculating win/loss record and per-game averages
        team_games = schedule[
            ((schedule['home_team'] == lookup_team) | (schedule['away_team'] == lookup_team)) &
            (~pd.isna(schedule['home_score'])) &
            (~pd.isna(schedule['away_score']))
        ]

        # Calculate record
        wins = len(team_games[
            ((team_games['home_team'] == lookup_team) & (team_games['home_score'] > team_games['away_score'])) |
            ((team_games['away_team'] == lookup_team) & (team_games['away_score'] > team_games['home_score']))
        ])
        losses = len(team_games[
            ((team_games['home_team'] == lookup_team) & (team_games['home_score'] < team_games['away_score'])) |
            ((team_games['away_team'] == lookup_team) & (team_games['away_score'] < team_games['home_score']))
        ])
        ties = len(team_games[
            ((team_games['home_team'] == lookup_team) & (team_games['home_score'] == team_games['away_score'])) |
            ((team_games['away_team'] == lookup_team) & (team_games['away_score'] == team_games['home_score']))
        ])

        points_for = (
            team_games[team_games['home_team'] == lookup_team]['home_score'].sum() +
            team_games[team_games['away_team'] == lookup_team]['away_score'].sum()
        )
        points_against = (
            team_games[team_games['home_team'] == lookup_team]['away_score'].sum() +
            team_games[team_games['away_team'] == lookup_team]['home_score'].sum()
        )

        games_played = len(team_games)

        # Add/override record and points stats
        team_stats[team].update({
            'games_played': games_played,
            'wins': wins,
            'losses': losses,
            'ties': ties,
            'win_pct': (wins + 0.5 * ties) / (wins + losses + ties) if (wins + losses + ties) > 0 else 0,
            'points_for': points_for,
            'points_against': points_against,
            'point_diff': points_for - points_against,
            'points_per_game': points_for / games_played if games_played > 0 else 0,
            'points_against_per_game': points_against / games_played if games_played > 0 else 0
        })

        # Add conversion rates (calculated from play-by-play)
        team_stats[team].update(calculate_conversion_rates(pbp_data, lookup_team))

        # Add red zone stats (calculated from play-by-play)
        team_stats[team].update(calculate_red_zone_stats(pbp_data, lookup_team))

        # Calculate derived stats
        passing_yards = team_stats[team].get('passing_yards', 0)
        rushing_yards = team_stats[team].get('rushing_yards', 0)
        passing_first_downs = team_stats[team].get('passing_first_downs', 0)
        rushing_first_downs = team_stats[team].get('rushing_first_downs', 0)
        attempts = team_stats[team].get('attempts', 0)
        carries = team_stats[team].get('carries', 0)
        completions = team_stats[team].get('completions', 0)
        passing_epa = team_stats[team].get('passing_epa', 0)
        rushing_epa = team_stats[team].get('rushing_epa', 0)

        # Rename some fields for clarity
        team_stats[team]['sacks_taken'] = team_stats[team].get('sacks_suffered', 0)
        team_stats[team]['interceptions'] = team_stats[team].get('passing_interceptions', 0)

        # Add calculated fields
        team_stats[team].update({
            'completion_pct': (completions / attempts * 100) if attempts > 0 else 0,
            'yards_per_attempt': (passing_yards / attempts) if attempts > 0 else 0,
            'yards_per_carry': (rushing_yards / carries) if carries > 0 else 0,
            'passer_rating': 0,  # Placeholder
            'total_yards': passing_yards + rushing_yards,
            'yards_per_game': (passing_yards + rushing_yards) / games_played if games_played > 0 else 0,
            'total_first_downs': passing_first_downs + rushing_first_downs,
            'first_downs_per_game': (passing_first_downs + rushing_first_downs) / games_played if games_played > 0 else 0,
            'total_epa': passing_epa + rushing_epa,
            'epa_per_game': (passing_epa + rushing_epa) / games_played if games_played > 0 else 0,
            'total_turnovers': team_stats[team].get('passing_interceptions', 0) + team_stats[team].get('rushing_fumbles_lost', 0) + team_stats[team].get('receiving_fumbles_lost', 0),
            'turnover_margin': team_stats[team].get('passing_interceptions', 0) + team_stats[team].get('rushing_fumbles_lost', 0) + team_stats[team].get('receiving_fumbles_lost', 0)
        })
    
    # Convert to DataFrame
    stats_df = pd.DataFrame.from_dict(team_stats, orient='index')
    
    # Before adding ESPN data, create a DataFrame for the ESPN stats
    espn_df = pd.DataFrame.from_dict(espn_stats, orient='index')
    espn_df.index = espn_df.index.map(lambda x: espn_id_to_abbr.get(str(x)))
    espn_df = espn_df.drop(columns=['espn_api_id'])  # Remove the ID column since we used it for mapping
    
    # Merge the ESPN data with the existing stats
    stats_df = pd.concat([stats_df, espn_df], axis=1)
    
    # Fill NaN values with appropriate defaults before type conversion
    default_values = {
        'ot_wins': 0,
        'ot_losses': 0,
        'playoff_seed': 0,
        'games_behind': 0,
        'league_win_pct': 0,
        'div_win_pct': 0,
        'clincher': '',
        'streak_display': '',
        'road_record': '',
        'conf_record': '',
        'div_record': ''
    }
    
    for col, default in default_values.items():
        if col in stats_df.columns:
            stats_df[col] = stats_df[col].fillna(default)
    
    # Convert integer columns
    int_columns = [
        'espn_api_id', 'wins', 'losses', 'ties',
        'games_played', 'points', 'points_for', 
        'points_against', 'point_diff', 'differential',
        'div_wins', 'div_losses', 'div_ties',
        'ot_wins', 'ot_losses', 'playoff_seed'
    ]
    
    # Convert float columns
    float_columns = [
        'win_pct', 'league_win_pct', 'div_win_pct',
        'avg_points_for', 'avg_points_against',
        'games_behind'
    ]
    
    # Format integer columns
    for col in int_columns:
        if col in stats_df.columns:
            stats_df[col] = pd.to_numeric(stats_df[col], errors='coerce').fillna(0).astype(int)
    
    # Format float columns
    for col in float_columns:
        if col in stats_df.columns:
            stats_df[col] = pd.to_numeric(stats_df[col], errors='coerce').fillna(0).round(4)
    
    # Create a copy to defragment the DataFrame
    stats_df = stats_df.copy()

    # sort by win pct descending then point difference descending   
    stats_df = stats_df.sort_values(by=['win_pct', 'point_diff'], ascending=[False, False])

    return stats_df

def save_team_stats():
    try:
        team_stats_df = generate_team_stats()
        
        # Save to CSV with ties included
        output_file = "data/team_stats.csv"
        team_stats_df.to_csv(output_file, index_label='team_abbr')
        return True
    except Exception as e:
        print(f"Error saving team_stats.csv: {e}")
        return False

if __name__ == "__main__":
    # Generate stats
    print("Generating team statistics...")
    team_stats_df = generate_team_stats()
    
    # Save to CSV with ties included
    output_file = "data/team_stats.csv"
    team_stats_df.to_csv(output_file, index_label='team_abbr')
    print(f"\nTeam statistics saved to {output_file}")
    
    # Display summary of key stats for each team
    print("\nKey Team Statistics:")
    print("=" * 80)
    
    for team in sorted(team_stats_df.index):
        stats = team_stats_df.loc[team]
        print(f"\n{team}:")
        print("-" * 40)
        print(f"Record: {int(stats['wins'])}-{int(stats['losses'])}-{int(stats['ties'])} ({stats['win_pct']:.3f})")
        print(f"Points: {int(stats['points_for'])} For / {int(stats['points_against'])} Against (Diff: {int(stats['point_diff']):+d})")
    
    print("\nComplete statistics available in team_stats.csv")
    
