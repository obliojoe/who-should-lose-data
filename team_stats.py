import pandas as pd
import logging
import json
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Sequence

from raw_data_manifest import RawDataManifest

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

TEAM_ALIAS = {
    'LA': 'LAR',
    'WSH': 'WAS',
}

TEAM_STATS_FALLBACK_NUMERIC_COLUMNS = [
    'season',
    'week',
    'completions',
    'attempts',
    'passing_yards',
    'passing_tds',
    'passing_interceptions',
    'sacks_suffered',
    'sack_yards_lost',
    'sack_fumbles',
    'sack_fumbles_lost',
    'passing_air_yards',
    'passing_yards_after_catch',
    'passing_first_downs',
    'passing_epa',
    'passing_cpoe',
    'passing_2pt_conversions',
    'carries',
    'rushing_yards',
    'rushing_tds',
    'rushing_fumbles',
    'rushing_fumbles_lost',
    'rushing_first_downs',
    'rushing_epa',
    'rushing_2pt_conversions',
    'receptions',
    'targets',
    'receiving_yards',
    'receiving_tds',
    'receiving_fumbles',
    'receiving_fumbles_lost',
    'receiving_air_yards',
    'receiving_yards_after_catch',
    'receiving_first_downs',
    'receiving_epa',
    'receiving_2pt_conversions',
    'special_teams_tds',
    'def_tackles_solo',
    'def_tackles_with_assist',
    'def_tackle_assists',
    'def_tackles_for_loss',
    'def_tackles_for_loss_yards',
    'def_fumbles_forced',
    'def_sacks',
    'def_sack_yards',
    'def_qb_hits',
    'def_interceptions',
    'def_interception_yards',
    'def_pass_defended',
    'def_tds',
    'def_fumbles',
    'def_safeties',
    'misc_yards',
    'fumble_recovery_own',
    'fumble_recovery_yards_own',
    'fumble_recovery_opp',
    'fumble_recovery_yards_opp',
    'fumble_recovery_tds',
    'penalties',
    'penalty_yards',
    'timeouts',
    'punt_returns',
    'punt_return_yards',
    'kickoff_returns',
    'kickoff_return_yards',
    'fg_made',
    'fg_att',
    'fg_missed',
    'fg_blocked',
    'fg_long',
    'fg_pct',
    'fg_made_0_19',
    'fg_made_20_29',
    'fg_made_30_39',
    'fg_made_40_49',
    'fg_made_50_59',
    'fg_made_60_',
    'fg_missed_0_19',
    'fg_missed_20_29',
    'fg_missed_30_39',
    'fg_missed_40_49',
    'fg_missed_50_59',
    'fg_missed_60_',
    'fg_made_list',
    'fg_missed_list',
    'fg_blocked_list',
    'fg_made_distance',
    'fg_missed_distance',
    'fg_blocked_distance',
    'pat_made',
    'pat_att',
    'pat_missed',
    'pat_blocked',
    'pat_pct',
    'gwfg_made',
    'gwfg_att',
    'gwfg_missed',
    'gwfg_blocked',
    'gwfg_distance',
]

TEAM_STATS_CONVERSION_COLUMNS = [
    'third_down_attempts',
    'third_down_conversions',
    'third_down_pct',
    'fourth_down_attempts',
    'fourth_down_conversions',
    'fourth_down_pct',
    'third_down_attempts_against',
    'third_down_conversions_against',
    'third_down_pct_against',
    'fourth_down_attempts_against',
    'fourth_down_conversions_against',
    'fourth_down_pct_against',
]

TEAM_STATS_RED_ZONE_COLUMNS = [
    'red_zone_trips',
    'red_zone_tds',
    'red_zone_pct',
    'red_zone_trips_against',
    'red_zone_tds_against',
    'red_zone_pct_against',
]

TEAM_STATS_DERIVED_COLUMNS = [
    'games_played',
    'wins',
    'losses',
    'ties',
    'win_pct',
    'points_for',
    'points_against',
    'point_diff',
    'points_per_game',
    'points_against_per_game',
    'completion_pct',
    'yards_per_attempt',
    'yards_per_carry',
    'passer_rating',
    'total_yards',
    'yards_per_game',
    'total_first_downs',
    'first_downs_per_game',
    'total_epa',
    'epa_per_game',
    'total_turnovers',
    'turnover_margin',
    'sacks_taken',
    'interceptions',
]

TEAM_STATS_ESPN_COLUMNS = [
    'espn_api_id',
    'league_win_pct',
    'div_win_pct',
    'games_behind',
    'ot_wins',
    'ot_losses',
    'playoff_seed',
    'clincher',
    'streak_display',
    'road_record',
    'conf_record',
    'div_record',
]

TEAM_STATS_TEXT_COLUMNS = {
    'clincher',
    'conf_record',
    'div_record',
    'road_record',
    'streak_display',
}

TEAM_STATS_EXPECTED_OUTPUT_COLUMNS = sorted(
    set(TEAM_STATS_FALLBACK_NUMERIC_COLUMNS)
    | set(TEAM_STATS_CONVERSION_COLUMNS)
    | set(TEAM_STATS_RED_ZONE_COLUMNS)
    | set(TEAM_STATS_DERIVED_COLUMNS)
    | set(TEAM_STATS_ESPN_COLUMNS)
)


def _require_manifest(manifest: Optional[RawDataManifest]) -> RawDataManifest:
    if manifest:
        return manifest
    manifest = RawDataManifest.from_latest()
    if manifest is None:
        raise RuntimeError(
            "Raw data manifest not found. Run collect_raw_data.py before generating team stats."
        )
    return manifest


def _load_weekly_csvs(
    manifest: RawDataManifest,
    dataset: str,
    *,
    upto_week: Optional[int] = None,
    usecols: Optional[Sequence[str]] = None,
) -> Optional[pd.DataFrame]:
    entries = manifest.entries(dataset)
    if not entries:
        logger.warning("Dataset %s not available in manifest", dataset)
        return None

    base_dir = entries[0].path.parent
    season = manifest.season
    if season is None:
        logger.warning("Manifest missing season metadata; cannot resolve %s", dataset)
        return None

    if upto_week is None:
        upto_week = manifest.week

    frames = []
    pattern = f"season_{season}_week_*.csv"
    has_non_empty_frame = False

    for path in sorted(base_dir.glob(pattern)):
        week_value: Optional[int] = None
        try:
            week_value = int(path.stem.split('_')[-1])
        except (ValueError, IndexError):
            week_value = None

        if upto_week is not None and week_value is not None and week_value > upto_week:
            continue

        try:
            frame = pd.read_csv(path, usecols=usecols)
        except Exception as exc:
            logger.warning("Failed to load %s: %s", path, exc)
            continue

        frames.append(frame)
        if not frame.empty:
            has_non_empty_frame = True

    if not frames or not has_non_empty_frame:
        logger.warning("No usable weekly CSV data loaded for %s", dataset)

        season_entries = manifest.entries(f"{dataset}_season")
        if not season_entries:
            return None

        season_frames = []
        for entry in season_entries:
            try:
                frame = pd.read_csv(entry.path, usecols=usecols)
            except Exception as exc:
                logger.warning("Failed to load %s: %s", entry.path, exc)
                continue

            if upto_week is not None and 'week' in frame.columns:
                frame = frame[frame['week'] <= upto_week]
            season_frames.append(frame)
        frames = [frame for frame in season_frames if not frame.empty]
        if not frames:
            return None

    data = pd.concat(frames, ignore_index=True)

    if 'season' in data.columns:
        data = data[data['season'] == season]
    return data


def _load_single_csv(
    manifest: RawDataManifest,
    dataset: str,
    *,
    usecols: Optional[Sequence[str]] = None,
) -> Optional[pd.DataFrame]:
    entries = manifest.entries(dataset)
    if not entries:
        logger.warning("Dataset %s not available in manifest", dataset)
        return None

    path = entries[0].path
    try:
        frame = pd.read_csv(path, usecols=usecols)
    except Exception as exc:
        logger.warning("Failed to load %s: %s", path, exc)
        return None

    season = manifest.season
    if season is not None and 'season' in frame.columns:
        frame = frame[frame['season'] == season]

    return frame


def _ensure_team_stats_schema(
    stats_df: pd.DataFrame,
    season: Optional[int],
    week: Optional[int],
) -> pd.DataFrame:
    """Guarantee all downstream consumers see the expected columns."""

    season_value = season if season is not None else 0
    week_value = week if week is not None else 0

    missing_columns = []
    column_defaults = {}

    for column in TEAM_STATS_EXPECTED_OUTPUT_COLUMNS:
        if column in stats_df.columns:
            continue
        if column == 'season':
            default = season_value
        elif column == 'week':
            default = week_value
        elif column in TEAM_STATS_TEXT_COLUMNS:
            default = ''
        else:
            default = 0
        missing_columns.append(column)
        column_defaults[column] = default

    if missing_columns:
        filler = pd.DataFrame(
            {col: column_defaults[col] for col in missing_columns},
            index=stats_df.index,
        )
        stats_df = pd.concat([stats_df, filler], axis=1)

    return stats_df

def calculate_conversion_rates(pbp_data, team):
    """Calculate offensive and defensive conversion rates"""
    if pbp_data is None or pbp_data.empty:
        return {
            'third_down_attempts': 0,
            'third_down_conversions': 0,
            'third_down_pct': 0,
            'fourth_down_attempts': 0,
            'fourth_down_conversions': 0,
            'fourth_down_pct': 0,
            'third_down_attempts_against': 0,
            'third_down_conversions_against': 0,
            'third_down_pct_against': 0,
            'fourth_down_attempts_against': 0,
            'fourth_down_conversions_against': 0,
            'fourth_down_pct_against': 0,
        }

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
    if pbp_data is None or pbp_data.empty:
        return {
            'red_zone_trips': 0,
            'red_zone_tds': 0,
            'red_zone_pct': 0,
            'red_zone_trips_against': 0,
            'red_zone_tds_against': 0,
            'red_zone_pct_against': 0,
        }

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
        (offense_rz['posteam'] == team)
    ].groupby(['game_id', 'drive']).size())
    
    # Defensive red zone stats
    defense_rz = red_zone_plays[red_zone_plays['defteam'] == team]
    # Count unique drives against that reached red zone
    rz_trips_against = len(defense_rz.groupby(['game_id', 'drive']).size())
    # Count touchdowns allowed in red zone (credit scores where opponent possessed the ball)
    rz_tds_against = len(defense_rz[
        (defense_rz['touchdown'] == 1) &
        (defense_rz['posteam'] != team)
    ].groupby(['game_id', 'drive']).size())
    
    return {
        'red_zone_trips': rz_trips,
        'red_zone_tds': rz_tds,
        'red_zone_pct': (rz_tds / rz_trips * 100) if rz_trips > 0 else 0,
        'red_zone_trips_against': rz_trips_against,
        'red_zone_tds_against': rz_tds_against,
        'red_zone_pct_against': (rz_tds_against / rz_trips_against * 100) if rz_trips_against > 0 else 0
    }

def get_espn_standings_data(
    manifest: RawDataManifest,
    teams_df: pd.DataFrame,
) -> Dict[str, Dict]:
    """Load ESPN standings details from captured raw data."""

    def _parse_team_id(team_ref: Optional[str]) -> Optional[int]:
        if not team_ref:
            return None
        tail = team_ref.rstrip('/').split('/')[-1]
        token = tail.split('?')[0]
        try:
            return int(token)
        except ValueError:
            return None

    def _find_record(records, name: str) -> Optional[Dict]:
        for record in records or []:
            if record.get('name') == name:
                return record
        return None

    stats_by_team: Dict[str, Dict] = {}

    for label in ('afc', 'nfc'):
        payload = manifest.load_json(f'espn_standings_{label}')
        if not payload:
            continue

        for team_entry in payload.get('standings', []) or []:
            team_ref = (team_entry.get('team') or {}).get('$ref')
            team_id = _parse_team_id(team_ref)
            if team_id is None:
                continue

            team_row = teams_df[teams_df['espn_api_id'] == team_id]
            if team_row.empty:
                continue

            team_abbr = team_row.iloc[0]['team_abbr']
            team_abbr = TEAM_ALIAS.get(team_abbr, team_abbr)

            overall_record = _find_record(team_entry.get('records'), 'overall') or {}
            stats = {stat.get('name'): stat for stat in (overall_record.get('stats') or [])}

            div_record = _find_record(team_entry.get('records'), 'vs. Div.') or {}
            conf_record = _find_record(team_entry.get('records'), 'vs. Conf.') or {}
            road_record = _find_record(team_entry.get('records'), 'Road') or {}

            def _value(name: str, default=0):
                entry = stats.get(name) or {}
                return entry.get('value', default)

            def _display(name: str) -> str:
                entry = stats.get(name) or {}
                return entry.get('displayValue') or ''

            stats_by_team[team_abbr] = {
                'espn_api_id': team_id,
                'league_win_pct': _value('leagueWinPercent', 0.0),
                'div_win_pct': _value('divisionWinPercent', 0.0),
                'games_behind': _value('gamesBehind', 0.0),
                'ot_wins': int(_value('OTWins', 0)),
                'ot_losses': int(_value('OTLosses', 0)),
                'road_record': road_record.get('displayValue', ''),
                'conf_record': conf_record.get('displayValue', ''),
                'div_record': div_record.get('displayValue', ''),
                'playoff_seed': int(_value('playoffSeed', 0)),
                'clincher': _display('clincher'),
                'streak_display': _display('streak'),
            }

    return stats_by_team

def generate_team_stats(manifest: Optional[RawDataManifest] = None) -> pd.DataFrame:
    """Generate comprehensive team statistics from stored raw snapshots."""

    manifest = _require_manifest(manifest)
    season = manifest.season
    week = manifest.week

    with open('data/teams.json', 'r', encoding='utf-8') as fh:
        teams_df = pd.DataFrame(json.load(fh))

    weekly_team_stats = _load_weekly_csvs(manifest, 'nflreadpy_team_stats', upto_week=week)

    non_numeric_cols = {'team', 'team_abbr', 'opponent_team', 'season_type'}

    if weekly_team_stats is not None and not weekly_team_stats.empty:
        weekly_team_stats = weekly_team_stats.copy()
        weekly_team_stats['team_abbr'] = weekly_team_stats['team'].replace(TEAM_ALIAS)

        # Convert numeric columns from string/object payloads so aggregation retains base stats
        for col in weekly_team_stats.columns:
            if col not in non_numeric_cols:
                weekly_team_stats[col] = pd.to_numeric(weekly_team_stats[col], errors='coerce')

        aggregated_stats = (
            weekly_team_stats.groupby('team_abbr').sum(numeric_only=True).fillna(0)
        )
    else:
        logger.warning(
            "No nflreadpy_team_stats data available for week %s; continuing with schedule-derived stats only",
            week,
        )
        aggregated_stats = pd.DataFrame(columns=TEAM_STATS_FALLBACK_NUMERIC_COLUMNS)
        aggregated_stats.index.name = 'team_abbr'

    schedule_df = _load_single_csv(manifest, 'nflreadpy_schedules')
    if schedule_df is None or schedule_df.empty:
        raise RuntimeError('No nflreadpy_schedules data available in raw snapshot')

    schedule_df = schedule_df[schedule_df.get('game_type') == 'REG'] if 'game_type' in schedule_df.columns else schedule_df
    for col in ('home_team', 'away_team'):
        if col in schedule_df.columns:
            schedule_df[col] = schedule_df[col].replace(TEAM_ALIAS)

    pbp_columns = [
        'season',
        'week',
        'posteam',
        'defteam',
        'down',
        'play_type',
        'yards_gained',
        'ydstogo',
        'two_point_attempt',
        'touchdown',
        'game_id',
        'drive',
        'yardline_100',
        'play_type_nfl',
    ]
    pbp_df = _load_weekly_csvs(manifest, 'nflreadpy_pbp', upto_week=week, usecols=pbp_columns)
    if pbp_df is not None and not pbp_df.empty:
        pbp_df = pbp_df.copy()
        for team_col in ('posteam', 'defteam'):
            if team_col in pbp_df.columns:
                pbp_df[team_col] = pbp_df[team_col].replace(TEAM_ALIAS)

        defaults = {
            'two_point_attempt': 0,
            'touchdown': 0,
        }
        for col, default in defaults.items():
            if col not in pbp_df.columns:
                pbp_df[col] = default

        if 'play_type_nfl' not in pbp_df.columns:
            pbp_df['play_type_nfl'] = ''

        for col in ('yards_gained', 'ydstogo', 'yardline_100'):
            if col in pbp_df.columns:
                pbp_df[col] = pd.to_numeric(pbp_df[col], errors='coerce').fillna(0)
    else:
        pbp_df = None

    teams = sorted({
        *(schedule_df['home_team'].dropna().unique() if 'home_team' in schedule_df else []),
        *(schedule_df['away_team'].dropna().unique() if 'away_team' in schedule_df else []),
    })

    team_stats: Dict[str, Dict] = {}

    for team in teams:
        team_record = aggregated_stats.loc[team].to_dict() if team in aggregated_stats.index else {}
        team_record = {k: (0 if pd.isna(v) else v) for k, v in team_record.items()}

        team_games = schedule_df[
            ((schedule_df.get('home_team') == team) | (schedule_df.get('away_team') == team))
        ].copy()
        completed_games = team_games.dropna(subset=['home_score', 'away_score']) if {'home_score', 'away_score'} <= set(team_games.columns) else team_games.iloc[0:0]

        wins = len(
            completed_games[
                ((completed_games.get('home_team') == team) & (completed_games.get('home_score') > completed_games.get('away_score')))
                |
                ((completed_games.get('away_team') == team) & (completed_games.get('away_score') > completed_games.get('home_score')))
            ]
        )
        losses = len(
            completed_games[
                ((completed_games.get('home_team') == team) & (completed_games.get('home_score') < completed_games.get('away_score')))
                |
                ((completed_games.get('away_team') == team) & (completed_games.get('away_score') < completed_games.get('home_score')))
            ]
        )
        ties = len(
            completed_games[
                ((completed_games.get('home_team') == team) & (completed_games.get('home_score') == completed_games.get('away_score')))
                |
                ((completed_games.get('away_team') == team) & (completed_games.get('away_score') == completed_games.get('home_score')))
            ]
        )

        points_for = (
            completed_games.loc[completed_games.get('home_team') == team, 'home_score'].sum()
            + completed_games.loc[completed_games.get('away_team') == team, 'away_score'].sum()
        )
        points_against = (
            completed_games.loc[completed_games.get('home_team') == team, 'away_score'].sum()
            + completed_games.loc[completed_games.get('away_team') == team, 'home_score'].sum()
        )

        games_played = len(completed_games)

        team_record.update({
            'games_played': games_played,
            'wins': wins,
            'losses': losses,
            'ties': ties,
            'win_pct': (wins + 0.5 * ties) / (wins + losses + ties) if (wins + losses + ties) > 0 else 0,
            'points_for': points_for,
            'points_against': points_against,
            'point_diff': points_for - points_against,
            'points_per_game': points_for / games_played if games_played > 0 else 0,
            'points_against_per_game': points_against / games_played if games_played > 0 else 0,
        })

        team_record.update(calculate_conversion_rates(pbp_df, team))
        team_record.update(calculate_red_zone_stats(pbp_df, team))

        passing_yards = team_record.get('passing_yards', 0)
        rushing_yards = team_record.get('rushing_yards', 0)
        passing_first_downs = team_record.get('passing_first_downs', 0)
        rushing_first_downs = team_record.get('rushing_first_downs', 0)
        attempts = team_record.get('attempts', 0)
        carries = team_record.get('carries', 0)
        completions = team_record.get('completions', 0)
        passing_epa = team_record.get('passing_epa', 0)
        rushing_epa = team_record.get('rushing_epa', 0)

        team_record['sacks_taken'] = team_record.get('sacks_suffered', 0)
        team_record['interceptions'] = team_record.get('passing_interceptions', 0)

        turnovers_given = (
            team_record.get('passing_interceptions', 0)
            + team_record.get('sack_fumbles_lost', 0)
            + team_record.get('rushing_fumbles_lost', 0)
            + team_record.get('receiving_fumbles_lost', 0)
        )

        takeaways = (
            team_record.get('def_interceptions', 0)
            + team_record.get('fumble_recovery_opp', 0)
        )

        team_record.update({
            'completion_pct': (completions / attempts * 100) if attempts > 0 else 0,
            'yards_per_attempt': (passing_yards / attempts) if attempts > 0 else 0,
            'yards_per_carry': (rushing_yards / carries) if carries > 0 else 0,
            'passer_rating': 0,
            'total_yards': passing_yards + rushing_yards,
            'yards_per_game': (passing_yards + rushing_yards) / games_played if games_played > 0 else 0,
            'total_first_downs': passing_first_downs + rushing_first_downs,
            'first_downs_per_game': (passing_first_downs + rushing_first_downs) / games_played if games_played > 0 else 0,
            'total_epa': passing_epa + rushing_epa,
            'epa_per_game': (passing_epa + rushing_epa) / games_played if games_played > 0 else 0,
            'total_turnovers': turnovers_given,
            'turnover_margin': takeaways - turnovers_given,
        })

        team_stats[team] = team_record

    stats_df = pd.DataFrame.from_dict(team_stats, orient='index')

    espn_stats = get_espn_standings_data(manifest, teams_df)
    if espn_stats:
        espn_df = pd.DataFrame.from_dict(espn_stats, orient='index')
        stats_df = stats_df.join(espn_df, how='left')

    stats_df = _ensure_team_stats_schema(stats_df, season, week)

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
        'div_record': '',
    }

    for col, default in default_values.items():
        if col in stats_df.columns:
            stats_df[col] = stats_df[col].fillna(default)

    int_columns = [
        'espn_api_id',
        'wins',
        'losses',
        'ties',
        'games_played',
        'points_for',
        'points_against',
        'point_diff',
        'ot_wins',
        'ot_losses',
        'playoff_seed',
        'total_turnovers',
    ]

    float_columns = [
        'win_pct',
        'league_win_pct',
        'div_win_pct',
        'games_behind',
        'points_per_game',
        'points_against_per_game',
    ]

    for col in int_columns:
        if col in stats_df.columns:
            stats_df[col] = pd.to_numeric(stats_df[col], errors='coerce').fillna(0).astype(int)

    for col in float_columns:
        if col in stats_df.columns:
            stats_df[col] = pd.to_numeric(stats_df[col], errors='coerce').fillna(0).round(4)

    stats_df.index.name = 'team_abbr'
    stats_df = stats_df.sort_values(by=['win_pct', 'point_diff'], ascending=[False, False])

    return stats_df


def save_team_stats(manifest: Optional[RawDataManifest] = None) -> bool:
    try:
        team_stats_df = generate_team_stats(manifest=manifest)
        output_path = Path("data/team_stats.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        records = (
            team_stats_df.reset_index()
            .replace({np.nan: None})
            .to_dict(orient='records')
        )
        with output_path.open('w', encoding='utf-8') as fh:
            json.dump(records, fh, indent=2)
        return True
    except Exception as exc:
        logger.error("Error saving team_stats.json: %s", exc)
        return False

if __name__ == "__main__":
    print("Generating team statistics...")
    if save_team_stats():
        print("\nTeam statistics saved to data/team_stats.json")
    else:
        print("Failed to generate team statistics")
    
