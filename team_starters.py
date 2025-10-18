import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from raw_data_manifest import RawDataManifest


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

TEAM_ALIAS = {
    'LA': 'LAR',
    'WSH': 'WAS',
}

RAW_DEPTHCHART_DIR = Path('data/raw/espn/depthchart')

PLAYER_STAT_CATEGORIES: Dict[str, List[str]] = {
    'passing': [
        'completions', 'attempts', 'passing_yards', 'passing_tds', 'passing_interceptions',
        'sacks_suffered', 'sack_yards_lost', 'passing_air_yards', 'passing_yards_after_catch',
        'passing_first_downs', 'passing_epa', 'passing_cpoe', 'passing_2pt_conversions', 'pacr'
    ],
    'rushing': [
        'carries', 'rushing_yards', 'rushing_tds', 'rushing_fumbles', 'rushing_fumbles_lost',
        'rushing_first_downs', 'rushing_epa', 'rushing_2pt_conversions'
    ],
    'receiving': [
        'receptions', 'targets', 'receiving_yards', 'receiving_tds', 'receiving_fumbles',
        'receiving_fumbles_lost', 'receiving_air_yards', 'receiving_yards_after_catch',
        'receiving_first_downs', 'receiving_epa', 'receiving_2pt_conversions', 'racr',
        'target_share', 'air_yards_share', 'wopr'
    ],
    'defense': [
        'def_tackles_solo', 'def_tackles_with_assist', 'def_tackle_assists',
        'def_tackles_for_loss', 'def_tackles_for_loss_yards', 'def_fumbles_forced',
        'def_sacks', 'def_sack_yards', 'def_qb_hits', 'def_interceptions',
        'def_interception_yards', 'def_pass_defended', 'def_tds', 'def_fumbles', 'def_safeties'
    ],
    'returns': [
        'punt_returns', 'punt_return_yards', 'punt_return_tds',
        'kickoff_returns', 'kickoff_return_yards', 'kickoff_return_tds',
        'special_teams_tds', 'misc_yards'
    ],
    'kicking': [
        'fg_made', 'fg_att', 'fg_missed', 'fg_blocked', 'fg_long', 'fg_pct',
        'fg_made_0_19', 'fg_made_20_29', 'fg_made_30_39', 'fg_made_40_49', 'fg_made_50_59', 'fg_made_60_',
        'fg_missed_0_19', 'fg_missed_20_29', 'fg_missed_30_39', 'fg_missed_40_49',
        'fg_missed_50_59', 'fg_missed_60_', 'fg_made_list', 'fg_missed_list', 'fg_blocked_list',
        'fg_made_distance', 'fg_missed_distance', 'fg_blocked_distance',
        'pat_made', 'pat_att', 'pat_missed', 'pat_blocked', 'pat_pct',
        'gwfg_made', 'gwfg_att', 'gwfg_missed', 'gwfg_blocked', 'gwfg_distance'
    ],
    'misc': [
        'penalties', 'penalty_yards', 'fumble_recovery_own', 'fumble_recovery_yards_own',
        'fumble_recovery_opp', 'fumble_recovery_yards_opp', 'fumble_recovery_tds'
    ],
    'fantasy': ['fantasy_points', 'fantasy_points_ppr'],
}


def _clean_value(value: Any) -> Optional[Any]:
    if value is None:
        return None
    if isinstance(value, (np.integer, np.int64, np.int32)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if math.isnan(value) or math.isinf(value):
            return None
        return float(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _extract_stat_categories(row: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    categories: Dict[str, Dict[str, Any]] = {}
    for category, fields in PLAYER_STAT_CATEGORIES.items():
        payload: Dict[str, Any] = {}
        for field in fields:
            if field not in row:
                continue
            cleaned = _clean_value(row.get(field))
            if cleaned is None:
                continue
            if isinstance(cleaned, (int, float)) and cleaned == 0:
                continue
            if isinstance(cleaned, str) and cleaned == '':
                continue
            payload[field] = cleaned
        if payload:
            categories[category] = payload
    return categories


def _require_manifest(manifest: Optional[RawDataManifest]) -> RawDataManifest:
    if manifest:
        return manifest
    manifest = RawDataManifest.from_latest()
    if manifest is None:
        raise RuntimeError(
            "Raw data manifest not found. Run collect_raw_data.py before generating team starters."
        )
    return manifest


def _resolve_week_path(manifest: RawDataManifest, dataset: str, week: Optional[int] = None) -> Optional[Path]:
    entries = manifest.entries(dataset)
    if not entries:
        logger.warning("Dataset %s not available in manifest", dataset)
        return None

    base_dir = entries[0].path.parent
    season = manifest.season
    if season is None:
        logger.warning("Manifest missing season metadata; cannot resolve %s", dataset)
        return None

    target_week = week or manifest.week
    if target_week is not None:
        candidate = base_dir / f"season_{season}_week_{target_week}.csv"
        if candidate.exists():
            return candidate

    return entries[0].path if entries else None


def _load_week_csv(
    manifest: RawDataManifest,
    dataset: str,
    *,
    week: Optional[int] = None,
    usecols: Optional[Sequence[str]] = None,
) -> Optional[pd.DataFrame]:
    path = _resolve_week_path(manifest, dataset, week)
    if path is None or not path.exists():
        return None
    try:
        frame = pd.read_csv(path, usecols=usecols)
    except Exception as exc:
        logger.warning("Failed to load %s: %s", path, exc)
        return None

    season = manifest.season
    if season is not None and 'season' in frame.columns:
        frame = frame[frame['season'] == season]

    return frame.reset_index(drop=True)


def _load_weekly_history(
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

    if not frames:
        return None

    data = pd.concat(frames, ignore_index=True)
    if 'season' in data.columns:
        data = data[data['season'] == season]
    return data.reset_index(drop=True)


def _prepare_depth_chart(depth_df: pd.DataFrame) -> pd.DataFrame:
    if depth_df is None or depth_df.empty:
        raise RuntimeError('No depth chart data available in raw snapshot')

    depth_df = depth_df.copy()
    depth_df['pos_rank'] = pd.to_numeric(depth_df.get('pos_rank'), errors='coerce').fillna(0).astype(int)
    starters = depth_df[
        (depth_df['pos_rank'] == 1)
        | ((depth_df['pos_rank'] == 2) & (depth_df['pos_abb'] == 'RB'))
    ].copy()

    starters['is_special_teams'] = starters['pos_grp'].str.contains('Special Teams', case=False, na=False)
    starters = starters.sort_values('is_special_teams')
    starters = starters.drop_duplicates(subset=['gsis_id', 'team'], keep='first')
    starters = starters.drop(columns=['is_special_teams'])
    starters = starters[starters['team'].notna() & (starters['team'] != '')]
    starters = starters.dropna(subset=['gsis_id'])

    return starters.reset_index(drop=True)


def _merge_weekly_stats(
    starters: pd.DataFrame,
    weekly_stats: Optional[pd.DataFrame],
) -> pd.DataFrame:
    if weekly_stats is None or weekly_stats.empty:
        return starters

    weekly_fields = {
        'completions': 'completions_week',
        'attempts': 'attempts_week',
        'passing_yards': 'passing_yards_week',
        'passing_tds': 'passing_tds_week',
        'passing_interceptions': 'interceptions_week',
        'carries': 'carries_week',
        'rushing_yards': 'rushing_yards_week',
        'rushing_tds': 'rushing_tds_week',
        'receptions': 'receptions_week',
        'targets': 'targets_week',
        'receiving_yards': 'receiving_yards_week',
        'receiving_tds': 'receiving_tds_week',
    }

    available_fields = [col for col in weekly_fields if col in weekly_stats.columns]
    stats_subset = weekly_stats[['player_id'] + available_fields].copy()
    stats_subset = stats_subset.rename(columns={col: weekly_fields[col] for col in available_fields})

    starters = starters.merge(stats_subset, left_on='gsis_id', right_on='player_id', how='left')
    starters = starters.drop(columns=['player_id'], errors='ignore')
    return starters


def _merge_season_stats(
    starters: pd.DataFrame,
    season_stats: Optional[pd.DataFrame],
) -> pd.DataFrame:
    if season_stats is None or season_stats.empty:
        return starters

    season_fields = {
        'completions': 'completions_season',
        'attempts': 'attempts_season',
        'passing_yards': 'passing_yards_season',
        'passing_tds': 'passing_tds_season',
        'passing_interceptions': 'interceptions_season',
        'carries': 'carries_season',
        'rushing_yards': 'rushing_yards_season',
        'rushing_tds': 'rushing_tds_season',
        'receptions': 'receptions_season',
        'targets': 'targets_season',
        'receiving_yards': 'receiving_yards_season',
        'receiving_tds': 'receiving_tds_season',
    }

    available_fields = [col for col in season_fields if col in season_stats.columns]
    stats_subset = season_stats[['player_id'] + available_fields].copy()
    stats_subset = stats_subset.rename(columns={col: season_fields[col] for col in available_fields})

    starters = starters.merge(stats_subset, left_on='gsis_id', right_on='player_id', how='left')
    starters = starters.drop(columns=['player_id'], errors='ignore')
    return starters


def get_current_starters(manifest: Optional[RawDataManifest] = None) -> pd.DataFrame:
    manifest = _require_manifest(manifest)

    roster_df = _load_week_csv(manifest, 'nflreadpy_rosters_weekly')
    depth_df = _load_week_csv(manifest, 'nflreadpy_depth_charts')
    weekly_player_stats = _load_week_csv(manifest, 'nflreadpy_player_stats')
    season_player_stats = _load_weekly_history(manifest, 'nflreadpy_player_stats')

    weekly_stats_raw = weekly_player_stats.copy() if weekly_player_stats is not None else None
    season_stats_raw = season_player_stats.copy() if season_player_stats is not None else None

    if roster_df is None or roster_df.empty:
        raise RuntimeError('No roster data available in raw snapshot')
    if depth_df is None or depth_df.empty:
        raise RuntimeError('No depth chart data available in raw snapshot')
    if weekly_player_stats is None or weekly_player_stats.empty:
        logger.warning('Weekly player stats missing; starters will omit recent game stats')
        weekly_player_stats = None
    if season_player_stats is None or season_player_stats.empty:
        season_player_stats = weekly_player_stats

    latest_week = manifest.week
    if latest_week is None and 'week' in roster_df.columns:
        latest_week = roster_df['week'].max()

    roster_week = roster_df.copy()
    if latest_week is not None and 'week' in roster_week.columns:
        roster_week = roster_week[roster_week['week'] == latest_week].copy()

    if weekly_player_stats is not None and latest_week is not None and 'week' in weekly_player_stats.columns:
        weekly_player_stats = weekly_player_stats[weekly_player_stats['week'] == latest_week].copy()

    starters = _prepare_depth_chart(depth_df)

    starters = starters.merge(
        roster_week,
        on='gsis_id',
        how='left',
        suffixes=('_depth', '_roster')
    )

    starters = _merge_weekly_stats(starters, weekly_player_stats)

    if season_player_stats is not None and not season_player_stats.empty:
        stat_fields = {
            'completions',
            'attempts',
            'passing_yards',
            'passing_tds',
            'passing_interceptions',
            'carries',
            'rushing_yards',
            'rushing_tds',
            'receptions',
            'targets',
            'receiving_yards',
            'receiving_tds',
        }

        stats_source = season_player_stats.copy()
        available_stats = [col for col in stat_fields if col in stats_source.columns]
        for col in available_stats:
            stats_source[col] = pd.to_numeric(stats_source[col], errors='coerce').fillna(0)

        season_totals = (
            stats_source.groupby('player_id')[available_stats]
            .sum()
            .reset_index()
        )

        starters = _merge_season_stats(starters, season_totals)

        games_played = (
            season_player_stats.groupby('player_id')['week']
            .nunique()
            .reset_index()
            .rename(columns={'player_id': 'player_id', 'week': 'games_played'})
        )
        starters = starters.merge(games_played, left_on='gsis_id', right_on='player_id', how='left')
        starters = starters.drop(columns=['player_id'], errors='ignore')
    else:
        starters['games_played'] = 0

    int_columns = [
        'jersey_number',
        'years_exp',
        'entry_year',
        'rookie_year',
        'draft_number',
        'weight',
        'games_played',
    ]
    for col in int_columns:
        if col in starters.columns:
            starters[col] = pd.to_numeric(starters[col], errors='coerce').fillna(0).astype(int)

    if 'birth_date' in starters.columns:
        starters['age'] = pd.Timestamp('today').year - pd.to_datetime(
            starters['birth_date'], errors='coerce'
        ).dt.year

    if 'position_group' not in starters.columns and 'position_group_roster' in starters.columns:
        starters['position_group'] = starters['position_group_roster']

    if 'player_name' not in starters.columns:
        for candidate in (
            'full_name_depth',
            'full_name',
            'player_display_name',
        ):
            if candidate in starters.columns:
                starters['player_name'] = starters[candidate]
                break

    team_map = TEAM_ALIAS
    if 'team_depth' in starters.columns:
        starters['team'] = starters['team_depth'].replace(team_map)
    elif 'team' in starters.columns:
        starters['team'] = starters['team'].replace(team_map)

    base_columns = {
        'team': 'team_abbr',
        'gsis_id': 'player_id',
        'pos_abb': 'position',
        'position_group': 'position_group',
        'pos_grp': 'formation',
        'player_name': 'player_name',
        'jersey_number': 'number',
        'status': 'status',
        'height': 'height',
        'weight': 'weight',
        'college': 'college',
        'years_exp': 'experience',
        'age': 'age',
        'birth_date': 'birth_date',
        'entry_year': 'entry_year',
        'draft_number': 'draft_pick',
    }

    rename_map = {col: base_columns[col] for col in base_columns if col in starters.columns}

    # Drop pre-existing columns that would collide with rename targets to avoid duplicates
    for src_col, target_col in rename_map.items():
        if src_col != target_col and target_col in starters.columns:
            starters = starters.drop(columns=[target_col])

    starters = starters.rename(columns=rename_map)

    # Ensure we do not carry duplicate column names forward (pandas warns otherwise)
    starters = starters.loc[:, ~starters.columns.duplicated()]

    stat_columns = [
        col
        for col in starters.columns
        if col.endswith('_season') or col.endswith('_week')
    ]
    optional_columns = [col for col in ('games_played', 'week', 'weekly_opponent') if col in starters.columns]

    ordered_columns: List[str] = []
    for col in list(rename_map.values()) + stat_columns + optional_columns:
        if col not in ordered_columns:
            ordered_columns.append(col)

    if not ordered_columns:
        ordered_columns = list(starters.columns)

    result = starters[ordered_columns].copy()

    if 'number' in result.columns:
        result['number'] = result['number'].apply(
            lambda val: int(val) if pd.notna(val) else None
        )

    mapping = build_espn_player_map()
    if mapping:
        espn_ids = []
        for _, row in result.iterrows():
            team = row.get('team_abbr')
            name = normalize_player_name(row.get('player_name'))
            jersey = row.get('number')
            jersey_key = None
            if pd.notna(jersey):
                try:
                    jersey_key = str(int(jersey))
                except (TypeError, ValueError):
                    jersey_key = str(jersey).strip() if isinstance(jersey, str) else None

            espn_id = None
            if team and name:
                key_exact = (team, name, jersey_key)
                key_general = (team, name, None)
                espn_id = mapping.get(key_exact) or mapping.get(key_general)

            espn_ids.append(str(espn_id) if espn_id else '')

        result['espn_player_id'] = espn_ids
    else:
        result['espn_player_id'] = ''

    player_stats = build_player_stats_records(
        result,
        weekly_player_stats,
        season_stats_raw,
        latest_week,
    )

    return result, player_stats


def build_player_stats_records(
    starters_df: pd.DataFrame,
    weekly_stats: Optional[pd.DataFrame],
    season_stats_raw: Optional[pd.DataFrame],
    latest_week: Optional[int],
) -> List[Dict[str, Any]]:
    starters_records = starters_df.replace({np.nan: None}).to_dict(orient='records')
    player_ids = {str(record.get('player_id')) for record in starters_records if record.get('player_id')}

    weekly_map: Dict[str, Dict[str, Any]] = {}
    if weekly_stats is not None and not weekly_stats.empty:
        weekly_copy = weekly_stats.copy()
        if 'player_id' in weekly_copy.columns:
            weekly_copy['player_id'] = weekly_copy['player_id'].astype(str)
            weekly_filtered = weekly_copy[weekly_copy['player_id'].isin(player_ids)]
            weekly_map = {
                row['player_id']: {k: _clean_value(v) for k, v in row.items()}
                for row in weekly_filtered.to_dict(orient='records')
            }

    season_map: Dict[str, Dict[str, Any]] = {}
    games_played_map: Dict[str, int] = {}
    if season_stats_raw is not None and not season_stats_raw.empty:
        season_copy = season_stats_raw.copy()
        if 'player_id' in season_copy.columns:
            season_copy['player_id'] = season_copy['player_id'].astype(str)
            season_filtered = season_copy[season_copy['player_id'].isin(player_ids)]

            if not season_filtered.empty:
                games_played_map = (
                    season_filtered.groupby('player_id')['week']
                    .nunique()
                    .to_dict()
                )

                numeric_cols = season_filtered.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    season_totals = (
                        season_filtered.groupby('player_id')[numeric_cols]
                        .sum(min_count=1)
                        .reset_index()
                    )
                else:
                    season_totals = pd.DataFrame({'player_id': season_filtered['player_id'].unique()})

                latest_info = (
                    season_filtered.sort_values(['player_id', 'week'])
                    .groupby('player_id')
                    .tail(1)[['player_id', 'team', 'position', 'position_group']]
                )

                season_totals = season_totals.merge(latest_info, on='player_id', how='left')
                season_map = {
                    row['player_id']: {k: _clean_value(v) for k, v in row.items()}
                    for row in season_totals.to_dict(orient='records')
                }

    player_stats: List[Dict[str, Any]] = []

    for starter in starters_records:
        player_id = starter.get('player_id')
        if not player_id:
            continue

        record: Dict[str, Any] = {
            'player_id': player_id,
            'team_abbr': starter.get('team_abbr'),
            'player_name': starter.get('player_name'),
            'number': starter.get('number'),
            'position': starter.get('position'),
            'position_group': starter.get('position_group'),
            'espn_player_id': starter.get('espn_player_id') or None,
        }

        weekly_row = weekly_map.get(player_id)
        if weekly_row:
            record['weekly'] = _extract_stat_categories(weekly_row)
            week_value = weekly_row.get('week')
            cleaned_week = _clean_value(week_value)
            if cleaned_week is not None:
                record['week'] = cleaned_week
            opponent = weekly_row.get('opponent_team')
            if opponent:
                record['weekly_opponent'] = opponent

        season_row = season_map.get(player_id)
        if season_row:
            record['season'] = _extract_stat_categories(season_row)
            if season_row.get('team') and not record.get('team_abbr'):
                record['team_abbr'] = season_row.get('team')

        games_played = games_played_map.get(player_id)
        if games_played is not None:
            record['games_played'] = int(games_played)
        elif record.get('weekly'):
            record['games_played'] = 1
        else:
            record['games_played'] = 0

        player_stats.append(record)

    return player_stats


def save_team_starters(manifest: Optional[RawDataManifest] = None) -> bool:
    try:
        team_starters_df, player_stats = get_current_starters(manifest)
        output_path = Path('data/team_starters.json')
        output_path.parent.mkdir(parents=True, exist_ok=True)

        records = team_starters_df.replace({np.nan: None}).to_dict(orient='records')
        with output_path.open('w', encoding='utf-8') as fh:
            json.dump(records, fh, indent=2)

        stats_path = Path('data/player_stats.json')
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with stats_path.open('w', encoding='utf-8') as fh:
            json.dump(player_stats, fh, indent=2)
        return True
    except Exception as exc:
        logger.error("Error saving team_starters.json: %s", exc)
        return False


if __name__ == "__main__":
    save_team_starters()


def normalize_player_name(name: Optional[str]) -> str:
    if not name or not isinstance(name, str):
        return ''
    normalized = name.lower()
    for suffix in [' jr', ' sr', ' iii', ' ii', ' iv']:
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]
    normalized = normalized.replace('.', '').replace("'", '').replace('-', ' ')
    normalized = ''.join(ch for ch in normalized if ch.isalnum() or ch.isspace())
    normalized = normalized.replace(' ', '')
    return normalized


def build_espn_player_map() -> Dict[Tuple[str, str, Optional[str]], str]:
    mapping: Dict[Tuple[str, str, Optional[str]], str] = {}
    if not RAW_DEPTHCHART_DIR.exists():
        return mapping

    for depth_path in RAW_DEPTHCHART_DIR.glob('season_*_week_*/*.json'):
        stem = depth_path.stem
        team_abbr = stem.split('-', 1)[1] if '-' in stem else None
        if not team_abbr:
            continue

        try:
            depth_data = json.loads(depth_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue

        for item in depth_data.get('items', []) or []:
            positions = item.get('positions', {}) or {}
            for pos_data in positions.values():
                for athlete in pos_data.get('athletes', []) or []:
                    espn_id = athlete.get('id')
                    full_name = athlete.get('fullName') or athlete.get('displayName')
                    jersey = athlete.get('jersey')
                    if not espn_id or not full_name:
                        continue

                    key = (team_abbr, normalize_player_name(full_name), str(jersey) if jersey else None)
                    mapping[key] = espn_id
                    if jersey:
                        general_key = (team_abbr, normalize_player_name(full_name), None)
                        mapping.setdefault(general_key, espn_id)

    return mapping
