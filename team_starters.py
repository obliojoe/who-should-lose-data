import logging
import os
from pathlib import Path
from typing import Dict, Optional, Sequence

import pandas as pd

from raw_data_manifest import RawDataManifest


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

TEAM_ALIAS = {
    'LA': 'LAR',
    'WSH': 'WAS',
}


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
        'pos_abb': 'position',
        'player_name': 'player_name',
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
        'completions_week': 'completions_week',
        'attempts_week': 'attempts_week',
        'passing_yards_week': 'passing_yards_week',
        'passing_tds_week': 'passing_tds_week',
        'interceptions_week': 'interceptions_week',
        'carries_week': 'carries_week',
        'rushing_yards_week': 'rushing_yards_week',
        'rushing_tds_week': 'rushing_tds_week',
        'receptions_week': 'receptions_week',
        'targets_week': 'targets_week',
        'receiving_yards_week': 'receiving_yards_week',
        'receiving_tds_week': 'receiving_tds_week',
        'completions_season': 'completions_season',
        'attempts_season': 'attempts_season',
        'passing_yards_season': 'passing_yards_season',
        'passing_tds_season': 'passing_tds_season',
        'interceptions_season': 'interceptions_season',
        'carries_season': 'carries_season',
        'rushing_yards_season': 'rushing_yards_season',
        'rushing_tds_season': 'rushing_tds_season',
        'receptions_season': 'receptions_season',
        'targets_season': 'targets_season',
        'receiving_yards_season': 'receiving_yards_season',
        'receiving_tds_season': 'receiving_tds_season',
        'games_played': 'games_played',
    }

    available_columns = [col for col in base_columns if col in starters.columns]
    result = starters[available_columns].rename(columns={col: base_columns[col] for col in available_columns})

    stat_columns = [
        col for col in result.columns if any(suffix in col for suffix in ('_week', '_season', 'pct', 'per', 'rating'))
    ]
    for col in stat_columns:
        result[col] = pd.to_numeric(result[col], errors='coerce').fillna(0)

    return result


def save_team_starters(filename: str, manifest: Optional[RawDataManifest] = None) -> bool:
    try:
        team_starters_df = get_current_starters(manifest)
        os.makedirs('data', exist_ok=True)
        csv_path = Path('data') / filename
        team_starters_df.to_csv(csv_path, index=False)
        return True
    except Exception as exc:
        logger.error("Error saving %s: %s", filename, exc)
        return False


if __name__ == "__main__":
    starters_df = get_current_starters()
    save_team_starters('team_starters.csv')
