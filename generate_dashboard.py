#!/usr/bin/env python3
"""
Generate dashboard_content.json with code-based data selection and AI-based creative text.
Refactored to minimize AI usage and maximize deterministic code.
"""

import json
import logging
import os
from datetime import datetime
import pandas as pd
import numpy as np
from ai_service import AIService

# Set up logging
logger = logging.getLogger(__name__)

def get_current_week(schedule_df):
    """Determine the current NFL week based on schedule"""
    upcoming_games = schedule_df[
        (schedule_df['away_score'].isna()) | (schedule_df['home_score'].isna())
    ]

    if len(upcoming_games) > 0:
        return int(upcoming_games['week_num'].min())

    return int(schedule_df['week_num'].max())


def select_stat_leaders(team_stats_df, sagarin_df):
    """
    Select top 5 stats for each category using code logic.
    Returns structured data ready for AI to add context.
    """
    # team_stats.json already has season totals (week 15 is aggregated row)
    # Just select the columns we need - no need to aggregate
    season_stats = team_stats_df[['team_abbr', 'points_per_game', 'points_against_per_game',
                                   'yards_per_game', 'passing_yards', 'rushing_yards',
                                   'completion_pct', 'third_down_pct', 'red_zone_pct',
                                   'turnover_margin', 'def_sacks', 'def_interceptions',
                                   'games_played', 'points_for', 'points_against',
                                   'attempts', 'carries']].copy()

    # Calculate per-game stats for yards (CSV has season totals for these)
    season_stats['passing_yards_per_game'] = season_stats['passing_yards'] / season_stats['games_played']
    season_stats['rushing_yards_per_game'] = season_stats['rushing_yards'] / season_stats['games_played']

    stat_leaders = {
        'offense': [],
        'defense': [],
        'efficiency': []
    }

    # Offense leaders (use per-game stats from CSV)
    offense_stats = [
        ('points_per_game', 'Points Per Game', ''),
        ('yards_per_game', 'Total Yards Per Game', ' yds'),
        ('passing_yards_per_game', 'Passing Yards Per Game', ' yds'),
        ('rushing_yards_per_game', 'Rushing Yards Per Game', ' yds'),
        ('red_zone_pct', 'Red Zone %', '%')
    ]

    for stat_col, stat_name, unit in offense_stats:
        top_team = season_stats.nlargest(1, stat_col).iloc[0]
        stat_leaders['offense'].append({
            'team': top_team['team_abbr'],
            'stat': stat_name,
            'value': round(float(top_team[stat_col]), 1),
            'unit': unit,
            'rank': 1
        })

    # Defense leaders
    defense_stats = [
        ('points_against_per_game', 'Points Allowed Per Game', '', True),  # Lower is better
        ('def_sacks', 'Sacks', '', False),
        ('def_interceptions', 'Interceptions', '', False),
        ('third_down_pct', 'Third Down Defense', '%', True)  # Lower is better for defense
    ]

    for stat_col, stat_name, unit, ascending in defense_stats:
        if ascending:
            top_team = season_stats.nsmallest(1, stat_col).iloc[0]
        else:
            top_team = season_stats.nlargest(1, stat_col).iloc[0]

        stat_leaders['defense'].append({
            'team': top_team['team_abbr'],
            'stat': stat_name,
            'value': round(float(top_team[stat_col]), 1),
            'unit': unit,
            'rank': 1
        })

    # Add one more defense stat to get 5
    top_turnover = season_stats.nlargest(1, 'turnover_margin').iloc[0]
    stat_leaders['defense'].append({
        'team': top_turnover['team_abbr'],
        'stat': 'Turnover Margin',
        'value': int(top_turnover['turnover_margin']),
        'unit': '',
        'rank': 1
    })

    # Efficiency leaders
    efficiency_stats = [
        ('third_down_pct', 'Third Down Conversion', '%'),
        ('completion_pct', 'Completion Percentage', '%'),
        ('red_zone_pct', 'Red Zone Efficiency', '%')
    ]

    for stat_col, stat_name, unit in efficiency_stats:
        top_team = season_stats.nlargest(1, stat_col).iloc[0]
        stat_leaders['efficiency'].append({
            'team': top_team['team_abbr'],
            'stat': stat_name,
            'value': round(float(top_team[stat_col]), 1),
            'unit': unit,
            'rank': 1
        })

    # Add yards per play (efficiency metric) as 4th efficiency stat
    if 'attempts' in season_stats.columns and 'carries' in season_stats.columns:
        season_stats['yards_per_play'] = (season_stats['passing_yards'] + season_stats['rushing_yards']) / (season_stats['attempts'] + season_stats['carries'])
        top_ypp = season_stats.nlargest(1, 'yards_per_play').iloc[0]
        stat_leaders['efficiency'].append({
            'team': top_ypp['team_abbr'],
            'stat': 'Yards Per Play',
            'value': round(float(top_ypp['yards_per_play']), 1),
            'unit': '',
            'rank': 1
        })

    # Add point differential as 5th efficiency stat
    season_stats['point_diff'] = season_stats['points_for'] - season_stats['points_against']
    top_diff = season_stats.nlargest(1, 'point_diff').iloc[0]
    stat_leaders['efficiency'].append({
        'team': top_diff['team_abbr'],
        'stat': 'Point Differential',
        'value': int(top_diff['point_diff']),
        'unit': '',
        'rank': 1
    })

    return stat_leaders


def select_individual_highlights(starters_df):
    """
    Select top 6 individual performers using code logic.
    Returns structured data ready for AI to add context.
    """
    highlights = []

    # Top QB by passing yards
    qbs = starters_df[starters_df['position'] == 'QB'].copy()
    if len(qbs) > 0:
        top_qb_yards = qbs.nlargest(1, 'passing_yards_season').iloc[0]
        if pd.notna(top_qb_yards['passing_yards_season']) and top_qb_yards['passing_yards_season'] > 0:
            stat_line = f"{int(top_qb_yards['passing_yards_season'])} pass yds, {int(top_qb_yards['passing_tds_season'])} TDs"
            if pd.notna(top_qb_yards['interceptions_season']):
                stat_line += f", {int(top_qb_yards['interceptions_season'])} INTs"

            highlights.append({
                'player': top_qb_yards['player_name'],
                'team': top_qb_yards['team_abbr'],
                'position': 'QB',
                'leader_stat': 'Passing Yards',
                'stat_line': stat_line,
                'season_projection': f"{int(top_qb_yards['passing_yards_season'] * 17 / 5)} pass yds, {int(top_qb_yards['passing_tds_season'] * 17 / 5)} TDs"
            })

    # Top QB by TDs
    if len(qbs) > 0:
        top_qb_tds = qbs.nlargest(1, 'passing_tds_season').iloc[0]
        if pd.notna(top_qb_tds['passing_tds_season']) and top_qb_tds['passing_tds_season'] > 0:
            stat_line = f"{int(top_qb_tds['passing_yards_season'])} pass yds, {int(top_qb_tds['passing_tds_season'])} TDs"
            if pd.notna(top_qb_tds['interceptions_season']):
                stat_line += f", {int(top_qb_tds['interceptions_season'])} INTs"

            highlights.append({
                'player': top_qb_tds['player_name'],
                'team': top_qb_tds['team_abbr'],
                'position': 'QB',
                'leader_stat': 'Passing TDs',
                'stat_line': stat_line,
                'season_projection': f"{int(top_qb_tds['passing_yards_season'] * 17 / 5)} pass yds, {int(top_qb_tds['passing_tds_season'] * 17 / 5)} TDs"
            })

    # Top RB by rushing yards
    rbs = starters_df[starters_df['position'] == 'RB'].copy()
    if len(rbs) > 0:
        top_rb_yards = rbs.nlargest(1, 'rushing_yards_season').iloc[0]
        if pd.notna(top_rb_yards['rushing_yards_season']) and top_rb_yards['rushing_yards_season'] > 0:
            stat_line = f"{int(top_rb_yards['rushing_yards_season'])} rush yds, {int(top_rb_yards['rushing_tds_season'])} TDs"

            highlights.append({
                'player': top_rb_yards['player_name'],
                'team': top_rb_yards['team_abbr'],
                'position': 'RB',
                'leader_stat': 'Rushing Yards',
                'stat_line': stat_line,
                'season_projection': f"{int(top_rb_yards['rushing_yards_season'] * 17 / 5)} rush yds, {int(top_rb_yards['rushing_tds_season'] * 17 / 5)} TDs"
            })

    # Top RB by TDs
    if len(rbs) > 0:
        top_rb_tds = rbs.nlargest(1, 'rushing_tds_season').iloc[0]
        if pd.notna(top_rb_tds['rushing_tds_season']) and top_rb_tds['rushing_tds_season'] > 0:
            stat_line = f"{int(top_rb_tds['rushing_yards_season'])} rush yds, {int(top_rb_tds['rushing_tds_season'])} TDs"

            highlights.append({
                'player': top_rb_tds['player_name'],
                'team': top_rb_tds['team_abbr'],
                'position': 'RB',
                'leader_stat': 'Rushing TDs',
                'stat_line': stat_line,
                'season_projection': f"{int(top_rb_tds['rushing_yards_season'] * 17 / 5)} rush yds, {int(top_rb_tds['rushing_tds_season'] * 17 / 5)} TDs"
            })

    # Top WR/TE by receiving yards
    wrs = starters_df[starters_df['position'].isin(['WR', 'TE'])].copy()
    if len(wrs) > 0:
        top_wr_yards = wrs.nlargest(1, 'receiving_yards_season').iloc[0]
        if pd.notna(top_wr_yards['receiving_yards_season']) and top_wr_yards['receiving_yards_season'] > 0:
            stat_line = f"{int(top_wr_yards['receiving_yards_season'])} rec yds, {int(top_wr_yards['receiving_tds_season'])} TDs"

            highlights.append({
                'player': top_wr_yards['player_name'],
                'team': top_wr_yards['team_abbr'],
                'position': top_wr_yards['position'],
                'leader_stat': 'Receiving Yards',
                'stat_line': stat_line,
                'season_projection': f"{int(top_wr_yards['receiving_yards_season'] * 17 / 5)} rec yds, {int(top_wr_yards['receiving_tds_season'] * 17 / 5)} TDs"
            })

    # Top WR/TE by TDs
    if len(wrs) > 0:
        top_wr_tds = wrs.nlargest(1, 'receiving_tds_season').iloc[0]
        if pd.notna(top_wr_tds['receiving_tds_season']) and top_wr_tds['receiving_tds_season'] > 0:
            stat_line = f"{int(top_wr_tds['receiving_yards_season'])} rec yds, {int(top_wr_tds['receiving_tds_season'])} TDs"

            highlights.append({
                'player': top_wr_tds['player_name'],
                'team': top_wr_tds['team_abbr'],
                'position': top_wr_tds['position'],
                'leader_stat': 'Receiving TDs',
                'stat_line': stat_line,
                'season_projection': f"{int(top_wr_tds['receiving_yards_season'] * 17 / 5)} rec yds, {int(top_wr_tds['receiving_tds_season'] * 17 / 5)} TDs"
            })

    return highlights[:6]  # Return top 6


def select_power_rankings(sagarin_df, team_records, teams_df, teams_lookup):
    """
    Select power rankings using R1+SOV algorithm.
    Returns top 5, bottom 5, biggest riser, biggest faller.
    """
    # Build team name lookup
    team_names = {}
    for _, team in teams_df.iterrows():
        team_names[team['team_abbr']] = f"{team['city']} {team['mascot']}"

    # Rankings already have rank and movement calculated
    # Just ensure movement column exists
    if 'movement' not in sagarin_df.columns:
        sagarin_df['movement'] = 0

    movement_series = pd.to_numeric(sagarin_df['movement'], errors='coerce').fillna(0)
    sagarin_df = sagarin_df.assign(_movement=movement_series)

    # Top 5
    top_5 = []
    for idx, row in sagarin_df.head(5).iterrows():
        team_abbr = row['team_abbr']

        # Use record from dataframe if available, otherwise look up
        if 'record' in row and pd.notna(row['record']):
            record = row['record']
        else:
            record_data = team_records.get(team_abbr, {'wins': 0, 'losses': 0, 'ties': 0})
            record = f"{record_data['wins']}-{record_data['losses']}"
            if record_data['ties'] > 0:
                record += f"-{record_data['ties']}"

        # Determine trend
        trend = 'steady'
        movement = int(row['_movement']) if pd.notna(row['_movement']) else 0
        if movement > 0:
            trend = 'up'
        elif movement < 0:
            trend = 'down'

        # Format movement with sign
        movement_str = None
        if 'previous_rank' in row and pd.notna(row['previous_rank']):
            if movement > 0:
                movement_str = f"+{movement}"
            elif movement < 0:
                movement_str = f"{movement}"  # Already has minus sign
            else:
                movement_str = "0"

        top_5.append({
            'rank': int(row['rank']) if 'rank' in row else idx + 1,
            'team': team_abbr,
            'team_name': team_names.get(team_abbr, team_abbr),
            'rating': round(float(row['rating']), 2),
            'record': record,
            'trend': trend,
            'previous_rank': int(row['previous_rank']) if 'previous_rank' in row and pd.notna(row['previous_rank']) else None,
            'movement': movement_str,
            'head_coach': teams_lookup.get(team_abbr, {}).get('head_coach', ''),
            'offensive_coordinator': teams_lookup.get(team_abbr, {}).get('offensive_coordinator', ''),
            'defensive_coordinator': teams_lookup.get(team_abbr, {}).get('defensive_coordinator', '')
        })

    # Bottom 5
    bottom_5 = []
    for idx, row in sagarin_df.tail(5).iterrows():
        team_abbr = row['team_abbr']

        # Use record from dataframe if available, otherwise look up
        if 'record' in row and pd.notna(row['record']):
            record = row['record']
        else:
            record_data = team_records.get(team_abbr, {'wins': 0, 'losses': 0, 'ties': 0})
            record = f"{record_data['wins']}-{record_data['losses']}"
            if record_data['ties'] > 0:
                record += f"-{record_data['ties']}"

        # Determine trend
        trend = 'steady'
        movement = int(row['_movement']) if pd.notna(row['_movement']) else 0
        if movement > 0:
            trend = 'up'
        elif movement < 0:
            trend = 'down'

        # Format movement with sign
        movement_str = None
        if 'previous_rank' in row and pd.notna(row['previous_rank']):
            if movement > 0:
                movement_str = f"+{movement}"
            elif movement < 0:
                movement_str = f"{movement}"  # Already has minus sign
            else:
                movement_str = "0"

        bottom_5.append({
            'rank': idx + 1,
            'team': team_abbr,
            'team_name': team_names.get(team_abbr, team_abbr),
            'rating': round(float(row['rating']), 2),
            'record': record,
            'trend': trend,
            'previous_rank': int(row['previous_rank']) if 'previous_rank' in row and pd.notna(row['previous_rank']) else None,
            'movement': movement_str,
            'head_coach': teams_lookup.get(team_abbr, {}).get('head_coach', ''),
            'offensive_coordinator': teams_lookup.get(team_abbr, {}).get('offensive_coordinator', ''),
            'defensive_coordinator': teams_lookup.get(team_abbr, {}).get('defensive_coordinator', '')
        })

    def format_profile(row: pd.Series) -> dict:
        record_data = team_records.get(row['team_abbr'], {'wins': 0, 'losses': 0, 'ties': 0})
        record = f"{record_data['wins']}-{record_data['losses']}"
        if record_data['ties'] > 0:
            record += f"-{record_data['ties']}"

        movement_val = int(row['_movement']) if pd.notna(row['_movement']) else 0
        if movement_val > 0:
            movement_str = f"+{movement_val}"
        elif movement_val < 0:
            movement_str = f"{movement_val}"
        else:
            movement_str = "0"

        team_profile = teams_lookup.get(row['team_abbr'], {})

        return {
            'team': row['team_abbr'],
            'previous_rank': int(row['previous_rank']) if pd.notna(row.get('previous_rank')) else None,
            'current_rank': int(row['rank']) if pd.notna(row.get('rank')) else None,
            'movement': movement_str,
            'rating': round(float(row['rating']), 2),
            'record': record,
            'head_coach': team_profile.get('head_coach', ''),
            'offensive_coordinator': team_profile.get('offensive_coordinator', ''),
            'defensive_coordinator': team_profile.get('defensive_coordinator', '')
        }

    # Find biggest riser and faller (movement already calculated above)
    biggest_riser = None
    biggest_faller = None

    if not sagarin_df.empty:
        if movement_series.max() > 0:
            riser_row = sagarin_df.loc[movement_series.idxmax()]
            biggest_riser = format_profile(riser_row)
        else:
            biggest_riser = format_profile(sagarin_df.iloc[0])

        if movement_series.min() < 0:
            faller_row = sagarin_df.loc[movement_series.idxmin()]
            biggest_faller = format_profile(faller_row)
        else:
            biggest_faller = format_profile(sagarin_df.iloc[-1])

    # Full rankings (all 32 teams)
    all_rankings = []
    for idx, row in sagarin_df.iterrows():
        team_abbr = row['team_abbr']
        record_data = team_records.get(team_abbr, {'wins': 0, 'losses': 0, 'ties': 0})
        record = f"{record_data['wins']}-{record_data['losses']}"
        if record_data['ties'] > 0:
            record += f"-{record_data['ties']}"

        # Determine trend
        trend = 'steady'
        movement = int(row['_movement'])
        if movement > 0:
            trend = 'up'
        elif movement < 0:
            trend = 'down'

        # Format movement with sign
        movement_str = None
        if 'previous_rank' in row and pd.notna(row['previous_rank']):
            if movement > 0:
                movement_str = f"+{movement}"
            elif movement < 0:
                movement_str = f"{movement}"  # Already has minus sign
            else:
                movement_str = "0"

        team_profile = teams_lookup.get(team_abbr, {})

        all_rankings.append({
            'rank': idx + 1,
            'team': team_abbr,
            'team_name': team_names.get(team_abbr, team_abbr),
            'rating': round(float(row['rating']), 2),
            'record': record,
            'trend': trend,
            'previous_rank': int(row['previous_rank']) if 'previous_rank' in row and pd.notna(row['previous_rank']) else None,
            'movement': movement_str,
            'head_coach': team_profile.get('head_coach', ''),
            'offensive_coordinator': team_profile.get('offensive_coordinator', ''),
            'defensive_coordinator': team_profile.get('defensive_coordinator', '')
        })

    return {
        'top_5': top_5,
        'bottom_5': bottom_5,
        'all_rankings': all_rankings,
        'biggest_riser': biggest_riser,
        'biggest_faller': biggest_faller
    }


def select_playoff_snapshot(analysis_cache, team_records, playoff_seeds):
    """
    Select playoff snapshot based on playoff probabilities.
    Returns top 4, middle 4, bottom 4.
    """
    # Extract playoff probabilities
    playoff_probs = []
    for team_abbr, team_data in analysis_cache['team_analyses'].items():
        prob = team_data.get('playoff_chance', 0)
        record_data = team_records.get(team_abbr, {'wins': 0, 'losses': 0, 'ties': 0})
        record = f"{record_data['wins']}-{record_data['losses']}"
        if record_data['ties'] > 0:
            record += f"-{record_data['ties']}"

        playoff_probs.append({
            'team': team_abbr,
            'probability': round(prob, 1),
            'current_seed': playoff_seeds.get(team_abbr),
            'record': record
        })

    # Sort by probability
    playoff_probs.sort(key=lambda x: x['probability'], reverse=True)

    return {
        'top_4': playoff_probs[:4],
        'middle_4': playoff_probs[4:8],
        'bottom_4': playoff_probs[-4:]
    }


def select_game_of_week_and_meek(upcoming_games, team_records, sagarin_df, analysis_cache, game_analyses):
    """
    Select game of the week (best) and game of the meek (worst) using scoring logic.
    """
    sagarin_ranks = {row['team_abbr']: idx + 1 for idx, row in sagarin_df.iterrows()}

    game_scores = []

    for game in upcoming_games:
        away_team = game['away_team']
        home_team = game['home_team']

        # Get data
        away_prob = analysis_cache['team_analyses'].get(away_team, {}).get('playoff_chance', 0)
        home_prob = analysis_cache['team_analyses'].get(home_team, {}).get('playoff_chance', 0)
        away_rank = sagarin_ranks.get(away_team, 32)
        home_rank = sagarin_ranks.get(home_team, 32)

        # Scoring logic for game of the week
        score = 0
        score += (away_prob + home_prob) / 2  # Higher combined playoff probability
        score += max(0, 32 - away_rank)  # Higher rank (lower number)
        score += max(0, 32 - home_rank)

        if game['is_divisional']:
            score += 20  # Bonus for divisional games

        # Primetime bonuses - SNF/MNF get more than TNF (network TV vs streaming)
        primetime_type = game.get('primetime_type')
        if primetime_type == 'sunday_night':
            score += 15  # Sunday Night Football (NBC - biggest audience)
        elif primetime_type == 'monday_night':
            score += 12  # Monday Night Football (ESPN)
        elif primetime_type == 'thursday':
            score += 8   # Thursday Night Football (Amazon Prime)

        if game.get('is_thanksgiving', False):
            score += 20  # Extra bonus for Thanksgiving games (tradition + family viewing)

        # Get records
        away_record_data = team_records.get(away_team, {'wins': 0, 'losses': 0, 'ties': 0})
        away_record = f"{away_record_data['wins']}-{away_record_data['losses']}"
        if away_record_data['ties'] > 0:
            away_record += f"-{away_record_data['ties']}"

        home_record_data = team_records.get(home_team, {'wins': 0, 'losses': 0, 'ties': 0})
        home_record = f"{home_record_data['wins']}-{home_record_data['losses']}"
        if home_record_data['ties'] > 0:
            home_record += f"-{home_record_data['ties']}"

        # Get betting info
        game_analysis = game_analyses.get(str(game['espn_id']), {})
        betting = game_analysis.get('betting', {})

        # Convert spread to be relative to away team for clarity
        # If KC is favorite at -2.5, and KC is home, then away_spread = +2.5
        spread = betting.get('spread')
        favorite = betting.get('favorite')
        away_spread = None
        if spread is not None and favorite:
            if favorite == away_team:
                away_spread = spread  # Away team is favorite, keep negative spread
            elif favorite == home_team:
                away_spread = -spread  # Home team is favorite, flip to positive (away gets points)

        game_scores.append({
            'espn_id': str(game['espn_id']),
            'away_team': away_team,
            'away_record': away_record,
            'away_rank': away_rank,
            'away_playoff_prob': round(away_prob, 1),
            'home_team': home_team,
            'home_record': home_record,
            'home_rank': home_rank,
            'home_playoff_prob': round(home_prob, 1),
            'betting_line': away_spread,  # Now relative to away team
            'over_under': betting.get('over_under'),
            'score': score,
            'inverse_score': -score,  # For finding worst game
            'stadium': game.get('stadium', ''),
            'gametime': game.get('gametime', ''),
            'away_head_coach': game.get('away_coach', ''),
            'home_head_coach': game.get('home_coach', '')
        })

    # Log all game scores for debugging
    logger.info("Game quality scores for this week:")
    for g in sorted(game_scores, key=lambda x: x['score'], reverse=True):
        logger.info(f"  {g['away_team']} @ {g['home_team']}: {g['score']:.1f}")

    # Best game (highest score)
    game_of_week = max(game_scores, key=lambda x: x['score']).copy()
    game_of_week['quality_score'] = round(game_of_week['score'], 1)
    del game_of_week['score']
    del game_of_week['inverse_score']

    # Worst game (lowest score)
    game_of_meek = min(game_scores, key=lambda x: x['score']).copy()
    game_of_meek['quality_score'] = round(game_of_meek['score'], 1)
    del game_of_meek['score']
    del game_of_meek['inverse_score']

    return game_of_week, game_of_meek


def select_week_preview(upcoming_games, game_of_week_id, team_records, sagarin_df, analysis_cache, teams_lookup):
    """
    Select Thursday night game and 2-3 Sunday spotlight games.
    """
    sagarin_ranks = {row['team_abbr']: idx + 1 for idx, row in sagarin_df.iterrows()}

    thursday_night = None
    sunday_games = []

    for game in upcoming_games:
        # Get records
        away_record_data = team_records.get(game['away_team'], {'wins': 0, 'losses': 0, 'ties': 0})
        away_record = f"{away_record_data['wins']}-{away_record_data['losses']}"
        if away_record_data['ties'] > 0:
            away_record += f"-{away_record_data['ties']}"

        home_record_data = team_records.get(game['home_team'], {'wins': 0, 'losses': 0, 'ties': 0})
        home_record = f"{home_record_data['wins']}-{home_record_data['losses']}"
        if home_record_data['ties'] > 0:
            home_record += f"-{home_record_data['ties']}"

        team_profile_away = teams_lookup.get(game['away_team'], {})
        team_profile_home = teams_lookup.get(game['home_team'], {})

        game_data = {
            'espn_id': str(game['espn_id']),
            'away_team': game['away_team'],
            'away_record': away_record,
            'home_team': game['home_team'],
            'home_record': home_record,
            'away_head_coach': team_profile_away.get('head_coach', ''),
            'home_head_coach': team_profile_home.get('head_coach', ''),
            'away_offensive_coordinator': team_profile_away.get('offensive_coordinator', ''),
            'home_offensive_coordinator': team_profile_home.get('offensive_coordinator', ''),
            'away_defensive_coordinator': team_profile_away.get('defensive_coordinator', ''),
            'home_defensive_coordinator': team_profile_home.get('defensive_coordinator', ''),
            'stadium': team_profile_home.get('stadium', ''),
            'gametime': game.get('gametime', '')
        }

        if game['is_thursday']:
            thursday_night = game_data
        elif str(game['espn_id']) != game_of_week_id:  # Don't duplicate game of week
            # Score for Sunday spotlight
            away_prob = analysis_cache['team_analyses'].get(game['away_team'], {}).get('playoff_chance', 0)
            home_prob = analysis_cache['team_analyses'].get(game['home_team'], {}).get('playoff_chance', 0)
            away_rank = sagarin_ranks.get(game['away_team'], 32)
            home_rank = sagarin_ranks.get(game['home_team'], 32)

            score = (away_prob + home_prob) / 2
            score += max(0, 32 - away_rank)
            score += max(0, 32 - home_rank)

            game_data['score'] = score
            sunday_games.append(game_data)

    # Sort Sunday games and take top 3
    sunday_games.sort(key=lambda x: x['score'], reverse=True)
    sunday_spotlight = sunday_games[:3]
    for game in sunday_spotlight:
        del game['score']

    return thursday_night, sunday_spotlight


def generate_dashboard_content(ai_model=None):
    """
    Generate dashboard_content.json with code-based data selection and AI-based creative text.
    """
    try:
        logger.info("Preparing dashboard data...")

        # Load all data files
        with open('data/schedule.json', 'r', encoding='utf-8') as fh:
            schedule_df = pd.DataFrame(json.load(fh))

        # Normalize score columns: blank strings -> NaN, numeric strings -> numbers
        for score_col in ('away_score', 'home_score'):
            if score_col in schedule_df.columns:
                schedule_df[score_col] = pd.to_numeric(schedule_df[score_col], errors='coerce')
        with open('data/team_stats.json', 'r', encoding='utf-8') as fh:
            team_stats_df = pd.DataFrame(json.load(fh))

        # Load R1+SOV power rankings instead of Sagarin
        from power_rankings_dashboard import get_power_rankings_df
        current_week = schedule_df[schedule_df['away_score'].notna()]['week_num'].max()
        logger.info(f"Current week: {current_week}")
        sagarin_df = get_power_rankings_df(int(current_week))
        logger.info(f"Loaded {len(sagarin_df)} R1+SOV power rankings")

        with open('data/team_starters.json', 'r', encoding='utf-8') as fh:
            starters_df = pd.DataFrame(json.load(fh))
        with open('data/teams.json', 'r', encoding='utf-8') as fh:
            teams_records = json.load(fh)
        teams_df = pd.DataFrame(teams_records)
        teams_lookup = {record['team_abbr']: record for record in teams_records}

        with open('data/analysis_cache.json', 'r') as f:
            analysis_cache = json.load(f)

        with open('data/standings_cache.json', 'r') as f:
            standings_cache = json.load(f)

        with open('data/game_analyses.json', 'r') as f:
            game_analyses = json.load(f)

        # Power rankings metadata (no longer using Sagarin cache)
        with open('data/power_rankings_history.json', 'r') as f:
            pr_history = json.load(f)

        # Create a cache-like object for compatibility
        sagarin_cache = {
            'last_update': pr_history.get('last_updated', datetime.now().isoformat()),
            'last_content_update': pr_history.get('last_updated', datetime.now().isoformat())
        }

        # Build team records lookup
        team_records = {}
        for conf_name, conf_standings in standings_cache['standings']['conference'].items():
            for team_data in conf_standings:
                team_abbr = team_data['team']
                team_records[team_abbr] = {
                    'wins': team_data['wins'],
                    'losses': team_data['losses'],
                    'ties': team_data['ties']
                }

        # Build playoff seeds lookup (includes ALL teams 1-16 per conference)
        playoff_seeds = {}
        for conf_name, conf_playoffs in standings_cache['standings']['playoff'].items():
            for team_data in conf_playoffs.get('division_winners', []):
                playoff_seeds[team_data['team']] = team_data['seed']
            for team_data in conf_playoffs.get('wild_cards', []):
                playoff_seeds[team_data['team']] = team_data['seed']
            for team_data in conf_playoffs.get('eliminated', []):
                playoff_seeds[team_data['team']] = team_data['seed']

        # Get current week
        current_week = get_current_week(schedule_df)
        logger.info(f"Current week: {current_week}")

        # Get all games from current week (for game of week/meek selection)
        all_week_games = schedule_df[schedule_df['week_num'] == current_week].to_dict('records')

        # Get only unplayed games (for "REMAINING GAMES THIS WEEK" section)
        upcoming_games = schedule_df[
            (schedule_df['week_num'] == current_week) &
            (schedule_df['away_score'].isna() | schedule_df['home_score'].isna())
        ].to_dict('records')

        # Add flags to all week games
        divisions = dict(zip(teams_df['team_abbr'], teams_df['division']))
        for game in all_week_games:
            away_div = divisions.get(game['away_team'])
            home_div = divisions.get(game['home_team'])
            game['is_divisional'] = (away_div == home_div and away_div is not None)

            try:
                game_datetime = pd.to_datetime(game['game_date'])
                game_time = pd.to_datetime(game.get('gametime', '00:00'), format='%H:%M').time() if game.get('gametime') else None

                # Primetime detection: Thursday/Sunday/Monday night games
                day_of_week = game_datetime.dayofweek
                game['is_thursday'] = (day_of_week == 3)  # Keep for backward compatibility

                # Primetime breakdown by type
                if day_of_week == 3:  # Thursday Night Football (Amazon Prime)
                    game['is_primetime'] = True
                    game['primetime_type'] = 'thursday'
                elif day_of_week == 6 and game_time and game_time.hour >= 20:  # Sunday Night Football (NBC)
                    game['is_primetime'] = True
                    game['primetime_type'] = 'sunday_night'
                elif day_of_week == 0 and game_time and game_time.hour >= 19:  # Monday Night Football (ESPN) - can start as early as 7pm ET
                    game['is_primetime'] = True
                    game['primetime_type'] = 'monday_night'
                else:
                    game['is_primetime'] = False
                    game['primetime_type'] = None

                # Thanksgiving detection (Thursday in November, specifically around 4th Thursday)
                game['is_thanksgiving'] = (day_of_week == 3 and game_datetime.month == 11 and 22 <= game_datetime.day <= 28)

            except:
                game['is_thursday'] = False
                game['is_primetime'] = False
                game['primetime_type'] = None
                game['is_thanksgiving'] = False

        # Add flags to upcoming games as well
        for game in upcoming_games:
            away_div = divisions.get(game['away_team'])
            home_div = divisions.get(game['home_team'])
            game['is_divisional'] = (away_div == home_div and away_div is not None)

            try:
                game_datetime = pd.to_datetime(game['game_date'])
                game_time = pd.to_datetime(game.get('gametime', '00:00'), format='%H:%M').time() if game.get('gametime') else None

                # Primetime detection: Thursday/Sunday/Monday night games
                day_of_week = game_datetime.dayofweek
                game['is_thursday'] = (day_of_week == 3)  # Keep for backward compatibility

                # Primetime breakdown by type
                if day_of_week == 3:  # Thursday Night Football (Amazon Prime)
                    game['is_primetime'] = True
                    game['primetime_type'] = 'thursday'
                elif day_of_week == 6 and game_time and game_time.hour >= 20:  # Sunday Night Football (NBC)
                    game['is_primetime'] = True
                    game['primetime_type'] = 'sunday_night'
                elif day_of_week == 0 and game_time and game_time.hour >= 19:  # Monday Night Football (ESPN) - can start as early as 7pm ET
                    game['is_primetime'] = True
                    game['primetime_type'] = 'monday_night'
                else:
                    game['is_primetime'] = False
                    game['primetime_type'] = None

                # Thanksgiving detection (Thursday in November, specifically around 4th Thursday)
                game['is_thanksgiving'] = (day_of_week == 3 and game_datetime.month == 11 and 22 <= game_datetime.day <= 28)

            except:
                game['is_thursday'] = False
                game['is_primetime'] = False
                game['primetime_type'] = None
                game['is_thanksgiving'] = False

        # CODE-BASED DATA SELECTION
        logger.info("Selecting data using code logic...")

        stat_leaders = select_stat_leaders(team_stats_df, sagarin_df)
        individual_highlights = select_individual_highlights(starters_df)

        # Check if any games have been played this week
        any_games_played = any(
            pd.notna(g.get('away_score')) and pd.notna(g.get('home_score'))
            for g in all_week_games
        )

        # Always generate power rankings data for the dashboard JSON
        power_rankings = select_power_rankings(sagarin_df, team_records, teams_df, teams_lookup)

        # But don't include them in the AI prompt context if games have been played (they're stale)
        include_power_rankings_in_prompt = not any_games_played

        if include_power_rankings_in_prompt:
            logger.info("No games played yet this week - will include power rankings in AI prompt context")
        else:
            logger.info("Games have been played this week - will exclude power rankings from AI prompt context (Sagarin ratings are stale)")

        playoff_snapshot = select_playoff_snapshot(analysis_cache, team_records, playoff_seeds)
        game_of_week, game_of_meek = select_game_of_week_and_meek(
            all_week_games, team_records, sagarin_df, analysis_cache, game_analyses
        )
        thursday_night, sunday_spotlight = select_week_preview(
            upcoming_games,
            game_of_week['espn_id'],
            team_records,
            sagarin_df,
            analysis_cache,
            teams_lookup
        )

        # Add game scores and analysis to game of week/meek
        for game_obj in [game_of_week, game_of_meek]:
            espn_id = game_obj['espn_id']
            game_analysis = game_analyses.get(espn_id, {})

            # Check if game has been played
            away_score = game_analysis.get('away_score')
            home_score = game_analysis.get('home_score')

            if away_score is not None and home_score is not None:
                # Game has been played
                game_obj['status'] = 'completed'
                game_obj['away_score'] = away_score
                game_obj['home_score'] = home_score
                game_obj['analysis'] = game_analysis.get('analysis', '')
                game_obj['analysis_type'] = 'post_game'
            else:
                # Game hasn't been played yet
                game_obj['status'] = 'upcoming'
                game_obj['away_score'] = None
                game_obj['home_score'] = None
                game_obj['analysis'] = game_analysis.get('analysis', '')
                game_obj['analysis_type'] = 'pre_game'

            away_profile = teams_lookup.get(game_obj['away_team'], {})
            home_profile = teams_lookup.get(game_obj['home_team'], {})

            game_obj.setdefault('away_head_coach', away_profile.get('head_coach', ''))
            game_obj.setdefault('home_head_coach', home_profile.get('head_coach', ''))
            game_obj.setdefault('away_offensive_coordinator', away_profile.get('offensive_coordinator', ''))
            game_obj.setdefault('home_offensive_coordinator', home_profile.get('offensive_coordinator', ''))
            game_obj.setdefault('away_defensive_coordinator', away_profile.get('defensive_coordinator', ''))
            game_obj.setdefault('home_defensive_coordinator', home_profile.get('defensive_coordinator', ''))
            if not game_obj.get('stadium'):
                game_obj['stadium'] = game_analysis.get('stadium') or home_profile.get('stadium', '')

        # AI-BASED CREATIVE TEXT GENERATION
        logger.info("Generating creative text with AI...")

        # Build comprehensive league context for AI
        # Build full standings with conference, division, and playoff info for every team
        full_standings = []

        # Build lookup maps
        conference_ranks = {}
        division_ranks = {}

        # Get conference ranks
        for conf_name, conf_teams in standings_cache['standings']['conference'].items():
            for team_data in conf_teams:
                conference_ranks[team_data['team']] = team_data['rank']

        # Get division ranks from divisional standings
        for conf_name, divisions in standings_cache['standings']['divisional'].items():
            for div_name, div_teams in divisions.items():
                for team_data in div_teams:
                    division_ranks[team_data['team']] = team_data['rank']

        # Build full standings for all 32 teams
        for conf_name, divisions in standings_cache['standings']['divisional'].items():
            for div_name, div_teams in divisions.items():
                for team_data in div_teams:
                    team_abbr = team_data['team']
                    record = f"{team_data['wins']}-{team_data['losses']}"
                    if team_data['ties'] > 0:
                        record += f"-{team_data['ties']}"

                    full_standings.append({
                        'team': team_abbr,
                        'conference': conf_name,
                        'division': div_name,
                        'record': record,
                        'division_rank': team_data['rank'],
                        'conference_rank': conference_ranks.get(team_abbr),
                        'playoff_seed': playoff_seeds.get(team_abbr)
                    })

        # Prepare all teams data sorted by power ranking (for power rankings context)
        all_teams_context = []
        for idx, row in sagarin_df.iterrows():
            team_abbr = row['team_abbr']
            record_data = team_records.get(team_abbr, {'wins': 0, 'losses': 0, 'ties': 0})
            record = f"{record_data['wins']}-{record_data['losses']}"
            if record_data['ties'] > 0:
                record += f"-{record_data['ties']}"

            playoff_prob = analysis_cache['team_analyses'].get(team_abbr, {}).get('playoff_chance', 0)
            playoff_seed = playoff_seeds.get(team_abbr)
            team_profile = teams_lookup.get(team_abbr, {})

            all_teams_context.append({
                'team': team_abbr,
                'rank': idx + 1,
                'rating': round(float(row['rating']), 2),
                'record': record,
                'playoff_prob': round(playoff_prob, 1),
                'playoff_seed': playoff_seed,
                'head_coach': team_profile.get('head_coach', ''),
                'offensive_coordinator': team_profile.get('offensive_coordinator', ''),
                'defensive_coordinator': team_profile.get('defensive_coordinator', ''),
                'stadium': team_profile.get('stadium', '')
            })

        # Get week chaos data
        week_chaos = analysis_cache.get('week_chaos', {})
        week_chaos_score = week_chaos.get('score', 0)

        # Build chaos context if week is volatile
        chaos_context = ""
        if week_chaos_score >= 65:
            chaos_context = f"""
WEEK VOLATILITY ALERT:
This is an unusually chaotic week (Chaos Index: {week_chaos_score}/100).
{week_chaos.get('description', '')}
High-chaos teams: {', '.join([t['team'] for t in week_chaos.get('highest_chaos_teams', [])[:5]])}

You may reference this volatility in your league_pulse.summary if it adds to the narrative,
but don't force it. Let the drama emerge naturally.
"""

        # Get completed games from this week
        def _int_or_none(value):
            """Convert score fields to ints, tolerating blanks/NaN."""
            if value in (None, '', 'nan'):
                return None
            try:
                return int(value)
            except (ValueError, TypeError):
                try:
                    return int(float(value))
                except (ValueError, TypeError):
                    return None

        completed_games = []
        for game in all_week_games:
            away_score = _int_or_none(game.get('away_score'))
            home_score = _int_or_none(game.get('home_score'))
            if away_score is not None and home_score is not None:
                completed_games.append({
                    'away': game['away_team'],
                    'away_score': away_score,
                    'home': game['home_team'],
                    'home_score': home_score
                })

        def _format_spread(team_abbr: str, value):
            """Create a friendly spread string for prompt output."""
            if value in (None, '', 'N/A'):
                return f"{team_abbr} N/A"
            if isinstance(value, str) and value.strip().upper() in {'PK', 'PICK', 'EVEN'}:
                return f"{team_abbr} PK"
            try:
                spread = float(value)
            except (TypeError, ValueError):
                return f"{team_abbr} {value}"

            if abs(spread) < 1e-6:
                return f"{team_abbr} PK"

            formatted = f"{abs(spread):.1f}".rstrip('0').rstrip('.')
            if spread > 0:
                return f"{team_abbr} +{formatted}"
            return f"{team_abbr} -{formatted}"

        def _format_over_under(value):
            """Normalize over/under values for display."""
            if value in (None, '', 'N/A'):
                return 'N/A'
            try:
                total = float(value)
            except (TypeError, ValueError):
                return str(value)
            return f"{total:.1f}".rstrip('0').rstrip('.')

        def _build_game_details(game_record, include_playoff_probs=False):
            lines = []
            status = game_record.get('status')
            away_team = game_record['away_team']
            home_team = game_record['home_team']
            away_record = game_record.get('away_record', 'N/A')
            home_record = game_record.get('home_record', 'N/A')
            away_head_coach = game_record.get('away_head_coach') or 'N/A'
            home_head_coach = game_record.get('home_head_coach') or 'N/A'
            away_oc = game_record.get('away_offensive_coordinator') or 'N/A'
            home_oc = game_record.get('home_offensive_coordinator') or 'N/A'
            away_dc = game_record.get('away_defensive_coordinator') or 'N/A'
            home_dc = game_record.get('home_defensive_coordinator') or 'N/A'
            stadium_name = game_record.get('stadium') or ''

            def _format_playoff_prob(value):
                if value in (None, '', 'N/A'):
                    return 'N/A'
                try:
                    pct = float(value)
                except (TypeError, ValueError):
                    return str(value)
                formatted = f"{pct:.1f}".rstrip('0').rstrip('.')
                return f"{formatted}%"

            if status == 'completed':
                away_score = game_record.get('away_score', 'N/A')
                home_score = game_record.get('home_score', 'N/A')
                lines.append(f"FINAL SCORE: {away_team} {away_score} - {home_team} {home_score}")
            else:
                spread_text = _format_spread(away_team, game_record.get('betting_line'))
                ou_text = _format_over_under(game_record.get('over_under'))
                lines.append(f"Spread: {spread_text}, Over/Under: {ou_text}")

            if include_playoff_probs:
                away_prob = _format_playoff_prob(game_record.get('away_playoff_prob'))
                home_prob = _format_playoff_prob(game_record.get('home_playoff_prob'))
                lines.append(f"{away_team} Record: {away_record}, Playoff Prob: {away_prob}")
                lines.append(f"{home_team} Record: {home_record}, Playoff Prob: {home_prob}")
            else:
                lines.append(f"{away_team} Record: {away_record}")
                lines.append(f"{home_team} Record: {home_record}")

            lines.append(
                f"Head Coaches: {away_team} - {away_head_coach}, {home_team} - {home_head_coach}"
            )
            lines.append(
                f"Coordinators (OC/DC): {away_team} - {away_oc}/{away_dc}; {home_team} - {home_oc}/{home_dc}"
            )
            if stadium_name:
                lines.append(f"Venue: {stadium_name}")

            return '\n'.join(lines)

        game_of_week_details = _build_game_details(game_of_week, include_playoff_probs=True)
        game_of_meek_details = _build_game_details(game_of_meek, include_playoff_probs=False)

        # Build prompt with comprehensive context
        # REFACTORED STRUCTURE: Put data inline with each request to reduce cognitive load
        ai_prompt = f"""You are generating creative text for an NFL dashboard. Generate ONLY the requested text fields in JSON format.

Current Week: {current_week}

=== GENERAL LEAGUE CONTEXT ===
This data is available for reference throughout all your responses.

FULL LEAGUE STANDINGS (all 32 teams with records):
{json.dumps(full_standings, separators=(',', ':'))}

{f'''COMPLETED GAMES THIS WEEK:
{json.dumps(completed_games, separators=(',', ':'))}

''' if completed_games else '''NOTE: No games have been completed this week yet.

'''}
{f'''ALL 32 TEAMS (Power Rankings):
{json.dumps(all_teams_context, separators=(',', ':'))}

''' if include_power_rankings_in_prompt else ''}{chaos_context}
=== END GENERAL CONTEXT ===

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Now generate the following creative text fields. Each section has its own data context.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SECTION 1: LEAGUE PULSE
Generate two fields summarizing the current league landscape:

1a. league_pulse_summary
   - 3-6 engaging sentences with markdown formatting (bold for team names, italics for emphasis)
   - Focus on NARRATIVE and STORYLINES, not raw statistics
   - Tell a compelling story about what's happening in the league
   - Minimize numbers - they're shown elsewhere on the page

1b. key_storylines (array of 3)
   - Each storyline has: title (catchy 3-6 word headline), description (2-3 sentences)
   - Focus on narrative over numbers. Use stats sparingly and only when essential.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SECTION 2: GAME OF THE WEEK

Game: {game_of_week['away_team']} @ {game_of_week['home_team']}
Status: {game_of_week.get('status', 'upcoming')}
{game_of_week_details}

2. game_of_week_tagline (1-2 sentences)
   - If completed: Brief recap highlighting what made it the best game or why it did not live up to expectations
   - If upcoming: Explain why it's THE game to watch
   - Do NOT reference past seasons, Super Bowls, or historical matchups

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SECTION 3: GAME OF THE MEEK

Game: {game_of_meek['away_team']} @ {game_of_meek['home_team']}
Status: {game_of_meek.get('status', 'upcoming')}
{game_of_meek_details}

3. game_of_meek_tagline (1-2 sentences)
   - If completed: Humorous recap of why it was forgettable or why it was better than expected
   - If upcoming: Explain humorously why it's skippable

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SECTION 4: STAT LEADER CONTEXTS
Generate brief context phrases (2-6 words each) for the following stat leaders:

4a. OFFENSE LEADERS (provide exactly 5 context phrases in order):
{json.dumps(stat_leaders['offense'], indent=2)}

4b. DEFENSE LEADERS (provide exactly 5 context phrases in order):
{json.dumps(stat_leaders['defense'], indent=2)}

4c. EFFICIENCY LEADERS (provide exactly 5 context phrases in order):
{json.dumps(stat_leaders['efficiency'], indent=2)}

Return format: {{"offense": ["phrase1", "phrase2", "phrase3", "phrase4", "phrase5"], "defense": [...], "efficiency": [...]}}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SECTION 5: INDIVIDUAL HIGHLIGHTS
Generate context for the following 6 individual stat leaders (PLAIN STRINGS explaining why performance matters):

{json.dumps(individual_highlights, indent=2)}

5. individual_contexts (array of 6 strings)
   Return format: ["string1", "string2", "string3", "string4", "string5", "string6"]
   CRITICAL: Focus ONLY on their current season performance and stats shown above
   Do NOT make assumptions about career length, years in league, or draft status

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SECTION 6: POWER RANKINGS
Generate 1-sentence reasons for biggest movers:

6a. Biggest Riser: {power_rankings['biggest_riser']['team'] if power_rankings['biggest_riser'] else 'N/A'}
{f"   Moved from #{power_rankings['biggest_riser']['previous_rank']} to #{power_rankings['biggest_riser']['current_rank']} (Record: {power_rankings['biggest_riser']['record']})" if power_rankings['biggest_riser'] else ''}

6b. Biggest Faller: {power_rankings['biggest_faller']['team'] if power_rankings['biggest_faller'] else 'N/A'}
{f"   Moved from #{power_rankings['biggest_faller']['previous_rank']} to #{power_rankings['biggest_faller']['current_rank']} (Record: {power_rankings['biggest_faller']['record']})" if power_rankings['biggest_faller'] else ''}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SECTION 7: PLAYOFF PICTURE
Generate brief context (PLAIN STRING) for these specific 12 teams in playoff race:

7a. TOP 4 PLAYOFF TEAMS (provide contexts in this exact order):
{chr(10).join([f"   {i+1}. {team['team']} - Record: {team['record']}, Playoff Prob: {team['probability']}%, Current Seed: {team.get('current_seed', 'N/A')}" for i, team in enumerate(playoff_snapshot['top_4'])])}

7b. MIDDLE 4 PLAYOFF TEAMS (provide contexts in this exact order):
{chr(10).join([f"   {i+1}. {team['team']} - Record: {team['record']}, Playoff Prob: {team['probability']}%, Current Seed: {team.get('current_seed', 'N/A')}" for i, team in enumerate(playoff_snapshot['middle_4'])])}

7c. BOTTOM 4 PLAYOFF TEAMS (provide contexts in this exact order):
{chr(10).join([f"   {i+1}. {team['team']} - Record: {team['record']}, Playoff Prob: {team['probability']}%, Current Seed: {team.get('current_seed', 'N/A')}" for i, team in enumerate(playoff_snapshot['bottom_4'])])}

Return format: {{"top_4": ["string1", "string2", "string3", "string4"], "middle_4": [...], "bottom_4": [...]}}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SECTION 8: WEEK PREVIEW
Generate brief taglines (5-8 words) for the following games:

8a. Thursday Night Game:
{json.dumps(thursday_night, indent=2) if thursday_night else 'None'}

8b. Sunday Spotlight Games (provide {len(sunday_spotlight)} taglines):
{json.dumps(sunday_spotlight, indent=2)}

Return format: "thursday_tagline": "...", "sunday_taglines": [...]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

IMPORTANT: All context fields must be PLAIN STRINGS in arrays, not nested objects. Do not include team names, stats, or other data - ONLY the creative text.

Return ONLY valid JSON with this exact structure:
{{
  "league_pulse_summary": "...",
  "key_storylines": [{{"title": "...", "description": "..."}}],
  "game_of_week_tagline": "...",
  "game_of_meek_tagline": "...",
  "stat_leader_contexts": {{"offense": ["context string", ...], "defense": [...], "efficiency": [...]}},
  "individual_contexts": ["context string", ...],
  "biggest_riser_reason": "...",
  "biggest_faller_reason": "...",
  "playoff_key_infos": {{"top_4": ["context string", ...], "middle_4": [...], "bottom_4": [...]}},
  "thursday_tagline": "...",
  "sunday_taglines": [...]
}}
"""

        # Initialize AI service
        from ai_service import resolve_model_name, detect_provider_from_model

        model_override = None
        if ai_model:
            model_override = resolve_model_name(ai_model)
            detected_provider = detect_provider_from_model(model_override)
            if detected_provider:
                import ai_service as ai_service_module
                ai_service_module.model_provider = detected_provider

        ai_service = AIService(model_override=model_override)

        logger.info(f"Generating creative text with AI (model: {ai_service.model})...")

        # Save the prompt to data/prompts for debugging/review
        os.makedirs('data/prompts', exist_ok=True)
        prompt_path = 'data/prompts/dashboard_generation.txt'
        with open(prompt_path, 'w') as f:
            f.write(ai_prompt)

        system_message = "You are an expert NFL analyst generating creative text for a dashboard. Return only valid JSON with the requested creative text fields."

        response, status = ai_service.generate_analysis(ai_prompt, system_message=system_message)

        if status != 'success':
            logger.error(f"Failed to generate creative text: {status}")
            return False

        # Parse AI response
        try:
            ai_creative = json.loads(response)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse AI response as JSON: {e}")
            logger.error(f"Response: {response[:500]}...")
            return False

        # Validate AI response has all required fields
        required_fields = [
            'league_pulse_summary', 'key_storylines', 'game_of_week_tagline',
            'game_of_meek_tagline', 'stat_leader_contexts', 'individual_contexts',
            'biggest_riser_reason', 'biggest_faller_reason', 'playoff_key_infos',
            'thursday_tagline', 'sunday_taglines'
        ]

        missing_fields = [f for f in required_fields if f not in ai_creative]
        if missing_fields:
            logger.error(f"AI response missing required fields: {missing_fields}")
            logger.error(f"AI response keys: {list(ai_creative.keys())}")
            logger.error(f"Full response saved to data/prompts/dashboard_generation.txt")
            with open('data/prompts/dashboard_ai_response.json', 'w') as f:
                json.dump(ai_creative, f, indent=2)
            return False

        # Validate stat_leader_contexts structure
        stat_contexts = ai_creative.get('stat_leader_contexts', {})
        if not isinstance(stat_contexts, dict):
            logger.error(f"stat_leader_contexts should be dict, got {type(stat_contexts)}")
            return False
        for category in ['offense', 'defense', 'efficiency']:
            if category not in stat_contexts:
                logger.error(f"stat_leader_contexts missing '{category}' key")
                logger.error(f"Available keys: {list(stat_contexts.keys())}")
                with open('data/prompts/dashboard_ai_response.json', 'w') as f:
                    json.dump(ai_creative, f, indent=2)
                logger.error(f"Full AI response saved to data/prompts/dashboard_ai_response.json")
                return False
            if not isinstance(stat_contexts[category], list) or len(stat_contexts[category]) != 5:
                logger.error(f"stat_leader_contexts['{category}'] should be list of 5 strings, got {type(stat_contexts[category])} with length {len(stat_contexts[category]) if isinstance(stat_contexts[category], list) else 'N/A'}")
                return False

        # COMBINE CODE-SELECTED DATA WITH AI-GENERATED TEXT
        logger.info("Combining code-selected data with AI-generated text...")

        dashboard_content = {
            'timestamp': datetime.now().isoformat(),
            'week': current_week,
            'week_chaos': week_chaos,
            'league_pulse': {
                'summary': ai_creative['league_pulse_summary'],
                'key_storylines': ai_creative['key_storylines']
            },
            'game_of_the_week': {
                **game_of_week,
                'tagline': ai_creative['game_of_week_tagline']
            },
            'game_of_the_meek': {
                **game_of_meek,
                'tagline': ai_creative['game_of_meek_tagline']
            },
            'stat_leaders': {
                'offense': [
                    {**stat, 'context': ctx}
                    for stat, ctx in zip(stat_leaders['offense'], ai_creative['stat_leader_contexts']['offense'])
                ],
                'defense': [
                    {**stat, 'context': ctx}
                    for stat, ctx in zip(stat_leaders['defense'], ai_creative['stat_leader_contexts']['defense'])
                ],
                'efficiency': [
                    {**stat, 'context': ctx}
                    for stat, ctx in zip(stat_leaders['efficiency'], ai_creative['stat_leader_contexts']['efficiency'])
                ]
            },
            'individual_highlights': [
                {**player, 'context': ctx}
                for player, ctx in zip(individual_highlights, ai_creative['individual_contexts'])
            ],
            'power_rankings': {
                'last_updated': sagarin_cache.get('last_content_update', sagarin_cache.get('last_update')),
                'top_5': power_rankings['top_5'],
                'bottom_5': power_rankings['bottom_5'],
                'all_rankings': power_rankings['all_rankings'],
                'biggest_riser': {
                    **power_rankings['biggest_riser'],
                    'reason': ai_creative['biggest_riser_reason']
                } if power_rankings['biggest_riser'] else None,
                'biggest_faller': {
                    **power_rankings['biggest_faller'],
                    'reason': ai_creative['biggest_faller_reason']
                } if power_rankings['biggest_faller'] else None
            },
            'playoff_snapshot': {
                'top_4': [
                    {**team, 'key_info': info}
                    for team, info in zip(playoff_snapshot['top_4'], ai_creative['playoff_key_infos']['top_4'])
                ],
                'middle_4': [
                    {**team, 'key_info': info}
                    for team, info in zip(playoff_snapshot['middle_4'], ai_creative['playoff_key_infos']['middle_4'])
                ],
                'bottom_4': [
                    {**team, 'key_info': info}
                    for team, info in zip(playoff_snapshot['bottom_4'], ai_creative['playoff_key_infos']['bottom_4'])
                ]
            },
            'week_preview': {
                'thursday_night': {
                    **thursday_night,
                    'tagline': ai_creative['thursday_tagline']
                } if thursday_night else None,
                'sunday_spotlight': [
                    {**game, 'tagline': tagline}
                    for game, tagline in zip(sunday_spotlight, ai_creative['sunday_taglines'])
                ],
                'total_games': len(upcoming_games)
            }
        }

        # Save to file
        output_path = 'data/dashboard_content.json'
        with open(output_path, 'w') as f:
            json.dump(dashboard_content, f, indent=2)

        logger.info(f"Successfully generated {output_path}")
        return True

    except Exception as e:
        logger.error(f"Error generating dashboard content: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    # Set up logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    success = generate_dashboard_content()
    exit(0 if success else 1)
