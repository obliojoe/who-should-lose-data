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
    # team_stats.csv already has season totals (week 15 is aggregated row)
    # Just select the columns we need - no need to aggregate
    season_stats = team_stats_df[['team_abbr', 'points_per_game', 'points_against_per_game',
                                   'yards_per_game', 'passing_yards', 'rushing_yards',
                                   'completion_pct', 'third_down_pct', 'red_zone_pct',
                                   'turnover_margin', 'def_sacks', 'def_interceptions',
                                   'games_played', 'points_for', 'points_against']].copy()

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
        ('red_zone_pct', 'Red Zone Efficiency', '%'),
        ('turnover_margin', 'Turnover Margin', '')
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


def select_power_rankings(sagarin_df, team_records, teams_df):
    """
    Select power rankings using Sagarin ratings (already calculated).
    Returns top 5, bottom 5, biggest riser, biggest faller.
    """
    # Build team name lookup
    team_names = {}
    for _, team in teams_df.iterrows():
        team_names[team['team_abbr']] = f"{team['city']} {team['mascot']}"

    # Top 5
    top_5 = []
    for idx, row in sagarin_df.head(5).iterrows():
        team_abbr = row['team_abbr']
        record_data = team_records.get(team_abbr, {'wins': 0, 'losses': 0, 'ties': 0})
        record = f"{record_data['wins']}-{record_data['losses']}"
        if record_data['ties'] > 0:
            record += f"-{record_data['ties']}"

        # Determine trend
        trend = 'steady'
        if 'previous_rank' in row and pd.notna(row['previous_rank']):
            current_rank = idx + 1
            prev_rank = int(row['previous_rank'])
            if current_rank < prev_rank:
                trend = 'up'
            elif current_rank > prev_rank:
                trend = 'down'

        top_5.append({
            'rank': idx + 1,
            'team': team_abbr,
            'team_name': team_names.get(team_abbr, team_abbr),
            'rating': round(float(row['rating']), 2),
            'record': record,
            'trend': trend
        })

    # Bottom 5
    bottom_5 = []
    for idx, row in sagarin_df.tail(5).iterrows():
        team_abbr = row['team_abbr']
        record_data = team_records.get(team_abbr, {'wins': 0, 'losses': 0, 'ties': 0})
        record = f"{record_data['wins']}-{record_data['losses']}"
        if record_data['ties'] > 0:
            record += f"-{record_data['ties']}"

        # Determine trend
        trend = 'steady'
        if 'previous_rank' in row and pd.notna(row['previous_rank']):
            current_rank = idx + 1
            prev_rank = int(row['previous_rank'])
            if current_rank < prev_rank:
                trend = 'up'
            elif current_rank > prev_rank:
                trend = 'down'

        bottom_5.append({
            'rank': idx + 1,
            'team': team_abbr,
            'team_name': team_names.get(team_abbr, team_abbr),
            'rating': round(float(row['rating']), 2),
            'record': record,
            'trend': trend
        })

    # Find biggest riser and faller
    biggest_riser = None
    biggest_faller = None

    if 'previous_rank' in sagarin_df.columns:
        sagarin_df['rank'] = range(1, len(sagarin_df) + 1)
        sagarin_df['movement'] = sagarin_df.apply(
            lambda row: int(row['previous_rank']) - row['rank'] if pd.notna(row['previous_rank']) else 0,
            axis=1
        )

        # Biggest riser (positive movement)
        if sagarin_df['movement'].max() > 0:
            riser_row = sagarin_df.loc[sagarin_df['movement'].idxmax()]
            record_data = team_records.get(riser_row['team_abbr'], {'wins': 0, 'losses': 0, 'ties': 0})
            record = f"{record_data['wins']}-{record_data['losses']}"
            if record_data['ties'] > 0:
                record += f"-{record_data['ties']}"

            biggest_riser = {
                'team': riser_row['team_abbr'],
                'previous_rank': int(riser_row['previous_rank']),
                'current_rank': int(riser_row['rank']),
                'movement': f"+{int(riser_row['movement'])}",
                'rating': round(float(riser_row['rating']), 2),
                'record': record
            }

        # Biggest faller (negative movement)
        if sagarin_df['movement'].min() < 0:
            faller_row = sagarin_df.loc[sagarin_df['movement'].idxmin()]
            record_data = team_records.get(faller_row['team_abbr'], {'wins': 0, 'losses': 0, 'ties': 0})
            record = f"{record_data['wins']}-{record_data['losses']}"
            if record_data['ties'] > 0:
                record += f"-{record_data['ties']}"

            biggest_faller = {
                'team': faller_row['team_abbr'],
                'previous_rank': int(faller_row['previous_rank']),
                'current_rank': int(faller_row['rank']),
                'movement': f"{int(riser_row['movement'])}",  # Will be negative
                'rating': round(float(faller_row['rating']), 2),
                'record': record
            }

    return {
        'top_5': top_5,
        'bottom_5': bottom_5,
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

        if game['is_thursday']:
            score += 10  # Bonus for primetime

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

        game_scores.append({
            'espn_id': str(game['espn_id']),
            'away_team': away_team,
            'away_record': away_record,
            'home_team': home_team,
            'home_record': home_record,
            'betting_line': betting.get('spread'),
            'over_under': betting.get('over_under'),
            'score': score,
            'inverse_score': -score  # For finding worst game
        })

    # Best game (highest score)
    game_of_week = max(game_scores, key=lambda x: x['score']).copy()
    del game_of_week['score']
    del game_of_week['inverse_score']

    # Worst game (lowest score)
    game_of_meek = min(game_scores, key=lambda x: x['score']).copy()
    del game_of_meek['score']
    del game_of_meek['inverse_score']

    return game_of_week, game_of_meek


def select_week_preview(upcoming_games, game_of_week_id, team_records, sagarin_df, analysis_cache):
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

        game_data = {
            'espn_id': str(game['espn_id']),
            'away_team': game['away_team'],
            'away_record': away_record,
            'home_team': game['home_team'],
            'home_record': home_record
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
        schedule_df = pd.read_csv('data/schedule.csv')
        team_stats_df = pd.read_csv('data/team_stats.csv')
        sagarin_df = pd.read_csv('data/sagarin.csv')
        starters_df = pd.read_csv('data/team_starters.csv')
        teams_df = pd.read_csv('data/teams.csv')

        with open('data/analysis_cache.json', 'r') as f:
            analysis_cache = json.load(f)

        with open('data/standings_cache.json', 'r') as f:
            standings_cache = json.load(f)

        with open('data/game_analyses.json', 'r') as f:
            game_analyses = json.load(f)

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

        # Build playoff seeds lookup
        playoff_seeds = {}
        for conf_name, conf_playoffs in standings_cache['standings']['playoff'].items():
            for team_data in conf_playoffs.get('division_winners', []):
                playoff_seeds[team_data['team']] = team_data['seed']
            for team_data in conf_playoffs.get('wild_cards', []):
                playoff_seeds[team_data['team']] = team_data['seed']

        # Get current week
        current_week = get_current_week(schedule_df)
        logger.info(f"Current week: {current_week}")

        # Get upcoming games
        upcoming_games = schedule_df[schedule_df['week_num'] == current_week].to_dict('records')

        # Add flags to games
        divisions = dict(zip(teams_df['team_abbr'], teams_df['division']))
        for game in upcoming_games:
            away_div = divisions.get(game['away_team'])
            home_div = divisions.get(game['home_team'])
            game['is_divisional'] = (away_div == home_div and away_div is not None)

            try:
                game_datetime = pd.to_datetime(game['game_date'])
                game['is_thursday'] = (game_datetime.dayofweek == 3)
            except:
                game['is_thursday'] = False

        # CODE-BASED DATA SELECTION
        logger.info("Selecting data using code logic...")

        stat_leaders = select_stat_leaders(team_stats_df, sagarin_df)
        individual_highlights = select_individual_highlights(starters_df)
        power_rankings = select_power_rankings(sagarin_df, team_records, teams_df)
        playoff_snapshot = select_playoff_snapshot(analysis_cache, team_records, playoff_seeds)
        game_of_week, game_of_meek = select_game_of_week_and_meek(
            upcoming_games, team_records, sagarin_df, analysis_cache, game_analyses
        )
        thursday_night, sunday_spotlight = select_week_preview(
            upcoming_games, game_of_week['espn_id'], team_records, sagarin_df, analysis_cache
        )

        # AI-BASED CREATIVE TEXT GENERATION
        logger.info("Generating creative text with AI...")

        # Load the prompt template for better context
        with open('_design/dashboard_generation_prompt.md', 'r') as f:
            prompt_template = f.read()

        # Build prompt with template context
        ai_prompt = f"""You are generating creative text for an NFL dashboard. Generate ONLY the requested text fields in JSON format.

Current Week: {current_week}

IMPORTANT: For the league_pulse.summary field, use markdown formatting (bold for team names, italics for emphasis) to make it more engaging.

Generate:
1. league_pulse.summary - 3-6 engaging sentences with markdown formatting about the league's current state
2. league_pulse.key_storylines - Array of 3 storylines, each with:
   - title: Catchy 3-6 word headline
   - description: 2-3 sentences with stats
3. game_of_the_week.tagline - 1-2 sentences why {game_of_week['away_team']} @ {game_of_week['home_team']} is THE game to watch
4. game_of_the_meek.tagline - 1-2 humorous sentences why {game_of_meek['away_team']} @ {game_of_meek['home_team']} is skippable
5. stat_leader_contexts - For each stat, provide a brief 2-6 word context phrase (PLAIN STRING, not object):
{json.dumps(stat_leaders, indent=2)}
   Return format: {{"offense": ["string1", "string2", ...], "defense": ["string1", ...], "efficiency": ["string1", ...]}}

6. individual_contexts - For each player, provide context explaining why their performance matters (PLAIN STRING, not object):
{json.dumps(individual_highlights, indent=2)}
   Return format: ["string1", "string2", "string3", ...]

7. power_rankings reasons - Provide 1-sentence reasons for biggest_riser and biggest_faller:
   Riser: {power_rankings.get('biggest_riser', {}).get('team', 'N/A')}
   Faller: {power_rankings.get('biggest_faller', {}).get('team', 'N/A')}

8. playoff_key_infos - For each of the 12 teams, provide brief context (PLAIN STRING, not object):
{json.dumps(playoff_snapshot, indent=2)}
   Return format: {{"top_4": ["string1", "string2", ...], "middle_4": ["string1", ...], "bottom_4": ["string1", ...]}}

9. week_preview taglines - Brief 5-8 word taglines for Thursday night and {len(sunday_spotlight)} Sunday games

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

        # COMBINE CODE-SELECTED DATA WITH AI-GENERATED TEXT
        logger.info("Combining code-selected data with AI-generated text...")

        dashboard_content = {
            'timestamp': datetime.now().isoformat(),
            'week': current_week,
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
                'top_5': power_rankings['top_5'],
                'bottom_5': power_rankings['bottom_5'],
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
