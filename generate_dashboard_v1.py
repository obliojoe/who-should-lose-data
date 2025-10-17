#!/usr/bin/env python3
"""
Generate dashboard_content.json using AI analysis.
This file is called as part of generate_cache.py pipeline.
"""

import json
import logging
import os
from datetime import datetime
import pandas as pd
from ai_service import AIService

# Set up logging
logger = logging.getLogger(__name__)

def get_current_week(schedule_df):
    """Determine the current NFL week based on schedule"""
    today = pd.Timestamp.now()

    # Find the earliest week with games that haven't been played yet (no scores)
    upcoming_games = schedule_df[
        (schedule_df['away_score'].isna()) | (schedule_df['home_score'].isna())
    ]

    if len(upcoming_games) > 0:
        return int(upcoming_games['week_num'].min())

    # If all games have scores, return the last week
    return int(schedule_df['week_num'].max())

def get_top_players_for_team(team_abbr, starters_df):
    """Get top 2-3 players for a team based on season stats"""
    team_players = starters_df[starters_df['team_abbr'] == team_abbr].copy()

    top_players = []

    # Get top QB (by passing yards)
    qbs = team_players[team_players['position'] == 'QB']
    if len(qbs) > 0:
        top_qb = qbs.nlargest(1, 'passing_yards_season')
        for _, player in top_qb.iterrows():
            if pd.notna(player['passing_yards_season']) and player['passing_yards_season'] > 0:
                stats = f"{int(player['passing_yards_season'])} pass yds, {int(player['passing_tds_season'])} TDs"
                if pd.notna(player['interceptions_season']):
                    stats += f", {int(player['interceptions_season'])} INTs"
                top_players.append({
                    'name': player['player_name'],
                    'position': 'QB',
                    'stats': stats
                })

    # Get top RB (by rushing yards)
    rbs = team_players[team_players['position'] == 'RB']
    if len(rbs) > 0:
        top_rb = rbs.nlargest(1, 'rushing_yards_season')
        for _, player in top_rb.iterrows():
            if pd.notna(player['rushing_yards_season']) and player['rushing_yards_season'] > 0:
                stats = f"{int(player['rushing_yards_season'])} rush yds, {int(player['rushing_tds_season'])} TDs"
                top_players.append({
                    'name': player['player_name'],
                    'position': 'RB',
                    'stats': stats
                })

    # Get top WR (by receiving yards)
    wrs = team_players[team_players['position'].isin(['WR', 'TE'])]
    if len(wrs) > 0:
        top_wr = wrs.nlargest(1, 'receiving_yards_season')
        for _, player in top_wr.iterrows():
            if pd.notna(player['receiving_yards_season']) and player['receiving_yards_season'] > 0:
                stats = f"{int(player['receiving_yards_season'])} rec yds, {int(player['receiving_tds_season'])} TDs"
                top_players.append({
                    'name': player['player_name'],
                    'position': player['position'],
                    'stats': stats
                })

    return top_players[:3]  # Limit to top 3

def prepare_dashboard_data():
    """
    Load and prepare all data needed for dashboard generation.
    Returns a dict formatted for the AI prompt.
    """
    logger.info("Preparing dashboard data...")

    # Load all required data files
    with open('data/schedule.json', 'r', encoding='utf-8') as fh:
        schedule_df = pd.DataFrame(json.load(fh))
    with open('data/team_stats.json', 'r', encoding='utf-8') as fh:
        team_stats_df = pd.DataFrame(json.load(fh))
    sagarin_df = pd.read_csv('data/sagarin.csv')
    with open('data/team_starters.json', 'r', encoding='utf-8') as fh:
        starters_df = pd.DataFrame(json.load(fh))
    with open('data/teams.json', 'r', encoding='utf-8') as fh:
        teams_df = pd.DataFrame(json.load(fh))

    # Create division lookup
    divisions = dict(zip(teams_df['team_abbr'], teams_df['division']))

    with open('data/analysis_cache.json', 'r') as f:
        analysis_cache = json.load(f)

    with open('data/standings_cache.json', 'r') as f:
        standings_cache = json.load(f)

    with open('data/game_analyses.json', 'r') as f:
        game_analyses = json.load(f)

    # Build a flat lookup of team records from standings_cache
    team_records = {}
    for conf_name, conf_standings in standings_cache['standings']['conference'].items():
        for team_data in conf_standings:
            team_abbr = team_data['team']
            team_records[team_abbr] = {
                'wins': team_data['wins'],
                'losses': team_data['losses'],
                'ties': team_data['ties']
            }

    # Build a lookup of current playoff seeds
    playoff_seeds = {}
    for conf_name, conf_playoffs in standings_cache['standings']['playoff'].items():
        # Division winners (seeds 1-4)
        for team_data in conf_playoffs.get('division_winners', []):
            playoff_seeds[team_data['team']] = team_data['seed']
        # Wild cards (seeds 5-7)
        for team_data in conf_playoffs.get('wild_cards', []):
            playoff_seeds[team_data['team']] = team_data['seed']

    # Determine current week
    current_week = get_current_week(schedule_df)
    logger.info(f"Current week: {current_week}")

    # Build teams data
    teams_data = {}
    for team_abbr in sagarin_df['team_abbr']:
        # Get Sagarin data
        sag_data = sagarin_df[sagarin_df['team_abbr'] == team_abbr].iloc[0]
        current_rank = int(sagarin_df[sagarin_df['team_abbr'] == team_abbr].index[0]) + 1

        # Get standings data
        standings = team_records.get(team_abbr, {'wins': 0, 'losses': 0, 'ties': 0})
        record = f"{standings.get('wins', 0)}-{standings.get('losses', 0)}"
        if standings.get('ties', 0) > 0:
            record += f"-{standings.get('ties', 0)}"

        # Get playoff probabilities
        team_analysis = analysis_cache['team_analyses'].get(team_abbr, {})

        # Get current playoff seed
        current_seed = playoff_seeds.get(team_abbr)

        # Get team stats (aggregate across weeks)
        team_season_stats = team_stats_df[team_stats_df['team_abbr'] == team_abbr]
        if len(team_season_stats) > 0:
            stats = {
                'points_per_game': round(team_season_stats['points_for'].sum() / len(team_season_stats), 1),
                'points_allowed_per_game': round(team_season_stats['points_against'].sum() / len(team_season_stats), 1),
                'total_yards_per_game': round(team_season_stats['total_yards'].mean(), 1),
                'rushing_yards_per_game': round(team_season_stats['rushing_yards'].mean(), 1),
                'passing_yards_per_game': round(team_season_stats['passing_yards'].mean(), 1),
                'completion_pct': round(team_season_stats['completion_pct'].mean(), 1),
                'third_down_pct': round(team_season_stats['third_down_pct'].mean(), 1),
                'red_zone_pct': round(team_season_stats['red_zone_pct'].mean(), 1),
                'turnover_margin': int(team_season_stats['turnover_margin'].sum()),
            }
        else:
            stats = {}

        # Get top players
        top_players = get_top_players_for_team(team_abbr, starters_df)

        teams_data[team_abbr] = {
            'record': record,
            'playoff_probability': round(team_analysis.get('playoff_chance', 0), 1),
            'division_probability': round(team_analysis.get('division_chance', 0), 1),
            'current_seed': current_seed,
            'sagarin_rating': round(sag_data['rating'], 2),
            'sagarin_rank': current_rank,
            'stats': stats,
            'top_players': top_players
        }

    # Get upcoming games (current week)
    upcoming_games = schedule_df[schedule_df['week_num'] == current_week].to_dict('records')
    upcoming_games_data = []
    for game in upcoming_games:
        # Add betting line and preview if available
        game_analysis = game_analyses.get(str(game['espn_id']), {})
        betting = game_analysis.get('betting', {})

        # Get preview from analysis field if this is a preview-type analysis
        preview = ''
        if game_analysis.get('analysis_type') == 'preview':
            preview = game_analysis.get('analysis', '')

        # Check if divisional game
        away_div = divisions.get(game['away_team'])
        home_div = divisions.get(game['home_team'])
        is_divisional = (away_div == home_div and away_div is not None)

        # Check if Thursday game (simple check based on day of week)
        is_thursday = False
        try:
            game_datetime = pd.to_datetime(game['game_date'])
            is_thursday = (game_datetime.dayofweek == 3)  # 3 = Thursday
        except:
            pass

        upcoming_games_data.append({
            'espn_id': str(game['espn_id']),
            'week': int(game['week_num']),
            'away_team': game['away_team'],
            'home_team': game['home_team'],
            'game_date': game['game_date'],
            'is_divisional': is_divisional,
            'is_thursday': is_thursday,
            'betting_spread': betting.get('spread'),
            'betting_over_under': betting.get('over_under'),
            'betting_favorite': betting.get('favorite'),
            'preview': preview
        })

    # Get recent results (last week)
    last_week = current_week - 1
    recent_results = schedule_df[
        (schedule_df['week_num'] == last_week) &
        (schedule_df['away_score'].notna()) &
        (schedule_df['home_score'].notna())
    ].to_dict('records')

    recent_results_data = []
    for game in recent_results:
        recent_results_data.append({
            'espn_id': str(game['espn_id']),
            'week': int(game['week_num']),
            'away_team': game['away_team'],
            'away_score': int(game['away_score']),
            'home_team': game['home_team'],
            'home_score': int(game['home_score'])
        })

    # Get power rankings previous week (from sagarin previous_rank if available)
    power_rankings_previous = {}
    if 'previous_rank' in sagarin_df.columns:
        for _, row in sagarin_df.iterrows():
            if pd.notna(row['previous_rank']):
                power_rankings_previous[row['team_abbr']] = int(row['previous_rank'])

    return {
        'current_week': current_week,
        'timestamp': datetime.now().isoformat(),
        'teams': teams_data,
        'upcoming_games': upcoming_games_data,
        'recent_results': recent_results_data,
        'power_rankings_previous_week': power_rankings_previous
    }

def generate_dashboard_content(ai_model=None):
    """
    Generate dashboard_content.json using AI analysis.

    Args:
        ai_model: Optional AI model override

    Returns:
        bool: Success status
    """
    try:
        # Prepare data
        dashboard_data = prepare_dashboard_data()

        # Load the prompt template
        with open('_design/dashboard_generation_prompt.md', 'r') as f:
            prompt_template = f.read()

        # Build the full prompt with inline data (minified JSON to save tokens)
        prompt = f"""{prompt_template}

---
DATA FOR WEEK {dashboard_data['current_week']}:
{json.dumps(dashboard_data, separators=(',', ':'))}
---

Generate the dashboard_content.json based on the above data and instructions.
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

        logger.info(f"Generating dashboard content with AI (model: {ai_service.model})...")

        # Save the prompt to data/prompts for debugging/review
        import os
        os.makedirs('data/prompts', exist_ok=True)
        prompt_path = 'data/prompts/dashboard_generation.txt'
        with open(prompt_path, 'w') as f:
            f.write(prompt)

        # Generate the dashboard content
        system_message = "You are an expert NFL analyst generating structured JSON data for a dashboard. Follow the instructions precisely and return valid JSON only."

        response, status = ai_service.generate_analysis(prompt, system_message=system_message)

        if status != 'success':
            logger.error(f"Failed to generate dashboard content: {status}")
            return False

        # Parse and validate the response
        try:
            dashboard_content = json.loads(response)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse AI response as JSON: {e}")
            logger.error(f"Response: {response[:500]}...")
            return False

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
