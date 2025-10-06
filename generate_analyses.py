from datetime import date, datetime, timedelta
import json
import os
import time
import requests
import argparse
import pandas as pd
from anthropic import Anthropic
from dotenv import load_dotenv
import logging
import signal
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Don't add handlers here - let the parent logger handle it
# This prevents duplicate log messages
if not logger.handlers:
    logger.propagate = True

# Load environment variables from .env file
load_dotenv()

def fetch_game_json(game_id):
    """
    Fetch game JSON data from ESPN API for a given game ID.
    Returns the raw JSON data or None if request fails.
    """
    url = f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary?event={game_id}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise exception for bad status codes
        return response.json()
    except requests.RequestException as e:
        print(f"Error fetching game data: {e}")
        return None

def clean_article(article, game_team_ids):
    """
    Clean individual article and check if it should be kept.
    Only keeps articles specifically about teams playing in this game.
    
    Args:
        article: The article to clean
        game_team_ids: Set of team IDs for teams playing in this game
    """
    if not article.get('categories'):
        return None
        
    # Check if this article has any categories about teams in this game
    has_relevant_team = False
    for category in article['categories']:
        if (category.get('type') == 'team' and 
            category.get('sportId') == 28 and 
            category.get('teamId') in game_team_ids):
            has_relevant_team = True
            break
    
    # Only keep articles about teams in this game
    if has_relevant_team:
        return {
            'headline': article.get('headline'),
            'description': article.get('description'),
            'published': article.get('published'),
            'lastModified': article.get('lastModified'),
            'type': article.get('type'),
            'categories': [
                {
                    'type': cat.get('type'),
                    'teamId': cat.get('teamId'),
                    'description': cat.get('description')
                }
                for cat in article.get('categories', [])
                if cat.get('type') == 'team' and cat.get('teamId') in game_team_ids
            ]
        }
    return None

def clean_game_json(data):
    """
    Clean ESPN game JSON by removing unnecessary fields like links, logos, images, etc.
    Takes either a JSON dict or a file path as input.
    Returns cleaned JSON data.
    """
    # If input is a file path, load the JSON
    if isinstance(data, str):
        with open(data, 'r', encoding='utf-8') as f:
            data = json.load(f)

    # Get team IDs for this game from the boxscore
    game_team_ids = set()
    try:
        for team in data.get('boxscore', {}).get('teams', []):
            team_id = int(team.get('team', {}).get('id'))
            if team_id:
                game_team_ids.add(team_id)
    except (KeyError, ValueError, TypeError):
        logger.error("Failed to extract team IDs from game data")
        game_team_ids = set()

    def clean_string(s):
        """Clean string by removing extra whitespace"""
        if isinstance(s, str):
            return ' '.join(s.split())
        return s

    def clean_dict(d):
        # Fields to remove
        fields_to_remove = {
            # Visual elements
            'logo', 'logos', 'href', 'links', 'link', 'image', 'images', 'url', 
            'thumbnail', 'headshot', 'icon', 'uid', 'guid', 'alternateIds',
            'color', 'alternateColor', 'flag',
            
            # Content sections
            'videos', 'standings', 'odds', 'broadcasts',
            'predictor', 'gamecastAvailable', 'header', 'media', 'notes',
            'shop', 'tickets', 'deviceRestrictions', 'pbpInnings',
            'winprobability',
            
            # Metadata and References
            'tracking', 'analytics', 'meta', 'lang', 'site', 'sportBroadcasts',
            'preferences', 'type', 'uid', 'lastModified', 'premium',
            'gameSource', 'appLinks', '$ref'
        }
        
        if isinstance(d, dict):
            # Special handling for news articles
            if 'articles' in d:
                articles = d.get('articles', [])
                cleaned_articles = []
                for article in articles:
                    cleaned = clean_article(article, game_team_ids)
                    if cleaned:
                        cleaned_articles.append(cleaned)
                if cleaned_articles:
                    return {
                        k: clean_dict(v) if k != 'articles' else cleaned_articles
                        for k, v in d.items()
                        if k not in fields_to_remove and v is not None
                    }
            
            return {
                k: clean_dict(v)
                for k, v in d.items()
                if k not in fields_to_remove and v is not None
            }
        elif isinstance(d, list):
            return [clean_dict(item) for item in d if item is not None]
        else:
            return clean_string(d)

    return clean_dict(data)

def fetch_and_clean_game(game_id, save_to_file=False):
    """
    Fetch game data from ESPN API, clean it, and optionally save to file.
    Returns the cleaned JSON data.
    """
    # Fetch raw game data
    raw_data = fetch_game_json(game_id)
    if raw_data is None:
        return None
        
    # Clean the data
    cleaned_data = raw_data
    cleaned_data = clean_game_json(raw_data)
    
    # Optionally save to file
    if save_to_file:
        output_file = f'data/game_{game_id}_cleaned.json'
        raw_output_file = f'data/game_{game_id}_raw.json'
        os.makedirs('data', exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(
                cleaned_data,
                f,
                indent=None,
                separators=(',', ':'),
                ensure_ascii=False
            )
        with open(raw_output_file, 'w', encoding='utf-8') as f:
            json.dump(
                raw_data,
                f,
                indent=None,
                separators=(',', ':'),
                ensure_ascii=False
            )

        # Print file size comparison
        raw_size = len(json.dumps(raw_data))
        cleaned_size = os.path.getsize(output_file)
        reduction = (1 - cleaned_size/raw_size) * 100
        
        print(f"Cleaned JSON saved to: {output_file}")
        print(f"Original size: {raw_size:,} bytes")
        print(f"Cleaned size: {cleaned_size:,} bytes")
        print(f"Size reduction: {reduction:.1f}%")
    
    return cleaned_data

def send_to_claude(game_data, game_id, prompt_template=None, model="sonnet"):
    """
    Send game data to Claude API with an optional custom prompt template.
    Returns Claude's response.
    """
    from ai_service import AIService

    # Default prompt if none provided
    if prompt_template is None:
        prompt_template = """You are an award-winning NFL analyst and writer, renowned for combining razor-sharp statistical accuracy with entertaining, personality-filled writing. You have a unique talent for finding fascinating stories in the data and telling them with style. Think of yourself as a cross between John Madden's enthusiasm, Tony Romo's insight, and a stand-up comedian's wit. Don't hesitate to throw in some sarcasm or sharp jabs.

Generate a 1 or 2 paragraph summary of the game included in the data below. Start by reporting the final result, then discuss some of the more intersting things that happened, then talk about the impact and any impacts it might have on the teams involved and the league in general.

Follow up with a few important or surprising statistics that are relevant to the game.

IMPORTANT: The game data includes detailed statistics, play-by-play, and betting lines (if available). You can reference the pre-game betting spread to discuss how the game compared to expectations.

Do not follow up with any questions. Consider this a final draft that will be shared as-is with the public.

THIS IS THE 2025/2026 NFL SEASON

Game Data:
{game_json}
"""

    # Format the prompt with the game data
    try:
        prompt = prompt_template.format(game_json=json.dumps(game_data))

        system_prompt = "You are an NFL writer. Your expertise is in writing brief game analyses in multiple voices. You can be a brilliant and serious analyst as easily as a surrealist sports comic. Whether you are writing something serious or ridiculous, you are always accurate in your use of NFL references and statistics. Consider all of the data provided."

        ai_service = AIService()
        # Use generate_text for plain text response (not JSON)
        analysis, status = ai_service.generate_text(prompt, system_message=system_prompt)
        return analysis

    except Exception as e:
        logger.error(f"Error sending to Claude API: {e}")
        return None

def send_preview_to_claude(game_data, game_id, prompt_template=None, model="sonnet"):
    """
    Send game data to Claude API for generating a preview of an upcoming game.
    """
    from ai_service import AIService

    # Get this week's completed games
    try:
        schedule_df = pd.read_csv('data/schedule.csv')
        
        # Get the current game's info
        current_game = schedule_df[schedule_df['espn_id'] == int(game_id)].iloc[0]
        current_week = current_game['week_num']  # Using week_num from schedule.csv
        
        # Get completed games from same week
        this_week_games = schedule_df[
            (schedule_df['week_num'] == current_week) &  # Same week
            (schedule_df['espn_id'] != int(game_id)) &  # Not the current game
            (schedule_df['away_score'].notna()) &  # Has scores
            (schedule_df['home_score'].notna())
        ]
        
        # Format game results
        game_results = []
        for _, game in this_week_games.iterrows():
            result = f"{game['away_team']} {int(game['away_score'])} @ {game['home_team']} {int(game['home_score'])}"
            game_results.append(result)
            
        week_results = "\n".join(game_results)
        
    except Exception as e:
        logger.error(f"Error getting week's results: {e}")
        logger.error(f"Exception details: {str(e)}")
        week_results = ""

    # Default preview prompt
    if prompt_template is None:
        prompt_template = """You are an award-winning NFL analyst and writer, renowned for combining sharp statistical analysis with entertaining, personality-filled writing. You're previewing an upcoming NFL game.

Generate a 2-3 paragraph preview of the upcoming game described in the data below. Include:
1. Key storylines and matchups to watch
2. Recent performance of both teams
3. Key injuries or factors that could impact the game
4. A prediction for how the game might play out (but don't predict a specific score)

Make it engaging and entertaining while maintaining analytical accuracy. Feel free to use some personality and wit in your writing.

IMPORTANT: The game data includes multiple sections:
- **team_season_stats**: Comprehensive season statistics for both teams including efficiency metrics (3rd down %, red zone %), advanced stats (EPA, completion %), records (conference, division, road), and more
- **betting_lines**: Spread, over/under, and money lines
- **espn_predictor**: Win probability predictions (home_win_prob, away_win_prob, matchup_quality)
- **leaders**: Top players in key statistical categories
- **injuries**: Current injury reports
- **boxscore.teams[].statistics**: Basic per-game averages from ESPN

Use the team_season_stats section for detailed statistical analysis - it has the most comprehensive data.

Consider this a print-ready final draft, so do not include any pre-text like "Here is my preview of the game..." or follow up questions.

THIS IS THE 2024/2025 NFL SEASON

Game Data:
{game_json}
"""

    try:
        prompt = prompt_template.format(
            game_json=json.dumps(game_data),
            week_results=week_results
        )

        # save prompt to file with game id
        # with open(f'data/game_{game_id}_prompt.txt', 'w', encoding='utf-8') as f:
        #     f.write(prompt)

        system_prompt = "You are an NFL writer specializing in game previews and analysis. You combine statistical insight with entertaining writing, whether serious or playful. You're always accurate with NFL references and statistics. Respond with ONLY the preview text, no JSON, no additional formatting."

        ai_service = AIService()
        # Use generate_text for plain text response (not JSON)
        analysis, status = ai_service.generate_text(prompt, system_message=system_prompt)
        return analysis

    except Exception as e:
        logger.error(f"Error sending to Claude API: {e}")
        return None

def get_team_abbreviation(team_name):
    """Convert ESPN team name to standard NFL abbreviation"""
    nfl_abbreviations = {
        'Arizona Cardinals': 'ARI',
        'Atlanta Falcons': 'ATL',
        'Baltimore Ravens': 'BAL',
        'Buffalo Bills': 'BUF',
        'Carolina Panthers': 'CAR',
        'Chicago Bears': 'CHI',
        'Cincinnati Bengals': 'CIN',
        'Cleveland Browns': 'CLE',
        'Dallas Cowboys': 'DAL',
        'Denver Broncos': 'DEN',
        'Detroit Lions': 'DET',
        'Green Bay Packers': 'GB',
        'Houston Texans': 'HOU',
        'Indianapolis Colts': 'IND',
        'Jacksonville Jaguars': 'JAX',
        'Kansas City Chiefs': 'KC',
        'Las Vegas Raiders': 'LV',
        'Los Angeles Chargers': 'LAC',
        'Los Angeles Rams': 'LAR',
        'Miami Dolphins': 'MIA',
        'Minnesota Vikings': 'MIN',
        'New England Patriots': 'NE',
        'New Orleans Saints': 'NO',
        'New York Giants': 'NYG',
        'New York Jets': 'NYJ',
        'Philadelphia Eagles': 'PHI',
        'Pittsburgh Steelers': 'PIT',
        'San Francisco 49ers': 'SF',
        'Seattle Seahawks': 'SEA',
        'Tampa Bay Buccaneers': 'TB',
        'Tennessee Titans': 'TEN',
        'Washington Commanders': 'WAS'
    }
    return nfl_abbreviations.get(team_name, team_name)

def load_team_season_stats(away_team_abbr, home_team_abbr):
    """
    Load comprehensive season statistics for both teams from team_stats.csv.
    Returns dict with stats for both teams, filtered to most relevant metrics.
    """
    try:
        stats_df = pd.read_csv('data/team_stats.csv')

        # Fields to include for AI analysis
        stat_fields = [
            # Record & Context
            'wins', 'losses', 'ties', 'win_pct',
            'conf_record', 'div_record', 'road_record',
            'playoff_seed', 'clincher', 'streak_display',

            # Performance
            'points_per_game', 'points_against_per_game', 'point_diff',

            # Efficiency Metrics
            'third_down_pct', 'third_down_pct_against',
            'fourth_down_pct', 'fourth_down_pct_against',
            'red_zone_pct', 'red_zone_pct_against',

            # Passing
            'completion_pct', 'yards_per_attempt', 'passer_rating',
            'passing_yards', 'passing_tds', 'interceptions', 'sacks_taken',

            # Rushing
            'rushing_yards', 'rushing_tds', 'yards_per_carry',

            # Receiving
            'receiving_yards', 'receiving_tds', 'receiving_first_downs',
            'targets', 'receptions', 'receiving_yards_after_catch',

            # Defensive
            'def_sacks', 'def_tackles_solo', 'def_tackles_for_loss',
            'def_interceptions', 'def_fumbles_forced', 'def_pass_defended',
            'def_qb_hits', 'def_tds', 'def_safeties',

            # Fumbles & Penalties
            'fumble_recovery_own', 'fumble_recovery_opp', 'fumble_recovery_tds',
            'penalty_yards', 'penalties',

            # Field Goals & Kicking
            'fg_made', 'fg_att', 'fg_pct', 'fg_long',
            'pat_made', 'pat_att', 'pat_pct',

            # Special Teams
            'punt_returns', 'punt_return_yards',
            'kickoff_returns', 'kickoff_return_yards',
            'special_teams_tds',

            # Totals
            'total_yards', 'yards_per_game',
            'total_first_downs', 'first_downs_per_game',

            # Turnovers & Advanced
            'total_turnovers', 'turnover_margin',
            'total_epa', 'passing_epa', 'rushing_epa', 'receiving_epa'
        ]

        team_stats = {}
        for team_abbr in [away_team_abbr, home_team_abbr]:
            team_row = stats_df[stats_df['team_abbr'] == team_abbr]
            if not team_row.empty:
                team_stats[team_abbr] = {}
                for field in stat_fields:
                    if field in team_row.columns:
                        value = team_row.iloc[0][field]
                        # Convert to native Python types for JSON serialization
                        if pd.isna(value):
                            team_stats[team_abbr][field] = None
                        elif isinstance(value, (int, float)):
                            # Already native Python type
                            team_stats[team_abbr][field] = value
                        elif hasattr(value, 'item'):
                            # pandas/numpy numeric types - convert to native Python
                            team_stats[team_abbr][field] = value.item()
                        else:
                            # String or other types
                            team_stats[team_abbr][field] = str(value) if value is not None else None
            else:
                logger.warning(f"No stats found for team: {team_abbr}")
                team_stats[team_abbr] = None

        return team_stats

    except Exception as e:
        logger.error(f"Error loading team stats: {e}")
        return {}

def _process_single_game(game, analyses, force_reanalyze, current_date, week_from_now):
    """
    Worker function to process a single game analysis.
    Returns tuple: (game_id, analysis_dict, matchup_str) or (game_id, None, matchup_str) if failed/skipped.
    matchup_str is formatted as "AWAY@HOME" for display in progress bar.
    """
    game_id = str(game['espn_id'])
    game_date = datetime.strptime(game['game_date'], '%Y-%m-%d')
    matchup_str = f"{game['away_team']}@{game['home_team']}"

    # Determine if game is completed or upcoming
    is_completed = not (pd.isna(game['home_score']) or pd.isna(game['away_score']) or \
                      str(game['home_score']).strip() == '' or str(game['away_score']).strip() == '')

    is_upcoming = not is_completed and game_date >= current_date and game_date <= week_from_now

    # Check if we need to regenerate:
    # 1. If force_reanalyze is True
    # 2. If no analysis exists yet
    # 3. If existing analysis is a preview but game is now completed
    needs_analysis = (
        force_reanalyze or
        game_id not in analyses or
        (is_completed and analyses[game_id].get('analysis_type') == 'preview')
    )

    if not needs_analysis:
        return (game_id, None, matchup_str)

    if not (is_completed or is_upcoming):
        return (game_id, None, matchup_str)

    # Fetch and clean game data
    game_data = fetch_and_clean_game(game_id)
    if not game_data:
        return (game_id, None, matchup_str)

    # Fetch ESPN supplemental data BEFORE generating analysis
    espn_context = None
    predictor_data = None
    leaders_data = None
    broadcast_info = None

    try:
        from espn_api import ESPNAPIService
        espn_service = ESPNAPIService()

        # Get betting/weather (available for all games)
        espn_context = espn_service.get_game_context(game_id, game['home_team'], game['away_team'])

        # Get predictor data for upcoming games
        if not is_completed:
            predictor_data = espn_service.get_predictor_data(game_id)

        # Get leaders data for completed games
        if is_completed:
            leaders_data = espn_service.get_game_leaders(game_id)

        # Get broadcast info (available for all)
        broadcast_info = espn_service.get_broadcast_info(game_id)

    except Exception as e:
        logger.debug(f"Failed to fetch ESPN data for {game_id}: {str(e)}")

    # Add ESPN betting lines to game_data for AI context
    if espn_context and espn_context.get('betting'):
        game_data['betting_lines'] = espn_context.get('betting')

    # Add predictor data for previews only
    if not is_completed and predictor_data:
        game_data['espn_predictor'] = predictor_data

    # Add comprehensive team season stats from team_stats.csv (previews only)
    # Note: Only add for upcoming games to avoid anachronistic stats in historical game analysis
    if not is_completed:
        team_season_stats = load_team_season_stats(game['away_team'], game['home_team'])
        if team_season_stats:
            game_data['team_season_stats'] = team_season_stats

    # Generate analysis or preview with ESPN context
    # Get AI service info for metadata
    from ai_service import AIService
    ai_service = AIService()

    try:
        if is_completed:
            analysis = send_to_claude(game_data, game_id)
            analysis_type = "post_game"
        else:
            analysis = send_preview_to_claude(game_data, game_id)
            analysis_type = "preview"

        if not analysis:
            logger.error(f"Failed to generate {analysis_type} for {matchup_str} - AI returned empty response")
            return (game_id, None, matchup_str)
    except Exception as e:
        logger.error(f"Error generating {analysis_type if 'analysis_type' in locals() else 'analysis'} for {matchup_str}: {str(e)}")
        return (game_id, None, matchup_str)

    # Build analysis dict with metadata
    analysis_dict = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "game_date": game['game_date'],
        "home_team": game['home_team'],
        "away_team": game['away_team'],
        "home_score": int(game['home_score']) if is_completed else None,
        "away_score": int(game['away_score']) if is_completed else None,
        "stadium": game.get('stadium', ''),
        "home_coach": game.get('home_coach', ''),
        "away_coach": game.get('away_coach', ''),
        "analysis_type": analysis_type,
        "analysis": analysis,
        "ai_provider": ai_service.model_provider,
        "ai_model": ai_service.model
    }

    # Add ESPN context if available
    if espn_context:
        if espn_context.get('betting'):
            analysis_dict['betting'] = espn_context['betting']
        if espn_context.get('weather'):
            analysis_dict['weather'] = espn_context['weather']

    # Add predictor data
    if predictor_data:
        analysis_dict['predictor'] = predictor_data

    # Add leaders data
    if leaders_data:
        analysis_dict['leaders'] = leaders_data

    # Add broadcast info
    if broadcast_info:
        analysis_dict['broadcast'] = broadcast_info

    return (game_id, analysis_dict, matchup_str)


def batch_analyze_games(output_file='data/game_analyses.json', force_reanalyze=False, game_ids=None, regenerate_type=None):
    """
    Analyze completed games and preview upcoming games, saving to a JSON file.

    Args:
        output_file: Path to save analysis JSON
        force_reanalyze: Force regeneration (overrides existing analysis)
        game_ids: List of ESPN game IDs to regenerate (if specified, only these games are processed)
        regenerate_type: "analysis", "preview", or "all" to filter by analysis type
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm import tqdm

    logger.info("Starting batch_analyze_games function")
    analyses = {}
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                analyses = json.load(f)
        except json.JSONDecodeError:
            analyses = {}

    schedule_df = pd.read_csv('data/schedule.csv')
    games = schedule_df.to_dict('records')

    # Get current date without time part
    current_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    week_from_now = current_date + timedelta(days=7)

    # Filter games based on criteria
    games_to_process = []
    for game in games:
        game_id = str(game['espn_id'])
        game_date = datetime.strptime(game['game_date'], '%Y-%m-%d')

        # Determine if game is completed or upcoming
        is_completed = not (pd.isna(game['home_score']) or pd.isna(game['away_score']) or \
                          str(game['home_score']).strip() == '' or str(game['away_score']).strip() == '')

        is_upcoming = not is_completed and game_date >= current_date and game_date <= week_from_now

        # Filter by game IDs if specified
        if game_ids is not None and game_id not in game_ids:
            continue

        # Filter by regenerate type if specified
        if regenerate_type is not None:
            if regenerate_type == 'analysis' and not is_completed:
                continue
            elif regenerate_type == 'preview' and not is_upcoming:
                continue
            # 'all' means process everything (no filter)

        games_to_process.append(game)

    if not games_to_process:
        logger.info("No games to process")
        return analyses

    # Parallelize game analysis
    max_workers = int(os.environ.get('GAME_ANALYSIS_WORKERS', '3'))
    if max_workers < 1:
        max_workers = 1
    total_games = len(games_to_process)
    max_workers = min(max_workers, total_games)

    # Show AI provider/model info once before starting
    from ai_service import AIService
    ai_test = AIService()
    provider_info = f"Using AI provider: {ai_test.model_provider}, model: {ai_test.model}"
    logger.info(provider_info)
    logger.info(f"Processing {total_games} game(s) with up to {max_workers} worker(s)")

    # Temporarily increase log level to suppress AI service initialization logs
    ai_logger = logging.getLogger('ai_service')
    original_level = ai_logger.level
    ai_logger.setLevel(logging.WARNING)

    # Also suppress httpx and anthropic retry logs
    httpx_logger = logging.getLogger('httpx')
    httpx_original_level = httpx_logger.level
    httpx_logger.setLevel(logging.WARNING)

    anthropic_logger = logging.getLogger('anthropic')
    anthropic_original_level = anthropic_logger.level
    anthropic_logger.setLevel(logging.WARNING)

    executor = ThreadPoolExecutor(max_workers=max_workers)
    try:
        # Submit all game analysis tasks
        future_to_game = {
            executor.submit(_process_single_game, game, analyses, force_reanalyze, current_date, week_from_now): game
            for game in games_to_process
        }

        with tqdm(total=total_games, desc="Generating game analyses", unit="game") as pbar:
            for future in as_completed(future_to_game):
                try:
                    game_id, analysis_dict, matchup_str = future.result()

                    # Update progress bar with current game
                    pbar.set_postfix_str(matchup_str)

                    if analysis_dict is not None:
                        # Update analyses dict
                        analyses[game_id] = analysis_dict

                        # Save after each successful analysis
                        with open(output_file, 'w', encoding='utf-8') as f:
                            json.dump(analyses, f, indent=2, ensure_ascii=False)
                except Exception as e:
                    game = future_to_game[future]
                    matchup = f"{game.get('away_team', '?')}@{game.get('home_team', '?')}"
                    logger.error(f"Error processing game {matchup}: {str(e)}")
                finally:
                    pbar.update(1)

        logger.info(f"Batch analysis complete. Processed {total_games} game(s)")
    except KeyboardInterrupt:
        logger.warning("\n\nKeyboardInterrupt received! Cancelling remaining tasks...")
        # Cancel all pending futures
        for future in future_to_game:
            future.cancel()
        # Shutdown executor immediately without waiting for running tasks
        executor.shutdown(wait=False, cancel_futures=True)
        logger.info("Forced shutdown. Exiting immediately...")
        # Use os._exit to forcefully terminate all threads (including blocked API calls)
        os._exit(1)
    finally:
        # Restore original log levels
        ai_logger.setLevel(original_level)
        httpx_logger.setLevel(httpx_original_level)
        anthropic_logger.setLevel(anthropic_original_level)
        # Ensure executor is always cleaned up
        executor.shutdown(wait=True)

    return analyses

def main():
    parser = argparse.ArgumentParser(description='Fetch and clean ESPN NFL game data')
    parser.add_argument('game_id', nargs='?', default='401671814',
                      help='ESPN game ID (e.g., 401671814)')
    parser.add_argument('--analyze', action='store_true',
                      help='Send cleaned data to Claude for analysis')
    parser.add_argument('--save', action='store_true',
                      help='Save the cleaned game data to a file')
    parser.add_argument('--force-reanalyze', action='store_true',
                      help='Force reanalysis of games that already have analyses')
    parser.add_argument('--process-schedule', action='store_true',
                      help='Process the schedule.csv file and save analyses to game_analyses.json')
    
    args = parser.parse_args()

    if args.process_schedule:
        batch_analyze_games(force_reanalyze=args.force_reanalyze)
        return

    if args.game_id:
        game_data = fetch_and_clean_game(args.game_id, save_to_file=args.save)
    
    analysis = None
    
    if args.analyze and game_data:
        analysis = send_to_claude(game_data)
        if analysis:
            print("\nClaude's Analysis:")
            print(analysis)
        return
    elif args.save and analysis:
        with open(f'data/game_{args.game_id}_analysis.txt', 'w', encoding='utf-8') as f:
            f.write(analysis)   
    else:
        print(analysis)

if __name__ == "__main__":
    main()
