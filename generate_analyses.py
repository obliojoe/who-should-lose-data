from datetime import date, datetime, timedelta
import json
import os
import time
import argparse
import logging
import signal
import sys
from pathlib import Path
from typing import Dict, List, Optional
from functools import lru_cache

import pandas as pd
from anthropic import Anthropic
from dotenv import load_dotenv

from raw_data_manifest import RawDataManifest
from playoff_utils import load_teams
from team_metadata import TEAM_METADATA

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Don't add handlers here - let the parent logger handle it
# This prevents duplicate log messages
if not logger.handlers:
    logger.propagate = True

# Load environment variables from .env file
load_dotenv()

RAW_DATA_MANIFEST: Optional[RawDataManifest] = None
try:
    TEAMS_METADATA = load_teams()
except Exception:
    TEAMS_METADATA = {team['team_abbr']: team for team in TEAM_METADATA}


def _truncate_text(value: Optional[str], max_length: int = 180) -> Optional[str]:
    """Return a trimmed ASCII-safe string with ellipsis when needed."""
    if not value:
        return None

    text = str(value).strip()
    if len(text) <= max_length:
        return text

    return text[: max_length - 3].rstrip() + "..."


def _get_espn_team_id(team_abbr: str) -> Optional[int]:
    profile = TEAMS_METADATA.get(team_abbr)
    if not profile:
        return None
    espn_id = profile.get('espn_api_id')
    if espn_id is None:
        return None
    try:
        return int(espn_id)
    except (TypeError, ValueError):
        return None


def _normalize_name(value: Optional[str]) -> str:
    if not value or not isinstance(value, str):
        return ''
    return ''.join(value.lower().replace("'", '').replace('-', ' ').split())


def _load_schedule_dataframe() -> Optional[pd.DataFrame]:
    try:
        with open('data/schedule.json', 'r', encoding='utf-8') as fh:
            schedule_df = pd.DataFrame(json.load(fh))
        schedule_df['home_score'] = pd.to_numeric(schedule_df['home_score'], errors='coerce')
        schedule_df['away_score'] = pd.to_numeric(schedule_df['away_score'], errors='coerce')
        schedule_df['game_datetime'] = pd.to_datetime(
            schedule_df['game_date'] + ' ' + schedule_df['gametime'].fillna('00:00'),
            errors='coerce'
        )
        return schedule_df
    except Exception:
        return None


def _load_team_recent_form(team_abbr: str, limit: int = 5) -> List[dict]:
    schedule_df = _load_schedule_dataframe()
    if schedule_df is None:
        return []

    team_games = schedule_df[
        (schedule_df['home_team'] == team_abbr) | (schedule_df['away_team'] == team_abbr)
    ]
    team_games = team_games[team_games['home_score'].notna() & team_games['away_score'].notna()]
    if team_games.empty:
        return []

    team_games = team_games.sort_values('game_datetime').tail(limit)
    recent = []
    for _, row in team_games.iterrows():
        is_home = row['home_team'] == team_abbr
        opponent = row['away_team'] if is_home else row['home_team']
        home_score = int(row['home_score'])
        away_score = int(row['away_score'])
        result = 'W' if (is_home and home_score > away_score) or (not is_home and away_score > home_score) else 'L'
        if home_score == away_score:
            result = 'T'

        recent.append({
            'week': int(row.get('week_num', 0) or 0),
            'date': row.get('game_date'),
            'opponent': opponent,
            'location': 'home' if is_home else 'away',
            'score': f"{row['home_team']} {home_score}-{away_score} {row['away_team']}",
            'result': result
        })

    return recent


def _load_team_injury_notes(team_abbr: str, limit: int = 6) -> List[dict]:
    if RAW_DATA_MANIFEST is None:
        return []
    espn_id = _get_espn_team_id(team_abbr)
    if espn_id is None:
        return []

    try:
        data = RAW_DATA_MANIFEST.load_json('espn_injuries', str(espn_id))
    except ValueError:
        data = None

    if not data:
        return []

    items = []
    for item in data.get('items', []) or []:
        athlete = item.get('athlete', {}) or {}
        detail = item.get('details', {}) or {}
        status = item.get('status') or detail.get('description') or detail.get('detail')

        short_comment = _truncate_text(item.get('shortComment'), max_length=160)
        long_comment = _truncate_text(item.get('longComment'), max_length=220)

        details = {
            'player': athlete.get('fullName') or athlete.get('displayName'),
            'position': athlete.get('position'),
            'status': status,
            'injury': detail.get('type') or detail.get('detail'),
            'last_updated': item.get('date')
        }

        if short_comment:
            details['short_comment'] = short_comment
        if long_comment:
            details['long_comment'] = long_comment

        items.append(details)

    return items[:limit]


def _load_team_headlines(team_abbr: str, limit: int = 2) -> List[dict]:
    if RAW_DATA_MANIFEST is None:
        return []
    espn_id = _get_espn_team_id(team_abbr)
    if espn_id is None:
        return []

    try:
        data = RAW_DATA_MANIFEST.load_json('espn_news', str(espn_id))
    except ValueError:
        data = None

    if not data:
        return []

    headlines = []
    for article in data.get('articles', []) or []:
        headline = _truncate_text(article.get('headline'), max_length=160)
        description = _truncate_text(article.get('description'), max_length=200)
        if not headline:
            continue

        entry = {
            'headline': headline,
            'published': article.get('published')
        }
        if description:
            entry['description'] = description

        headlines.append(entry)
        if len(headlines) >= limit:
            break

    return headlines


def _load_depth_chart_flags(team_abbr: str, limit: int = 3) -> List[dict]:
    if RAW_DATA_MANIFEST is None:
        return []
    espn_id = _get_espn_team_id(team_abbr)
    if espn_id is None:
        return []

    try:
        data = RAW_DATA_MANIFEST.load_json('espn_depthchart', str(espn_id))
    except ValueError:
        data = None

    if not data:
        return []

    notable_statuses = {
        'Out', 'Questionable', 'Doubtful', 'Injured Reserve', 'Suspended', 'Physically Unable to Perform',
        'Non-Football Injury', 'Practice Squad', 'Limited', 'Game-Time Decision', 'Not Active'
    }

    alerts = []
    for grouping in data.get('items', []) or []:
        positions = grouping.get('positions', {}) or {}
        for pos_key, pos_info in positions.items():
            for athlete in pos_info.get('athletes', []) or []:
                status_info = athlete.get('status', {}) or {}
                status_name = status_info.get('name') or status_info.get('abbreviation') or status_info.get('type')
                if status_name and status_name not in notable_statuses:
                    continue
                if status_name:
                    alerts.append({
                        'position': pos_info.get('position', {}).get('displayName') or pos_key.upper(),
                        'player': athlete.get('fullName') or athlete.get('displayName'),
                        'jersey': athlete.get('jersey'),
                        'status': status_name
                    })
    return alerts[:limit]


def _load_scoreboard_context(event_id: str) -> Dict:
    if RAW_DATA_MANIFEST is None:
        return {}
    identifiers = RAW_DATA_MANIFEST.list_identifiers('espn_scoreboard')
    if not identifiers:
        return {}

    try:
        scoreboard = RAW_DATA_MANIFEST.load_json('espn_scoreboard', identifiers[-1])
    except ValueError:
        scoreboard = None

    if not scoreboard:
        return {}

    for event in scoreboard.get('events', []) or []:
        if str(event.get('id')) != str(event_id):
            continue
        competition = (event.get('competitions') or [None])[0] or {}
        context: Dict = {}
        notes = []
        for note in competition.get('notes', []) or []:
            text = note.get('headline') or note.get('text')
            text = _truncate_text(text, max_length=200)
            if not text:
                continue
            notes.append(text)
            if len(notes) >= 3:
                break
        if notes:
            context['notes'] = notes
        if competition.get('attendance') is not None:
            context['attendance'] = competition.get('attendance')
        context['neutral_site'] = competition.get('neutralSite', False)
        if competition.get('broadcasts'):
            context['broadcasts'] = [
                ', '.join(b.get('names') or []) or b.get('shortName') or b.get('type', {}).get('shortName')
                for b in competition.get('broadcasts', [])
            ]
        venue = competition.get('venue') or {}
        if venue.get('fullName'):
            context['venue'] = venue.get('fullName')
        if venue.get('address'):
            context['venue_city'] = venue.get('address', {}).get('city')
        status = competition.get('status', {}).get('type', {}) or {}
        if status.get('detail'):
            context['status_detail'] = status.get('detail')
        weather = competition.get('weather') or {}
        if weather:
            context.setdefault('weather', {})
            context['weather'].update({
                'temperature': weather.get('temperature'),
                'condition': weather.get('displayValue') or weather.get('condition'),
                'wind': weather.get('wind', {}).get('displayValue'),
                'humidity': weather.get('humidity')
            })
        return context
    return {}


@lru_cache(maxsize=1)
def _load_schedule_index() -> Dict[str, Dict[str, int]]:
    try:
        with open('data/schedule.json', 'r', encoding='utf-8') as fh:
            schedule = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}

    index: Dict[str, Dict[str, int]] = {}
    for game in schedule or []:
        game_id = str(game.get('espn_id', '')).strip()
        if not game_id:
            continue
        week = game.get('week_num')
        game_date = game.get('game_date')
        try:
            season = int(str(game_date)[:4]) if game_date else None
        except ValueError:
            season = None
        index[game_id] = {
            'week': int(week) if week not in (None, '') else None,
            'season': season,
        }
    return index


def fetch_game_json(game_id):
    """Load stored ESPN summary JSON for a given game ID."""
    manifest = RAW_DATA_MANIFEST or RawDataManifest.from_latest()
    if manifest:
        payload = manifest.load_json('espn_summary', str(game_id))
        if payload is not None:
            return payload

    # Fallback: search raw snapshot directories directly (supports older weeks no longer in manifest)
    base_dir = Path('data/raw/espn/games')
    if not base_dir.exists():
        logger.warning("Raw ESPN directory missing when loading game %s", game_id)
        return None

    schedule_index = _load_schedule_index()
    info = schedule_index.get(str(game_id))
    matches = []
    if info and info.get('season') and info.get('week'):
        candidate = base_dir / f"season_{info['season']}" / f"week_{info['week']}" / str(game_id) / 'summary.json'
        if candidate.exists():
            matches = [candidate]
    if not matches:
        matches = sorted(base_dir.glob(f'season_*/week_*/{game_id}/summary.json'))
    if not matches:
        return None

    summary_path = matches[-1]
    try:
        with summary_path.open('r', encoding='utf-8') as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        logger.error("Failed to read fallback summary for game %s at %s: %s", game_id, summary_path, exc)
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

def extract_key_plays(data):
    """
    Extract key plays (turnovers, sacks, big plays 20+ yards) from drives data.
    This gives us important plays without including every single play.
    """
    key_plays = []

    if 'drives' not in data or 'previous' not in data['drives']:
        return key_plays

    for drive in data['drives']['previous']:
        if 'plays' not in drive:
            continue

        for play in drive['plays']:
            is_key_play = False
            play_tags = []

            # Check if it's a turnover (interception or fumble)
            play_text = play.get('text', '').lower()
            if 'intercept' in play_text or 'fumble' in play_text:
                is_key_play = True
                if 'intercept' in play_text:
                    play_tags.append('INTERCEPTION')
                if 'fumble' in play_text:
                    play_tags.append('FUMBLE')

            # Check if it's a sack
            if 'sack' in play_text:
                is_key_play = True
                play_tags.append('SACK')

            # Check for big plays (20+ yards)
            stat_yardage = play.get('statYardage', 0)
            if stat_yardage >= 20:
                is_key_play = True
                play_tags.append(f'BIG PLAY ({stat_yardage} yards)')

            # Add key plays to the list
            if is_key_play:
                key_plays.append({
                    'text': play.get('text'),
                    'tags': play_tags,
                    'period': play.get('period', {}).get('number'),
                    'clock': play.get('clock', {}).get('displayValue'),
                    'scoringPlay': play.get('scoringPlay', False),
                    'yards': stat_yardage
                })

    return key_plays

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

    # Extract key plays BEFORE we remove the drives data
    key_plays = extract_key_plays(data)

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

            # Play-by-play data (too verbose, we keep scoringPlays instead)
            'drives',

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

    cleaned_data = clean_dict(data)

    # Add the extracted key plays to the cleaned data
    if key_plays:
        cleaned_data['keyPlays'] = key_plays

    return cleaned_data

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

IMPORTANT: The game data includes detailed statistics, scoring plays, key plays, and betting lines (if available). You can reference the pre-game betting spread to discuss how the game compared to expectations.
- 'scoringPlays': All touchdowns and field goals with descriptions
- 'keyPlays': Important non-scoring plays including turnovers (interceptions, fumbles), sacks, and big plays (20+ yards)

FORMATTING:
- Use markdown formatting for better readability
- Use **bold** for emphasis on key points or player names
- Use bullet lists (- or *) when listing multiple statistics or plays
- Separate paragraphs with blank lines

Do not follow up with any questions. Consider this a final draft that will be shared as-is with the public.

THIS IS THE 2025/2026 NFL SEASON

Game Data:
{game_json}
"""

    # Format the prompt with the game data
    try:
        prompt = prompt_template.format(game_json=json.dumps(game_data))

        # Save prompt to data/prompts for debugging/review
        import os
        os.makedirs('data/prompts', exist_ok=True)
        prompt_path = f'data/prompts/game_analysis_{game_id}.txt'
        with open(prompt_path, 'w', encoding='utf-8') as f:
            f.write(prompt)

        system_prompt = "You are an NFL writer. Your expertise is in writing brief game analyses in multiple voices. You can be a brilliant and serious analyst as easily as a surrealist sports comic. Whether you are writing something serious or ridiculous, you are always accurate in your use of NFL references and statistics. Consider all of the data provided."

        ai_service = AIService()
        # Use generate_text for plain text response (not JSON)
        analysis, status = ai_service.generate_text(prompt, system_message=system_prompt)
        if status != "success":
            logger.error(
                "AI provider failed to generate analysis for game %s (status=%s): %s",
                game_id,
                status,
                (analysis[:200] + "...") if isinstance(analysis, str) and len(analysis) > 200 else analysis,
            )
            return None
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
        with open('data/schedule.json', 'r', encoding='utf-8') as fh:
            schedule_df = pd.DataFrame(json.load(fh))

        if schedule_df.empty:
            logger.warning("schedule.json is empty; game previews will omit weekly context")
        elif 'espn_id' not in schedule_df.columns:
            logger.warning("schedule.json missing 'espn_id' column; cannot match games to weeks")

        # Normalize ESPN IDs to strings for reliable matching
        game_id_str = str(game_id).strip()
        espn_id_series = None
        if 'espn_id' in schedule_df.columns:
            espn_id_series = schedule_df['espn_id'].astype(str).str.strip()
            espn_id_series = espn_id_series.replace({'': pd.NA, 'nan': pd.NA, 'None': pd.NA})
            current_games = schedule_df[espn_id_series == game_id_str]
        else:
            current_games = pd.DataFrame()
        if current_games.empty:
            logger.warning(
                "Game %s not found in schedule.json",
                game_id,
            )
            raise ValueError(f"Game {game_id} not found in schedule.json")

        current_game = current_games.iloc[0]
        current_week = current_game.get('week_num')
        if pd.isna(current_week):
            raise ValueError(f"Week number missing for game {game_id}")
        
        # Get completed games from same week
        this_week_games = schedule_df[
            (schedule_df['week_num'] == current_week) &  # Same week
            (schedule_df.index != current_game.name) &  # Not the current game
            (schedule_df['away_score'].notna()) &  # Has scores
            (schedule_df['home_score'].notna())
        ]
        
        # Format game results
        game_results = []
        for _, game in this_week_games.iterrows():
            try:
                away_score = int(float(game['away_score']))
                home_score = int(float(game['home_score']))
            except (TypeError, ValueError):
                away_score = game.get('away_score')
                home_score = game.get('home_score')

            result = f"{game['away_team']} {away_score} @ {game['home_team']} {home_score}"
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
- **team_records**: AUTHORITATIVE overall season records (wins-losses-ties) for both teams - USE THIS for team records
- **team_season_stats**: Comprehensive season statistics for both teams including efficiency metrics (3rd down %, red zone %), advanced stats (EPA, completion %), records (conference, division, road), and more
- **betting_lines**: Spread, over/under, and money lines
- **espn_predictor**: Win probability predictions (home_win_prob, away_win_prob, matchup_quality)
- **leaders**: Top players in key statistical categories
- **injuries**: Current injury reports
- **gameInfo**: Venue, weather conditions
- **team_profiles/head_coaches/coordinators**: Coaching context and stadium information for each team
- **injury_report**: Extended injury notes with detailed comments and last-updated timestamps
- **recent_news**: Latest team-specific headlines
- **depth_chart_alerts**: Notable starter/lineup status changes from the latest depth chart snapshot
- **recent_form**: Last five game results for each team
- **event_context**: Broadcast, neutral-site, attendance, and special event notes pulled from the scoreboard feed

Use the team_season_stats section for detailed statistical analysis - it has the most comprehensive data.

FORMATTING:
- Use markdown formatting for better readability
- Use **bold** for emphasis on key points, player names, or team names
- Use bullet lists (- or *) when listing multiple matchups or key factors
- Separate paragraphs with blank lines

Consider this a print-ready final draft, so do not include any pre-text like "Here is my preview of the game..." or follow up questions.

THIS IS THE 2025/2026 NFL SEASON

Game Data:
{game_json}
"""

    try:
        prompt = prompt_template.format(
            game_json=json.dumps(game_data),
            week_results=week_results
        )

        # Save prompt to data/prompts for debugging/review
        import os
        os.makedirs('data/prompts', exist_ok=True)
        prompt_path = f'data/prompts/game_preview_{game_id}.txt'
        with open(prompt_path, 'w', encoding='utf-8') as f:
            f.write(prompt)

        system_prompt = "You are an NFL writer specializing in game previews and analysis. You combine statistical insight with entertaining writing, whether serious or playful. You're always accurate with NFL references and statistics. Respond with ONLY the preview text, no JSON, no additional formatting."

        ai_service = AIService()
        # Use generate_text for plain text response (not JSON)
        analysis, status = ai_service.generate_text(prompt, system_message=system_prompt)
        if status != "success":
            logger.error(
                "AI provider failed to generate preview for game %s (status=%s): %s",
                game_id,
                status,
                (analysis[:200] + "...") if isinstance(analysis, str) and len(analysis) > 200 else analysis,
            )
            return None
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
    Load comprehensive season statistics for both teams from team_stats.json.
    Returns dict with stats for both teams, filtered to most relevant metrics.
    """
    try:
        with open('data/team_stats.json', 'r', encoding='utf-8') as fh:
            stats_df = pd.DataFrame(json.load(fh))

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

def _process_single_game(game, analyses, force_reanalyze, current_date, week_from_now, ai_model=None):
    """
    Worker function to process a single game analysis.
    Returns tuple: (game_id, analysis_dict, matchup_str, status)
    - status can be: "success", "skipped", "error"
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
        return (game_id, None, matchup_str, "skipped")

    if not (is_completed or is_upcoming):
        return (game_id, None, matchup_str, "skipped")

    # Fetch and clean game data
    game_data = fetch_and_clean_game(game_id)
    if not game_data:
        logger.error(
            "No ESPN summary data found for %s (game_id=%s). Ensure collect_raw_data captured espn_summary files.",
            matchup_str,
            game_id,
        )
        return (game_id, None, matchup_str, "error")

    # Fetch ESPN supplemental data BEFORE generating analysis
    espn_context = None
    predictor_data = None
    leaders_data = None
    broadcast_info = None

    try:
        from espn_api import ESPNAPIService
        espn_service = ESPNAPIService(RAW_DATA_MANIFEST)

        # Get betting/weather (available for all games)
        espn_context = espn_service.get_game_context(game_id, game['home_team'], game['away_team'])
        scoreboard_context = _load_scoreboard_context(game_id)
        if scoreboard_context:
            if espn_context is None:
                espn_context = {}
            espn_context.setdefault('event', {}).update(scoreboard_context)
            game_data['event_context'] = scoreboard_context
        if scoreboard_context.get('venue') and not game_data.get('stadium'):
            game_data['stadium'] = scoreboard_context.get('venue')

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

    # Attach team metadata for AI context
    away_profile = TEAMS_METADATA.get(game['away_team'], {})
    home_profile = TEAMS_METADATA.get(game['home_team'], {})
    game_data['team_profiles'] = {
        game['away_team']: away_profile,
        game['home_team']: home_profile,
    }
    game_data['stadium'] = game_data.get('stadium') or home_profile.get('stadium', '')
    game_data['head_coaches'] = {
        game['away_team']: away_profile.get('head_coach', ''),
        game['home_team']: home_profile.get('head_coach', ''),
    }
    game_data['coordinators'] = {
        game['away_team']: {
            'offensive': away_profile.get('offensive_coordinator', ''),
            'defensive': away_profile.get('defensive_coordinator', ''),
        },
        game['home_team']: {
            'offensive': home_profile.get('offensive_coordinator', ''),
            'defensive': home_profile.get('defensive_coordinator', ''),
        },
    }
    game_data['injury_report'] = {
        game['away_team']: _load_team_injury_notes(game['away_team']),
        game['home_team']: _load_team_injury_notes(game['home_team'])
    }
    game_data['recent_news'] = {
        game['away_team']: _load_team_headlines(game['away_team']),
        game['home_team']: _load_team_headlines(game['home_team'])
    }
    game_data['depth_chart_alerts'] = {
        game['away_team']: _load_depth_chart_flags(game['away_team']),
        game['home_team']: _load_depth_chart_flags(game['home_team'])
    }
    game_data['recent_form'] = {
        game['away_team']: _load_team_recent_form(game['away_team'], limit=4),
        game['home_team']: _load_team_recent_form(game['home_team'], limit=4)
    }

    # Add ESPN betting lines to game_data for AI context
    if espn_context and espn_context.get('betting'):
        game_data['betting_lines'] = espn_context.get('betting')

    # Add predictor data for previews only
    if not is_completed and predictor_data:
        game_data['espn_predictor'] = predictor_data

    # Add comprehensive team season stats from team_stats.json (previews only)
    # Note: Only add for upcoming games to avoid anachronistic stats in historical game analysis
    if not is_completed:
        team_season_stats = load_team_season_stats(game['away_team'], game['home_team'])
        if team_season_stats:
            game_data['team_season_stats'] = team_season_stats

            # Add explicit team records prominently for AI to see
            game_data['team_records'] = {
                game['away_team']: {
                    'wins': team_season_stats[game['away_team']].get('wins'),
                    'losses': team_season_stats[game['away_team']].get('losses'),
                    'ties': team_season_stats[game['away_team']].get('ties'),
                    'record_display': f"{team_season_stats[game['away_team']].get('wins')}-{team_season_stats[game['away_team']].get('losses')}" +
                                     (f"-{team_season_stats[game['away_team']].get('ties')}" if team_season_stats[game['away_team']].get('ties', 0) > 0 else "")
                },
                game['home_team']: {
                    'wins': team_season_stats[game['home_team']].get('wins'),
                    'losses': team_season_stats[game['home_team']].get('losses'),
                    'ties': team_season_stats[game['home_team']].get('ties'),
                    'record_display': f"{team_season_stats[game['home_team']].get('wins')}-{team_season_stats[game['home_team']].get('losses')}" +
                                     (f"-{team_season_stats[game['home_team']].get('ties')}" if team_season_stats[game['home_team']].get('ties', 0) > 0 else "")
                }
            }

        # Remove redundant/confusing sections for previews
        sections_to_remove = [
            'boxscore',  # Redundant with team_season_stats
            'lastFiveGames',  # Confusing and incomplete
            'againstTheSpread',  # Usually empty
            'wallclockAvailable',  # ESPN metadata
            'ticketsInfo'  # Not relevant for analysis
        ]
        for section in sections_to_remove:
            game_data.pop(section, None)

        # Remove streak_display from team_season_stats (causes confusion)
        if 'team_season_stats' in game_data:
            for team in game_data['team_season_stats']:
                game_data['team_season_stats'][team].pop('streak_display', None)

        # Keep only one betting source (prefer betting_lines, remove pickcenter duplicate)
        if 'betting_lines' in game_data and 'pickcenter' in game_data:
            game_data.pop('pickcenter', None)

    # Generate analysis or preview with ESPN context
    # Get AI service info for metadata
    from ai_service import AIService, resolve_model_name, detect_provider_from_model

    # Resolve model override if provided
    model_override = None
    if ai_model:
        model_override = resolve_model_name(ai_model)
        # Auto-detect and switch provider based on model
        detected_provider = detect_provider_from_model(model_override)
        if detected_provider:
            import ai_service as ai_service_module
            ai_service_module.model_provider = detected_provider

    ai_service = AIService(model_override=model_override)

    try:
        if is_completed:
            analysis = send_to_claude(game_data, game_id)
            analysis_type = "post_game"
        else:
            analysis = send_preview_to_claude(game_data, game_id)
            analysis_type = "preview"

        if not analysis:
            logger.error(f"Failed to generate {analysis_type} for {matchup_str} - AI returned empty response")
            return (game_id, None, matchup_str, "error")

        # Check if the analysis is actually an error message (not JSON)
        if isinstance(analysis, str) and analysis.startswith("RATE_LIMIT_ERROR"):
            logger.error(f"Rate limit hit while processing {matchup_str}")
            # Return special tuple to signal rate limit
            return (game_id, None, matchup_str, "RATE_LIMIT")

    except Exception as e:
        error_msg = str(e)
        if 'rate' in error_msg.lower() or '429' in error_msg:
            logger.error(f"Rate limit error for {matchup_str}: {error_msg}")
            return (game_id, None, matchup_str, "RATE_LIMIT")
        logger.error(f"Error generating {analysis_type if 'analysis_type' in locals() else 'analysis'} for {matchup_str}: {error_msg}")
        return (game_id, None, matchup_str, "error")

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

    return (game_id, analysis_dict, matchup_str, "success")


def batch_analyze_games(
    output_file='data/game_analyses.json',
    force_reanalyze=False,
    game_ids=None,
    regenerate_type=None,
    ai_model=None,
    manifest: Optional[RawDataManifest] = None,
):
    """
    Analyze completed games and preview upcoming games, saving to a JSON file.

    Args:
        output_file: Path to save analysis JSON
        force_reanalyze: Force regeneration (overrides existing analysis)
        game_ids: List of ESPN game IDs to regenerate (if specified, only these games are processed)
        regenerate_type: "analysis", "preview", "all", or "weekly-refresh" to filter by analysis type
            - "analysis": only regenerate completed games
            - "preview": only regenerate upcoming games
            - "all": regenerate everything
            - "weekly-refresh": refresh all current week's games + all existing previews
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm import tqdm

    global RAW_DATA_MANIFEST
    if manifest is not None:
        RAW_DATA_MANIFEST = manifest
    elif RAW_DATA_MANIFEST is None:
        RAW_DATA_MANIFEST = RawDataManifest.from_latest()

    logger.info("Starting batch_analyze_games function")
    analyses = {}
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                analyses = json.load(f)
        except json.JSONDecodeError:
            analyses = {}

    with open('data/schedule.json', 'r', encoding='utf-8') as fh:
        schedule_df = pd.DataFrame(json.load(fh))
    games = schedule_df.to_dict('records')

    # Get current date without time part
    current_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    week_from_now = current_date + timedelta(days=7)

    completed_games = schedule_df[
        schedule_df['away_score'].notna() & schedule_df['home_score'].notna()
    ]
    current_week = None
    if not completed_games.empty and 'week_num' in completed_games.columns:
        current_week = int(float(completed_games['week_num'].max()))

    if current_week is None:
        week_series = schedule_df['week_num'] if 'week_num' in schedule_df.columns else pd.Series(dtype=int)
        if not week_series.empty:
            current_week = int(float(pd.to_numeric(week_series, errors='coerce').dropna().min()))
        else:
            current_week = 1

    allowed_preview_weeks = {current_week}
    # Allow next week's previews starting Tuesday (weekday() >= 1)
    if current_date.weekday() >= 1:
        allowed_preview_weeks.add(current_week + 1)

    # Filter games based on criteria
    games_to_process = []
    for game in games:
        game_id = str(game['espn_id'])
        game_date = datetime.strptime(game['game_date'], '%Y-%m-%d')

        # Determine if game is completed or upcoming
        is_completed = not (pd.isna(game['home_score']) or pd.isna(game['away_score']) or \
                          str(game['home_score']).strip() == '' or str(game['away_score']).strip() == '')

        try:
            game_week = int(float(game.get('week_num')))
        except (TypeError, ValueError):
            game_week = 0
        is_upcoming = (
            not is_completed
            and game_date >= current_date
            and game_date <= week_from_now
            and game_week in allowed_preview_weeks
        )

        # Filter by game IDs if specified
        if game_ids is not None and game_id not in game_ids:
            continue

        # Filter by regenerate type if specified
        if regenerate_type is not None:
            if regenerate_type == 'analysis' and not is_completed:
                continue
            elif regenerate_type == 'preview' and not is_upcoming:
                continue
            elif regenerate_type == 'weekly-refresh':
                # Include games from this week OR games that already have a preview
                in_current_week = (
                    game_date >= current_date
                    and game_date <= week_from_now
                    and game_week in allowed_preview_weeks
                )
                has_existing_preview = game_id in analyses and analyses[game_id].get('analysis_type') == 'preview'

                if not (in_current_week or has_existing_preview):
                    continue
            # 'all' means process everything (no filter)

        # Skip games that already have the correct analysis unless we're forcing
        existing_analysis = analyses.get(game_id)
        needs_analysis = (
            force_reanalyze or
            existing_analysis is None or
            (is_completed and existing_analysis.get('analysis_type') == 'preview')
        )

        if not needs_analysis:
            continue

        # Only process games that are either completed or inside the preview window
        if not (is_completed or is_upcoming):
            continue

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
    from ai_service import AIService, resolve_model_name, detect_provider_from_model

    # Resolve model override if provided (same logic as in _process_single_game)
    model_override = None
    if ai_model:
        model_override = resolve_model_name(ai_model)
        # Auto-detect and switch provider based on model
        detected_provider = detect_provider_from_model(model_override)
        if detected_provider:
            import ai_service as ai_service_module
            ai_service_module.model_provider = detected_provider

    ai_test = AIService(model_override=model_override)
    provider_info = f"Using AI provider: {ai_test.model_provider}, model: {ai_test.model}"
    logger.info(provider_info)
    logger.info(f"GAME_ANALYSIS_WORKERS env var: {os.environ.get('GAME_ANALYSIS_WORKERS', '3')}")
    logger.info(f"Calculated max_workers: {max_workers}")
    logger.info(f"Processing {total_games} game(s) with {max_workers} parallel worker(s)")

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

    # For weekly-refresh, force regeneration of all selected games
    should_force_reanalyze = force_reanalyze or (regenerate_type == 'weekly-refresh')

    executor = ThreadPoolExecutor(max_workers=max_workers)
    try:
        # Submit all game analysis tasks
        future_to_game = {
            executor.submit(_process_single_game, game, analyses, should_force_reanalyze, current_date, week_from_now, ai_model): game
            for game in games_to_process
        }

        error_games = []
        with tqdm(total=total_games, desc="Generating game analyses", unit="game") as pbar:
            for future in as_completed(future_to_game):
                try:
                    result = future.result()

                    # Check if rate limit was hit
                    if len(result) == 4 and result[3] == "RATE_LIMIT":
                        logger.error("\n🚨 RATE LIMIT DETECTED - Cancelling all remaining tasks...")
                        # Cancel all pending futures
                        for f in future_to_game:
                            f.cancel()
                        executor.shutdown(wait=False, cancel_futures=True)
                        logger.error("All tasks cancelled. Exiting...")
                        os._exit(1)

                    game_id, analysis_dict, matchup_str, status = result[0], result[1], result[2], result[3]

                    # Update progress bar with current game
                    pbar.set_postfix_str(matchup_str)

                    if status == "success" and analysis_dict is not None:
                        # Update analyses dict
                        analyses[game_id] = analysis_dict

                        # Save after each successful analysis
                        with open(output_file, 'w', encoding='utf-8') as f:
                            json.dump(analyses, f, indent=2, ensure_ascii=False)
                    elif status == "error":
                        # Track failed games (not skipped ones)
                        error_games.append(matchup_str)
                except Exception as e:
                    game = future_to_game[future]
                    matchup = f"{game.get('away_team', '?')}@{game.get('home_team', '?')}"
                    logger.error(f"Error processing game {matchup}: {str(e)}")
                    error_games.append(matchup)
                finally:
                    pbar.update(1)

        logger.info(f"Batch analysis complete. Processed {total_games} game(s)")

        # Report errors if any occurred
        if error_games:
            logger.error(f"Game AI errors occurred for {len(error_games)} game(s): {', '.join(error_games)}")
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
                      help='Process the schedule.json file and save analyses to game_analyses.json')
    
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
