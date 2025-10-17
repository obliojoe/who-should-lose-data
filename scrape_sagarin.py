import json
import logging
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup

from raw_data_manifest import RawDataManifest

# Logging setup ------------------------------------------------------------
try:
    os.makedirs('logs', exist_ok=True)
    file_handler = logging.FileHandler('logs/sagarin.log')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
except Exception as exc:  # pragma: no cover - defensive
    print(f"Warning: Could not set up file logging: {exc}")
    file_handler = logging.StreamHandler()
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)

# File locations -----------------------------------------------------------
SAGARIN_JSON = Path('data/sagarin.json')
HISTORY_LIMIT = 52  # roughly one full season of weekly snapshots

# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def _load_existing_payload() -> Dict:
    if SAGARIN_JSON.exists():
        try:
            with SAGARIN_JSON.open('r', encoding='utf-8') as fh:
                payload = json.load(fh)
                if isinstance(payload, dict):
                    payload.setdefault('ratings', {})
                    payload.setdefault('history', [])
                    return payload
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to read %s: %s", SAGARIN_JSON, exc)
    return {
        'home_field_advantage': None,
        'last_scraped': None,
        'last_content_update': None,
        'ratings': {},
        'history': [],
    }

def _timestamp() -> str:
    return datetime.now().isoformat()

def _determine_current_week() -> Optional[int]:
    schedule_path = Path('data/schedule.json')
    if not schedule_path.exists():
        return None
    try:
        with schedule_path.open('r', encoding='utf-8') as fh:
            games = json.load(fh)
        completed_weeks = [game.get('week_num') for game in games
                           if game.get('home_score') not in (None, '', 'nan')
                           and game.get('away_score') not in (None, '', 'nan')]
        if completed_weeks:
            return int(max(completed_weeks))
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Unable to determine current week from schedule: %s", exc)
    return None

# -------------------------------------------------------------------------
# Raw HTML sourcing helpers
# -------------------------------------------------------------------------

def load_raw_sagarin_html(manifest: Optional[RawDataManifest]) -> Optional[str]:
    """Return the most recent raw Sagarin HTML captured by collect_raw_data."""
    if manifest is None:
        return None

    try:
        entries = manifest.entries('sagarin_html')
    except Exception as exc:  # Manifest may not support the dataset yet
        logger.debug("Unable to enumerate sagarin_html entries from manifest: %s", exc)
        return None

    if not entries:
        return None

    def _entry_sort_key(entry):
        try:
            return entry.path.stat().st_mtime
        except OSError:
            return 0

    latest_entry = max(entries, key=_entry_sort_key)
    try:
        html_content = latest_entry.path.read_text(encoding='utf-8')
        logger.info("Loaded Sagarin HTML from raw snapshot at %s", latest_entry.path)
        return html_content
    except OSError as exc:
        logger.warning("Failed to read raw Sagarin HTML at %s: %s", latest_entry.path, exc)
        return None

# -------------------------------------------------------------------------
# Parsing helpers
# -------------------------------------------------------------------------

def load_team_abbrs() -> Dict[str, str]:
    teams = {}
    with open('data/teams.json', 'r', encoding='utf-8') as fh:
        records = json.load(fh)

    for record in records:
        city = record.get('city', '')
        mascot = record.get('mascot', '')
        team_abbr = record.get('team_abbr', '')
        if not team_abbr or not city or not mascot:
            continue

        lookup_mascot = 'Redskins' if mascot == 'Commanders' else mascot
        teams[f"{city} {lookup_mascot}"] = team_abbr

    return teams


def scrape_home_advantage(html_content: str) -> float:
    pattern = r'HOME ADVANTAGE=\[\s*([0-9]+\.[0-9]+)\s*\]'
    matches = re.findall(pattern, html_content)
    if not matches:
        return 0.0
    try:
        return float(matches[0])
    except ValueError:
        return 0.0


def scrape_team_ratings(html_content: str) -> List[Tuple[str, float]]:
    soup = BeautifulSoup(html_content, 'html.parser')
    pre_blocks = soup.find_all('pre')
    if len(pre_blocks) < 3:
        return []

    data = pre_blocks[2].get_text()
    pattern = r'^\s*\d+\s+([\w\s]+)\s+=\s+(\d+\.\d+)'

    results: List[Tuple[str, float]] = []
    for line in data.split('\n'):
        match = re.search(pattern, line)
        if match:
            team_name = match.group(1).strip()
            rating_str = match.group(2)
            try:
                results.append((team_name, float(rating_str)))
            except ValueError:
                continue
    return results


def extract_sagarin_metrics(html_content: str) -> Tuple[float, Dict[str, float]]:
    home_advantage = scrape_home_advantage(html_content)
    team_abbrs = load_team_abbrs()
    ratings: Dict[str, float] = {}
    for team_name, rating in scrape_team_ratings(html_content):
        abbr = team_abbrs.get(team_name)
        if abbr:
            ratings[abbr] = rating
    return home_advantage, ratings

# -------------------------------------------------------------------------
# Cache + JSON helpers
# -------------------------------------------------------------------------

def should_update_cache() -> bool:
    if not SAGARIN_JSON.exists():
        logger.info("No sagarin.json file found – will fetch fresh data")
        return True
    try:
        with SAGARIN_JSON.open('r', encoding='utf-8') as fh:
            cache_data = json.load(fh)
        last_scraped_str = cache_data.get('last_scraped') or cache_data.get('last_update')
        if not last_scraped_str:
            return True
        last_scraped = datetime.fromisoformat(last_scraped_str)
        if datetime.now() - last_scraped > timedelta(days=1):
            logger.info("Sagarin snapshot is older than 24h – refreshing")
            return True
        return False
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to read %s: %s", SAGARIN_JSON, exc)
        return True


def load_from_cache() -> Tuple[Optional[float], Dict[str, float]]:
    try:
        with SAGARIN_JSON.open('r', encoding='utf-8') as fh:
            payload = json.load(fh)
        home_advantage = payload.get('home_field_advantage')
        ratings_section = payload.get('ratings', {})
        ratings = {team: info.get('rating') for team, info in ratings_section.items() if info}
        logger.info("Loaded Sagarin data from sagarin.json")
        return home_advantage, ratings
    except Exception as exc:
        logger.error("Error loading sagarin.json: %s", exc)
        return None, {}


def _ratings_changed(new_ratings: Dict[str, float], existing_ratings: Dict[str, Dict],
                     tolerance: float = 1e-3) -> bool:
    if len(new_ratings) != len(existing_ratings):
        return True
    for team, rating in new_ratings.items():
        existing = existing_ratings.get(team, {}).get('rating')
        if existing is None or abs(existing - rating) > tolerance:
            return True
    return False


def save_to_json(home_advantage: float, team_ratings: Dict[str, float]) -> None:
    existing_payload = _load_existing_payload()
    existing_ratings = existing_payload.get('ratings', {})
    existing_home_adv = existing_payload.get('home_field_advantage')

    ratings_changed = _ratings_changed(team_ratings, existing_ratings)
    home_adv_changed = existing_home_adv is None or abs(existing_home_adv - home_advantage) > 1e-6

    if not ratings_changed and not home_adv_changed:
        logger.info("Sagarin ratings and home advantage unchanged – leaving sagarin.json untouched")
        return

    now = _timestamp()
    current_week = _determine_current_week()

    sorted_ratings = sorted(team_ratings.items(), key=lambda item: item[1], reverse=True)

    new_ratings_section: Dict[str, Dict] = {}
    for idx, (team, rating) in enumerate(sorted_ratings, start=1):
        previous_entry = existing_ratings.get(team, {})
        team_history: List[Dict] = previous_entry.get('history', [])
        if ratings_changed:
            team_history = team_history[-9:]
            team_history.append({
                'timestamp': now,
                'rating': rating,
                'rank': idx,
            })
        new_entry = {
            'rating': rating,
            'rank': idx,
            'previous_rank': previous_entry.get('rank'),
            'previous_rating': previous_entry.get('rating'),
        }
        if team_history:
            new_entry['history'] = team_history
        new_ratings_section[team] = new_entry

    history_snapshots: List[Dict] = existing_payload.get('history', [])
    if ratings_changed or home_adv_changed:
        snapshot = {
            'timestamp': now,
            'home_field_advantage': home_advantage,
            'ratings': {team: rating for team, rating in sorted_ratings},
        }
        if current_week is not None:
            snapshot['week'] = current_week
        history_snapshots.append(snapshot)
        history_snapshots = history_snapshots[-HISTORY_LIMIT:]

    payload = {
        'home_field_advantage': home_advantage,
        'last_scraped': now,
        'last_content_update': now if (ratings_changed or home_adv_changed)
        else existing_payload.get('last_content_update', now),
        'ratings': new_ratings_section,
        'history': history_snapshots,
    }

    SAGARIN_JSON.parent.mkdir(parents=True, exist_ok=True)
    with SAGARIN_JSON.open('w', encoding='utf-8') as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    logger.info("Saved Sagarin ratings to %s", SAGARIN_JSON)

# -------------------------------------------------------------------------
# Main scrape orchestration
# -------------------------------------------------------------------------

def scrape_sagarin(force_rescrape: bool = False,
                   manifest: Optional[RawDataManifest] = None) -> float:
    """Fetch Sagarin ratings, updating sagarin.json when necessary."""
    if not force_rescrape:
        html_content = load_raw_sagarin_html(manifest)
        if html_content:
            home_advantage, team_ratings = extract_sagarin_metrics(html_content)
            save_to_json(home_advantage, team_ratings)
            return home_advantage

        if not should_update_cache():
            cached_value, _ = load_from_cache()
            if cached_value is not None:
                logger.info("Using cached Sagarin ratings (json up-to-date)")
                return cached_value

    if force_rescrape:
        logger.info("Force rescrape requested – fetching live data from sagarin.com")

    try:
        logger.info("Attempting to scrape new Sagarin ratings...")
        response = requests.get('http://sagarin.com/sports/nflsend.htm', timeout=15)
        response.raise_for_status()
        home_advantage, team_ratings = extract_sagarin_metrics(response.text)
        logger.info("Successfully scraped Sagarin ratings (home advantage %.2f)", home_advantage)
        save_to_json(home_advantage, team_ratings)
        return home_advantage
    except Exception as exc:
        logger.error("Error scraping Sagarin ratings: %s", exc)
        cached_value, _ = load_from_cache()
        if cached_value is not None:
            logger.info("Using cached Sagarin ratings after scrape failure")
            return cached_value
        logger.warning("No cache available – using default home advantage 2.5")
        return 2.5

# -------------------------------------------------------------------------
# CLI entrypoint (legacy)
# -------------------------------------------------------------------------
if __name__ == '__main__':
    force = '--force' in os.sys.argv
    scrape_sagarin(force_rescrape=force, manifest=RawDataManifest.from_latest())
