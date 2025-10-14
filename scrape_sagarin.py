import re
import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
from datetime import datetime, timedelta
import json
import logging

# Create logs directory if it doesn't exist
try:
    os.makedirs('logs', exist_ok=True)
    # File logging setup
    file_handler = logging.FileHandler('logs/sagarin.log')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
except Exception as e:
    # Fallback to console logging if file logging fails
    print(f"Warning: Could not set up file logging: {e}")
    file_handler = logging.StreamHandler()
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

# Set up the logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)

CACHE_FILE = 'data/sagarin_cache.json'
CSV_FILE = 'data/sagarin.csv'

def load_team_abbrs():
    teams = {}
    with open("data/teams.csv") as f:
        for line in f:
            team_abbr, city, mascot = line.strip().split(",")[:3]
            if mascot == "Commanders":
                mascot = "Redskins"
            teams[city + " " + mascot] = team_abbr
    return teams

def should_update_cache():
    """Check if we should update the cache based on last scrape time"""
    if not os.path.exists(CACHE_FILE):
        logger.info("No cache file found - will scrape new data")
        return True

    try:
        with open(CACHE_FILE, 'r') as f:
            cache_data = json.load(f)
            # Use last_scraped if available, fall back to last_update for backwards compatibility
            last_scraped_str = cache_data.get('last_scraped', cache_data.get('last_update'))
            last_scraped = datetime.fromisoformat(last_scraped_str)
            time_since_scrape = datetime.now() - last_scraped

            if time_since_scrape > timedelta(days=1):
                logger.info(f"Cache is {time_since_scrape.total_seconds() / 3600:.1f} hours old - will scrape new data")
                return True
            else:
                # logger.info(f"Cache is only {time_since_scrape.total_seconds() / 3600:.1f} hours old - will use cached data")
                return False
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        logger.error(f"Error reading cache file: {str(e)}")
        return True

def save_to_cache(home_advantage, team_ratings):
    """Save the home advantage value, team ratings, and current timestamp to cache"""
    # Load existing cache to preserve last_content_update if ratings haven't changed
    existing_last_content_update = None
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as f:
                old_cache = json.load(f)
                existing_last_content_update = old_cache.get('last_content_update')
        except Exception:
            pass

    # Load existing CSV to check if ratings have changed
    ratings_changed = False
    previous_data = {}

    if os.path.exists(CSV_FILE):
        try:
            existing_df = pd.read_csv(CSV_FILE)
            if 'rating' in existing_df.columns and 'team_abbr' in existing_df.columns:
                # Check if ratings have actually changed
                existing_ratings = dict(zip(existing_df['team_abbr'], existing_df['rating']))
                for team, new_rating in team_ratings.items():
                    old_rating = existing_ratings.get(team)
                    if old_rating is None or abs(float(old_rating) - float(new_rating)) > 0.001:
                        ratings_changed = True
                        break

                # If ratings changed, store current ratings/ranks as "previous"
                if ratings_changed:
                    sorted_existing = existing_df.sort_values('rating', ascending=False).reset_index(drop=True)
                    for idx, row in sorted_existing.iterrows():
                        previous_data[row['team_abbr']] = {
                            'previous_rank': idx + 1,
                            'previous_rating': row['rating']
                        }
                else:
                    # Ratings haven't changed - preserve existing previous_rank and previous_rating
                    logger.info("Ratings unchanged - preserving historical data in CSV")
                    for _, row in existing_df.iterrows():
                        previous_data[row['team_abbr']] = {
                            'previous_rank': row.get('previous_rank'),
                            'previous_rating': row.get('previous_rating')
                        }
        except Exception as e:
            logger.warning(f"Could not load previous rankings: {e}")
            ratings_changed = True  # Assume changed if we can't read existing
    else:
        ratings_changed = True  # No existing file, so this is new data

    # Determine last_content_update timestamp
    now = datetime.now().isoformat()
    if ratings_changed:
        last_content_update = now
        logger.info(f"Ratings CHANGED - updating last_content_update to now")
    else:
        last_content_update = existing_last_content_update or now
        logger.info(f"Ratings UNCHANGED - preserving last_content_update: {last_content_update}")

    # Save cache with both timestamps
    cache_data = {
        'home_advantage': home_advantage,
        'team_ratings': team_ratings,
        'last_scraped': now,
        'last_content_update': last_content_update,
        'last_update': now  # Keep for backwards compatibility
    }
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache_data, f)
    logger.info(f"Saved home advantage value to cache: {home_advantage}")

    # Create new dataframe with current ratings
    teams_df = pd.DataFrame(list(team_ratings.items()), columns=['team_abbr', 'rating'])
    teams_df = teams_df.sort_values('rating', ascending=False).reset_index(drop=True)

    # Add previous rank and rating columns
    teams_df['previous_rank'] = teams_df['team_abbr'].map(lambda x: previous_data.get(x, {}).get('previous_rank', None))
    teams_df['previous_rating'] = teams_df['team_abbr'].map(lambda x: previous_data.get(x, {}).get('previous_rating', None))

    # Reorder columns for readability
    teams_df = teams_df[['team_abbr', 'rating', 'previous_rank', 'previous_rating']]

    teams_df.to_csv(CSV_FILE, index=False)
    if ratings_changed:
        logger.info(f"Saved NEW team ratings to {CSV_FILE} with updated historical data")
    else:
        logger.info(f"Saved team ratings to {CSV_FILE} (ratings unchanged, historical data preserved)")

def load_from_cache():
    """Load home advantage value and team ratings from cache"""
    try:
        with open(CACHE_FILE, 'r') as f:
            cache_data = json.load(f)
            home_advantage = cache_data['home_advantage']
            team_ratings = cache_data.get('team_ratings', {})
            logger.info(f"Loaded home advantage value from cache: {home_advantage}")
            return home_advantage, team_ratings
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        logger.error(f"Error loading from cache: {str(e)}")
        return None, None

def scrape_teams_and_ratings(html_content):
    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Find the table
    data = soup.find_all('pre')[2].get_text()
    pattern = r'^\s*\d+\s+([\w\s]+)\s+=\s+(\d+\.\d+)'

    results = []
    for line in data.split('\n'):
        match = re.search(pattern, line)
        if match:
            team_name = match.group(1).strip()
            rating = match.group(2)
            results.append((team_name, rating))

    return results

# Function to extract the first HOME ADVANTAGE number from an HTML file
def scrape_home_advantage(html_content):
    # Regex pattern to match the HOME ADVANTAGE number considering HTML tags
    pattern = r'HOME ADVANTAGE=\[\s*([0-9]+\.[0-9]+)\s*\]'
    matches = re.findall(pattern, html_content)

    return matches[0] if matches else 0

def scrape_sagarin(force_rescrape=False):
    """Main function to get Sagarin ratings, using cache when appropriate

    Args:
        force_rescrape: If True, ignores cache and forces a fresh scrape from website
    """
    if not force_rescrape and not should_update_cache():
        cached_value, team_ratings = load_from_cache()
        if cached_value is not None:
            # Don't overwrite CSV when using cache - preserve historical data
            logger.info(f"Using cached Sagarin ratings (CSV preserved)")
            return cached_value

    # If we need to update, proceed with scraping
    if force_rescrape:
        logger.info("Force rescrape requested - fetching fresh data from Sagarin website")
    try:
        logger.info("Attempting to scrape new Sagarin ratings...")
        response = requests.get('http://sagarin.com/sports/nflsend.htm')
        home_advantage = float(scrape_home_advantage(response.text))
        logger.info(f"Successfully scraped new home advantage value: {home_advantage}")
        
        # Scrape team ratings
        team_abbrs = load_team_abbrs()
        team_ratings = {}
        for team_name, rating in scrape_teams_and_ratings(response.text):
            if team_name in team_abbrs:
                team_ratings[team_abbrs[team_name]] = float(rating)
        
        # Save both to cache and CSV
        save_to_cache(home_advantage, team_ratings)
        return home_advantage
    except Exception as e:
        logger.error(f"Error scraping Sagarin ratings: {str(e)}")
        # If scraping fails, try to use cached value as fallback
        cached_value, team_ratings = load_from_cache()
        if cached_value is not None:
            logger.info("Using cached value as fallback after scraping failed (CSV preserved)")
            return cached_value
        # If no cache available, return a default value
        logger.warning("No cache available - using default home advantage value of 2.5")
        return 2.5  # Default NFL home field advantage
