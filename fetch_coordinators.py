#!/usr/bin/env python3
"""Scrape NFL coordinator data from Wikipedia."""

import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Team name mapping from Wikipedia format to our abbreviations
TEAM_MAPPING = {
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


def fetch_coordinators(coordinator_type='offensive'):
    """
    Fetch coordinator data from Wikipedia.

    Args:
        coordinator_type: 'offensive' or 'defensive'

    Returns:
        dict: {team_abbr: coordinator_name}
    """
    if coordinator_type == 'offensive':
        url = 'https://en.wikipedia.org/wiki/List_of_current_NFL_offensive_coordinators'
    else:
        url = 'https://en.wikipedia.org/wiki/List_of_current_NFL_defensive_coordinators'

    logger.debug(f"Fetching {coordinator_type} coordinators from {url}")

    # Add User-Agent header to avoid 403 errors
    headers = {
        'User-Agent': 'WhoShouldLose/1.0 (https://whoshouldlose.com; educational/research purposes)'
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        # Find all wikitables (there may be multiple for AFC/NFC)
        tables = soup.find_all('table', {'class': 'wikitable'})

        if not tables:
            logger.error(f"Could not find wikitable on {url}")
            return {}

        coordinators = {}

        # Parse all tables
        for table in tables:
            rows = table.find_all('tr')[1:]  # Skip header row

            for row in rows:
                cells = row.find_all('td')

                if len(cells) < 2:
                    continue

                # First cell is usually team name
                team_cell = cells[0]
                team_name = team_cell.get_text(strip=True)

                # Second cell is usually coordinator name
                coordinator_cell = cells[1]
                coordinator_name = coordinator_cell.get_text(strip=True)

                # Map team name to abbreviation
                if team_name in TEAM_MAPPING:
                    team_abbr = TEAM_MAPPING[team_name]
                    coordinators[team_abbr] = coordinator_name
                    logger.debug(f"{team_abbr}: {coordinator_name}")
                else:
                    logger.warning(f"Unknown team name: {team_name}")

        logger.debug(f"Found {len(coordinators)} {coordinator_type} coordinators")
        return coordinators

    except Exception as e:
        logger.error(f"Error fetching {coordinator_type} coordinators: {e}")
        return {}


def fetch_all_coordinators():
    """Return a combined offensive/defensive coordinator mapping.

    Returns:
        dict: {team_abbr: {"offensive_coordinator": str, "defensive_coordinator": str,
                          "last_updated": str}}
    """

    offensive = fetch_coordinators('offensive')
    defensive = fetch_coordinators('defensive')

    if not offensive and not defensive:
        logger.error("Failed to fetch coordinator data")
        return {}

    all_teams = set(offensive.keys()) | set(defensive.keys())
    timestamp = datetime.now().strftime('%Y-%m-%d')

    combined = {}
    for team_abbr in sorted(all_teams):
        combined[team_abbr] = {
            'offensive_coordinator': offensive.get(team_abbr, ''),
            'defensive_coordinator': defensive.get(team_abbr, ''),
            'last_updated': timestamp,
        }

    return combined


def save_coordinators_csv(output_path='data/coordinators.csv'):
    """Fetch coordinators and persist them to CSV (legacy helper)."""

    data = fetch_all_coordinators()
    if not data:
        return False

    df = pd.DataFrame([
        {
            'team_abbr': team_abbr,
            'offensive_coordinator': info.get('offensive_coordinator', ''),
            'defensive_coordinator': info.get('defensive_coordinator', ''),
            'last_updated': info.get('last_updated', ''),
        }
        for team_abbr, info in data.items()
    ])

    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df)} coordinators to {output_path}")

    print("\n" + "=" * 70)
    print("COORDINATOR DATA")
    print("=" * 70)
    print(df.to_string(index=False))
    print("=" * 70)

    return True


if __name__ == '__main__':
    import sys
    success = save_coordinators_csv()
    sys.exit(0 if success else 1)
