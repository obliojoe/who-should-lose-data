import csv
import hashlib
import json
import os
from pathlib import Path

data_dir = "data/"

def load_teams():
    """Load team data from CSV file"""
    teams = {}
    with open(f"{data_dir}teams.csv", 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            teams[row['team_abbr']] = row
    return teams

def load_schedule():
    """Load schedule data from CSV file"""
    schedule = []
    with open(f"{data_dir}schedule.csv", 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            schedule.append(row)
    return schedule

def load_ratings():
    """Load Sagarin ratings from CSV file"""
    ratings = {}
    with open(f"{data_dir}sagarin.csv", 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ratings[row['team_abbr']] = float(row['rating'])
    return ratings

def get_data_dir(data_dir=None):
    """Resolve the data directory path"""
    if data_dir:
        path = Path(data_dir)
    else:
        # Default to data/ directory next to script
        path = Path(__file__).parent / 'data'
    
    if not path.exists():
        raise FileNotFoundError(f"Data directory not found: {path}")
    
    return path

def generate_sagarin_hash():
    """Generate hash of current Sagarin ratings"""
    ratings = load_ratings()
    ratings_str = json.dumps(ratings, sort_keys=True)
    return hashlib.md5(ratings_str.encode()).hexdigest()

def format_percentage(value):
    """Format percentage values exactly as the website does"""
    if value == 100:
        return 100.0
    elif value > 99.9:
        return 99.9
    elif value < 0.1 and value > 0:
        return 0.1
    elif value == 0:
        return 0.0
    else:
        return round(value, 1)
