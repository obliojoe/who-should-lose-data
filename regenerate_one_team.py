#!/usr/bin/env python3
"""
Quick script to regenerate analysis for a single team and update the cache.
This mimics the website's regenerate button but from the command line.

Usage:
    python regenerate_one_team.py DET
    python regenerate_one_team.py MIN
"""

import sys
import os
import json
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.generate_cache.generate_cache import generate_team_analysis_prompt
from scripts.generate_cache.ai_service import AIService
from scripts.generate_cache.playoff_utils import load_teams, load_schedule
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def regenerate_team_analysis(team_abbr):
    """
    Regenerate AI analysis for a specific team and update the cache

    Args:
        team_abbr: Team abbreviation (e.g., 'DET', 'MIN')
    """
    cache_path = Path(__file__).parent.parent.parent / 'persist' / 'analysis_cache.json'

    # Load the existing cache
    if not cache_path.exists():
        logger.error(f"Cache file not found at {cache_path}")
        return False

    with open(cache_path, 'r') as f:
        cache_data = json.load(f)

    # Verify team exists in cache
    if team_abbr not in cache_data['team_analyses']:
        logger.error(f"Team {team_abbr} not found in analysis cache")
        return False

    # Load required data
    teams = load_teams()
    team_info = teams.get(team_abbr)
    if not team_info:
        logger.error(f"Team {team_abbr} not found in teams data")
        return False

    logger.info(f"Regenerating analysis for {team_info['city']} {team_info['mascot']}...")

    # Get team record from cache (simpler than recalculating)
    team_data = cache_data['team_analyses'][team_abbr]
    team_record = {
        'wins': team_data.get('wins', 0),
        'losses': team_data.get('losses', 0),
        'ties': team_data.get('ties', 0)
    }

    # Generate the prompt
    prompt = generate_team_analysis_prompt(team_abbr, team_info, team_record, teams, cache_data)

    # Initialize AI service
    ai_service = AIService()

    # Test connection
    success, msg = ai_service.test_connection()
    if not success:
        logger.error(f"AI service connection failed: {msg}")
        return False

    model_name = ai_service.get_model()
    provider_name = ai_service.get_provider()
    logger.info(f"Using {provider_name} ({model_name})")

    # Generate new AI analysis
    logger.info("Generating AI analysis...")
    start_time = datetime.now()
    ai_text, status = ai_service.generate_analysis(prompt)
    elapsed = (datetime.now() - start_time).total_seconds()

    logger.info(f"Analysis completed in {elapsed:.2f}s with status: {status}")

    if status != "success":
        logger.error(f"Analysis generation failed: {ai_text}")
        return False

    # Parse and display the analysis
    try:
        analysis_json = json.loads(ai_text)

        print("\n" + "="*70)
        print(f"{team_info['city'].upper()} {team_info['mascot'].upper()} - REGENERATED ANALYSIS")
        print("="*70)
        print(f"\nGeneration time: {elapsed:.2f}s")
        print(f"Model: {model_name}")
        print("\n" + "="*70)
        print("THE VERDICT")
        print("="*70)
        print(analysis_json.get('ai_verdict', 'N/A'))
        print("\n" + "="*70)
        print("THE X-FACTOR")
        print("="*70)
        print(analysis_json.get('ai_xfactor', 'N/A'))
        print("\n" + "="*70)
        print("THE REALITY CHECK")
        print("="*70)
        print(analysis_json.get('ai_reality_check', 'N/A'))
        print("\n" + "="*70)
        print("THE QUOTE")
        print("="*70)
        print(f'"{analysis_json.get("ai_quote", "N/A")}"')
        print("="*70 + "\n")

    except json.JSONDecodeError as e:
        logger.warning(f"Could not parse analysis as JSON: {e}")
        print("\n" + "="*70)
        print("RAW ANALYSIS")
        print("="*70)
        print(ai_text)
        print("="*70 + "\n")

    # Update only the AI-related fields in the cache
    cache_data['team_analyses'][team_abbr]['ai_analysis'] = ai_text
    cache_data['team_analyses'][team_abbr]['ai_status'] = status

    # Save the updated cache file
    logger.info(f"Updating cache at {cache_path}...")
    with open(cache_path, 'w') as f:
        json.dump(cache_data, f, indent=2)

    logger.info("✓ Cache updated successfully!")
    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python regenerate_one_team.py <TEAM_ABBR>")
        print("Example: python regenerate_one_team.py DET")
        sys.exit(1)

    team_abbr = sys.argv[1].upper()
    success = regenerate_team_analysis(team_abbr)

    sys.exit(0 if success else 1)
