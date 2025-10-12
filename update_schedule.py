#!/usr/bin/env python3
"""Quick script to update schedule.csv with latest scores and dates/times from ESPN."""

from update_scores import update_scores_and_dates
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == '__main__':
    print("Updating schedule.csv with latest scores and dates/times...")
    updated = update_scores_and_dates(update_dates=True)
    print(f"Updated {updated} games")