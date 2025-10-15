#!/usr/bin/env python3
"""
Power Rankings Dashboard Integration

Provides helper functions to integrate R1+SOV power rankings into the dashboard,
replacing Sagarin ratings.
"""

import pandas as pd
from power_rankings_tracker import PowerRankingsTracker


def get_power_rankings_df(week_num: int) -> pd.DataFrame:
    """
    Get power rankings as a DataFrame compatible with dashboard code

    Returns DataFrame with columns:
    - team_abbr: Team abbreviation
    - rating: Composite score (for compatibility)
    - rank: Current rank (1-32)
    - previous_rank: Previous week's rank
    - movement: Rank change from previous week

    Args:
        week_num: Week number to get rankings for

    Returns:
        DataFrame formatted like sagarin_df for compatibility
    """
    tracker = PowerRankingsTracker()

    # Get current week rankings
    rankings = tracker.get_current_rankings(week_num)

    # Convert to DataFrame
    data = []
    for r in rankings:
        data.append({
            'team_abbr': r['team'],
            'rating': r['composite_score'],  # Use composite score as "rating"
            'rank': r['rank'],
            'previous_rank': r.get('previous_rank'),
            'movement': r.get('movement', 0),
            'record': r['record'],
            'win_pct': r['win_pct'],
            'playoff_prob': r['playoff_prob'],
            'sov': r['sov']
        })

    df = pd.DataFrame(data)

    # Set index to match Sagarin format
    df = df.set_index(pd.RangeIndex(len(df)))

    return df


def update_power_rankings_for_week(week_num: int, force: bool = False) -> pd.DataFrame:
    """
    Update power rankings for a specific week and return as DataFrame

    Args:
        week_num: Week number to update
        force: Force recalculation even if already exists

    Returns:
        DataFrame of rankings
    """
    tracker = PowerRankingsTracker()

    # Update current week (uses real playoff probabilities)
    tracker.update_current_week(week_num, force=force, is_historical=False)

    # Return as DataFrame
    return get_power_rankings_df(week_num)


if __name__ == "__main__":
    # Test the integration
    import sys

    week = int(sys.argv[1]) if len(sys.argv) > 1 else 6

    print(f"Getting power rankings for week {week}...")
    df = get_power_rankings_df(week)

    print(f"\nPower Rankings DataFrame (week {week}):")
    print(df.head(10))

    print(f"\nColumns: {list(df.columns)}")
    print(f"\nData types:\n{df.dtypes}")
