"""
Simple configuration file to track playoff clinching and elimination status
"""

PLAYOFF_STATUS = {
    # Format: 'team_abbr': {
    #   'playoffs': bool | None,  # True=clinched, False=eliminated, None=still in contention
    #   'division': bool | None,  # True=clinched, False=eliminated, None=still in contention
    #   'top_seed': bool | None   # True=clinched, False=eliminated, None=still in contention
    # }
    'DET': {'playoffs': True},
    'PHI': {'playoffs': True},
    'BUF': {'playoffs': True, 'division': True},
    'KC': {'playoffs': True, 'division': True},
    'NYG': {'playoffs': False},
    'NYJ': {'playoffs': False},
    'NE': {'playoffs': False},
    'LV': {'playoffs': False},
    'JAX': {'playoffs': False},
    'TEN': {'playoffs': False},
    'CLE': {'playoffs': False},
} 