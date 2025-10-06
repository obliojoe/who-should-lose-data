import csv
import numpy as np
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import logging
import os
from tiebreakers import apply_tiebreakers
from playoff_analysis import determine_wild_cards_with_tiebreakers

# Set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create logs directory if it doesn't exist
if not os.path.exists('logs'):
    os.makedirs('logs')

# Set up file handler
file_handler = logging.FileHandler('logs/simulate_season.log')
file_handler.setLevel(logging.WARNING)  # Only log warnings and errors to file
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add handler to logger
logger.addHandler(file_handler)
logger.propagate = False  # Don't propagate to parent loggers

def load_schedule():
    schedule = []
    with open('data/schedule.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            schedule.append(row)
    return schedule

def load_teams():
    teams = {}
    with open('data/teams.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            teams[row['team_abbr']] = row
    return teams

def load_ratings():
    ratings = {}
    with open('data/sagarin.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ratings[row['team_abbr']] = float(row['rating'])
    return ratings

# Pre-compute valid scores and their weights once
VALID_SCORES = np.array([
    0, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37,
    38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50
])

COMMON_SCORES = {0, 3, 6, 7, 10, 13, 14, 17, 20, 21, 24, 27, 28, 31, 34, 35}
SCORE_WEIGHTS = np.array([3 if score in COMMON_SCORES else 1 for score in VALID_SCORES])

# Default RNG used when no specific generator is provided
GLOBAL_RNG = np.random.default_rng()


def _head_to_head_default():
    return {'wins': 0, 'losses': 0, 'ties': 0}


def _common_games_default():
    return {'wins': 0, 'losses': 0, 'ties': 0}


def _team_record_default():
    return {
        'wins': 0,
        'losses': 0,
        'ties': 0,
        'division_wins': 0,
        'division_losses': 0,
        'division_ties': 0,
        'conference_wins': 0,
        'conference_losses': 0,
        'conference_ties': 0,
        'head_to_head': defaultdict(_head_to_head_default),
        'opponents': set(),
        'defeated_opponents': set(),
        'common_games': defaultdict(_common_games_default),
        'points_for': 0,
        'points_against': 0
    }

def _fast_score_calculation(expected_total, predicted_spread, volatility=13.5, rng=None):
    """Optimized core score calculation"""
    rng = GLOBAL_RNG if rng is None else rng

    actual_spread = rng.normal(predicted_spread, volatility)
    actual_total = rng.normal(expected_total, 10)
    
    home_score = (actual_total + actual_spread) / 2
    away_score = (actual_total - actual_spread) / 2
    
    return max(0, round(home_score)), max(0, round(away_score))

def generate_random_score(expected_total, predicted_spread, volatility=13.5, rng=None):
    """
    Generate a random but plausible NFL score with reduced ties
    
    Parameters:
    predicted_spread (float): Predicted point differential
    volatility (float): Standard deviation of actual vs predicted spread
    
    Returns:
    tuple: (home_score, away_score)
    """
    max_attempts = 3  # Try up to 3 times to generate non-tie scores
    
    rng = GLOBAL_RNG if rng is None else rng

    home_score, away_score = _fast_score_calculation(expected_total, predicted_spread, volatility, rng=rng)
        
        # Adjust to valid football scores
    home_score = adjust_to_football_score(home_score, rng=rng)
    away_score = adjust_to_football_score(away_score, rng=rng)
    
    # If we still have a tie after max attempts, adjust one team's score
    if home_score == away_score:
        if rng.random() > 0.01: # rarely keep it a tie
            # Get valid scores within 3 points (typical FG range) of current score
            nearby_scores = [s for s in VALID_SCORES if 0 < abs(s - home_score) <= 3]
            if nearby_scores:
                # Adjust score based on predicted spread direction
                if predicted_spread > 0:
                    home_score = int(rng.choice(nearby_scores))
                else:
                    away_score = int(rng.choice(nearby_scores))
    
    return home_score, away_score

def simulate_game(home_team, away_team, ratings, home_field_advantage, rng=None):
    """
    Simulate NFL game using Sagarin ratings with realistic scoring
    
    Parameters:
    home_team (str): Home team abbreviation
    away_team (str): Away team abbreviation
    ratings (dict): Dictionary of team Sagarin ratings
    home_field_advantage (float): Home field advantage points
    
    Returns:
    dict: Game result with scores and win indicator
    """
    home_rating = ratings[home_team]
    away_rating = ratings[away_team]
    
    # Calculate predicted spread and win probability
    raw_diff = home_rating - away_rating
    predicted_spread = raw_diff + home_field_advantage
    win_prob = 0.50 + (predicted_spread * 0.04)
    win_prob = max(0, min(1, win_prob))
    
    # Generate random but plausible scores
    expected_total = 44  # Average NFL game total
    volatility = 13.5  # Standard deviation of spread
    
    rng = GLOBAL_RNG if rng is None else rng

    home_score, away_score = generate_random_score(expected_total, predicted_spread, volatility, rng=rng)
        
    # Adjust scores to common NFL numbers
    home_score = adjust_to_football_score(home_score, rng=rng)
    away_score = adjust_to_football_score(away_score, rng=rng)
    
    return {
        'home_score': home_score,
        'away_score': away_score,
        'home_win': home_score > away_score
    }

def adjust_to_football_score(score, rng=None):
    """
    Adjust a score to valid NFL scoring combinations - maintained original logic
    """
    # Find closest valid score
    distances = np.abs(VALID_SCORES - score)
    min_distance = np.min(distances)
    closest_indices = np.where(distances == min_distance)[0]
    
    # If multiple scores are equally close, weight toward common scores
    if len(closest_indices) > 1:
        closest_scores = VALID_SCORES[closest_indices]
        rng = GLOBAL_RNG if rng is None else rng
        score_weights = [3 if s in COMMON_SCORES else 1 for s in closest_scores]
        return int(rng.choice(closest_scores, p=np.array(score_weights)/sum(score_weights)))

    return int(VALID_SCORES[closest_indices[0]])


def determine_division_winners(standings, teams):
    division_winners = []
    
    # Group teams by division
    division_teams = defaultdict(list)
    for team in standings:
        division = teams[team]['division']
        division_teams[division].append(team)
    
    # For each division, apply tiebreakers and get winner
    for division, div_teams in division_teams.items():
        # First sort teams by win percentage
        div_teams_by_pct = defaultdict(list)
        for team in div_teams:
            win_pct = standings[team]['wins'] / (standings[team]['wins'] + standings[team]['losses'] + standings[team].get('ties', 0))
            div_teams_by_pct[win_pct].append(team)
        
        # Get teams with best win percentage
        best_pct = max(div_teams_by_pct.keys())
        tied_teams = div_teams_by_pct[best_pct]
        
        # Only apply tiebreakers if multiple teams have the same record
        if len(tied_teams) > 1:
            winner = apply_tiebreakers(tied_teams, standings, division=division)[0]
        else:
            winner = tied_teams[0]
            
        division_winners.append(winner)
    
    return division_winners

def initialize_standings(schedule, teams):
    standings = defaultdict(_team_record_default)
    
    for game in schedule:
        # Skip games that haven't been played
        if not game['away_score'] or not game['home_score']:
            continue
            
        try:
            away_score = int(game['away_score'])
            home_score = int(game['home_score'])
            away_team = game['away_team']
            home_team = game['home_team']
            
            # Check if same division
            same_division = teams[away_team]['division'] == teams[home_team]['division']
            # Check if same conference
            same_conference = teams[away_team]['conference'] == teams[home_team]['conference']
            
            if home_score > away_score:
                # Regular wins/losses
                standings[home_team]['wins'] += 1
                standings[away_team]['losses'] += 1
                standings[home_team]['head_to_head'][away_team]['wins'] += 1
                standings[away_team]['head_to_head'][home_team]['losses'] += 1
                
                # Division records
                if same_division:
                    standings[home_team]['division_wins'] += 1
                    standings[away_team]['division_losses'] += 1
                
                # Conference records
                if same_conference:
                    standings[home_team]['conference_wins'] += 1
                    standings[away_team]['conference_losses'] += 1
                    
            elif away_score > home_score:
                # Regular wins/losses
                standings[away_team]['wins'] += 1
                standings[home_team]['losses'] += 1
                standings[away_team]['head_to_head'][home_team]['wins'] += 1
                standings[home_team]['head_to_head'][away_team]['losses'] += 1
                
                # Division records
                if same_division:
                    standings[away_team]['division_wins'] += 1
                    standings[home_team]['division_losses'] += 1
                
                # Conference records
                if same_conference:
                    standings[away_team]['conference_wins'] += 1
                    standings[home_team]['conference_losses'] += 1
                    
            else:  # Tie
                standings[home_team]['ties'] += 1
                standings[away_team]['ties'] += 1
                standings[home_team]['head_to_head'][away_team]['ties'] += 1
                standings[away_team]['head_to_head'][home_team]['ties'] += 1
                
                if same_division:
                    standings[home_team]['division_ties'] += 1
                    standings[away_team]['division_ties'] += 1
                
                if same_conference:
                    standings[home_team]['conference_ties'] += 1
                    standings[away_team]['conference_ties'] += 1
                    
        except (ValueError, TypeError):
            logger.error(f"Error processing game: {game}")
            continue
            
    return standings

def calculate_rating_adjustment(home_score, away_score, base_adjustment=0.5, rng=None):
    """
    Calculate rating adjustment based on margin of victory and randomness
    
    Args:
        home_score: Points scored by home team
        away_score: Points scored by away team
        base_adjustment: Base rating adjustment value
        
    Returns:
        float: Rating adjustment value
    """
    # Calculate margin of victory
    margin = abs(home_score - away_score)
    
    # Base adjustment increases with margin but caps out
    # Using log function to prevent huge swings from blowouts
    margin_factor = np.log(margin + 1) / 2.5  # +1 to handle 0 margin
    
    # Add some randomness (between 0.8 and 1.2 of calculated adjustment)
    rng = GLOBAL_RNG if rng is None else rng

    random_factor = rng.uniform(0.8, 1.2)
    
    # Calculate final adjustment
    adjustment = base_adjustment * margin_factor * random_factor

    # round to 2 decimal places
    adjustment = min(adjustment, base_adjustment * 2)
    adjustment = round(adjustment, 2)
    
    # Cap maximum adjustment at 2x base_adjustment
    return adjustment

def _simulate_single_season(schedule, teams, base_ratings, home_field_advantage, seed, include_schedule):
    rng = np.random.default_rng(seed)

    ratings = dict(base_ratings)
    standings = initialize_standings(schedule, teams)
    game_results = []
    schedule_output = [game.copy() for game in schedule] if include_schedule else None

    for game in schedule:
        if game['away_score'] and game['home_score']:
            game_results.append({
                'home_team': game['home_team'],
                'away_team': game['away_team'],
                'home_score': int(game['home_score']),
                'away_score': int(game['away_score'])
            })

    for game in schedule:
        if not game['away_score'] and not game['home_score']:
            home_team = game['home_team']
            away_team = game['away_team']
            result = simulate_game(home_team, away_team, ratings, home_field_advantage, rng=rng)

            adjustment = calculate_rating_adjustment(
                result['home_score'],
                result['away_score'],
                rng=rng
            )

            if result['home_win']:
                ratings[home_team] += adjustment
                ratings[away_team] -= adjustment
            else:
                ratings[away_team] += adjustment
                ratings[home_team] -= adjustment

            ratings[home_team] = round(ratings[home_team], 2)
            ratings[away_team] = round(ratings[away_team], 2)

            if result['home_win']:
                standings[home_team]['wins'] += 1
                standings[away_team]['losses'] += 1
                standings[home_team]['head_to_head'][away_team]['wins'] += 1
                standings[away_team]['head_to_head'][home_team]['losses'] += 1
            else:
                standings[away_team]['wins'] += 1
                standings[home_team]['losses'] += 1
                standings[home_team]['head_to_head'][away_team]['losses'] += 1
                standings[away_team]['head_to_head'][home_team]['wins'] += 1

            game_results.append({
                'home_team': home_team,
                'away_team': away_team,
                'home_score': result['home_score'],
                'away_score': result['away_score']
            })

            if include_schedule and schedule_output is not None:
                for game_entry in schedule_output:
                    if (game_entry['home_team'] == home_team and
                        game_entry['away_team'] == away_team):
                        game_entry['home_score'] = str(result['home_score'])
                        game_entry['away_score'] = str(result['away_score'])
                        break

    division_winners = determine_division_winners(standings, teams)

    wild_cards = determine_wild_cards_with_tiebreakers(standings, teams, division_winners, {
        'game_results': game_results
    })

    playoff_order = {}
    for conference in ['AFC', 'NFC']:
        conf_div_winners = [team for team in division_winners
                            if teams[team]['conference'] == conference]

        div_winner_groups = defaultdict(list)
        for team in conf_div_winners:
            win_pct = standings[team]['wins'] / (standings[team]['wins'] + standings[team]['losses'] + standings[team].get('ties', 0))
            div_winner_groups[win_pct].append(team)

        sorted_div_winners = []
        for win_pct in sorted(div_winner_groups.keys(), reverse=True):
            teams_in_group = div_winner_groups[win_pct]
            if len(teams_in_group) > 1:
                sorted_group = apply_tiebreakers(teams_in_group, standings)
                sorted_div_winners.extend(sorted_group)
            else:
                sorted_div_winners.extend(teams_in_group)

        conf_wild_cards = wild_cards.get(conference, [])

        logger.info(f"\n{conference} Final Playoff Seeds:")
        for i, team in enumerate(sorted_div_winners + conf_wild_cards, 1):
            record = standings[team]
            logger.info(f"#{i} Seed: {team} ({record['wins']}-{record['losses']}-{record.get('ties', 0)})")

        playoff_order[conference] = sorted_div_winners + conf_wild_cards

    afc_champion = determine_conference_champion('AFC', division_winners, wild_cards, standings, teams, ratings, home_field_advantage, rng=rng)
    nfc_champion = determine_conference_champion('NFC', division_winners, wild_cards, standings, teams, ratings, home_field_advantage, rng=rng)

    super_bowl_result = {
        'teams': [afc_champion, nfc_champion],
        'winner': simulate_super_bowl(afc_champion, nfc_champion, ratings, home_field_advantage, rng=rng)
    }

    result_payload = {
        'standings': standings,
        'division_winners': division_winners,
        'wild_cards': wild_cards,
        'playoff_order': playoff_order,
        'game_results': game_results,
        'super_bowl': super_bowl_result
    }

    return result_payload, schedule_output


def _simulate_single_season_wrapper(args):
    return _simulate_single_season(*args)


def simulate_season(
    test_schedule=None,
    num_simulations=1,
    home_field_advantage=2.5,
    random_seed=None,
    parallel=True,
    max_workers=None
):
    schedule = load_schedule()
    teams = load_teams()
    base_ratings = load_ratings()

    include_schedule = test_schedule is not None

    seed_sequence = np.random.SeedSequence(random_seed)
    child_seeds = seed_sequence.spawn(num_simulations)
    seed_values = [int(child.generate_state(1)[0]) for child in child_seeds] if num_simulations > 0 else []

    args_list = [
        (
            schedule,
            teams,
            base_ratings,
            home_field_advantage,
            seed_values[sim_idx],
            include_schedule and sim_idx == 0
        )
        for sim_idx in range(num_simulations)
    ]

    if parallel and num_simulations > 1:
        worker_count = min(max_workers or os.cpu_count() or 1, num_simulations)
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            outputs = list(executor.map(_simulate_single_season_wrapper, args_list))
    else:
        outputs = [_simulate_single_season_wrapper(args) for args in args_list]

    results = [payload for payload, _ in outputs]

    if include_schedule and test_schedule is not None and outputs:
        schedule_output = outputs[0][1]
        if schedule_output is not None:
            test_schedule.clear()
            test_schedule.extend(schedule_output)

    return results

def simulate_playoff_game(home_team, away_team, ratings, home_field_advantage, rng=None):
    """Simulate a playoff game with increased variance"""
    # Use higher volatility for playoff games
    return simulate_game(home_team, away_team, ratings, home_field_advantage, rng=rng)

def determine_conference_champion(conference, division_winners, wild_cards, standings, teams, ratings, home_field_advantage, rng=None):
    """
    Fix the playoff bracket to simulate wildcard -> divisional -> conference championship with the 7-team format.
    """
    # Gather all playoff teams from this conference
    playoff_teams = [t for t in division_winners if teams[t]['conference'] == conference]
    playoff_teams.extend(wild_cards.get(conference, []))

    # Sort them by best record: (wins, -losses, -ties) descending
    playoff_teams.sort(
        key=lambda t: (
            standings[t]['wins'],
            -standings[t]['losses'],
            -standings[t].get('ties', 0)
        ),
        reverse=True
    )

    # Expect 7 teams in the bracket
    # 1) Wildcard round:
    #    - #1 gets a bye
    #    - (2 vs 7), (3 vs 6), (4 vs 5)
    seed1 = playoff_teams[0]
    wc_pairs = [
        (playoff_teams[1], playoff_teams[6]),
        (playoff_teams[2], playoff_teams[5]),
        (playoff_teams[3], playoff_teams[4]),
    ]

    wildcard_winners = []
    for home, away in wc_pairs:
        result = simulate_playoff_game(home, away, ratings, home_field_advantage, rng=rng)
        winner = home if result['home_win'] else away
        wildcard_winners.append(winner)

    # 2) Divisional round:
    #    - #1 seed vs lowest remaining wildcard
    #    - The other two wildcard winners face each other
    # Sort the wildcard winners again by record:
    wildcard_winners.sort(
        key=lambda t: (
            standings[t]['wins'],
            -standings[t]['losses'],
            -standings[t].get('ties', 0)
        ),
        reverse=True
    )

    # #1 seed hosts the lowest seed of the wildcard winners (i.e. last in this sorted list)
    dr_game1_home = seed1
    dr_game1_away = wildcard_winners[-1]
    result1 = simulate_playoff_game(dr_game1_home, dr_game1_away, ratings, home_field_advantage, rng=rng)
    dr_winner1 = dr_game1_home if result1['home_win'] else dr_game1_away

    # The other two wildcard winners face off
    dr_game2_home = wildcard_winners[0]
    dr_game2_away = wildcard_winners[1]
    result2 = simulate_playoff_game(dr_game2_home, dr_game2_away, ratings, home_field_advantage, rng=rng)
    dr_winner2 = dr_game2_home if result2['home_win'] else dr_game2_away

    # 3) Conference championship round:
    cc_home = dr_winner1
    cc_away = dr_winner2
    cc_result = simulate_playoff_game(cc_home, cc_away, ratings, home_field_advantage, rng=rng)
    conference_champion = cc_home if cc_result['home_win'] else cc_away

    return conference_champion

def simulate_super_bowl(afc_team, nfc_team, ratings, home_field_advantage, rng=None):
    """Simulate Super Bowl game"""
    # Neutral field, so use half home field advantage
    result = simulate_game(afc_team, nfc_team, ratings, home_field_advantage / 2, rng=rng)
    return afc_team if result['home_win'] else nfc_team
