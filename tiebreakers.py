import logging
import os
import sys
from playoff_utils import load_teams

# Set up logger
logger = logging.getLogger(__name__)

# Create logs directory if it doesn't exist
if not os.path.exists('logs'):
    os.makedirs('logs')

# Set up file handler for errors only
file_handler = logging.FileHandler('logs/tiebreakers.log')
file_handler.setLevel(logging.ERROR)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add handler to logger
logger.addHandler(file_handler)
logger.setLevel(logging.ERROR)

# Force logger to propagate messages
logger.propagate = True

teams = load_teams()

def get_h2h_wins(head_to_head, team1, team2):
    """Helper function to get head-to-head wins handling both data structures"""
    h2h_data = head_to_head.get(team1, {}).get(team2)
    if isinstance(h2h_data, dict):
        return h2h_data.get('wins', 0)
    return h2h_data or 0

def calculate_win_pct(team_record):
    """Calculate win percentage from a team record dict"""
    total_games = team_record['wins'] + team_record['losses'] + team_record['ties']
    if total_games == 0:
        return 0.0
    return (team_record['wins'] + 0.5 * team_record['ties']) / total_games

def calculate_strength_of_victory(team, standings):
    """Calculate strength of victory - win percentage of teams you've beaten"""
    if 'defeated_opponents' not in standings[team]:
        return 0.0
        
    defeated = standings[team]['defeated_opponents']
    if not defeated:
        return 0.0
        
    total_wins = sum(standings[opp]['wins'] for opp in defeated)
    total_losses = sum(standings[opp]['losses'] for opp in defeated)
    total_ties = sum(standings[opp].get('ties', 0) for opp in defeated)
    
    total_games = total_wins + total_losses + total_ties
    if total_games == 0:
        return 0.0
        
    return (total_wins + 0.5 * total_ties) / total_games

def calculate_strength_of_schedule(team, standings):
    """Calculate strength of schedule - win percentage of all opponents"""
    if 'opponents' not in standings[team]:
        return 0.0
        
    opponents = standings[team]['opponents']
    if not opponents:
        return 0.0
        
    total_wins = sum(standings[opp]['wins'] for opp in opponents)
    total_losses = sum(standings[opp]['losses'] for opp in opponents)
    total_ties = sum(standings[opp].get('ties', 0) for opp in opponents)
    
    total_games = total_wins + total_losses + total_ties
    if total_games == 0:
        return 0.0
        
    return (total_wins + 0.5 * total_ties) / total_games

def head_to_head_pct(team1, team2, head_to_head):
    """Calculate head-to-head win percentage between two teams"""
    if not head_to_head or team1 not in head_to_head or team2 not in head_to_head[team1]:
        return None
    
    wins = head_to_head[team1][team2]
    losses = head_to_head[team2][team1]
    total = wins + losses
    
    if total == 0:
        return None
        
    return wins / total

def common_games_pct(team1, team2, standings):
    """Calculate win percentage in common games between two teams"""
    if 'common_games' not in standings[team1] or 'common_games' not in standings[team2]:
        return None
        
    # Get list of common opponents
    team1_opponents = set(standings[team1]['common_games'].keys())
    team2_opponents = set(standings[team2]['common_games'].keys())
    common_opponents = team1_opponents & team2_opponents
    
    if len(common_opponents) < 4:  # NFL requires minimum 4 common games
        return None
    
    # Calculate records in common games
    team1_record = {'wins': 0, 'losses': 0, 'ties': 0}
    team2_record = {'wins': 0, 'losses': 0, 'ties': 0}
    
    for opponent in common_opponents:
        team1_games = standings[team1]['common_games'][opponent]
        team2_games = standings[team2]['common_games'][opponent]
        
        team1_record['wins'] += team1_games['wins']
        team1_record['losses'] += team1_games['losses']
        team1_record['ties'] += team1_games['ties']
        
        team2_record['wins'] += team2_games['wins']
        team2_record['losses'] += team2_games['losses']
        team2_record['ties'] += team2_games['ties']
    
    return calculate_win_pct(team1_record), calculate_win_pct(team2_record)

def calculate_points_ranking(team, standings, conference_only=False):
    """Calculate combined ranking in points scored and points allowed"""
    if 'points_for' not in standings[team] or 'points_against' not in standings[team]:
        return float('inf')  # Return infinity if data not available
        
    # Get all teams in same conference if conference_only=True
    if conference_only:
        teams_to_compare = [t for t, info in standings.items() 
                           if ('conference' in info and info['conference'] == standings[team]['conference'])]
    else:
        teams_to_compare = list(standings.keys())
    
    # Calculate points scored ranking
    points_scored_rank = sum(1 for other in teams_to_compare
                           if standings[other].get('points_for', 0) > standings[team]['points_for'])
    
    # Calculate points allowed ranking
    points_allowed_rank = sum(1 for other in teams_to_compare
                            if standings[other].get('points_against', 0) < standings[team]['points_against'])
    
    return points_scored_rank + points_allowed_rank

def calculate_net_points(team, standings, common_opponents=None):
    """Calculate net points (points scored - points allowed) in specified games"""
    if 'points_for' not in standings[team] or 'points_against' not in standings[team]:
        return 0
        
    if common_opponents:
        # Only count games against common opponents
        points_for = sum(standings[team].get('common_games', {}).get(opp, {}).get('points_for', 0) 
                        for opp in common_opponents)
        points_against = sum(standings[team].get('common_games', {}).get(opp, {}).get('points_against', 0) 
                           for opp in common_opponents)
    else:
        # All games
        points_for = standings[team]['points_for']
        points_against = standings[team]['points_against']
    
    return points_for - points_against

def head_to_head_sweep(teams, head_to_head):
    """
    Check if any team has beaten all other teams in the group.
    
    Args:
        teams: List of team abbreviations
        head_to_head: Dict of head-to-head results in either format
    
    Returns:
        Team abbreviation of sweep winner, or None if no sweep exists
    """
    if not head_to_head:
        return None
        
    for team in teams:
        beats_all = True
        for other in teams:
            if team != other:
                team_wins = get_h2h_wins(head_to_head, team, other)
                other_wins = get_h2h_wins(head_to_head, other, team)
                
                # If team doesn't have a winning record against any other team,
                # they can't have swept the group
                if team_wins <= other_wins:
                    beats_all = False
                    break
        
        if beats_all:
            return team
            
    return None

def apply_tiebreakers(teams_list, standings, teams_dict=None, division=None, return_explanations=False, force_division_rules=False, head_to_head=None):
    """
    Apply NFL tiebreakers to rank a list of teams
    """
    if len(teams_list) <= 1:
        if return_explanations:
            return [(team, None) for team in teams_list]
        return teams_list
        
    explanations = {team: [] for team in teams_list}
    remaining_teams = teams_list.copy()
    final_order = []
    
    while remaining_teams:
        if len(remaining_teams) == 1:
            final_order.append((remaining_teams[0], None) if return_explanations else remaining_teams[0])
            break
            
        # Head-to-head sweep check
        sweep_winner = head_to_head_sweep(remaining_teams, head_to_head)
        if sweep_winner:
            explanations[sweep_winner] = "Head-to-head vs all tied teams"
            final_order.append((sweep_winner, explanations[sweep_winner]) if return_explanations else sweep_winner)
            remaining_teams.remove(sweep_winner)
            continue
            
        # Division record
        if division:
            def div_win_pct(record):
                wins, losses, ties = record
                total = wins + losses + ties
                if total == 0:
                    return 0.0
                return (wins + 0.5 * ties) / total

            div_records = [(t, (standings[t].get('division_wins', 0),
                              standings[t].get('division_losses', 0),
                              standings[t].get('division_ties', 0))) for t in remaining_teams]

            best_div_pct = max(div_win_pct(r[1]) for r in div_records)
            same_record_teams = [t for t, r in div_records if div_win_pct(r) == best_div_pct]

            if len(same_record_teams) < len(remaining_teams):
                # Set explanations for the teams with better division record
                if return_explanations and len(same_record_teams) == 1:
                    team = same_record_teams[0]
                    rec = next(r for t, r in div_records if t == team)
                    explanations[team] = f"Better division record ({rec[0]}-{rec[1]}-{rec[2]})"

                # Split into best and remaining, recursively apply tiebreakers to both
                worse_teams = [t for t in remaining_teams if t not in same_record_teams]
                better_result = apply_tiebreakers(same_record_teams, standings, teams_dict, division, return_explanations, force_division_rules, head_to_head)
                worse_result = apply_tiebreakers(worse_teams, standings, teams_dict, division, return_explanations, force_division_rules, head_to_head)

                # Merge explanations if needed
                if return_explanations:
                    # Update better_result with division record explanation if we set it
                    if len(same_record_teams) == 1:
                        better_result = [(better_result[0][0], explanations[better_result[0][0]])]

                final_order.extend(better_result)
                final_order.extend(worse_result)
                break

        # Conference record
        def conf_win_pct(record):
            wins, losses, ties = record
            total = wins + losses + ties
            if total == 0:
                return 0.0
            return (wins + 0.5 * ties) / total

        conf_records = [(t, (standings[t].get('conference_wins', 0),
                           standings[t].get('conference_losses', 0),
                           standings[t].get('conference_ties', 0))) for t in remaining_teams]

        best_conf_pct = max(conf_win_pct(r[1]) for r in conf_records)
        same_record_teams = [t for t, r in conf_records if conf_win_pct(r) == best_conf_pct]

        if len(same_record_teams) < len(remaining_teams):
            # Set explanations for the teams with better conference record
            if return_explanations and len(same_record_teams) == 1:
                team = same_record_teams[0]
                rec = next(r for t, r in conf_records if t == team)
                explanations[team] = f"Better conference record ({rec[0]}-{rec[1]}-{rec[2]})"

            # Split into best and remaining, recursively apply tiebreakers to both
            worse_teams = [t for t in remaining_teams if t not in same_record_teams]
            better_result = apply_tiebreakers(same_record_teams, standings, teams_dict, division, return_explanations, force_division_rules, head_to_head)
            worse_result = apply_tiebreakers(worse_teams, standings, teams_dict, division, return_explanations, force_division_rules, head_to_head)

            # Merge explanations if needed
            if return_explanations:
                # Update better_result with conference record explanation if we set it
                if len(same_record_teams) == 1:
                    better_result = [(better_result[0][0], explanations[better_result[0][0]])]

            final_order.extend(better_result)
            final_order.extend(worse_result)
            break
            
        # Strength of Victory
        sov_values = [(t, calculate_strength_of_victory(t, standings)) for t in remaining_teams]
        best_sov = max(sov_values, key=lambda x: x[1])[1]
        best_sov_teams = [t for t, v in sov_values if v == best_sov]

        if len(best_sov_teams) < len(remaining_teams):
            # Set explanations for the teams with better SOV
            if return_explanations and len(best_sov_teams) == 1:
                team = best_sov_teams[0]
                explanations[team] = f"Better strength of victory ({best_sov:.3f})"

            # Split into best and remaining, recursively apply tiebreakers to both
            worse_teams = [t for t in remaining_teams if t not in best_sov_teams]
            better_result = apply_tiebreakers(best_sov_teams, standings, teams_dict, division, return_explanations, force_division_rules, head_to_head)
            worse_result = apply_tiebreakers(worse_teams, standings, teams_dict, division, return_explanations, force_division_rules, head_to_head)

            # Merge explanations if needed
            if return_explanations:
                # Update better_result with SOV explanation if we set it
                if len(best_sov_teams) == 1:
                    better_result = [(better_result[0][0], explanations[better_result[0][0]])]

            final_order.extend(better_result)
            final_order.extend(worse_result)
            break

        # Strength of Schedule
        sos_values = [(t, calculate_strength_of_schedule(t, standings)) for t in remaining_teams]
        best_sos = max(sos_values, key=lambda x: x[1])[1]
        best_sos_teams = [t for t, v in sos_values if v == best_sos]

        if len(best_sos_teams) < len(remaining_teams):
            # Set explanations for the teams with better SOS
            if return_explanations and len(best_sos_teams) == 1:
                team = best_sos_teams[0]
                explanations[team] = f"Better strength of schedule ({best_sos:.3f})"

            # Split into best and remaining, recursively apply tiebreakers to both
            worse_teams = [t for t in remaining_teams if t not in best_sos_teams]
            better_result = apply_tiebreakers(best_sos_teams, standings, teams_dict, division, return_explanations, force_division_rules, head_to_head)
            worse_result = apply_tiebreakers(worse_teams, standings, teams_dict, division, return_explanations, force_division_rules, head_to_head)

            # Merge explanations if needed
            if return_explanations:
                # Update better_result with SOS explanation if we set it
                if len(best_sos_teams) == 1:
                    better_result = [(better_result[0][0], explanations[better_result[0][0]])]

            final_order.extend(better_result)
            final_order.extend(worse_result)
            break

        # If no tiebreaker has resolved yet, add all remaining teams in current order
        final_order.extend((team, None) if return_explanations else team for team in remaining_teams)
        break
    
    return final_order

def apply_wildcard_tiebreakers(teams_list, standings, teams_dict, head_to_head=None, return_explanations=False):
    """Apply the specific NFL wildcard tiebreaker rules"""
    logger.info(f"\nStarting wildcard tiebreaker process for teams: {teams_list}")
    logger.info("Records:")
    for team in teams_list:
        logger.info(f"{team}: {standings[team]['wins']}-{standings[team]['losses']}-{standings[team].get('ties', 0)}")

    # Add head-to-head logging
    logger.info("\nHead-to-head records:")
    for team1 in teams_list:
        for team2 in teams_list:
            if team1 != team2:
                wins = (head_to_head.get(team1, {}).get(team2, {}).get('wins', 0)
                       if isinstance(head_to_head.get(team1, {}).get(team2), dict)
                       else head_to_head.get(team1, {}).get(team2, 0))
                logger.info(f"{team1} vs {team2}: {wins} wins")

    # Add conference record logging
    logger.info("\nConference records:")
    for team in teams_list:
        conf_wins = standings[team].get('conference_wins', 0)
        conf_losses = standings[team].get('conference_losses', 0)
        conf_ties = standings[team].get('conference_ties', 0)
        logger.info(f"{team}: {conf_wins}-{conf_losses}-{conf_ties}")

    # Add logging for NFL tiebreaker steps
    logger.info("\nApplying NFL tiebreaker steps in order:")
    logger.info("1. Head-to-head sweep among all tied teams")
    logger.info("2. Conference record among all tied teams")
    logger.info("3. Division tiebreakers (only if above steps don't resolve)")

    if len(teams_list) <= 1:
        if return_explanations:
            return [(team, None) for team in teams_list]
        return teams_list
    
    # For exactly two teams, check head-to-head record FIRST
    if len(teams_list) == 2 and head_to_head:
        team1, team2 = teams_list
        # Handle both possible head-to-head data structures
        h2h_wins1 = get_h2h_wins(head_to_head, team1, team2)
        h2h_wins2 = get_h2h_wins(head_to_head, team2, team1)

        if h2h_wins1 > h2h_wins2:
            if return_explanations:
                return [(team1, f"Head-to-head vs {team2}"), (team2, None)]
            return [team1, team2]
        elif h2h_wins2 > h2h_wins1:
            if return_explanations:
                return [(team2, f"Head-to-head vs {team1}"), (team1, None)]
            return [team2, team1]

    # Check head-to-head sweep for ALL teams
    logger.info("\nChecking for head-to-head sweep among all teams")
    sweep_winner = head_to_head_sweep(teams_list, head_to_head)
    if sweep_winner:
        logger.info(f"{sweep_winner} has beaten all other teams")
        remaining = [t for t in teams_list if t != sweep_winner]
        remaining_result = apply_wildcard_tiebreakers(remaining, standings, teams_dict, head_to_head, return_explanations)
        if return_explanations:
            return [(sweep_winner, "Head-to-head sweep")] + remaining_result
        return [sweep_winner] + remaining_result
    logger.info("No team has beaten all others")

    # Check head-to-head win percentage among all teams
    # NOTE: Head-to-head only applies if ALL teams have played each other (complete round-robin)
    logger.info("\nChecking head-to-head win percentage among all teams")
    h2h_records = {}
    all_teams_played_each_other = True

    for team in teams_list:
        wins = sum(get_h2h_wins(head_to_head, team, other)
                  for other in teams_list if other != team)
        losses = sum(get_h2h_wins(head_to_head, other, team)
                    for other in teams_list if other != team)

        # Check if this team has played all other teams in the group
        if wins + losses < len(teams_list) - 1:
            all_teams_played_each_other = False
            logger.info(f"{team} has not played all other teams in group ({wins + losses} games vs {len(teams_list) - 1} needed)")

        if wins + losses > 0:  # Only count if they played at least one game
            h2h_records[team] = wins / (wins + losses)
            logger.info(f"{team} head-to-head win pct: {h2h_records[team]:.3f} ({wins}-{losses})")

    # Only use head-to-head if ALL teams have played each other
    if h2h_records and all_teams_played_each_other:
        best_pct = max(h2h_records.values())
        best_teams = [t for t, pct in h2h_records.items() if pct == best_pct]

        if len(best_teams) < len(teams_list):
            logger.info(f"Teams with best head-to-head record: {best_teams}")
            remaining = [t for t in teams_list if t not in best_teams]
            best_result = apply_wildcard_tiebreakers(best_teams, standings, teams_dict, head_to_head, return_explanations)
            remaining_result = apply_wildcard_tiebreakers(remaining, standings, teams_dict, head_to_head, return_explanations)
            if return_explanations:
                # Add explanation only to the first team in best group if it's alone
                if len(best_teams) == 1:
                    best_result = [(best_teams[0], f"Better head-to-head win % ({best_pct:.3f})")]
            return best_result + remaining_result

    if not all_teams_played_each_other:
        logger.info("Head-to-head does not apply (not all teams have played each other)")
    else:
        logger.info("Head-to-head did not break tie")

    # Check conference records for ALL teams before division grouping
    logger.info("\nChecking conference records for all teams:")
    conf_records = [(t, calculate_win_pct({
        'wins': standings[t].get('conference_wins', 0),
        'losses': standings[t].get('conference_losses', 0),
        'ties': standings[t].get('conference_ties', 0)
    })) for t in teams_list]

    best_conf_pct = max(r[1] for r in conf_records)
    best_conf_teams = [t for t, pct in conf_records if pct == best_conf_pct]

    if len(best_conf_teams) < len(teams_list):
        logger.info(f"Teams with best conference record: {best_conf_teams}")
        remaining = [t for t in teams_list if t not in best_conf_teams]
        # Recursively apply tiebreakers to BOTH groups, not just remaining
        best_result = apply_wildcard_tiebreakers(best_conf_teams, standings, teams_dict, head_to_head, return_explanations)
        remaining_result = apply_wildcard_tiebreakers(remaining, standings, teams_dict, head_to_head, return_explanations)
        if return_explanations:
            # Add explanation only to the first team in best group if it's alone
            if len(best_conf_teams) == 1:
                team = best_conf_teams[0]
                rec = next(r for t, r in conf_records if t == team)
                conf_wins = standings[team].get('conference_wins', 0)
                conf_losses = standings[team].get('conference_losses', 0)
                conf_ties = standings[team].get('conference_ties', 0)
                best_result = [(team, f"Better conference record ({conf_wins}-{conf_losses}-{conf_ties})")]
        return best_result + remaining_result
    logger.info("Conference records did not break tie")

    # For wildcard seeding, we do NOT apply division tiebreakers to teams from the same division
    # Instead, all teams compete using wildcard tiebreaker rules
    # The division grouping logic was incorrectly applying division tiebreakers
    remaining_teams = teams_list[:]
    logger.info(f"\nStarting wildcard tiebreaker with teams: {remaining_teams}")

    # Track explanations for each team
    explanations = {team: None for team in teams_list}
    # Track pending explanation for teams that advanced together
    pending_explanation = None

    # Now apply wildcard tiebreakers to the highest ranked teams
    final_order = []
    while remaining_teams:
        if len(remaining_teams) == 1:
            # Last team - use pending explanation if available
            if return_explanations and pending_explanation:
                team = remaining_teams[0]
                if not explanations.get(team):
                    explanations[team] = pending_explanation
                pending_explanation = None
            final_order.extend(remaining_teams)
            break
            
        # For exactly two teams, check head-to-head record first
        if len(remaining_teams) == 2 and head_to_head:
            team1, team2 = remaining_teams
            h2h_wins1 = get_h2h_wins(head_to_head, team1, team2)
            h2h_wins2 = get_h2h_wins(head_to_head, team2, team1)

            logger.info(f"\nTwo-team head-to-head check: {team1} vs {team2}")
            logger.info(f"{team1} h2h wins: {h2h_wins1}, {team2} h2h wins: {h2h_wins2}")

            if h2h_wins1 > h2h_wins2:
                logger.info(f"{team1} wins head-to-head, placing [{team1}, {team2}]")
                if return_explanations:
                    explanations[team1] = f"Head-to-head vs {team2}"
                    explanations[team2] = f"Head-to-head vs {team1}"
                final_order.extend([team1, team2])
                break
            elif h2h_wins2 > h2h_wins1:
                logger.info(f"{team2} wins head-to-head, placing [{team2}, {team1}]")
                if return_explanations:
                    explanations[team2] = f"Head-to-head vs {team1}"
                    explanations[team1] = f"Head-to-head vs {team2}"
                final_order.extend([team2, team1])
                break
            else:
                logger.info("Head-to-head is tied or no games played, continuing to other tiebreakers")
                # Don't break here - continue to apply other tiebreakers below

        # Head-to-head sweep check (for 3+ teams)
        sweep_winner = head_to_head_sweep(remaining_teams, head_to_head)
        if sweep_winner:
            if return_explanations:
                explanations[sweep_winner] = "Head-to-head sweep"
            final_order.append(sweep_winner)
            remaining_teams.remove(sweep_winner)
            continue
        
        # For 3+ teams, check head-to-head first
        if len(remaining_teams) > 2 and head_to_head:
            # Create head-to-head matrix
            h2h_records = {}
            for team in remaining_teams:
                wins = sum(get_h2h_wins(head_to_head, team, other)
                          for other in remaining_teams if other != team)
                losses = sum(get_h2h_wins(head_to_head, other, team)
                            for other in remaining_teams if other != team)
                if wins + losses > 0:  # Only count if they played each other
                    h2h_records[team] = wins / (wins + losses)

            if h2h_records:  # If we have head-to-head games
                best_pct = max(h2h_records.values())
                best_teams = [t for t, pct in h2h_records.items() if pct == best_pct]

                if len(best_teams) < len(remaining_teams):
                    if len(best_teams) == 1:
                        if return_explanations:
                            explanations[best_teams[0]] = f"Better head-to-head win % ({best_pct:.3f})"
                        final_order.extend(best_teams)
                        remaining_teams = [t for t in remaining_teams if t not in best_teams]
                        continue
                    else:
                        # Multiple teams tied - store pending explanation
                        if return_explanations:
                            for team in best_teams:
                                if not explanations.get(team):
                                    explanations[team] = f"Better head-to-head win % ({best_pct:.3f})"
                        remaining_teams = best_teams
                        continue
        
        # Conference record
        conf_records = [
            (t, calculate_win_pct({
                'wins': standings[t].get('conference_wins', 0),
                'losses': standings[t].get('conference_losses', 0),
                'ties': standings[t].get('conference_ties', 0)
            }))
            for t in remaining_teams
        ]

        best_pct = max(r[1] for r in conf_records)
        best_teams = [t for t, r in conf_records if r == best_pct]

        if len(best_teams) < len(remaining_teams):
            # This tiebreaker separated teams - best_teams beat the rest
            if len(best_teams) == 1:
                # Single winner
                if return_explanations:
                    team = best_teams[0]
                    conf_wins = standings[team].get('conference_wins', 0)
                    conf_losses = standings[team].get('conference_losses', 0)
                    conf_ties = standings[team].get('conference_ties', 0)
                    explanations[team] = f"Better conference record ({conf_wins}-{conf_losses}-{conf_ties})"
                    # If this is a 2-team tiebreaker, also set explanation for the loser
                    if len(remaining_teams) == 2:
                        loser = [t for t in remaining_teams if t not in best_teams][0]
                        loser_conf_wins = standings[loser].get('conference_wins', 0)
                        loser_conf_losses = standings[loser].get('conference_losses', 0)
                        loser_conf_ties = standings[loser].get('conference_ties', 0)
                        explanations[loser] = f"Better conference record ({conf_wins}-{conf_losses}-{conf_ties})"
                final_order.extend(best_teams)
                remaining_teams = [t for t in remaining_teams if t not in best_teams]
                continue
            else:
                # Multiple teams tied - store explanation for all of them
                # They all beat the teams that were dropped
                if return_explanations:
                    for team in best_teams:
                        if not explanations.get(team):
                            conf_wins = standings[team].get('conference_wins', 0)
                            conf_losses = standings[team].get('conference_losses', 0)
                            conf_ties = standings[team].get('conference_ties', 0)
                            explanations[team] = f"Better conference record ({conf_wins}-{conf_losses}-{conf_ties})"
                # Continue to next tiebreaker to sort these teams
                remaining_teams = best_teams
                continue
        
        # Common games (minimum of 4)
        if len(remaining_teams) >= 2:
            common_games_results = []
            for team in remaining_teams:
                wins = losses = ties = 0
                common_opponents = set(standings[team]['opponents'])
                for other in remaining_teams:
                    if other != team:
                        common_opponents &= set(standings[other]['opponents'])

                if len(common_opponents) >= 4:
                    # Check if standings has common_games field
                    if 'common_games' not in standings[team]:
                        break

                    for opp in common_opponents:
                        games = standings[team]['common_games'].get(opp, {})
                        wins += games.get('wins', 0)
                        losses += games.get('losses', 0)
                        ties += games.get('ties', 0)

                    pct = calculate_win_pct({'wins': wins, 'losses': losses, 'ties': ties})
                    common_games_results.append((team, pct))

            if common_games_results and len(common_games_results) == len(remaining_teams):
                best_pct = max(r[1] for r in common_games_results)
                best_teams = [t for t, pct in common_games_results if pct == best_pct]

                if len(best_teams) < len(remaining_teams):
                    if len(best_teams) == 1:
                        if return_explanations:
                            explanations[best_teams[0]] = f"Better common games record ({best_pct:.3f})"
                        final_order.extend(best_teams)
                        remaining_teams = [t for t in remaining_teams if t not in best_teams]
                        continue
                    else:
                        # Multiple teams tied - store pending explanation
                        if return_explanations:
                            for team in best_teams:
                                if not explanations.get(team):
                                    explanations[team] = f"Better common games record ({best_pct:.3f})"
                        remaining_teams = best_teams
                        continue

        # Strength of victory
        sov_values = [(t, calculate_strength_of_victory(t, standings)) for t in remaining_teams]
        best_sov = max(sov_values, key=lambda x: x[1])[1]
        best_sov_teams = [t for t, v in sov_values if v == best_sov]

        if len(best_sov_teams) < len(remaining_teams):
            if len(best_sov_teams) == 1:
                if return_explanations:
                    explanations[best_sov_teams[0]] = f"Better strength of victory ({best_sov:.3f})"
                    # If this is a 2-team tiebreaker, also set explanation for the loser
                    if len(remaining_teams) == 2:
                        loser = [t for t in remaining_teams if t not in best_sov_teams][0]
                        explanations[loser] = f"Better strength of victory ({best_sov:.3f})"
                final_order.extend(best_sov_teams)
                remaining_teams = [t for t in remaining_teams if t not in best_sov_teams]
                continue
            else:
                # Multiple teams tied - store pending explanation
                if return_explanations:
                    for team in best_sov_teams:
                        if not explanations.get(team):
                            explanations[team] = f"Better strength of victory ({best_sov:.3f})"
                remaining_teams = best_sov_teams
                continue

        # Strength of Schedule
        sos_values = [(t, calculate_strength_of_schedule(t, standings)) for t in remaining_teams]
        best_sos = max(sos_values, key=lambda x: x[1])[1]
        best_sos_teams = [t for t, v in sos_values if v == best_sos]

        logger.info(f"\nStrength of Schedule tiebreaker:")
        for team, sos in sos_values:
            logger.info(f"{team}: SOS = {sos:.4f}")
        logger.info(f"Best SOS: {best_sos:.4f}, teams: {best_sos_teams}")

        if len(best_sos_teams) < len(remaining_teams):
            logger.info(f"SOS broke tie, advancing: {best_sos_teams}")
            if len(best_sos_teams) == 1:
                if return_explanations:
                    explanations[best_sos_teams[0]] = f"Better strength of schedule ({best_sos:.3f})"
                    # If this is a 2-team tiebreaker, also set explanation for the loser
                    if len(remaining_teams) == 2:
                        loser = [t for t in remaining_teams if t not in best_sos_teams][0]
                        explanations[loser] = f"Better strength of schedule ({best_sos:.3f})"
                final_order.extend(best_sos_teams)
                remaining_teams = [t for t in remaining_teams if t not in best_sos_teams]
                continue
            else:
                # Multiple teams tied - store pending explanation
                if return_explanations:
                    for team in best_sos_teams:
                        if not explanations.get(team):
                            explanations[team] = f"Better strength of schedule ({best_sos:.3f})"
                remaining_teams = best_sos_teams
                continue

        # Points differential as final tiebreaker
        sorted_remaining = sorted(remaining_teams,
                                key=lambda t: standings[t]['points_for'] - standings[t]['points_against'],
                                reverse=True)
        if return_explanations and len(sorted_remaining) > 0:
            # Only add explanation for the first team if there are multiple
            if len(sorted_remaining) > 1:
                best_team = sorted_remaining[0]
                diff = standings[best_team]['points_for'] - standings[best_team]['points_against']
                explanations[best_team] = f"Better point differential ({diff:+d})"
        final_order.extend(sorted_remaining)
        break

    # Return final order
    logger.info(f"Final wildcard tiebreaker order: {final_order}")
    if return_explanations:
        return [(team, explanations.get(team)) for team in final_order]
    return final_order