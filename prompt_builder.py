"""
Shared module for building AI analysis prompts.
This ensures consistency between cache generation and regeneration.
"""
import random
import json
from datetime import datetime


def build_team_stats_json(team_abbr, stats_row, teams_dict):
    """
    Convert team stats DataFrame row to structured JSON.

    Args:
        team_abbr: Team abbreviation (e.g., 'MIN')
        stats_row: Pandas Series with team stats
        teams_dict: Dictionary of team info (from load_teams())

    Returns:
        dict: Hierarchically structured team statistics
    """
    team_info = teams_dict[team_abbr]

    return {
        "team": {
            "abbr": team_abbr,
            "name": f"{team_info['city']} {team_info['mascot']}",
            "city": team_info['city'],
            "mascot": team_info['mascot'],
            "conference": team_info['conference'],
            "division": team_info['division']
        },
        "record": {
            "wins": int(stats_row['wins']),
            "losses": int(stats_row['losses']),
            "ties": int(stats_row['ties']),
            "win_pct": round(float(stats_row['win_pct']) * 100, 1),
            "games_played": int(stats_row['games_played']),
            "points_for": int(stats_row['points_for']),
            "points_against": int(stats_row['points_against']),
            "point_diff": int(stats_row['point_diff']),
            "streak": str(stats_row['streak_display']),
            "playoff_seed": int(stats_row['playoff_seed']) if stats_row['playoff_seed'] else None,
            "clincher": str(stats_row['clincher']) if stats_row['clincher'] else None,
            "division_record": str(stats_row['div_record']),
            "conference_record": str(stats_row['conf_record']),
            "road_record": str(stats_row['road_record'])
        },
        "offense": {
            "scoring": {
                "points_per_game": round(float(stats_row['points_per_game']), 1)
            },
            "passing": {
                "yards": int(stats_row['passing_yards']),
                "touchdowns": int(stats_row['passing_tds']),
                "interceptions": int(stats_row['interceptions']),
                "completion_pct": round(float(stats_row['completion_pct']), 1),
                "yards_per_attempt": round(float(stats_row['yards_per_attempt']), 2),
                "passer_rating": float(stats_row['passer_rating']),
                "sacks_taken": int(stats_row['sacks_taken']),
                "first_downs": int(stats_row['passing_first_downs'])
            },
            "rushing": {
                "yards": int(stats_row['rushing_yards']),
                "touchdowns": int(stats_row['rushing_tds']),
                "yards_per_carry": round(float(stats_row['yards_per_carry']), 2),
                "first_downs": int(stats_row['rushing_first_downs'])
            },
            "total": {
                "yards": int(stats_row['total_yards']),
                "yards_per_game": round(float(stats_row['yards_per_game']), 1),
                "first_downs": int(stats_row['total_first_downs']),
                "first_downs_per_game": round(float(stats_row['first_downs_per_game']), 1)
            },
            "efficiency": {
                "third_down_attempts": int(stats_row['third_down_attempts']),
                "third_down_conversions": int(stats_row['third_down_conversions']),
                "third_down_pct": round(float(stats_row['third_down_pct']), 1),
                "fourth_down_attempts": int(stats_row['fourth_down_attempts']),
                "fourth_down_conversions": int(stats_row['fourth_down_conversions']),
                "fourth_down_pct": round(float(stats_row['fourth_down_pct']), 1),
                "red_zone_attempts": int(stats_row['red_zone_trips']),
                "red_zone_touchdowns": int(stats_row['red_zone_tds']),
                "red_zone_td_pct": round(float(stats_row['red_zone_pct']), 1)
            },
            "turnovers": {
                "total": int(stats_row['total_turnovers']),
                "turnover_margin": int(stats_row['turnover_margin'])
            }
        },
        "defense": {
            "scoring": {
                "points_allowed_per_game": round(float(stats_row['points_against_per_game']), 1)
            },
            "efficiency": {
                "third_down_attempts_against": int(stats_row['third_down_attempts_against']),
                "third_down_conversions_against": int(stats_row['third_down_conversions_against']),
                "third_down_pct_against": round(float(stats_row['third_down_pct_against']), 1),
                "fourth_down_attempts_against": int(stats_row['fourth_down_attempts_against']),
                "fourth_down_conversions_against": int(stats_row['fourth_down_conversions_against']),
                "fourth_down_pct_against": round(float(stats_row['fourth_down_pct_against']), 1),
                "red_zone_attempts_against": int(stats_row['red_zone_trips_against']),
                "red_zone_touchdowns_against": int(stats_row['red_zone_tds_against']),
                "red_zone_td_pct_against": round(float(stats_row['red_zone_pct_against']), 1)
            }
        },
        "advanced": {
            "epa": {
                "total": round(float(stats_row['total_epa']), 1),
                "per_game": round(float(stats_row['epa_per_game']), 2),
                "passing": round(float(stats_row['passing_epa']), 1),
                "rushing": round(float(stats_row['rushing_epa']), 1)
            }
        }
    }


def build_injuries_json(team_injuries, opponent_injuries):
    """Convert injury lists to structured JSON."""
    def format_injuries(injury_list):
        if not injury_list:
            return []
        status_map = {'INA': 'Inactive', 'OUT': 'Out', 'DOUBTFUL': 'Doubtful', 'QUESTIONABLE': 'Questionable'}
        return [
            {
                "name": name,
                "position": pos,
                "status": status_map.get(status, status)
            }
            for name, pos, status in injury_list
        ]

    return {
        "team": format_injuries(team_injuries),
        "opponent": format_injuries(opponent_injuries)
    }


def build_news_json(team_news, opponent_news):
    """Convert news headline lists to structured JSON."""
    def format_news(news_list):
        if not news_list:
            return []
        return [
            {"headline": headline}
            for headline, _ in news_list[:3]  # Max 3 headlines
            if headline
        ]

    return {
        "team": format_news(team_news),
        "opponent": format_news(opponent_news)
    }


def build_schedule_json(team_schedule, team_abbr):
    """
    Convert schedule information to structured JSON.

    Args:
        team_schedule: List of all games for this team (dict with game info)
        team_abbr: Team abbreviation

    Returns:
        dict: Schedule with completed games and upcoming games
    """
    completed_games = []
    upcoming_games = []

    for game in team_schedule:
        # Check if this team is in the game
        if game['home_team'] != team_abbr and game['away_team'] != team_abbr:
            continue

        is_home = game['home_team'] == team_abbr
        opponent = game['away_team'] if is_home else game['home_team']

        game_info = {
            "week": game.get('week_num', 0),
            "date": game['game_date'],
            "opponent": opponent,
            "location": "home" if is_home else "away",
            "home_team": game['home_team'],
            "away_team": game['away_team']
        }

        # Check if game has been played
        if game.get('home_score') and game.get('away_score'):
            # Game completed
            game_info["result"] = "W" if (
                (is_home and int(game['home_score']) > int(game['away_score'])) or
                (not is_home and int(game['away_score']) > int(game['home_score']))
            ) else "L"
            game_info["score"] = f"{game['home_score']}-{game['away_score']}" if is_home else f"{game['away_score']}-{game['home_score']}"
            completed_games.append(game_info)
        else:
            # Game upcoming
            upcoming_games.append(game_info)

    return {
        "completed": completed_games,
        "upcoming": upcoming_games
    }


def build_playoff_odds_json(playoff_chance, division_chance, top_seed_chance, sb_appearance_chance, sb_win_chance):
    """Convert playoff probabilities to structured JSON."""
    return {
        "make_playoffs": playoff_chance,
        "win_division": division_chance,
        "get_top_seed": top_seed_chance,
        "reach_super_bowl": sb_appearance_chance,
        "win_super_bowl": sb_win_chance
    }


def build_standings_json(standings_data, teams_dict):
    """
    Convert standings data to structured JSON.

    Args:
        standings_data: Dict with structure {conference: {division: [(team, info), ...]}}
        teams_dict: Dictionary of team info

    Returns:
        dict: Hierarchically structured standings
    """
    standings = {}

    for conference in standings_data:
        standings[conference] = {}
        for division in standings_data[conference]:
            standings[conference][division] = []
            for team_abbr, info in standings_data[conference][division]:
                record = f"{info['wins']}-{info['losses']}"
                if info.get('ties', 0) > 0:
                    record += f"-{info['ties']}"

                standings[conference][division].append({
                    "team": team_abbr,
                    "name": f"{teams_dict[team_abbr]['city']} {teams_dict[team_abbr]['mascot']}",
                    "record": record,
                    "wins": info['wins'],
                    "losses": info['losses'],
                    "ties": info.get('ties', 0)
                })

    return standings


def build_team_analysis_prompt(
    team_abbr,
    team_stats_row,
    opponent_abbr,
    opponent_stats_row,
    teams_dict,
    team_injuries=None,
    opponent_injuries=None,
    team_news=None,
    opponent_news=None,
    team_schedule=None,
    playoff_chance=0,
    division_chance=0,
    top_seed_chance=0,
    sb_appearance_chance=0,
    sb_win_chance=0,
    current_week=1,
    standings_data=None,
    team_coordinators=None,
    opponent_coordinators=None
):
    """
    Build AI analysis prompt using structured JSON.

    This version uses hierarchical JSON instead of base64 CSV for better
    AI comprehension and statistical accuracy.

    Args:
        team_abbr: Team abbreviation (e.g., 'MIN')
        team_stats_row: Pandas Series with team stats
        opponent_abbr: Opponent abbreviation (or "NONE")
        opponent_stats_row: Pandas Series with opponent stats (or None)
        teams_dict: Dictionary of team info (from load_teams())
        team_injuries: List of (name, pos, status) tuples
        opponent_injuries: List of (name, pos, status) tuples
        team_news: List of (headline, date) tuples
        opponent_news: List of (headline, date) tuples
        team_schedule: List of all games for this team (from schedule)
        playoff_chance: Playoff probability %
        division_chance: Division win probability %
        top_seed_chance: #1 seed probability %
        sb_appearance_chance: Super Bowl appearance probability %
        sb_win_chance: Super Bowl win probability %
        current_week: Current NFL week number
        standings_data: Dict with {conference: {division: [(team, info), ...]}}
        team_coordinators: Pandas Series with coordinator info (or None)
        opponent_coordinators: Pandas Series with coordinator info (or None)

    Returns:
        str: The formatted prompt for AI analysis
    """
    team_info = teams_dict[team_abbr]

    # Build all JSON structures
    team_stats_json = build_team_stats_json(team_abbr, team_stats_row, teams_dict)
    opponent_stats_json = build_team_stats_json(opponent_abbr, opponent_stats_row, teams_dict) if opponent_abbr != "NONE" and opponent_stats_row is not None else {}
    injuries_json = build_injuries_json(team_injuries, opponent_injuries)
    news_json = build_news_json(team_news, opponent_news)
    schedule_json = build_schedule_json(team_schedule, team_abbr) if team_schedule else {}
    playoff_odds_json = build_playoff_odds_json(
        playoff_chance, division_chance, top_seed_chance,
        sb_appearance_chance, sb_win_chance
    )
    standings_json = build_standings_json(standings_data, teams_dict) if standings_data else {}

    # Convert to minified JSON strings for prompt
    team_stats_str = json.dumps(team_stats_json, separators=(',', ':'))
    opponent_stats_str = json.dumps(opponent_stats_json, separators=(',', ':')) if opponent_abbr != "NONE" else "{}"
    injuries_str = json.dumps(injuries_json, separators=(',', ':'))
    news_str = json.dumps(news_json, separators=(',', ':'))
    schedule_str = json.dumps(schedule_json, separators=(',', ':'))
    playoff_odds_str = json.dumps(playoff_odds_json, separators=(',', ':'))
    standings_str = json.dumps(standings_json, separators=(',', ':'))

    # Build coaching section
    coaching_section = ""
    if team_coordinators is not None:
        coaching_section = f"""
{'=' * 70}
COACHING STAFF
{'=' * 70}
{team_info['city']} {team_info['mascot']}:
  Head Coach: {team_info.get('head_coach', 'Unknown')}
  Offensive Coordinator: {team_coordinators.get('offensive_coordinator', 'Unknown')}
  Defensive Coordinator: {team_coordinators.get('defensive_coordinator', 'Unknown')}
"""
        if opponent_abbr != "NONE" and opponent_coordinators is not None:
            opponent_info = teams_dict[opponent_abbr]
            coaching_section += f"""
{opponent_info['city']} {opponent_info['mascot']}:
  Head Coach: {opponent_info.get('head_coach', 'Unknown')}
  Offensive Coordinator: {opponent_coordinators.get('offensive_coordinator', 'Unknown')}
  Defensive Coordinator: {opponent_coordinators.get('defensive_coordinator', 'Unknown')}
"""

    # Random comedian/analyst selections for variety
    roast_comedians_list = "Jeff Ross, Anthony Jeselnik, Dave Attell, Lisa Lampanelli, Jim Norton, Daniel Tosh, Bill Burr, Nikki Glaser, Greg Giraldo"
    other_comedians_list = "Norm Macdonald, John Mulaney, Aziz Ansari, Hannibal Buress, Whitney Cummings, Maria Bamford, Tig Notaro, Sarah Silverman, Amy Schumer, Jim Jefferies, Louis C.K., Jim Gaffigan, Seth Macfarlane, George Carlin, Steven Wright, Mitch Hedberg, Bill Hicks"

    # Random hot take opening styles for variety
    hottake_styles = [
        "Channel Stephen A. Smith's explosive energy",
        "Write like Skip Bayless defending his most controversial take",
        "Embody Pat McAfee's enthusiastic, larger-than-life personality",
        "Write with Shannon Sharpe's confident, folksy storytelling",
        "Channel Dan Orlovsky's passionate analytical breakdown"
    ]
    hottake_style = random.choice(hottake_styles)

    roast_comedians = " and ".join(random.sample(roast_comedians_list.split(", "), 3))
    other_comedians = " and ".join(random.sample(other_comedians_list.split(", "), 3))

    # Randomly choose comedy style
    ai_fun_rule = (
        f"Write in the style of these comedians and writers: {other_comedians}"
        if random.choice([True, False]) else
        "Write a fake news story in the style of The Onion or SNL's Weekend Update"
    )

    opponent_name = opponent_stats_json.get('team', {}).get('name', 'NO OPPONENT') if opponent_abbr != "NONE" else "NO OPPONENT"

    prompt = f"""{'=' * 70}
{team_info['city'].upper()} {team_info['mascot'].upper()} ANALYSIS
{'=' * 70}
Current Date: {datetime.now().strftime('%B %d, %Y')}
Current Season: 2025/26 NFL Season
Week: {current_week}

{'=' * 70}
ROLE
{'=' * 70}
You are an award-winning NFL analyst combining statistical accuracy with
entertaining, personality-filled writing. Channel John Madden's enthusiasm,
Tony Romo's insight, and a stand-up comedian's wit.

{'=' * 70}
OUTPUT REQUIREMENTS
{'=' * 70}
Generate analysis in JSON format with these sections:

1. ai_summary
   - One paragraph overview of the {team_info['city']} {team_info['mascot']} season
   - Reference current stats and playoff position
   - Make it engaging and readable

2. ai_hottake
   - One bold, passionate paragraph
   - Use facts to support a strong opinion
   - Style: {hottake_style}
   - IMPORTANT: Vary your opening - avoid repetitive phrases like "WAKE UP" or "LISTEN"

3. ai_stats
   - Three statistical insights (3-4 sentences each)
   - Find interesting narratives in the numbers
   - Add context and color to raw stats

4. ai_preview
   - Preview of next game (if applicable)
   - Discuss implications for playoff/division race
   - Reference head-to-head context

5. ai_fun
   - {ai_fun_rule}
   - Reference real players and stats
   - Keep it creative and entertaining

6. ai_roast
   - CRITICAL: Roast the {team_info['city']} {team_info['mascot']}, NOT their opponent
   - Channel roast comedians like {roast_comedians}
   - 7-10 jokes written as a comedy routine
   - Use their season stats and disappointments as material

{'=' * 70}
TEAM STATISTICS
{'=' * 70}
{team_stats_str}

{'=' * 70}
OPPONENT STATISTICS
{'=' * 70}
{opponent_stats_str}

{'=' * 70}
SCHEDULE
{'=' * 70}
{schedule_str}

{'=' * 70}
INJURIES
{'=' * 70}
{injuries_str}

{'=' * 70}
RECENT NEWS
{'=' * 70}
{news_str}

{'=' * 70}
PLAYOFF ODDS
{'=' * 70}
{playoff_odds_str}

{'=' * 70}
STANDINGS
{'=' * 70}
{standings_str}
{coaching_section}
{'=' * 70}
CRITICAL RULES
{'=' * 70}
1. ALL content must be about the {team_info['city']} {team_info['mascot']}, NOT their opponent

2. Only use stats from the JSON data provided above

3. Stats are in clear hierarchical format - reference them accurately:
   - Offensive red zone: offense.efficiency.red_zone_td_pct
   - Defensive red zone: defense.efficiency.red_zone_td_pct_against

4. Response must be valid JSON with proper escaping

5. Use \\n for line breaks in JSON strings

6. Current season is 2024/25 - focus on THIS season's performance

7. HISTORICAL CLAIMS WARNING:
   Your training data may contain outdated historical narratives about teams.
   If you reference playoff history or multi-year droughts, be EXTREMELY
   cautious - many teams have broken old patterns in recent years (2021-2024).

   When in doubt about historical claims:
   - Focus on the current season data provided above
   - Avoid specific "first time since [year]" claims unless certain
   - Don't assume old narratives still apply

   Examples of outdated narratives to AVOID:
   - "Haven't won a playoff game since [old year]" (may be outdated)
   - "Perennial losers" for teams that have recently succeeded
   - "Never won a [championship/playoff game]" without verification

{'=' * 70}
REQUIRED JSON FORMAT
{'=' * 70}
{{
    "ai_summary": "Single string with \\n for breaks",
    "ai_hottake": "Single string with \\n for breaks",
    "ai_stats": "Single string with \\n for breaks",
    "ai_preview": "Single string with \\n for breaks",
    "ai_fun": "Single string with \\n for breaks",
    "ai_roast": "Single string with \\n for breaks"
}}
{'=' * 70}
"""

    return prompt
