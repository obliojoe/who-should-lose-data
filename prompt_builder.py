"""
Redesigned prompt builder for NFL team analysis - Version 2.0

New 4-section structure focused on fan engagement:
1. THE VERDICT - Combined summary + hot take
2. THE X-FACTOR - Key matchup insights
3. THE REALITY CHECK - Statistical truth with humor
4. THE QUOTE THAT NAILS IT - Shareable one-liner
"""
import json
from datetime import datetime

# Handle imports from both relative and absolute paths
try:
    from stat_filter import StatFilter
except ImportError:
    from scripts.generate_cache.stat_filter import StatFilter


def calculate_league_rankings(all_stats_df):
    """
    Calculate league rankings for key stats across all teams.
    Returns a dict mapping team_abbr to their rankings.

    For offensive stats: Higher is better (rank 1 = best)
    For defensive stats: Lower is better (rank 1 = best)
    """
    rankings = {}

    # Key stats to rank
    offensive_stats = [
        'points_per_game',
        'total_yards',
        'passing_yards',
        'rushing_yards',
        'third_down_pct',
        'red_zone_pct',
        'total_epa'
    ]

    defensive_stats = [
        'points_against_per_game',
        'def_sacks',  # Higher is better for defense
        'def_interceptions'  # Higher is better for defense
    ]

    for team in all_stats_df['team_abbr']:
        rankings[team] = {}

    # Rank offensive stats (higher is better, so rank 1 = highest value)
    for stat in offensive_stats:
        if stat in all_stats_df.columns:
            all_stats_df[f'{stat}_rank'] = all_stats_df[stat].rank(ascending=False, method='min').astype(int)
            for idx, row in all_stats_df.iterrows():
                rankings[row['team_abbr']][f'{stat}_rank'] = int(row[f'{stat}_rank'])

    # Rank defensive stats
    # Points allowed: lower is better
    if 'points_against_per_game' in all_stats_df.columns:
        all_stats_df['points_against_per_game_rank'] = all_stats_df['points_against_per_game'].rank(ascending=True, method='min').astype(int)
        for idx, row in all_stats_df.iterrows():
            rankings[row['team_abbr']]['points_against_per_game_rank'] = int(row['points_against_per_game_rank'])

    # Sacks and interceptions: higher is better for defense
    for stat in ['def_sacks', 'def_interceptions']:
        if stat in all_stats_df.columns:
            all_stats_df[f'{stat}_rank'] = all_stats_df[stat].rank(ascending=False, method='min').astype(int)
            for idx, row in all_stats_df.iterrows():
                rankings[row['team_abbr']][f'{stat}_rank'] = int(row[f'{stat}_rank'])

    return rankings


def build_team_stats_json(team_abbr, stats_row, teams_dict, league_rankings=None):
    """
    Convert team stats DataFrame row to structured JSON.
    Reuses original structure from prompt_builder.py

    Args:
        team_abbr: Team abbreviation
        stats_row: Row from team stats DataFrame
        teams_dict: Dictionary of team info
        league_rankings: Optional dict of league rankings for this team
    """
    team_info = teams_dict[team_abbr]
    team_ranks = league_rankings.get(team_abbr, {}) if league_rankings else {}

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
                "points_per_game": round(float(stats_row['points_per_game']), 1),
                "points_per_game_rank": team_ranks.get('points_per_game_rank')
            },
            "passing": {
                "yards": int(stats_row['passing_yards']),
                "yards_rank": team_ranks.get('passing_yards_rank'),
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
                "yards_rank": team_ranks.get('rushing_yards_rank'),
                "touchdowns": int(stats_row['rushing_tds']),
                "yards_per_carry": round(float(stats_row['yards_per_carry']), 2),
                "first_downs": int(stats_row['rushing_first_downs'])
            },
            "total": {
                "yards": int(stats_row['total_yards']),
                "yards_rank": team_ranks.get('total_yards_rank'),
                "yards_per_game": round(float(stats_row['yards_per_game']), 1),
                "first_downs": int(stats_row['total_first_downs']),
                "first_downs_per_game": round(float(stats_row['first_downs_per_game']), 1)
            },
            "efficiency": {
                "third_down_attempts": int(stats_row['third_down_attempts']),
                "third_down_conversions": int(stats_row['third_down_conversions']),
                "third_down_pct": round(float(stats_row['third_down_pct']), 1),
                "third_down_pct_rank": team_ranks.get('third_down_pct_rank'),
                "fourth_down_attempts": int(stats_row['fourth_down_attempts']),
                "fourth_down_conversions": int(stats_row['fourth_down_conversions']),
                "fourth_down_pct": round(float(stats_row['fourth_down_pct']), 1),
                "red_zone_attempts": int(stats_row['red_zone_trips']),
                "red_zone_touchdowns": int(stats_row['red_zone_tds']),
                "red_zone_td_pct": round(float(stats_row['red_zone_pct']), 1),
                "red_zone_pct_rank": team_ranks.get('red_zone_pct_rank')
            },
            "turnovers": {
                "total": int(stats_row['total_turnovers']),
                "turnover_margin": int(stats_row['turnover_margin'])
            }
        },
        "defense": {
            "scoring": {
                "points_allowed_per_game": round(float(stats_row['points_against_per_game']), 1),
                "points_allowed_per_game_rank": team_ranks.get('points_allowed_per_game_rank')
            },
            "pass_rush": {
                "sacks": int(stats_row['def_sacks']),
                "sacks_rank": team_ranks.get('def_sacks_rank')
            },
            "takeaways": {
                "interceptions": int(stats_row['def_interceptions']),
                "interceptions_rank": team_ranks.get('def_interceptions_rank')
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
                "total_rank": team_ranks.get('total_epa_rank'),
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

        formatted = []
        for injury in injury_list:
            # Handle both dict format (from ESPN API) and tuple format (from fallback)
            if isinstance(injury, dict):
                name = injury.get('player', 'Unknown')
                pos = injury.get('position', 'N/A')
                status = injury.get('status', 'Unknown')
                injury_type = injury.get('type', '')

                # Build status string with injury type if available
                status_text = status_map.get(status, status)
                if injury_type:
                    status_text = f"{status_text} ({injury_type})"

                formatted.append({
                    "name": name,
                    "position": pos,
                    "status": status_text
                })
            else:
                # Handle legacy tuple format (name, pos, status)
                name, pos, status = injury
                formatted.append({
                    "name": name,
                    "position": pos,
                    "status": status_map.get(status, status)
                })

        return formatted

    return {
        "team": format_injuries(team_injuries),
        "opponent": format_injuries(opponent_injuries)
    }


def build_news_json(team_news, opponent_news):
    """Convert news headline lists to structured JSON."""
    def format_news(news_list):
        if not news_list:
            return []

        formatted = []
        for news in news_list[:3]:  # Max 3 headlines
            # Handle both dict format (new) and tuple format (legacy)
            if isinstance(news, dict):
                headline = news.get('headline', '')
            else:
                headline = news[0] if news else ''

            if headline:
                formatted.append({"headline": headline})

        return formatted

    return {
        "team": format_news(team_news),
        "opponent": format_news(opponent_news)
    }


def build_schedule_json(team_schedule, team_abbr):
    """Convert schedule information to structured JSON."""
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

        # Check if game has been played (handle NaN values)
        home_score = game.get('home_score')
        away_score = game.get('away_score')

        # Check if scores are valid (not None, not NaN, not empty string)
        has_scores = (
            home_score is not None and
            away_score is not None and
            str(home_score).strip() != '' and
            str(away_score).strip() != '' and
            str(home_score) != 'nan' and
            str(away_score) != 'nan'
        )

        if has_scores:
            # Game completed
            home_score = int(float(home_score))
            away_score = int(float(away_score))
            game_info["result"] = "W" if (
                (is_home and home_score > away_score) or
                (not is_home and away_score > home_score)
            ) else "L"
            game_info["score"] = f"{home_score}-{away_score}" if is_home else f"{away_score}-{home_score}"
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
    """Convert standings data to structured JSON."""
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


def build_espn_context_json(espn_context):
    """
    Build JSON for ESPN API data (betting lines, weather, etc.)

    Args:
        espn_context: Dict from ESPNAPIService.get_game_context()

    Returns:
        dict: Formatted ESPN data
    """
    if not espn_context:
        return {}

    context = {}

    # Betting lines
    if espn_context.get('betting'):
        betting = espn_context['betting']
        context['betting'] = {
            'spread': betting.get('spread'),
            'over_under': betting.get('over_under'),
            'favorite': betting.get('favorite'),
            'underdog': betting.get('underdog'),
            'moneyline_favorite': betting.get('moneyline_favorite'),
            'moneyline_underdog': betting.get('moneyline_underdog')
        }

    # Weather
    if espn_context.get('weather'):
        weather = espn_context['weather']
        context['weather'] = {
            'is_indoor': weather.get('is_indoor', False),
            'temperature': weather.get('temperature'),
            'condition': weather.get('condition'),
            'wind_speed': weather.get('wind_speed')
        }

    return context


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
    opponent_coordinators=None,
    espn_context=None,
    league_rankings=None
):
    """
    Build AI analysis prompt using 4-section structure.

    Sections:
    1. ai_verdict - Combined engaging summary + bold take
    2. ai_xfactor - Key matchup insights and what will decide the game
    3. ai_reality_check - Statistical truth with humor
    4. ai_quotes - Five shareable quotes that capture the team's season

    Returns:
        str: The formatted prompt for AI analysis
    """
    team_info = teams_dict[team_abbr]

    # Build all JSON structures
    team_stats_json = build_team_stats_json(team_abbr, team_stats_row, teams_dict, league_rankings)
    opponent_stats_json = build_team_stats_json(opponent_abbr, opponent_stats_row, teams_dict, league_rankings) if opponent_abbr != "NONE" and opponent_stats_row is not None else {}
    injuries_json = build_injuries_json(team_injuries, opponent_injuries)
    news_json = build_news_json(team_news, opponent_news)
    schedule_json = build_schedule_json(team_schedule, team_abbr) if team_schedule else {}
    playoff_odds_json = build_playoff_odds_json(
        playoff_chance, division_chance, top_seed_chance,
        sb_appearance_chance, sb_win_chance
    )
    standings_json = build_standings_json(standings_data, teams_dict) if standings_data else {}
    espn_context_json = build_espn_context_json(espn_context)

    # Pre-filter stats to identify impressive/concerning/notable
    stat_filter = StatFilter()
    filtered_stats = stat_filter.filter_all_stats(team_stats_row)

    # Get matchup edges if opponent exists
    matchup_edges = {}
    if opponent_abbr != "NONE" and opponent_stats_row is not None:
        matchup_edges = stat_filter.get_matchup_edges(team_stats_row, opponent_stats_row)

    # Convert to JSON strings
    team_stats_str = json.dumps(team_stats_json, separators=(',', ':'))
    opponent_stats_str = json.dumps(opponent_stats_json, separators=(',', ':')) if opponent_abbr != "NONE" else "{}"
    injuries_str = json.dumps(injuries_json, separators=(',', ':'))
    news_str = json.dumps(news_json, separators=(',', ':'))
    schedule_str = json.dumps(schedule_json, separators=(',', ':'))
    playoff_odds_str = json.dumps(playoff_odds_json, separators=(',', ':'))
    standings_str = json.dumps(standings_json, separators=(',', ':'))
    espn_context_str = json.dumps(espn_context_json, separators=(',', ':'))
    filtered_stats_str = json.dumps(filtered_stats, separators=(',', ':'))
    matchup_edges_str = json.dumps(matchup_edges, separators=(',', ':'))

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

    opponent_name = opponent_stats_json.get('team', {}).get('name', 'NO OPPONENT') if opponent_abbr != "NONE" else "NO OPPONENT"

    prompt = f"""{'=' * 70}
{team_info['city'].upper()} {team_info['mascot'].upper()} ANALYSIS - V2.0
{'=' * 70}
Current Date: {datetime.now().strftime('%B %d, %Y')}
Current Season: 2025/26 NFL Season
Week: {current_week}

{'=' * 70}
YOUR ROLE & VOICE
{'=' * 70}
You are "The Armchair Analyst" - a confident, witty NFL expert who combines
deep statistical knowledge with sharp humor. Think Bill Simmons meets Tony Romo.

Your voice is:
- Conversational but insightful
- Data-driven but never dry
- Honest (call out weaknesses, celebrate strengths)
- Funny without trying too hard
- Focused on what fans actually care about

CONSISTENCY: Maintain this EXACT voice across ALL sections. Don't shift personas.

{'=' * 70}
OUTPUT REQUIREMENTS - 4 SECTIONS
{'=' * 70}

1. ai_verdict
   - 2-3 paragraphs that tell the COMPLETE story
   - Start with the bottom line: Are they good? Bad? Frauds? Legit?
   - Back it up with key stats and context
   - Include one bold take based on the data
   - Be direct and engaging - this is what fans check first

   Quality bar:
   - Should answer "Is this team for real?" definitively
   - Must cite specific stats to support claims
   - Avoid generic phrases like "impressive showing" or "solid performance"
   - Give fans something to argue about

2. ai_xfactor
   - What will actually decide this game/season?
   - Focus on specific matchup advantages/disadvantages
   - Reference betting lines, weather, rest advantages if available
   - Identify THE key player or unit to watch
   - Be specific: not "the defense needs to step up" but "can the pass rush
     generate pressure without blitzing given their 1.8 sacks/game?"

   Quality bar:
   - Should give fans one concrete thing to watch for
   - Must be based on actual data, not generic analysis
   - Explain WHY it matters (stakes, playoff implications, etc.)

3. ai_reality_check
   - Lead with 2-3 pre-filtered key stats (see FILTERED_STATS below)
   - Add context and narrative to each stat
   - Then pivot to humor: what are fans fooling themselves about?
   - Call out any concerning trends with wit
   - Keep the ratio: 60% insightful / 40% humorous

   Quality bar:
   - Stats should reveal something non-obvious
   - Humor should be clever, not mean-spirited
   - Must roast the {team_info['city']} {team_info['mascot']}, NOT their opponent
   - Self-aware fans should nod along, not get defensive

4. ai_quotes
   - FIVE distinct quotes, each 1-3 sentences (don't default to one sentence!)
   - Mix of lengths: some can be punchy one-liners, others should be fuller 2-3 sentence observations
   - Witty observations with substance - NOT just statistical breakdowns
   - Use stats to INFORM the quip, but DON'T cite numbers directly
   - Think bar conversation, not broadcast booth analysis
   - Each should work as a standalone social media quote
   - Must be specific to THIS team's situation
   - Think "screenshot and share" quality
   - Vary the tone: mix clever insights with humor, sarcasm, and truth

   Quality bar:
   - Quotable and memorable - something a fan would text their friend
   - NO statistics or percentages in the quotes themselves
   - VARIETY IS KEY: Mix short punchy quotes with longer, more developed ones
   - Examples of the VIBE and LENGTH we want:
     * Short: "Playing like a team that googles 'prevent defense' during timeouts."
     * Medium: "Their playoff hopes look great on paper until you remember they have to actually play the games. Then it's like watching someone try to assemble IKEA furniture drunk."
     * Longer: "Every week it's the same story: dominate the first half, build a lead, then proceed to play defense like they're afraid of hurting the opponent's feelings. It's not prevent defense, it's 'please score' defense."
   - Avoid generic phrases that could apply to any team
   - NO RECYCLED JOKES: Don't use the same joke/angle you've used before
   - AVOID STALE TROPES: No "forward pass" jokes, "Madden on rookie mode", "haven't won since X" unless truly remarkable
   - Each captures a different aspect of their season
   - Fans should LAUGH, nod, or get angry enough to share

{'=' * 70}
TEAM STATISTICS (WITH LEAGUE RANKINGS)
{'=' * 70}
{team_stats_str}

IMPORTANT: League rankings are included for key stats (e.g., "points_per_game_rank": 5)
- Rankings are out of 32 teams
- Rank 1 = best in league, Rank 32 = worst
- For offense: Lower rank number = better (rank 1 = highest scoring)
- For defense: Lower rank number = better (rank 1 = fewest points allowed, most sacks, most INTs)
- ALWAYS use rankings to provide context:
  * Ranks 1-8: Elite/Top tier
  * Ranks 9-16: Above average to middle-of-pack
  * Ranks 17-24: Below average to middle-of-pack
  * Ranks 25-32: Bottom tier/Really struggling
- Example: "15th in points allowed" = middle-of-pack defense, NOT "really bad"
- Example: "28th in points allowed" = actually bad defense
- Use rankings to give accurate assessments instead of exaggerating weaknesses

{'=' * 70}
OPPONENT STATISTICS (WITH LEAGUE RANKINGS)
{'=' * 70}
{opponent_stats_str}

{'=' * 70}
PRE-FILTERED KEY STATS
{'=' * 70}
These stats have been pre-identified as impressive, concerning, or notable.
Use these as the foundation for ai_reality_check section:
{filtered_stats_str}

{'=' * 70}
MATCHUP EDGES
{'=' * 70}
Pre-analyzed advantages and disadvantages for this specific matchup:
{matchup_edges_str}

{'=' * 70}
ESPN CONTEXT (Betting Lines, Weather, etc.)
{'=' * 70}
{espn_context_str}

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

3. Maintain ONE consistent voice (The Armchair Analyst) across all sections
   - Don't suddenly become Stephen A. Smith in one section and Norm Macdonald in another
   - Keep the same conversational, witty, insightful tone throughout

4. Use the PRE-FILTERED STATS as your foundation for ai_reality_check
   - Don't just list them - add context and narrative
   - Explain WHY each stat matters

5. For ai_xfactor, reference ESPN context if available:
   - Mention betting spread/over-under
   - Note weather conditions for outdoor games
   - Consider rest advantages

6. Response must be valid JSON with proper escaping

7. Use \\n for line breaks in JSON strings

8. Current season is 2024/25 - focus on THIS season's performance

9. HISTORICAL CLAIMS WARNING:
   Your training data may contain outdated narratives. When in doubt:
   - Focus on current season data provided
   - Avoid "first time since [year]" claims unless certain
   - Don't assume old narratives still apply
   - BANNED PHRASES: Avoid tired/overused jokes like:
     * "discovered the forward pass" (especially for Detroit, Cleveland, etc.)
     * "first winning season since [ancient year]"
     * Generic "lovable losers" narratives
   - Be FRESH with your humor - find NEW angles based on THIS season's data

10. QUALITY OVER QUANTITY:
    - Better to write one great insight than three generic ones
    - Every sentence should add value
    - Cut anything that sounds like filler

{'=' * 70}
REQUIRED JSON FORMAT
{'=' * 70}
{{
    "ai_verdict": "2-3 paragraphs with \\n for breaks",
    "ai_xfactor": "2-3 paragraphs with \\n for breaks",
    "ai_reality_check": "2-3 paragraphs with \\n for breaks",
    "ai_quotes": [
        "First quote - insightful take on their season",
        "Second quote - humorous observation",
        "Third quote - bold prediction or claim",
        "Fourth quote - different angle or stat-based",
        "Fifth quote - captures current moment"
    ]
}}

IMPORTANT:
- NO trailing commas after the last field
- Ensure valid JSON format
- The ai_quotes field is an ARRAY of 5 strings
- Each quote should be 12-20 words (shorter is better)
- NO statistics, percentages, or numbers in the quotes
- Quotes should be witty observations, not analytical statements
- The ai_quotes field is LAST and should NOT have a comma after it
{'=' * 70}
"""

    return prompt
