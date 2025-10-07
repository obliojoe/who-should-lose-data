import requests
import logging
from typing import Dict, Optional, Tuple
import time

logger = logging.getLogger('espn_api')

class ESPNAPIService:
    """Service for fetching additional game data from ESPN API"""

    BASE_URL = "http://sports.core.api.espn.com/v2/sports/football/leagues/nfl"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    def get_event_data(self, event_id: str) -> Optional[Dict]:
        """
        Fetch complete event data from ESPN API

        Args:
            event_id: ESPN event ID for the game

        Returns:
            Dict containing event data or None if request fails
        """
        try:
            url = f"{self.BASE_URL}/events/{event_id}"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            # Only log 503 errors at debug level (ESPN temporary unavailability is normal)
            error_str = str(e)
            if '503' in error_str:
                logger.debug(f"ESPN API temporarily unavailable for event {event_id} (503)")
            else:
                logger.error(f"Failed to fetch event data for {event_id}: {error_str}")
            return None

    def get_betting_lines(self, event_id: str) -> Optional[Dict]:
        """
        Extract betting lines from event data

        Returns:
            Dict with keys: spread, over_under, favorite, underdog, moneyline_favorite, moneyline_underdog
        """
        event_data = self.get_event_data(event_id)
        if not event_data:
            return None

        try:
            # ESPN includes odds in the pickcenter or competitions data
            competitions = event_data.get('competitions', [])
            if not competitions:
                return None

            competition = competitions[0]

            # Try to get odds from pickcenter reference
            if 'pickcenter' in competition:
                pickcenter_url = competition['pickcenter'].get('$ref')
                if pickcenter_url:
                    pickcenter_response = self.session.get(pickcenter_url, timeout=10)
                    if pickcenter_response.ok:
                        pickcenter_data = pickcenter_response.json()

                        # Extract betting information
                        betting_info = {}

                        # Get spread
                        if 'againstTheSpread' in pickcenter_data:
                            spread_data = pickcenter_data['againstTheSpread']
                            current = spread_data.get('current', {})
                            betting_info['spread'] = current.get('spread')

                            # Try to get favorite team
                            favorite_obj = current.get('favorite', {})
                            if isinstance(favorite_obj, dict) and 'team' in favorite_obj:
                                betting_info['favorite'] = favorite_obj.get('team', {}).get('abbreviation')

                            # Try to get underdog team
                            underdog_obj = current.get('underdog', {})
                            if isinstance(underdog_obj, dict) and 'team' in underdog_obj:
                                betting_info['underdog'] = underdog_obj.get('team', {}).get('abbreviation')

                            # Log for debugging
                            logger.debug(f"Spread data structure: favorite={betting_info.get('favorite')}, underdog={betting_info.get('underdog')}, spread={betting_info.get('spread')}")

                        # Get over/under
                        if 'overUnder' in pickcenter_data:
                            betting_info['over_under'] = pickcenter_data['overUnder'].get('current')

                        # Get moneylines
                        if 'moneyLine' in pickcenter_data:
                            ml_data = pickcenter_data['moneyLine']
                            betting_info['moneyline_favorite'] = ml_data.get('current', {}).get('favorite', {}).get('odds')
                            betting_info['moneyline_underdog'] = ml_data.get('current', {}).get('underdog', {}).get('odds')

                        return betting_info if betting_info else None

            # Fallback: try to get odds from odds reference
            if 'odds' in competition:
                odds_url = competition['odds'].get('$ref')
                if odds_url:
                    odds_response = self.session.get(odds_url, timeout=10)
                    if odds_response.ok:
                        odds_data = odds_response.json()
                        if 'items' in odds_data and odds_data['items']:
                            odds_item = odds_data['items'][0]
                            betting_info = {
                                'spread': odds_item.get('spread'),
                                'over_under': odds_item.get('overUnder'),
                                'provider': odds_item.get('provider', {}).get('name', 'ESPN')
                            }

                            # Parse details field for favorite (e.g., "DET -10.5")
                            details = odds_item.get('details', '')
                            if details and '-' in details:
                                # Details format is "TEAM -SPREAD"
                                parts = details.split()
                                if len(parts) >= 2:
                                    betting_info['favorite'] = parts[0]

                            logger.debug(f"Extracted betting info from odds: {betting_info}")
                            return betting_info

            return None

        except Exception as e:
            logger.error(f"Failed to extract betting lines for {event_id}: {str(e)}")
            return None

    def get_weather(self, event_id: str) -> Optional[Dict]:
        """
        Extract weather data from event

        Returns:
            Dict with keys: temperature, condition, wind_speed, is_indoor
        """
        event_data = self.get_event_data(event_id)
        if not event_data:
            return None

        try:
            competitions = event_data.get('competitions', [])
            if not competitions:
                return None

            competition = competitions[0]
            venue = competition.get('venue', {})

            # Check if indoor stadium
            is_indoor = venue.get('indoor', False)

            # Get weather data if outdoor
            weather_info = {'is_indoor': is_indoor}

            if not is_indoor and 'weather' in competition:
                weather = competition['weather']
                weather_info['temperature'] = weather.get('temperature')
                weather_info['condition'] = weather.get('displayValue', 'Unknown')
                weather_info['wind_speed'] = weather.get('windSpeed')

            return weather_info

        except Exception as e:
            logger.error(f"Failed to extract weather for {event_id}: {str(e)}")
            return None

    def get_game_context(self, event_id: str, home_team: str, away_team: str) -> Dict:
        """
        Get comprehensive game context including betting lines and weather

        Args:
            event_id: ESPN event ID
            home_team: Home team abbreviation
            away_team: Away team abbreviation

        Returns:
            Dict with betting, weather, and other context
        """
        context = {
            'betting': None,
            'weather': None,
            'event_id': event_id
        }

        # Get betting lines
        betting = self.get_betting_lines(event_id)
        if betting:
            context['betting'] = betting

        # Get weather
        weather = self.get_weather(event_id)
        if weather:
            context['weather'] = weather

        return context

    def get_predictor_data(self, event_id: str) -> Optional[Dict]:
        """
        Get ESPN's win probability predictions for upcoming games

        Returns:
            Dict with keys: home_win_prob, away_win_prob, predicted_point_diff, matchup_quality
        """
        try:
            url = f"{self.BASE_URL}/events/{event_id}/competitions/{event_id}/predictor"
            response = self.session.get(url, timeout=10)

            if not response.ok:
                return None

            data = response.json()

            predictor_info = {}

            # Get home team stats
            if 'homeTeam' in data:
                for stat in data['homeTeam'].get('statistics', []):
                    if stat['name'] == 'gameProjection':
                        predictor_info['home_win_prob'] = float(stat['value'])
                    elif stat['name'] == 'teamPredPtDiff':
                        predictor_info['predicted_point_diff'] = float(stat['value'])
                    elif stat['name'] == 'matchupQuality':
                        predictor_info['matchup_quality'] = float(stat['value'])

            # Get away team win probability
            if 'awayTeam' in data:
                for stat in data['awayTeam'].get('statistics', []):
                    if stat['name'] == 'gameProjection':
                        predictor_info['away_win_prob'] = float(stat['value'])

            return predictor_info if predictor_info else None

        except Exception as e:
            logger.error(f"Failed to get predictor data for {event_id}: {str(e)}")
            return None

    def get_game_leaders(self, event_id: str) -> Optional[Dict]:
        """
        Get game leaders for completed games (passing, rushing, receiving)

        Returns:
            Dict with leader stats by category
        """
        try:
            event_data = self.get_event_data(event_id)
            if not event_data:
                return None

            competitions = event_data.get('competitions', [])
            if not competitions:
                return None

            competition = competitions[0]

            if 'leaders' not in competition:
                return None

            leaders_ref = competition['leaders'].get('$ref')
            if not leaders_ref:
                return None

            response = self.session.get(leaders_ref, timeout=10)
            if not response.ok:
                return None

            leaders_data = response.json()
            leaders_info = {}

            # Extract key leader categories
            for category in leaders_data.get('categories', []):
                cat_name = category.get('name', '')

                # Only keep the main leader categories
                if cat_name in ['passingLeader', 'rushingLeader', 'receivingLeader']:
                    leaders = category.get('leaders', [])
                    if leaders:
                        # Get top 2 leaders for this category with athlete names
                        category_leaders = []
                        for leader in leaders[:2]:
                            leader_data = {
                                'displayValue': leader.get('displayValue'),
                                'value': leader.get('value')
                            }

                            # Fetch athlete info
                            if 'athlete' in leader:
                                athlete_ref = leader['athlete'].get('$ref')
                                if athlete_ref:
                                    try:
                                        athlete_resp = self.session.get(athlete_ref, timeout=10)
                                        if athlete_resp.ok:
                                            athlete = athlete_resp.json()
                                            leader_data['player'] = athlete.get('displayName')
                                            leader_data['jersey'] = athlete.get('jersey')
                                            leader_data['position'] = athlete.get('position', {}).get('abbreviation')
                                    except:
                                        pass  # Skip if athlete fetch fails

                            # Fetch team info
                            if 'team' in leader:
                                team_ref = leader['team'].get('$ref')
                                if team_ref:
                                    try:
                                        team_resp = self.session.get(team_ref, timeout=10)
                                        if team_resp.ok:
                                            team = team_resp.json()
                                            leader_data['team'] = team.get('abbreviation')
                                    except:
                                        pass  # Skip if team fetch fails

                            category_leaders.append(leader_data)

                        leaders_info[cat_name] = category_leaders

            return leaders_info if leaders_info else None

        except Exception as e:
            logger.error(f"Failed to get game leaders for {event_id}: {str(e)}")
            return None

    def get_broadcast_info(self, event_id: str) -> Optional[str]:
        """
        Get broadcast network information

        Returns:
            Network name string
        """
        try:
            event_data = self.get_event_data(event_id)
            if not event_data:
                return None

            competitions = event_data.get('competitions', [])
            if not competitions:
                return None

            competition = competitions[0]

            if 'broadcasts' not in competition:
                return None

            broadcasts_ref = competition['broadcasts'].get('$ref')
            if not broadcasts_ref:
                return None

            response = self.session.get(broadcasts_ref, timeout=10)
            if not response.ok:
                return None

            broadcasts_data = response.json()

            if 'items' in broadcasts_data and broadcasts_data['items']:
                network = broadcasts_data['items'][0].get('type', {}).get('shortName')
                return network

            return None

        except Exception as e:
            logger.error(f"Failed to get broadcast info for {event_id}: {str(e)}")
            return None

    def find_event_id_by_teams(self, week: int, year: int, team1: str, team2: str) -> Optional[str]:
        """
        Find ESPN event ID for a game by week and teams

        Args:
            week: NFL week number
            year: Season year
            team1: First team abbreviation
            team2: Second team abbreviation

        Returns:
            Event ID string or None if not found
        """
        try:
            # ESPN season type: 2 = regular season, 3 = playoffs
            season_type = 2 if week <= 18 else 3

            # Get scoreboard for the week
            url = f"{self.BASE_URL}/seasons/{year}/types/{season_type}/weeks/{week}/events"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()

            events_data = response.json()

            # Look through events to find matching teams
            if 'items' in events_data:
                for event_ref in events_data['items']:
                    event_url = event_ref.get('$ref')
                    if event_url:
                        # Extract event ID from URL
                        event_id = event_url.split('/')[-1]

                        # Get event details
                        event = self.get_event_data(event_id)
                        if event and 'competitions' in event:
                            competition = event['competitions'][0]
                            competitors = competition.get('competitors', [])

                            # Get team abbreviations
                            teams_in_game = []
                            for comp in competitors:
                                if 'team' in comp:
                                    team_ref = comp['team'].get('$ref')
                                    if team_ref:
                                        team_response = self.session.get(team_ref, timeout=10)
                                        if team_response.ok:
                                            team_data = team_response.json()
                                            teams_in_game.append(team_data.get('abbreviation'))

                            # Check if both teams match
                            if team1 in teams_in_game and team2 in teams_in_game:
                                logger.info(f"Found event ID {event_id} for {team1} vs {team2} in week {week}")
                                return event_id

            logger.warning(f"Could not find event ID for {team1} vs {team2} in week {week}")
            return None

        except Exception as e:
            logger.error(f"Failed to find event ID: {str(e)}")
            return None
