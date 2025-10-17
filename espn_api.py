import json
import logging
from typing import Dict, List, Optional

from raw_data_manifest import RawDataManifest
from team_metadata import TEAM_METADATA


logger = logging.getLogger('espn_api')

TEAM_ALIAS = {
    'LA': 'LAR',
    'WSH': 'WAS',
}


class ESPNAPIService:
    """Service for reading ESPN game data from collected raw snapshots."""

    def __init__(self, manifest: Optional[RawDataManifest] = None):
        self.manifest = manifest or RawDataManifest.from_latest()
        self._scoreboard_cache: Optional[Dict] = None
        self._team_map = self._build_team_map()

    def _build_team_map(self) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        try:
            with open('data/teams.json', 'r', encoding='utf-8') as fh:
                records = json.load(fh)
            for record in records:
                espn_id = record.get('espn_api_id')
                abbr = record.get('team_abbr')
                if espn_id is None or not abbr:
                    continue
                try:
                    identifier = str(int(espn_id))
                except (TypeError, ValueError):
                    identifier = str(espn_id).strip()
                mapping[identifier] = abbr
        except FileNotFoundError:
            logger.warning("teams.json not found; falling back to static team metadata")
            for record in TEAM_METADATA:
                espn_id = record.get('espn_api_id')
                team_abbr = record.get('team_abbr')
                if espn_id is None or not team_abbr:
                    continue
                mapping[str(int(espn_id))] = team_abbr
        except Exception as exc:
            logger.warning("Failed to load team mapping: %s", exc)

        return mapping

    def _map_team(self, espn_id: Optional[str]) -> Optional[str]:
        if espn_id is None:
            return None
        return self._team_map.get(str(espn_id))

    def _load_json(self, dataset: str, identifier: Optional[str]) -> Optional[Dict]:
        if not self.manifest:
            return None
        try:
            return self.manifest.load_json(dataset, identifier)
        except ValueError:
            logger.debug("Dataset %s with identifier %s not found in manifest", dataset, identifier)
            return None

    def _load_scoreboard(self) -> Optional[Dict]:
        if self._scoreboard_cache is not None:
            return self._scoreboard_cache

        scoreboard = self._load_json('espn_scoreboard', None)
        if scoreboard:
            self._scoreboard_cache = scoreboard
        return self._scoreboard_cache

    def _find_scoreboard_event(self, event_id: str) -> Optional[Dict]:
        scoreboard = self._load_scoreboard()
        if not scoreboard:
            return None

        for event in scoreboard.get('events', []) or []:
            if str(event.get('id')) == str(event_id):
                return event
        return None

    def _parse_betting(self, event_id: str, home_team: str, away_team: str) -> Optional[Dict]:
        pickcenter = self._load_json('espn_pickcenter', str(event_id))
        if isinstance(pickcenter, list) and pickcenter:
            entry = pickcenter[0]

            def _team_name(odds: Dict, default: Optional[str]) -> Optional[str]:
                if odds.get('favorite'):
                    return default
                return None

            home_odds = entry.get('homeTeamOdds', {}) or {}
            away_odds = entry.get('awayTeamOdds', {}) or {}

            favorite = None
            underdog = None
            if home_odds.get('favorite'):
                favorite = home_team
                underdog = away_team
            elif away_odds.get('favorite'):
                favorite = away_team
                underdog = home_team

            betting_info = {
                'spread': entry.get('spread'),
                'over_under': entry.get('overUnder'),
                'provider': (entry.get('provider') or {}).get('name'),
                'favorite': favorite,
                'underdog': underdog,
                'moneyline_favorite': None,
                'moneyline_underdog': None,
            }

            if favorite == home_team:
                betting_info['moneyline_favorite'] = home_odds.get('moneyLine')
                betting_info['moneyline_underdog'] = away_odds.get('moneyLine')
            elif favorite == away_team:
                betting_info['moneyline_favorite'] = away_odds.get('moneyLine')
                betting_info['moneyline_underdog'] = home_odds.get('moneyLine')

            return betting_info

        # Fallback to odds collected inside summary.json if available
        summary = self._load_json('espn_summary', str(event_id))
        if isinstance(summary, dict):
            odds_entries = summary.get('odds') or []
            if odds_entries:
                item = odds_entries[0]
                favorite = None
                details = item.get('details') or ''
                if details:
                    parts = details.split()
                    if parts:
                        favorite = parts[0]
                return {
                    'spread': item.get('spread'),
                    'over_under': item.get('overUnder'),
                    'provider': (item.get('provider') or {}).get('name'),
                    'favorite': favorite,
                    'underdog': None,
                    'moneyline_favorite': None,
                    'moneyline_underdog': None,
                }

        return None

    def _parse_weather(self, event_id: str) -> Optional[Dict]:
        event = self._find_scoreboard_event(event_id)
        if not event:
            return None

        competitions = event.get('competitions', []) or []
        if not competitions:
            return None

        competition = competitions[0]
        venue = competition.get('venue', {}) or {}
        weather = competition.get('weather') or {}

        weather_info: Dict[str, Optional[float]] = {}
        if 'indoor' in venue:
            weather_info['is_indoor'] = bool(venue.get('indoor'))

        if weather:
            weather_info['temperature'] = weather.get('temperature')
            weather_info['condition'] = weather.get('displayValue') or weather.get('shortDescription')
            wind = weather.get('wind') or {}
            weather_info['wind_speed'] = wind.get('speed') or weather.get('windSpeed')

        if weather_info:
            weather_info.setdefault('is_indoor', False)
            return weather_info

        if 'indoor' in venue:
            return {'is_indoor': bool(venue.get('indoor'))}

        return None

    def get_game_context(self, event_id: str, home_team: str, away_team: str) -> Dict:
        context = {
            'event_id': event_id,
            'betting': None,
            'weather': None,
        }

        betting = self._parse_betting(event_id, home_team, away_team)
        if betting:
            context['betting'] = betting

        weather = self._parse_weather(event_id)
        if weather:
            context['weather'] = weather

        return context

    def get_predictor_data(self, event_id: str) -> Optional[Dict]:
        predictor = self._load_json('espn_predictor', str(event_id))
        if not predictor:
            return None

        predictor_info: Dict[str, float] = {}

        for stat in predictor.get('homeTeam', {}).get('statistics', []) or []:
            name = stat.get('name')
            value = stat.get('value')
            if value is None:
                continue
            if name == 'gameProjection':
                predictor_info['home_win_prob'] = float(value)
            elif name == 'teamPredPtDiff':
                predictor_info['predicted_point_diff'] = float(value)
            elif name == 'matchupQuality':
                predictor_info['matchup_quality'] = float(value)

        for stat in predictor.get('awayTeam', {}).get('statistics', []) or []:
            if stat.get('name') == 'gameProjection' and stat.get('value') is not None:
                predictor_info['away_win_prob'] = float(stat.get('value'))

        return predictor_info or None

    def get_game_leaders(self, event_id: str) -> Optional[Dict]:
        leaders_payload = self._load_json('espn_leaders', str(event_id))
        if not isinstance(leaders_payload, list):
            return None

        results: Dict[str, List[Dict]] = {}

        for team_entry in leaders_payload:
            team_info = team_entry.get('team') or {}
            team_abbr = team_info.get('abbreviation') or self._map_team(team_info.get('id'))

            for category in team_entry.get('leaders', []) or []:
                name = (category.get('name') or category.get('displayName') or '').lower()
                if name.startswith('passing'):
                    key = 'passingLeader'
                elif name.startswith('rushing'):
                    key = 'rushingLeader'
                elif name.startswith('receiving'):
                    key = 'receivingLeader'
                else:
                    continue

                for leader in category.get('leaders', []) or []:
                    athlete = leader.get('athlete') or {}
                    entry = {
                        'player': athlete.get('displayName'),
                        'team': team_abbr,
                        'position': (athlete.get('position') or {}).get('abbreviation'),
                        'jersey': athlete.get('jersey'),
                        'display_value': leader.get('displayValue'),
                        'summary': leader.get('summary'),
                    }
                    results.setdefault(key, []).append(entry)

        return results or None

    def get_broadcast_info(self, event_id: str) -> Optional[str]:
        broadcasts = self._load_json('espn_broadcasts', str(event_id))
        if isinstance(broadcasts, list) and broadcasts:
            primary = broadcasts[0]
            media = primary.get('media') or {}
            if media.get('shortName'):
                return media['shortName']
            return primary.get('station') or primary.get('type', {}).get('shortName')
        return None

    def find_event_id_by_teams(self, week: int, year: int, team1: str, team2: str) -> Optional[str]:
        scoreboard = self._load_scoreboard()
        if not scoreboard:
            return None

        target_teams = {team1, team2}
        target_teams = {TEAM_ALIAS.get(team, team) for team in target_teams}

        for event in scoreboard.get('events', []) or []:
            competition = (event.get('competitions') or [None])[0]
            if not competition:
                continue
            competitors = competition.get('competitors', []) or []
            seen = set()
            for competitor in competitors:
                team = competitor.get('team', {}) or {}
                abbr = team.get('abbreviation')
                abbr = TEAM_ALIAS.get(abbr, abbr)
                seen.add(abbr)

            if target_teams.issubset(seen):
                return str(event.get('id'))

        return None
