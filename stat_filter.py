"""
Stat filtering logic to identify impressive, concerning, and notable statistics
for pre-filtering before AI analysis
"""
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger('stat_filter')

class StatFilter:
    """Filter and categorize team statistics into impressive/concerning/notable"""

    # Define thresholds for various stats (league-relative)
    IMPRESSIVE_THRESHOLDS = {
        'points_per_game': 27.0,
        'total_yards_per_game': 380.0,
        'passing_yards_per_game': 260.0,
        'rushing_yards_per_game': 135.0,
        'third_down_pct': 44.0,
        'red_zone_pct': 60.0,
        'time_of_possession': 32.0,
        'turnover_margin': 0.5,
        'sacks': 3.0,
    }

    CONCERNING_THRESHOLDS = {
        'points_per_game': 18.0,
        'total_yards_per_game': 300.0,
        'passing_yards_per_game': 200.0,
        'rushing_yards_per_game': 90.0,
        'third_down_pct': 36.0,
        'red_zone_pct': 50.0,
        'time_of_possession': 28.0,
        'turnover_margin': -0.5,
        'sacks': 1.5,
    }

    # Defense stats (lower is better for yards allowed)
    IMPRESSIVE_DEFENSIVE_THRESHOLDS = {
        'points_allowed_per_game': 18.0,
        'total_yards_allowed_per_game': 310.0,
        'passing_yards_allowed_per_game': 210.0,
        'rushing_yards_allowed_per_game': 100.0,
    }

    CONCERNING_DEFENSIVE_THRESHOLDS = {
        'points_allowed_per_game': 26.0,
        'total_yards_allowed_per_game': 380.0,
        'passing_yards_allowed_per_game': 260.0,
        'rushing_yards_allowed_per_game': 135.0,
    }

    def __init__(self):
        pass

    def get_stat_value(self, stats_row: Dict, stat_name: str) -> float:
        """Safely extract stat value from row"""
        try:
            value = stats_row.get(stat_name)
            if value is None or value == '':
                return 0.0
            return float(value)
        except (ValueError, TypeError):
            return 0.0

    def filter_offensive_stats(self, stats_row: Dict) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Filter offensive stats into impressive, concerning, and notable

        Returns:
            Tuple of (impressive_stats, concerning_stats, notable_stats)
        """
        impressive = []
        concerning = []
        notable = []

        # Points per game
        ppg = self.get_stat_value(stats_row, 'points_per_game')
        if ppg >= self.IMPRESSIVE_THRESHOLDS['points_per_game']:
            impressive.append({
                'stat': 'Points Per Game',
                'value': ppg,
                'context': 'elite scoring offense'
            })
        elif ppg <= self.CONCERNING_THRESHOLDS['points_per_game']:
            concerning.append({
                'stat': 'Points Per Game',
                'value': ppg,
                'context': 'struggling to score'
            })

        # Total yards per game
        ypg = self.get_stat_value(stats_row, 'yards_per_game')
        if ypg >= self.IMPRESSIVE_THRESHOLDS['total_yards_per_game']:
            impressive.append({
                'stat': 'Total Yards Per Game',
                'value': ypg,
                'context': 'dominant in moving the ball'
            })
        elif ypg <= self.CONCERNING_THRESHOLDS['total_yards_per_game']:
            concerning.append({
                'stat': 'Total Yards Per Game',
                'value': ypg,
                'context': 'anemic offense'
            })

        # Third down conversion
        third_down = self.get_stat_value(stats_row, 'third_down_pct')
        if third_down >= self.IMPRESSIVE_THRESHOLDS['third_down_pct']:
            impressive.append({
                'stat': 'Third Down %',
                'value': third_down,
                'context': 'excellent at sustaining drives'
            })
        elif third_down <= self.CONCERNING_THRESHOLDS['third_down_pct']:
            concerning.append({
                'stat': 'Third Down %',
                'value': third_down,
                'context': 'can\'t convert on third down'
            })

        # Red zone efficiency
        red_zone = self.get_stat_value(stats_row, 'red_zone_pct')
        if red_zone >= self.IMPRESSIVE_THRESHOLDS['red_zone_pct']:
            impressive.append({
                'stat': 'Red Zone %',
                'value': red_zone,
                'context': 'cash in when they get close'
            })
        elif red_zone <= self.CONCERNING_THRESHOLDS['red_zone_pct']:
            concerning.append({
                'stat': 'Red Zone %',
                'value': red_zone,
                'context': 'settle for field goals too often'
            })

        # Turnover margin
        to_margin = self.get_stat_value(stats_row, 'turnover_margin')
        if to_margin >= self.IMPRESSIVE_THRESHOLDS['turnover_margin']:
            impressive.append({
                'stat': 'Turnover Margin',
                'value': to_margin,
                'context': 'winning the turnover battle'
            })
        elif to_margin <= self.CONCERNING_THRESHOLDS['turnover_margin']:
            concerning.append({
                'stat': 'Turnover Margin',
                'value': to_margin,
                'context': 'giving the ball away'
            })

        return impressive, concerning, notable

    def filter_defensive_stats(self, stats_row: Dict) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Filter defensive stats into impressive, concerning, and notable

        Returns:
            Tuple of (impressive_stats, concerning_stats, notable_stats)
        """
        impressive = []
        concerning = []
        notable = []

        # Points allowed per game (lower is better)
        papg = self.get_stat_value(stats_row, 'points_against_per_game')
        if papg <= self.IMPRESSIVE_DEFENSIVE_THRESHOLDS['points_allowed_per_game']:
            impressive.append({
                'stat': 'Points Allowed Per Game',
                'value': papg,
                'context': 'lockdown defense'
            })
        elif papg >= self.CONCERNING_DEFENSIVE_THRESHOLDS['points_allowed_per_game']:
            concerning.append({
                'stat': 'Points Allowed Per Game',
                'value': papg,
                'context': 'getting torched'
            })

        # Sacks per game (calculate from def_sacks and games_played)
        def_sacks = self.get_stat_value(stats_row, 'def_sacks')
        games_played = self.get_stat_value(stats_row, 'games_played')
        sacks_per_game = def_sacks / games_played if games_played > 0 else 0
        if sacks_per_game >= self.IMPRESSIVE_THRESHOLDS['sacks']:
            impressive.append({
                'stat': 'Sacks Per Game',
                'value': round(sacks_per_game, 2),
                'context': 'ferocious pass rush'
            })
        elif sacks_per_game <= self.CONCERNING_THRESHOLDS['sacks']:
            concerning.append({
                'stat': 'Sacks Per Game',
                'value': round(sacks_per_game, 2),
                'context': 'no pass rush'
            })

        # Interceptions per game (calculate from def_interceptions and games_played)
        def_ints = self.get_stat_value(stats_row, 'def_interceptions')
        ints_per_game = def_ints / games_played if games_played > 0 else 0
        if ints_per_game >= 1.2:  # More than 1 per game is excellent
            impressive.append({
                'stat': 'Interceptions Per Game',
                'value': round(ints_per_game, 2),
                'context': 'ball-hawking secondary'
            })

        return impressive, concerning, notable

    def filter_all_stats(self, stats_row: Dict) -> Dict:
        """
        Filter all team stats and return categorized results

        Returns:
            Dict with keys: offensive (impressive/concerning/notable),
                           defensive (impressive/concerning/notable)
        """
        off_impressive, off_concerning, off_notable = self.filter_offensive_stats(stats_row)
        def_impressive, def_concerning, def_notable = self.filter_defensive_stats(stats_row)

        return {
            'offensive': {
                'impressive': off_impressive,
                'concerning': off_concerning,
                'notable': off_notable
            },
            'defensive': {
                'impressive': def_impressive,
                'concerning': def_concerning,
                'notable': def_notable
            }
        }

    def get_matchup_edges(self, team_stats: Dict, opponent_stats: Dict) -> Dict:
        """
        Identify key matchup advantages/disadvantages

        Returns:
            Dict with keys: advantages, disadvantages, key_battles
        """
        advantages = []
        disadvantages = []
        key_battles = []

        # Offensive vs defensive matchups
        team_ppg = self.get_stat_value(team_stats, 'points_per_game')
        opp_papg = self.get_stat_value(opponent_stats, 'points_allowed_per_game')

        if team_ppg > 25 and opp_papg > 24:
            advantages.append({
                'category': 'Offense vs Defense',
                'detail': f'High-powered offense ({team_ppg:.1f} PPG) facing weak defense ({opp_papg:.1f} allowed)'
            })
        elif team_ppg < 20 and opp_papg < 20:
            disadvantages.append({
                'category': 'Offense vs Defense',
                'detail': f'Struggling offense ({team_ppg:.1f} PPG) against tough defense ({opp_papg:.1f} allowed)'
            })

        # Passing game comparison
        team_pass_yds = self.get_stat_value(team_stats, 'passing_yards')
        opp_pass_yds = self.get_stat_value(opponent_stats, 'passing_yards')

        if team_pass_yds > opp_pass_yds * 1.2:  # 20% more passing yards
            advantages.append({
                'category': 'Passing Attack',
                'detail': f'Strong passing game advantage ({team_pass_yds} vs {opp_pass_yds} yards)'
            })

        # Rushing game comparison
        team_rush_yds = self.get_stat_value(team_stats, 'rushing_yards')
        opp_rush_yds = self.get_stat_value(opponent_stats, 'rushing_yards')

        if team_rush_yds > opp_rush_yds * 1.2:  # 20% more rushing yards
            advantages.append({
                'category': 'Running Game',
                'detail': f'Strong rushing game advantage ({team_rush_yds} vs {opp_rush_yds} yards)'
            })

        # Turnover battle
        team_to_margin = self.get_stat_value(team_stats, 'turnover_margin')
        opp_to_margin = self.get_stat_value(opponent_stats, 'turnover_margin')

        if abs(team_to_margin - opp_to_margin) > 1.0:
            if team_to_margin > opp_to_margin:
                advantages.append({
                    'category': 'Turnover Battle',
                    'detail': f'Big edge in turnovers (+{team_to_margin:.1f} vs {opp_to_margin:.1f})'
                })
            else:
                disadvantages.append({
                    'category': 'Turnover Battle',
                    'detail': f'Losing the turnover battle ({team_to_margin:.1f} vs +{opp_to_margin:.1f})'
                })

        return {
            'advantages': advantages,
            'disadvantages': disadvantages,
            'key_battles': key_battles
        }
