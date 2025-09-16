import nfl_data_py as nfl
import pandas as pd
import json
from datetime import datetime, timedelta
import logging
import sys
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NFLMatchupAnalyzer:
    def __init__(self, performance_year=2024, baseline_year=2024, week_threshold=9, min_targets=3):
        """
        Initialize NFL Matchup Analyzer with flexible baseline years and production settings
        """
        self.performance_year = performance_year
        self.baseline_year = baseline_year
        self.week_threshold = week_threshold
        self.min_targets = min_targets
        self.baselines = {}
        self.pbp_data = None
        self.schedule_data = None
        self.data_loaded = False
        
        self.effective_baseline_year = self._determine_baseline_year()
        
        try:
            self.load_baselines()
            self.load_schedule()
            self.data_loaded = True
            logger.info(f"Successfully initialized analyzer: {performance_year} performance vs {self.effective_baseline_year} baselines")
        except Exception as e:
            logger.error(f"Failed to initialize analyzer: {str(e)}")
            self.data_loaded = False
    
    def _determine_baseline_year(self):
        """Determine which year to use for baselines"""
        if self.performance_year == self.baseline_year:
            return self.baseline_year
        
        if self.performance_year > self.baseline_year:
            logger.info(f"Using {self.baseline_year} baselines for more stable league averages")
            return self.baseline_year
        else:
            return self.baseline_year
    
    def load_baselines(self):
        """Load league baselines with error handling"""
        try:
            logger.info(f"Loading {self.effective_baseline_year} NFL baselines...")
            baseline_pbp = nfl.import_pbp_data([self.effective_baseline_year])
            
            if baseline_pbp.empty:
                raise ValueError(f"No baseline data available for {self.effective_baseline_year}")
            
            passes = baseline_pbp[(baseline_pbp["pass"] == 1) & 
                                 (baseline_pbp["receiver_player_name"].notna())].copy()
            
            if passes.empty:
                raise ValueError("No passing plays found in baseline data")
            
            passes.loc[:, "explosive"] = passes["receiving_yards"] >= 20
            
            baseline_stats = passes.groupby("pass_location").agg({
                "pass": "count",
                "complete_pass": "sum",
                "passing_yards": "sum", 
                "explosive": "sum"
            })
            
            baseline_stats["completion_pct"] = (baseline_stats["complete_pass"] / baseline_stats["pass"] * 100).round(1)
            baseline_stats["ypa"] = (baseline_stats["passing_yards"] / baseline_stats["pass"]).round(1)
            baseline_stats["explosive_pct"] = (baseline_stats["explosive"] / baseline_stats["pass"] * 100).round(1)
            
            completed = passes[passes["complete_pass"] == 1]
            yac_baseline = completed.groupby("pass_location")["yards_after_catch"].mean().round(1)
            
            self.baselines = {
                "completion": dict(baseline_stats["completion_pct"]),
                "ypa": dict(baseline_stats["ypa"]), 
                "explosive": dict(baseline_stats["explosive_pct"]),
                "yac": dict(yac_baseline)
            }
            
            logger.info(f"Baselines loaded successfully from {self.effective_baseline_year} season")
            
            # Load performance data
            logger.info(f"Loading {self.performance_year} performance data...")
            self.pbp_data = nfl.import_pbp_data([self.performance_year])
            
            if self.pbp_data.empty:
                raise ValueError(f"No performance data available for {self.performance_year}")
                
            logger.info(f"Performance data loaded: {len(self.pbp_data)} total plays")
            
        except Exception as e:
            logger.error(f"Failed to load data: {str(e)}")
            raise
    
    def load_schedule(self):
        """Load NFL schedule data"""
        try:
            logger.info(f"Loading {self.performance_year} NFL schedule...")
            self.schedule_data = nfl.import_schedules([self.performance_year])
            
            if self.schedule_data.empty:
                raise ValueError(f"No schedule data available for {self.performance_year}")
            
            # Convert game_id to ensure proper formatting
            self.schedule_data['gameday'] = pd.to_datetime(self.schedule_data['gameday'])
            logger.info(f"Schedule loaded: {len(self.schedule_data)} games")
            
        except Exception as e:
            logger.error(f"Failed to load schedule: {str(e)}")
            raise
    
    def get_current_week(self):
        """Determine current NFL week based on date"""
        try:
            # Get max week from play-by-play data (completed games)
            max_completed_week = self.pbp_data['week'].max()
            
            # For current week, look at schedule
            today = datetime.now().date()
            
            # Find the next upcoming games
            future_games = self.schedule_data[
                self.schedule_data['gameday'].dt.date >= today
            ].sort_values('gameday')
            
            if not future_games.empty:
                next_week = future_games['week'].iloc[0]
                logger.info(f"Current week determined: {next_week} (max completed: {max_completed_week})")
                return next_week
            else:
                # Fallback to max completed week + 1
                return max_completed_week + 1
                
        except Exception as e:
            logger.warning(f"Could not determine current week: {str(e)}")
            return 2  # Default fallback
    
    def get_week_matchups(self, week_num=None):
        """Get actual matchups for a specific week"""
        try:
            if week_num is None:
                week_num = self.get_current_week()
            
            week_games = self.schedule_data[self.schedule_data['week'] == week_num].copy()
            
            if week_games.empty:
                logger.warning(f"No games found for week {week_num}")
                return []
            
            matchups = []
            for _, game in week_games.iterrows():
                matchups.append({
                    'away_team': game['away_team'],
                    'home_team': game['home_team'],
                    'gameday': game['gameday'].strftime('%Y-%m-%d'),
                    'week': int(game['week'])
                })
            
            logger.info(f"Found {len(matchups)} games for week {week_num}")
            return matchups
            
        except Exception as e:
            logger.error(f"Error getting week {week_num} matchups: {str(e)}")
            return []
    
    def analyze_team_qbs(self, team):
        """Analyze QB performance with error handling"""
        try:
            qb_data = self.pbp_data[
                (self.pbp_data["posteam"] == team) & 
                (self.pbp_data["pass"] == 1) & 
                (self.pbp_data["receiver_player_name"].notna())
            ].copy()
            
            if qb_data.empty:
                return None
                
            qb_data.loc[:, "explosive"] = (qb_data["receiving_yards"] >= 20).astype(int)
            
            qb_by_location = qb_data.groupby(["passer_player_name", "pass_location"]).agg({
                "pass": "count",
                "complete_pass": "sum", 
                "passing_yards": "sum",
                "explosive": "sum"
            })
            
            qb_totals = qb_data.groupby("passer_player_name")["pass"].count()
            if qb_totals.empty:
                return None
                
            primary_qb = qb_totals.idxmax()
            
            if primary_qb not in qb_by_location.index:
                return None
                
            qb_stats = qb_by_location.loc[primary_qb].copy()
            qb_stats["completion_pct"] = (qb_stats["complete_pass"] / qb_stats["pass"] * 100).round(1)
            qb_stats["volume_pct"] = (qb_stats["pass"] / qb_stats["pass"].sum() * 100).round(1)
            qb_stats["ypa"] = (qb_stats["passing_yards"] / qb_stats["pass"]).round(1)
            qb_stats["explosive_pct"] = (qb_stats["explosive"] / qb_stats["pass"] * 100).round(1)
            
            # Calculate edges vs baseline
            for loc in qb_stats.index:
                if loc in self.baselines["completion"]:
                    qb_stats.loc[loc, "completion_edge"] = qb_stats.loc[loc, "completion_pct"] - self.baselines["completion"][loc]
                    qb_stats.loc[loc, "ypa_edge"] = qb_stats.loc[loc, "ypa"] - self.baselines["ypa"][loc]
                    qb_stats.loc[loc, "explosive_edge"] = qb_stats.loc[loc, "explosive_pct"] - self.baselines["explosive"][loc]
            
            return {"qb_name": primary_qb, "stats": qb_stats}
            
        except Exception as e:
            logger.error(f"Error analyzing QB for {team}: {str(e)}")
            return None
        
    def analyze_defense_by_location(self, team):
        """Analyze defense performance with error handling"""
        try:
            def_data = self.pbp_data[
                (self.pbp_data["defteam"] == team) & 
                (self.pbp_data["pass"] == 1) & 
                (self.pbp_data["receiver_player_name"].notna())
            ].copy()
            
            if def_data.empty:
                return None
                
            def_data.loc[:, "explosive"] = (def_data["receiving_yards"] >= 20).astype(int)
            
            def_stats = def_data.groupby("pass_location").agg({
                "pass": "count",
                "complete_pass": "sum",
                "receiving_yards": "sum",
                "yards_after_catch": "mean", 
                "explosive": "sum"
            })
            
            def_stats["completion_allowed"] = (def_stats["complete_pass"] / def_stats["pass"] * 100).round(1)
            def_stats["ypa_allowed"] = (def_stats["receiving_yards"] / def_stats["pass"]).round(1)
            def_stats["yac_allowed"] = def_stats["yards_after_catch"].round(1)
            def_stats["explosive_allowed"] = (def_stats["explosive"] / def_stats["pass"] * 100).round(1)
            
            # Calculate weaknesses vs baseline
            for loc in def_stats.index:
                if loc in self.baselines["completion"]:
                    def_stats.loc[loc, "completion_weakness"] = def_stats.loc[loc, "completion_allowed"] - self.baselines["completion"][loc]
                    def_stats.loc[loc, "ypa_weakness"] = def_stats.loc[loc, "ypa_allowed"] - self.baselines["ypa"][loc]
                    def_stats.loc[loc, "yac_weakness"] = def_stats.loc[loc, "yac_allowed"] - self.baselines["yac"][loc]
                    def_stats.loc[loc, "explosive_weakness"] = def_stats.loc[loc, "explosive_allowed"] - self.baselines["explosive"][loc]
            
            return def_stats
            
        except Exception as e:
            logger.error(f"Error analyzing defense for {team}: {str(e)}")
            return None
    
    def analyze_receiver_performance_by_location(self, team):
        """Analyze receiver performance with error handling and configurable threshold"""
        try:
            team_data = self.pbp_data[
                (self.pbp_data["posteam"] == team) & 
                (self.pbp_data["pass"] == 1) & 
                (self.pbp_data["receiver_player_name"].notna())
            ].copy()
            
            if team_data.empty:
                return {}
                
            team_data.loc[:, "explosive"] = (team_data["receiving_yards"] >= 20).astype(int)
            
            wr_by_location = team_data.groupby(["receiver_player_name", "pass_location"]).agg({
                "pass": "count",
                "complete_pass": "sum",
                "receiving_yards": "sum", 
                "yards_after_catch": "mean",
                "explosive": "sum"
            })
            
            receiver_analysis = {}
            target_counts = team_data["receiver_player_name"].value_counts()
            qualifying_receivers = target_counts[target_counts >= self.min_targets].index
            
            for receiver in qualifying_receivers:
                if receiver in wr_by_location.index:
                    wr_stats = wr_by_location.loc[receiver].copy()
                    wr_stats["catch_rate"] = (wr_stats["complete_pass"] / wr_stats["pass"] * 100).round(1)
                    wr_stats["ypa"] = (wr_stats["receiving_yards"] / wr_stats["pass"]).round(1)
                    wr_stats["explosive_pct"] = (wr_stats["explosive"] / wr_stats["pass"] * 100).round(1)
                    wr_stats["yards_after_catch"] = wr_stats["yards_after_catch"].round(1)
                    
                    for loc in wr_stats.index:
                        if loc in self.baselines["completion"]:
                            wr_stats.loc[loc, "catch_edge"] = wr_stats.loc[loc, "catch_rate"] - self.baselines["completion"][loc]
                            wr_stats.loc[loc, "ypa_edge"] = wr_stats.loc[loc, "ypa"] - self.baselines["ypa"][loc]
                            wr_stats.loc[loc, "explosive_edge"] = wr_stats.loc[loc, "explosive_pct"] - self.baselines["explosive"][loc]
                            wr_stats.loc[loc, "yac_edge"] = wr_stats.loc[loc, "yards_after_catch"] - self.baselines["yac"][loc]
                    
                    receiver_analysis[receiver] = wr_stats
            
            return receiver_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing receivers for {team}: {str(e)}")
            return {}
    
    def calculate_matchup_summary(self, qb_analysis, defense_analysis, receiver_analysis, offense_team):
        """Calculate volume-weighted edges with improved error handling"""
        try:
            if not qb_analysis or defense_analysis is None:
                return None
                
            qb_stats = qb_analysis["stats"]
            def_stats = defense_analysis
            
            # Cap extreme edges to prevent unrealistic percentages
            def cap_edge(value, max_val=50):
                return max(-max_val, min(max_val, value))
            
            betting_opportunities = []
            total_comp_weighted = 0
            total_ypa_weighted = 0
            total_explosive_weighted = 0
            
            team_data = self.pbp_data[
                (self.pbp_data["posteam"] == offense_team) & 
                (self.pbp_data["pass"] == 1) & 
                (self.pbp_data["receiver_player_name"].notna())
            ]
            
            for location in ["left", "middle", "right"]:
                if location in qb_stats.index and location in def_stats.index:
                    qb_volume_pct = qb_stats.loc[location, "volume_pct"]
                    qb_comp_edge = qb_stats.loc[location, "completion_edge"] if "completion_edge" in qb_stats.columns else 0
                    qb_ypa_edge = qb_stats.loc[location, "ypa_edge"] if "ypa_edge" in qb_stats.columns else 0
                    qb_explosive_edge = qb_stats.loc[location, "explosive_edge"] if "explosive_edge" in qb_stats.columns else 0
                    
                    def_comp_weak = def_stats.loc[location, "completion_weakness"] if "completion_weakness" in def_stats.columns else 0
                    def_ypa_weak = def_stats.loc[location, "ypa_weakness"] if "ypa_weakness" in def_stats.columns else 0
                    def_explosive_weak = def_stats.loc[location, "explosive_weakness"] if "explosive_weakness" in def_stats.columns else 0
                    
                    # Cap individual edges
                    net_comp_edge = cap_edge(qb_comp_edge + def_comp_weak)
                    net_ypa_edge = cap_edge(qb_ypa_edge + def_ypa_weak, 10)
                    net_explosive_edge = cap_edge(qb_explosive_edge + def_explosive_weak)
                    
                    weighted_comp = net_comp_edge * (qb_volume_pct / 100)
                    weighted_ypa = net_ypa_edge * (qb_volume_pct / 100)
                    weighted_explosive = net_explosive_edge * (qb_volume_pct / 100)
                    
                    total_comp_weighted += weighted_comp
                    total_ypa_weighted += weighted_ypa
                    total_explosive_weighted += weighted_explosive
                    
                    # Find top receiver for this location
                    location_passes = team_data[team_data["pass_location"] == location]
                    top_receiver = None
                    
                    if not location_passes.empty:
                        receiver_targets = location_passes["receiver_player_name"].value_counts()
                        if len(receiver_targets) > 0:
                            top_receiver_name = receiver_targets.index[0]
                            if top_receiver_name in receiver_analysis:
                                wr_stats = receiver_analysis[top_receiver_name]
                                if location in wr_stats.index:
                                    wr_catch_edge = wr_stats.loc[location, "catch_edge"] if "catch_edge" in wr_stats.columns else 0
                                    total_receiver_edge = cap_edge(net_comp_edge + wr_catch_edge)
                                    top_receiver = {
                                        "name": top_receiver_name,
                                        "targets": int(receiver_targets[top_receiver_name]),
                                        "edge": round(total_receiver_edge, 1)
                                    }
                    
                    betting_opportunities.append({
                        "location": location,
                        "volume": round(qb_volume_pct, 1),
                        "comp_edge": round(net_comp_edge, 1),
                        "ypa_edge": round(net_ypa_edge, 1),
                        "explosive_edge": round(net_explosive_edge, 1),
                        "top_receiver": top_receiver
                    })
            
            return {
                "opportunities": betting_opportunities,
                "total_comp_edge": round(total_comp_weighted, 1),
                "total_ypa_edge": round(total_ypa_weighted, 1),
                "total_explosive_edge": round(total_explosive_weighted, 1),
                "qb_name": qb_analysis["qb_name"]
            }
            
        except Exception as e:
            logger.error(f"Error calculating matchup summary for {offense_team}: {str(e)}")
            return None
    
    def analyze_week_matchups(self, week_num=None):
        """Analyze all actual matchups for a specific week"""
        if not self.data_loaded:
            logger.error("Analyzer not properly initialized")
            return []
        
        matchups = self.get_week_matchups(week_num)
        if not matchups:
            return []
        
        results = []
        
        logger.info(f"Analyzing {len(matchups)} actual matchups for week {week_num or 'current'}")
        
        for matchup in matchups:
            away_team = matchup['away_team']
            home_team = matchup['home_team']
            
            try:
                # Analyze away team offense vs home team defense
                away_qb_analysis = self.analyze_team_qbs(away_team)
                home_defense_analysis = self.analyze_defense_by_location(home_team)
                away_receiver_analysis = self.analyze_receiver_performance_by_location(away_team)
                
                away_result = self.calculate_matchup_summary(
                    away_qb_analysis, home_defense_analysis, away_receiver_analysis, away_team
                )
                
                # Analyze home team offense vs away team defense  
                home_qb_analysis = self.analyze_team_qbs(home_team)
                away_defense_analysis = self.analyze_defense_by_location(away_team)
                home_receiver_analysis = self.analyze_receiver_performance_by_location(home_team)
                
                home_result = self.calculate_matchup_summary(
                    home_qb_analysis, away_defense_analysis, home_receiver_analysis, home_team
                )
                
                if away_result or home_result:
                    game_result = {
                        "game": f"{away_team} @ {home_team}",
                        "gameday": matchup['gameday'],
                        "week": matchup['week'],
                        "away_team": away_team,
                        "home_team": home_team,
                        "away_analysis": away_result,
                        "home_analysis": home_result
                    }
                    results.append(game_result)
                    
            except Exception as e:
                logger.warning(f"Error analyzing {away_team} @ {home_team}: {str(e)}")
                continue
        
        # Sort by game date
        results.sort(key=lambda x: x['gameday'])
        
        logger.info(f"Successfully analyzed {len(results)} games")
        return results
    
    def generate_json_output(self, results, include_metadata=True):
        """Generate JSON output for API consumption"""
        try:
            output = {
                "games": results,
                "metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "performance_year": self.performance_year,
                    "baseline_year": self.effective_baseline_year,
                    "min_targets_threshold": self.min_targets,
                    "total_games": len(results),
                    "data_loaded": self.data_loaded,
                    "disclaimer": "For educational analysis only. Small sample sizes in early season may produce unreliable results. Past performance does not guarantee future results."
                } if include_metadata else None
            }
            
            if not include_metadata:
                output = {"games": results}
                
            return json.dumps(output, indent=2)
            
        except Exception as e:
            logger.error(f"Error generating JSON output: {str(e)}")
            return json.dumps({"error": "Failed to generate output", "message": str(e)})
    
    @classmethod
    def create_early_season_analyzer(cls, current_year, previous_year=None, min_targets=3):
        """Create analyzer optimized for early season with configurable thresholds"""
        if previous_year is None:
            previous_year = current_year - 1
            
        logger.info(f"Creating early season analyzer: {current_year} performance vs {previous_year} baselines")
        return cls(performance_year=current_year, baseline_year=previous_year, min_targets=min_targets)

def run_analysis():
    """Run analysis function - separates logic from main() for flask integration"""
    try:
        # Configuration - can be set via environment variables
        performance_year = int(os.getenv('PERFORMANCE_YEAR', 2025))
        baseline_year = int(os.getenv('BASELINE_YEAR', 2024))
        min_targets = int(os.getenv('MIN_TARGETS', 3))
        target_week = os.getenv('TARGET_WEEK')  # Optional specific week
        
        analyzer = NFLMatchupAnalyzer.create_early_season_analyzer(
            current_year=performance_year,
            previous_year=baseline_year,
            min_targets=min_targets
        )
        
        if not analyzer.data_loaded:
            logger.error("Failed to load data")
            return None
        
        # Analyze actual week matchups
        week_num = int(target_week) if target_week else None
        results = analyzer.analyze_week_matchups(week_num)
        
        if not results:
            logger.error("No valid games found")
            return None
        
        return {
            'results': results,
            'analyzer': analyzer,
            'week_num': week_num or analyzer.get_current_week()
        }
        
    except Exception as e:
        logger.error(f"Analysis execution failed: {str(e)}")
        return None

def main():
    """Main function for command-line execution"""
    analysis_data = run_analysis()
    
    if not analysis_data:
        sys.exit(1)
    
    results = analysis_data['results']
    analyzer = analysis_data['analyzer']
    current_week = analysis_data['week_num']
    
    # Output JSON for API consumption
    json_output = analyzer.generate_json_output(results)
    print(json_output)
    
    # Human-readable summary to stderr for logging
    print(f"\n=== WEEK {current_week} NFL MATCHUP ANALYSIS ===", file=sys.stderr)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", file=sys.stderr)
    print("="*50, file=sys.stderr)
    
    for game in results:
        print(f"\n{game['game']} ({game['gameday']})", file=sys.stderr)
        
        if game['away_analysis']:
            away = game['away_analysis']
            print(f"  {game['away_team']} ({away['qb_name']}): {away['total_comp_edge']:+.1f}% completion edge", file=sys.stderr)
        
        if game['home_analysis']:
            home = game['home_analysis']
            print(f"  {game['home_team']} ({home['qb_name']}): {home['total_comp_edge']:+.1f}% completion edge", file=sys.stderr)
    
    return results

# Flask Integration
try:
    from flask import Flask, jsonify, request
    from flask_cors import CORS
    
    app = Flask(__name__)
    CORS(app)  # Enable CORS for all routes
    
    @app.route('/dashboard')
    def dashboard():
        """Serve the premium NFL prop signals dashboard"""
        return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NFL Prop Signals Dashboard</title>
    <style>
        .webflow-betting-embed {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(180deg, #334155 0%, #1f2937 15%, #1f2937 100%);
            color: #ffffff;
            min-height: 100vh;
            margin: 0;
            padding: 0;
        }

        .component-container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
        }

        .component-header {
            text-align: center;
            margin-bottom: 4rem;
            padding: 2rem 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }

        .component-title {
            font-size: clamp(2rem, 5vw, 4rem);
            font-weight: 800;
            background: linear-gradient(135deg, #9ca3af, #ffffff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.02em;
            line-height: 1.1;
            margin-bottom: 1rem;
        }

        .component-subtitle {
            font-size: 0.875rem;
            font-weight: 600;
            color: #60a5fa;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            margin-bottom: 1rem;
        }

        .description-text {
            font-size: 1.125rem;
            color: #e5e7eb;
            line-height: 1.7;
        }

        .refresh-section {
            display: flex;
            justify-content: center;
            margin-bottom: 3rem;
        }

        .refresh-btn {
            background: linear-gradient(135deg, #1e3a8a, #60a5fa);
            color: #ffffff;
            border: none;
            padding: 1rem 2rem;
            border-radius: 12px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            backdrop-filter: blur(20px);
        }

        .refresh-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 25px rgba(30, 58, 138, 0.3);
        }

        .refresh-btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }

        .loading-state, .error-state {
            text-align: center;
            padding: 3rem;
            font-size: 1.125rem;
            color: #d1d5db;
        }

        .error-state {
            background: rgba(239, 68, 68, 0.1);
            border: 1px solid rgba(239, 68, 68, 0.3);
            border-radius: 12px;
            color: #fca5a5;
            margin-bottom: 2rem;
        }

        .warning-state {
            background: rgba(245, 158, 11, 0.1);
            border: 1px solid rgba(245, 158, 11, 0.3);
            border-radius: 12px;
            color: #fbbf24;
            margin-bottom: 2rem;
            text-align: center;
            padding: 1rem;
        }

        .component-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(600px, 1fr));
            gap: 3rem;
            margin-bottom: 4rem;
        }

        .component-card {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 24px;
            padding: 3rem;
            transition: all 0.3s ease;
        }

        .component-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
            border-color: rgba(96, 165, 250, 0.3);
            background: linear-gradient(135deg, rgba(96, 165, 250, 0.05), rgba(59, 130, 246, 0.02));
        }

        .game-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 2rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }

        .game-title {
            font-size: 1.75rem;
            font-weight: 700;
            color: #ffffff;
        }

        .game-date {
            color: #9ca3af;
            font-size: 0.875rem;
        }

        .team-section {
            margin-bottom: 2rem;
        }

        .team-header {
            display: flex;
            align-items: center;
            gap: 1rem;
            margin-bottom: 1.5rem;
        }

        .team-name {
            font-size: 1.25rem;
            font-weight: 700;
            color: #60a5fa;
        }

        .qb-name {
            color: #d1d5db;
            font-size: 0.875rem;
        }

        .overall-stats {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1rem;
            margin-bottom: 1.5rem;
        }

        .stat-box {
            background: rgba(31, 41, 55, 0.8);
            padding: 1rem;
            border-radius: 12px;
            text-align: center;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .stat-label {
            font-size: 0.75rem;
            color: #9ca3af;
            text-transform: uppercase;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }

        .stat-value {
            font-size: 1.125rem;
            font-weight: 700;
        }

        .stat-positive { color: #10b981; }
        .stat-negative { color: #ef4444; }
        .stat-neutral { color: #d1d5db; }

        .opportunities-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1rem;
        }

        .opportunity-card {
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 1.5rem;
            transition: all 0.2s ease;
        }

        .opportunity-card:hover {
            border-color: #60a5fa;
            background: rgba(96, 165, 250, 0.05);
        }

        .opportunity-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
        }

        .location-badge {
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.75rem;
            padding: 0.375rem 0.75rem;
            border-radius: 20px;
        }

        .location-left {
            background: rgba(16, 185, 129, 0.1);
            border: 1px solid rgba(16, 185, 129, 0.3);
            color: #10b981;
        }

        .location-middle {
            background: rgba(245, 158, 11, 0.1);
            border: 1px solid rgba(245, 158, 11, 0.3);
            color: #f59e0b;
        }

        .location-right {
            background: rgba(96, 165, 250, 0.1);
            border: 1px solid rgba(96, 165, 250, 0.3);
            color: #60a5fa;
        }

        .volume-text {
            font-size: 0.75rem;
            color: #9ca3af;
            font-weight: 600;
        }

        .mini-stats {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 0.5rem;
            margin-bottom: 1rem;
        }

        .mini-stat {
            text-align: center;
        }

        .mini-stat-label {
            font-size: 0.625rem;
            color: #9ca3af;
            text-transform: uppercase;
            font-weight: 600;
        }

        .mini-stat-value {
            font-size: 0.875rem;
            font-weight: 700;
        }

        .top-receiver {
            background: rgba(96, 165, 250, 0.1);
            border: 1px solid rgba(96, 165, 250, 0.3);
            border-radius: 8px;
            padding: 0.75rem;
        }

        .receiver-info {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .receiver-name {
            font-weight: 700;
            color: #60a5fa;
            font-size: 0.875rem;
        }

        .receiver-stats {
            display: flex;
            gap: 0.75rem;
            font-size: 0.75rem;
            color: #d1d5db;
        }

        .metadata-section {
            background: rgba(55, 65, 81, 0.6);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 2rem;
            text-align: center;
            color: #d1d5db;
            font-size: 0.875rem;
            line-height: 1.6;
        }

        .metadata-section strong {
            color: #ffffff;
        }

        @media (max-width: 768px) {
            .component-container {
                padding: 1rem;
            }
            
            .component-card {
                padding: 2rem;
                border-radius: 20px;
            }
            
            .component-grid {
                grid-template-columns: 1fr;
                gap: 2rem;
            }
            
            .opportunities-grid {
                grid-template-columns: 1fr;
            }
            
            .overall-stats {
                grid-template-columns: 1fr;
            }

            .game-header {
                flex-direction: column;
                align-items: flex-start;
                gap: 0.5rem;
            }
        }

        @media (max-width: 480px) {
            .component-card {
                padding: 1.5rem;
            }
            
            .component-title {
                font-size: 2rem;
            }
        }

        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .animate {
            animation: fadeInUp 1s ease-out;
        }
    </style>
</head>
<body>
    <div class="webflow-betting-embed">
        <div class="component-container">
            <div class="component-header">
                <h1 class="component-title">NFL Prop Signals</h1>
                <p class="component-subtitle">Data-Driven Insights</p>
                <p class="description-text">Location-based matchup analysis for informed prop betting decisions</p>
            </div>

            <div class="refresh-section">
                <button class="refresh-btn" onclick="loadData()">Refresh Analysis</button>
            </div>

            <div id="loading" class="loading-state">Loading NFL prop signals...</div>
            <div id="error" class="error-state" style="display: none;"></div>
            <div id="warning" class="warning-state" style="display: none;"></div>
            <div id="content"></div>
        </div>
    </div>

    <script>
        (function() {
            'use strict';
            
            async function loadData() {
                const loadingEl = document.getElementById('loading');
                const errorEl = document.getElementById('error');
                const warningEl = document.getElementById('warning');
                const contentEl = document.getElementById('content');
                const refreshBtn = document.querySelector('.refresh-btn');
                
                loadingEl.style.display = 'block';
                errorEl.style.display = 'none';
                warningEl.style.display = 'none';
                contentEl.innerHTML = '';
                refreshBtn.disabled = true;
                refreshBtn.textContent = 'Loading...';
                
                const API_URL = '/analyze';
                
                try {
                    console.log('Loading NFL prop analysis...');
                    const response = await fetch(API_URL);
                    
                    if (!response.ok) {
                        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                    }
                    
                    const data = await response.json();
                    console.log('Successfully loaded analysis data');
                    displayData(data);
                    loadingEl.style.display = 'none';
                    
                } catch (error) {
                    console.error('Failed to load analysis:', error);
                    errorEl.textContent = `Analysis temporarily unavailable: ${error.message}`;
                    errorEl.style.display = 'block';
                    loadingEl.style.display = 'none';
                    
                } finally {
                    refreshBtn.disabled = false;
                    refreshBtn.textContent = 'Refresh Analysis';
                }
            }
            
            function getEdgeClass(value) {
                if (value > 5) return 'stat-positive';
                if (value < -5) return 'stat-negative';
                return 'stat-neutral';
            }
            
            function formatEdge(value) {
                return value > 0 ? `+${value.toFixed(1)}` : value.toFixed(1);
            }
            
            function displayData(data) {
                const contentEl = document.getElementById('content');
                
                if (!data.games || data.games.length === 0) {
                    contentEl.innerHTML = '<div class="error-state">No game analysis available</div>';
                    return;
                }
                
                const gamesHtml = data.games.map(game => `
                    <div class="component-card animate">
                        <div class="game-header">
                            <div class="game-title">${game.game}</div>
                            <div class="game-date">${new Date(game.gameday).toLocaleDateString('en-US', { 
                                weekday: 'short', 
                                month: 'short', 
                                day: 'numeric' 
                            })}</div>
                        </div>
                        
                        <div class="team-section">
                            <div class="team-header">
                                <span class="team-name">${game.away_team}</span>
                                <span class="qb-name">${game.away_analysis?.qb_name || 'Unknown QB'}</span>
                            </div>
                            
                            ${game.away_analysis ? `
                                <div class="overall-stats">
                                    <div class="stat-box">
                                        <div class="stat-label">Completion Edge</div>
                                        <div class="stat-value ${getEdgeClass(game.away_analysis.total_comp_edge)}">${formatEdge(game.away_analysis.total_comp_edge)}%</div>
                                    </div>
                                    <div class="stat-box">
                                        <div class="stat-label">Explosive Edge</div>
                                        <div class="stat-value ${getEdgeClass(game.away_analysis.total_explosive_edge)}">${formatEdge(game.away_analysis.total_explosive_edge)}%</div>
                                    </div>
                                    <div class="stat-box">
                                        <div class="stat-label">YPA Edge</div>
                                        <div class="stat-value ${getEdgeClass(game.away_analysis.total_ypa_edge)}">${formatEdge(game.away_analysis.total_ypa_edge)}</div>
                                    </div>
                                </div>
                                
                                <div class="opportunities-grid">
                                    ${game.away_analysis.opportunities.map(opp => `
                                        <div class="opportunity-card">
                                            <div class="opportunity-header">
                                                <span class="location-badge location-${opp.location}">${opp.location}</span>
                                                <span class="volume-text">${opp.volume.toFixed(1)}% vol</span>
                                            </div>
                                            
                                            <div class="mini-stats">
                                                <div class="mini-stat">
                                                    <div class="mini-stat-label">Comp</div>
                                                    <div class="mini-stat-value ${getEdgeClass(opp.comp_edge)}">${formatEdge(opp.comp_edge)}%</div>
                                                </div>
                                                <div class="mini-stat">
                                                    <div class="mini-stat-label">Exp</div>
                                                    <div class="mini-stat-value ${getEdgeClass(opp.explosive_edge)}">${formatEdge(opp.explosive_edge)}%</div>
                                                </div>
                                                <div class="mini-stat">
                                                    <div class="mini-stat-label">YPA</div>
                                                    <div class="mini-stat-value ${getEdgeClass(opp.ypa_edge)}">${formatEdge(opp.ypa_edge)}</div>
                                                </div>
                                            </div>
                                            
                                            ${opp.top_receiver ? `
                                                <div class="top-receiver">
                                                    <div class="receiver-info">
                                                        <span class="receiver-name">${opp.top_receiver.name}</span>
                                                        <div class="receiver-stats">
                                                            <span>Edge: ${formatEdge(opp.top_receiver.edge)}%</span>
                                                            <span>Targets: ${opp.top_receiver.targets}</span>
                                                        </div>
                                                    </div>
                                                </div>
                                            ` : ''}
                                        </div>
                                    `).join('')}
                                </div>
                            ` : '<div class="stat-neutral">Analysis unavailable</div>'}
                        </div>
                        
                        <div class="team-section">
                            <div class="team-header">
                                <span class="team-name">${game.home_team}</span>
                                <span class="qb-name">${game.home_analysis?.qb_name || 'Unknown QB'}</span>
                            </div>
                            
                            ${game.home_analysis ? `
                                <div class="overall-stats">
                                    <div class="stat-box">
                                        <div class="stat-label">Completion Edge</div>
                                        <div class="stat-value ${getEdgeClass(game.home_analysis.total_comp_edge)}">${formatEdge(game.home_analysis.total_comp_edge)}%</div>
                                    </div>
                                    <div class="stat-box">
                                        <div class="stat-label">Explosive Edge</div>
                                        <div class="stat-value ${getEdgeClass(game.home_analysis.total_explosive_edge)}">${formatEdge(game.home_analysis.total_explosive_edge)}%</div>
                                    </div>
                                    <div class="stat-box">
                                        <div class="stat-label">YPA Edge</div>
                                        <div class="stat-value ${getEdgeClass(game.home_analysis.total_ypa_edge)}">${formatEdge(game.home_analysis.total_ypa_edge)}</div>
                                    </div>
                                </div>
                                
                                <div class="opportunities-grid">
                                    ${game.home_analysis.opportunities.map(opp => `
                                        <div class="opportunity-card">
                                            <div class="opportunity-header">
                                                <span class="location-badge location-${opp.location}">${opp.location}</span>
                                                <span class="volume-text">${opp.volume.toFixed(1)}% vol</span>
                                            </div>
                                            
                                            <div class="mini-stats">
                                                <div class="mini-stat">
                                                    <div class="mini-stat-label">Comp</div>
                                                    <div class="mini-stat-value ${getEdgeClass(opp.comp_edge)}">${formatEdge(opp.comp_edge)}%</div>
                                                </div>
                                                <div class="mini-stat">
                                                    <div class="mini-stat-label">Exp</div>
                                                    <div class="mini-stat-value ${getEdgeClass(opp.explosive_edge)}">${formatEdge(opp.explosive_edge)}%</div>
                                                </div>
                                                <div class="mini-stat">
                                                    <div class="mini-stat-label">YPA</div>
                                                    <div class="mini-stat-value ${getEdgeClass(opp.ypa_edge)}">${formatEdge(opp.ypa_edge)}</div>
                                                </div>
                                            </div>
                                            
                                            ${opp.top_receiver ? `
                                                <div class="top-receiver">
                                                    <div class="receiver-info">
                                                        <span class="receiver-name">${opp.top_receiver.name}</span>
                                                        <div class="receiver-stats">
                                                            <span>Edge: ${formatEdge(opp.top_receiver.edge)}%</span>
                                                            <span>Targets: ${opp.top_receiver.targets}</span>
                                                        </div>
                                                    </div>
                                                </div>
                                            ` : ''}
                                        </div>
                                    `).join('')}
                                </div>
                            ` : '<div class="stat-neutral">Analysis unavailable</div>'}
                        </div>
                    </div>
                `).join('');
                
                const metadataHtml = data.metadata ? `
                    <div class="metadata-section animate">
                        <p><strong>Analysis Generated:</strong> ${new Date(data.metadata.generated_at).toLocaleString()}</p>
                        <p style="margin-top: 1rem;"><strong>Baseline Year:</strong> ${data.metadata.baseline_year} | <strong>Performance Year:</strong> ${data.metadata.performance_year} | <strong>Games Analyzed:</strong> ${data.metadata.total_games}</p>
                        <p style="margin-top: 1rem; font-size: 0.75rem; font-style: italic; color: #9ca3af;">${data.metadata.disclaimer}</p>
                    </div>
                ` : '';
                
                contentEl.innerHTML = `
                    <div class="component-grid">${gamesHtml}</div>
                    ${metadataHtml}
                `;
                
                // Add staggered animation
                setTimeout(() => {
                    document.querySelectorAll('.animate').forEach((el, index) => {
                        el.style.animationDelay = `${index * 0.2}s`;
                    });
                }, 100);
            }
            
            // Make loadData globally available
            window.loadData = loadData;
            
            // Load data when page loads
            document.addEventListener('DOMContentLoaded', loadData);
        })();
    </script>
</body>
</html>'''
    
    @app.route('/analyze', methods=['GET'])
    def analyze():
        """Main endpoint to analyze current week's NFL matchups"""
        try:
            # Get parameters from query string
            performance_year = request.args.get('performance_year', '2025')
            baseline_year = request.args.get('baseline_year', '2024') 
            min_targets = request.args.get('min_targets', '3')
            target_week = request.args.get('week')
            
            # Set environment variables temporarily
            original_env = {}
            env_vars = {
                'PERFORMANCE_YEAR': performance_year,
                'BASELINE_YEAR': baseline_year,
                'MIN_TARGETS': min_targets
            }
            if target_week:
                env_vars['TARGET_WEEK'] = target_week
            
            # Store original values and set new ones
            for key, value in env_vars.items():
                original_env[key] = os.environ.get(key)
                os.environ[key] = value
            
            try:
                # Run analysis
                analysis_data = run_analysis()
                
                if not analysis_data:
                    return jsonify({
                        "error": "Analysis failed",
                        "message": "Could not complete analysis",
                        "timestamp": datetime.now().isoformat()
                    }), 500
                
                # Generate JSON output
                results = analysis_data['results']
                analyzer = analysis_data['analyzer']
                json_output = analyzer.generate_json_output(results)
                
                # Parse and return JSON
                return jsonify(json.loads(json_output))
                
            finally:
                # Restore original environment variables
                for key, original_value in original_env.items():
                    if original_value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = original_value
        
        except Exception as e:
            logger.error(f"Error in Flask endpoint: {str(e)}")
            return jsonify({
                "error": "Unexpected error",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }), 500
    
    @app.route('/health', methods=['GET'])
    def health_check():
        """Health check endpoint"""
        return jsonify({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "service": "NFL Matchup Analyzer"
        })
    
    @app.route('/', methods=['GET'])
    def root():
        """Root endpoint with API documentation"""
        return jsonify({
            "service": "NFL Matchup Analyzer API",
            "version": "1.0",
            "endpoints": {
                "/dashboard": {
                    "method": "GET",
                    "description": "Premium NFL prop signals dashboard"
                },
                "/analyze": {
                    "method": "GET",
                    "description": "Analyze current week's NFL matchups",
                    "parameters": {
                        "performance_year": "Year to analyze (default: 2025)",
                        "baseline_year": "Year to use for baselines (default: 2024)",
                        "min_targets": "Minimum targets for receiver analysis (default: 3)",
                        "week": "Specific week to analyze (optional, defaults to current week)"
                    },
                    "example": "/analyze?performance_year=2025&baseline_year=2024&min_targets=3&week=2"
                },
                "/health": {
                    "method": "GET", 
                    "description": "Health check endpoint"
                }
            },
            "timestamp": datetime.now().isoformat()
        })
    
    def run_flask_app():
        """Run Flask application"""
        port = int(os.environ.get('PORT', 10000))
        debug = os.environ.get('DEBUG', 'False').lower() == 'true'
        
        logger.info(f"Starting NFL Matchup Analyzer API on port {port}")
        app.run(host='0.0.0.0', port=port, debug=debug)

except ImportError:
    logger.info("Flask not available - running in CLI mode only")
    app = None
    def run_flask_app():
        logger.error("Flask not installed. Install with: pip install flask")
        sys.exit(1)

if __name__ == "__main__":
    # Check if Flask mode is requested
    if len(sys.argv) > 1 and sys.argv[1] == '--flask':
        if app is not None:
            run_flask_app()
        else:
            logger.error("Flask not available. Install with: pip install flask")
            sys.exit(1)
    else:
        main()
