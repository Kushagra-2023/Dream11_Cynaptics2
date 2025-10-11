import pandas as pd
import numpy as np
import pulp

class FantasySolver:
    """
    Encapsulates player data and optimization logic to find the best team.
    This solver now accepts a DataFrame of player stats to work with real data.
    """
    def __init__(self, player_df: pd.DataFrame):
        """
        Initializes the solver with a DataFrame of available players.
        """
        if player_df.empty:
            raise ValueError("Player DataFrame cannot be empty.")
        self.player_df = self._preprocess_data(player_df)

    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocesses the raw player data to calculate projected scores and variance.
        """
        # --- 1. Normalize Stats for Fair Comparison ---
        # We normalize columns so that a 10-point difference in one stat doesn't
        # unfairly outweigh a 10-point difference in another.
        stats_cols = [
            'total_points', 'total_points_opposition',
            'avg_batting_points_3', 'avg_bowling_points_3', 'avg_fielding_points_3',
            'avg_batting_points_10', 'avg_bowling_points_10', 'avg_fielding_points_10'
        ]
        for col in stats_cols:
            if df[col].max() > df[col].min():
                df[f'norm_{col}'] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
            else:
                df[f'norm_{col}'] = 0.5 # Assign a neutral value if all values are the same

        # --- 2. Calculate Projected Score ---
        # This formula weights different aspects of a player's performance.
        df['projected_score'] = (
            df['norm_total_points'] * 0.30 +                 # Overall performance
            df['norm_total_points_opposition'] * 0.15 +      # History vs opponent
            (df['norm_avg_batting_points_3'] + df['norm_avg_bowling_points_3'] + df['norm_avg_fielding_points_3']) / 3 * 0.35 + # Recent form (3 matches)
            (df['norm_avg_batting_points_10'] + df['norm_avg_bowling_points_10'] + df['norm_avg_fielding_points_10']) / 3 * 0.20  # Mid-term form (10 matches)
        )

        # --- 3. Calculate Variance (as a proxy for risk) ---
        # A simple measure of variance: the standard deviation of their recent points.
        df['variance'] = df[['avg_batting_points_3', 'avg_bowling_points_3', 'avg_fielding_points_3']].std(axis=1)
        df['variance'] = df['variance'].fillna(df['variance'].mean()) # Handle players with no variance

        return df

    def solve(self, params: dict):
        """
        Runs the PuLP solver based on the provided strategy parameters.
        """
        prob = pulp.LpProblem("FantasyTeamSelection", pulp.LpMaximize)
        player_indices = self.player_df.index
        player_vars = pulp.LpVariable.dicts("Player", player_indices, 0, 1, cat='Binary')
        captain_vars = pulp.LpVariable.dicts("Captain", player_indices, 0, 1, cat='Binary')
        vc_vars = pulp.LpVariable.dicts("ViceCaptain", player_indices, 0, 1, cat='Binary')

        # The objective is now to maximize the sum of projected scores
        prob += pulp.lpSum(
            [player_vars[i] * self.player_df.loc[i, 'projected_score'] for i in player_indices] +
            [captain_vars[i] * self.player_df.loc[i, 'projected_score'] for i in player_indices] +
            [vc_vars[i] * 0.5 * self.player_df.loc[i, 'projected_score'] for i in player_indices]
        ), "TotalProjectedScore"

        # --- CONSTRAINTS ---
        prob += pulp.lpSum([player_vars[p] for p in player_indices]) == params['total_players']
        prob += pulp.lpSum([self.player_df.loc[i, 'price'] * player_vars[i] for i in player_indices]) <= params['budget']

        for role, (min_val, max_val) in params['role_constraints'].items():
            role_players = self.player_df[self.player_df['role'].str.lower() == role.lower()].index
            prob += pulp.lpSum([player_vars[p] for p in role_players]) >= min_val
            prob += pulp.lpSum([player_vars[p] for p in role_players]) <= max_val

        # Team constraint using the provided team name
        team1_players = self.player_df[self.player_df['country'] == params['team1_name']].index
        prob += pulp.lpSum([player_vars[p] for p in team1_players]) == params['num_team1_players']

        # Captain and Vice-Captain constraints
        prob += pulp.lpSum([captain_vars[p] for p in player_indices]) == 1
        prob += pulp.lpSum([vc_vars[p] for p in player_indices]) == 1
        for i in player_indices:
            prob += captain_vars[i] <= player_vars[i]
            prob += vc_vars[i] <= player_vars[i]
            prob += captain_vars[i] + vc_vars[i] <= 1

        # Risk constraint based on variance
        if params['risk'] in ['stable', 'risky']:
            avg_variance = self.player_df['variance'].mean()
            baseline_total_variance = avg_variance * params['total_players']
            if params['risk'] == 'stable':
                prob += pulp.lpSum([self.player_df.loc[i, 'variance'] * player_vars[i] for i in player_indices]) <= baseline_total_variance * 0.9
            else: # risky
                prob += pulp.lpSum([self.player_df.loc[i, 'variance'] * player_vars[i] for i in player_indices]) >= baseline_total_variance * 1.1

        prob.solve(pulp.PULP_CBC_CMD(msg=0))

        if pulp.LpStatus[prob.status] == "Optimal":
            selected_indices = [i for i in player_indices if player_vars[i].varValue > 0]
            captain_index = [i for i in player_indices if captain_vars[i].varValue > 0][0]
            vc_index = [i for i in player_indices if vc_vars[i].varValue > 0][0]

            team_df = self.player_df.loc[selected_indices].copy()
            team_df['team_role'] = 'Player'
            team_df.loc[captain_index, 'team_role'] = 'Captain'
            team_df.loc[vc_index, 'team_role'] = 'Vice-Captain'

            summary = {
                'total_projected_score': float(round(pulp.value(prob.objective), 2)),
                'total_cost': int(team_df['price'].sum()),
                'total_variance': int(team_df['variance'].sum()),
            }
            return team_df, summary
        else:
            return None, None
