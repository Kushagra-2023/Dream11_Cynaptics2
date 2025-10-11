from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Tuple, Literal, List
import pandas as pd

from solver import FantasySolver

app = FastAPI(
    title="Fantasy Sports Team Solver API",
    description="An API to generate optimal fantasy cricket teams based on real player data."
)

# --- Define the Pydantic Models for the Request Body ---

class Player(BaseModel):
    name: str
    price: float
    country: str
    role: str
    total_batting_points: int
    total_bowling_points: int
    total_fielding_points: int
    total_points: int
    avg_batting_points_3: float
    avg_bowling_points_3: float
    avg_fielding_points_3: float
    avg_batting_points_10: float
    avg_bowling_points_10: float
    avg_fielding_points_10: float
    total_points_opposition: int

class TeamParams(BaseModel):
    budget: int = Field(90, description="Total salary cap for the team.")
    risk: Literal['stable', 'balanced', 'risky'] = Field('risky', description="The risk profile for the team selection.")
    total_players: int = Field(11, description="The total number of players in the team.")
    team1_name: str = Field(..., description="The name of the first team (country).")
    num_team1_players: int = Field(6, description="The number of players to select from the first team.")
    role_constraints: Dict[str, Tuple[int, int]] = Field(
        default={
            'batsman': (3, 5), 'bowler': (3, 5),
            'allrounder': (1, 3), 'wicketkeeper': (1, 2)
        },
        description="Dictionary with min/max constraints for each player role."
    )

class GenerateTeamRequest(BaseModel):
    players: List[Player]
    params: TeamParams


@app.post("/generate-team/", summary="Generate an Optimal Fantasy Team")
async def generate_team(request: GenerateTeamRequest):
    """
    Accepts a list of available players and team constraints to generate the optimal team.
    """
    try:
        # Convert the list of players into a DataFrame
        players_data = [p.model_dump() for p in request.players]
        if not players_data:
            raise HTTPException(status_code=400, detail="Player list cannot be empty.")
            
        player_df = pd.DataFrame(players_data)
        
        # Initialize the solver with the provided player data
        solver = FantasySolver(player_df)

        # Run the solver with the provided parameters
        params_dict = request.params.model_dump()
        team_df, summary = solver.solve(params_dict)

        if team_df is None:
            raise HTTPException(
                status_code=404,
                detail="Could not find an optimal team with the given constraints. Try adjusting the budget or role limits."
            )

        # Return the original columns for easier frontend mapping
        # Select columns that the frontend expects
        output_cols = [p.name for p in Player.model_fields.values()] + ['team_role']
        team_df_cleaned = team_df[[col for col in output_cols if col in team_df.columns]]
        
        team_json = team_df_cleaned.to_dict(orient='records')

        return {
            "solver_parameters": params_dict,
            "analysis_summary": summary,
            "selected_team": team_json
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
