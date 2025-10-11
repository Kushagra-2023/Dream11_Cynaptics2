import io
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- Configuration ---
matplotlib.use("Agg")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],  # your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Data Loading and Constants ---
PLAYER_DATA_FILE_PATH = "player_summary.csv"
STATS_TO_NORMALIZE = {
    "Batting S/R": "higher",
    "Bowling S/R": "lower",
    "Economy Rate": "lower",
    "Wickets": "higher",
    "Catches": "higher",
}
PLOT_CATEGORIES = list(STATS_TO_NORMALIZE.keys())

# --- Load Data on Startup ---
try:
    data = pd.read_csv(PLAYER_DATA_FILE_PATH, index_col=0)
    data.index.name = "Player"

    # After loading, keep only the columns we need for the plot.
    data = data[PLOT_CATEGORIES]
    data.dropna(subset=PLOT_CATEGORIES, inplace=True)

except FileNotFoundError:
    print(f"FATAL: The data file '{PLAYER_DATA_FILE_PATH}' was not found.")
    exit()
except (ValueError, KeyError) as e:
    print(
        f"FATAL: A required stat column was not found. Make sure these columns exist: {PLOT_CATEGORIES}. Details: {e}"
    )
    exit()

# --- Helper Functions ---


def normalize_zscore(player_stats_series: pd.Series) -> list:
    """
    Converts a player's raw stats into scaled Z-scores (0-100 range).
    A score of 50 is average.
    """
    scaled_scores = []
    for stat, preference in STATS_TO_NORMALIZE.items():
        stat_column = data[stat].dropna()
        player_value = player_stats_series[stat]

        mu = stat_column.mean()
        std_dev = stat_column.std()

        if std_dev == 0:
            z_score = 0
        else:
            z_score = (player_value - mu) / std_dev

        if preference == "lower":
            z_score *= -1

        scaled_score = 50 + (z_score * 15)
        scaled_score = np.clip(scaled_score, 0, 100)

        scaled_scores.append(round(scaled_score, 2))

    return scaled_scores


def generate_spider_plot(
    categories: list, player_values: list, player_names: list
) -> io.BytesIO:
    """Generates a spider plot and returns it as an in-memory buffer."""
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    ax.set_ylim(0, 100)

    # Player 1
    values1 = player_values[0]
    ax.plot(
        angles,
        values1,
        color="red",
        linewidth=2,
        linestyle="solid",
        label=player_names[0],
    )
    ax.fill(angles, values1, "r", alpha=0.1)

    # Player 2 (if provided)
    if len(player_values) > 1:
        values2 = player_values[1]
        ax.plot(
            angles,
            values2,
            color="blue",
            linewidth=2,
            linestyle="solid",
            label=player_names[1],
        )
        ax.fill(angles, values2, "b", alpha=0.1)
        plt.title(
            f"{player_names[0]} vs {player_names[1]} Comparison", size=20, color="gray"
        )
    # else:
    #     plt.title(f"Skill Summary", size=20, color="gray")

    ax.set_xticks(angles[:-1])

    # CHANGED: Increased font size and darkened color for better readability
    ax.set_xticklabels(categories, color="black", size=30)

    ax.set_yticklabels([])
    ax.grid(color="grey", linestyle="--", linewidth=0.5)
    # plt.legend(loc="upper right", bbox_to_anchor=(1.1, 1))
    buf = io.BytesIO()
    plt.savefig(buf, format="png", transparent=True, dpi=300, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)

    return buf


def get_player_stats(player_name: str) -> pd.Series:
    """Finds a player by index and returns their stats as a Pandas Series."""
    try:
        player_series = data.loc[player_name]
        return player_series
    except KeyError:
        raise HTTPException(
            status_code=404, detail=f"Player '{player_name}' not found."
        )


# --- API Endpoints ---


@app.post("/players/summary/{player_name}")
def endp_single_player(player_name: str):
    """Generates and returns a single player's skill plot as a PNG image."""
    player_stats_series = get_player_stats(player_name)
    values = normalize_zscore(player_stats_series)
    values += values[:1]

    image_buffer = generate_spider_plot(
        categories=PLOT_CATEGORIES, player_values=[values], player_names=[player_name]
    )

    # FIXED: Return the image directly as a response
    return Response(content=image_buffer.getvalue(), media_type="image/png")


# @app.post("/players/comparison/{player1_name}")
# def endp_double_player(player1_name: str, player2_name: str):
#     """Generates and returns a two-player comparison plot as a PNG image."""

class ComparisonRequest(BaseModel):
    player1_name: str
    player2_name: str
@app.post("/players/comparison")
def endp_double_player(req: ComparisonRequest):
    player1_name = req.player1_name
    player2_name = req.player2_name
    player1_stats = get_player_stats(player1_name)
    values1 = normalize_zscore(player1_stats)
    values1 += values1[:1]

    player2_stats = get_player_stats(player2_name)
    values2 = normalize_zscore(player2_stats)
    values2 += values2[:1]

    image_buffer = generate_spider_plot(
        categories=PLOT_CATEGORIES,
        player_values=[values1, values2],
        player_names=[player1_name, player2_name],
    )

    # FIXED: Return the image directly as a streaming response
    return Response(content=image_buffer.getvalue(), media_type="image/png")
