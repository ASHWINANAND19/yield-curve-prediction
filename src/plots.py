"""Common plot style so every figure in the project looks the same."""
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from src import config

# colour-blind friendly palette (fixed order)
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
GREY = "#8a8984"

TARGET_COLORS = {"DGS1MO": BLUE, "DGS5": ORANGE, "DGS30": AQUA}

# each model always gets the same colour in every chart
MODEL_COLORS = {"Naive": GREY, "Linear": BLUE, "XGBoost": ORANGE, "Keras": AQUA,
                "fastai": YELLOW, "TabNet": MAGENTA, "Flash": GREEN}

# 8 tenors are ordered (short to long), so they use light to dark shades of one colour
TENOR_COLORS = dict(zip(config.TENORS, ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
                                        "#2a78d6", "#256abf", "#184f95", "#0d366b"]))

# for correlations: blue = negative, grey = zero, red = positive
DIVERGING = LinearSegmentedColormap.from_list("blue_red", [BLUE, "#f0efec", RED])


def setup():
    plt.rcParams.update({
        "figure.figsize": (10, 4.5),
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e4e3df",
        "grid.linewidth": 0.8,
        "axes.edgecolor": "#8a8984",
        "axes.titleweight": "bold",
        "axes.titlesize": 12,
        "lines.linewidth": 1.5,
        "legend.frameon": False,
        "font.size": 10,
    })


def save(fig, name):
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.FIG_DIR / f"{name}.png")
