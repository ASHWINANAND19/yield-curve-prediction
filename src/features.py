"""Feature engineering, targets and the train / validation / test split."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src import config

# the 12 predictors used in the original case study
BOOK_FEATURES = config.TENORS + ["GOV_PCT", "HOM_PCT", "FOR_PCT", "BAA10Y"]
# shape of the curve and recent momentum
CURVE_FEATURES = ["SLOPE_10Y_2Y", "SLOPE_10Y_3M", "CURVATURE",
                  "DGS1MO_CHG5", "DGS5_CHG5", "DGS30_CHG5"]
# market sentiment from Yahoo Finance
MARKET_FEATURES = ["VIX", "SP500_RET5", "BRENT_RET5", "GOLD_RET5", "DXY_RET5"]
ALL_FEATURES = BOOK_FEATURES + CURVE_FEATURES + MARKET_FEATURES
# everything except the 8 raw yield levels (used in experiment C)
NO_LEVEL_FEATURES = [f for f in ALL_FEATURES if f not in config.TENORS]
FEATURE_SETS = {"all": ALL_FEATURES, "no_levels": NO_LEVEL_FEATURES}


def add_features(df):
    df = df.copy()
    df["SLOPE_10Y_2Y"] = df["DGS10"] - df["DGS2"]
    df["SLOPE_10Y_3M"] = df["DGS10"] - df["DGS3MO"]
    df["CURVATURE"] = 2 * df["DGS5"] - df["DGS2"] - df["DGS10"]
    for t in config.TARGETS:
        df[f"{t}_CHG5"] = df[t].diff(config.HORIZON)
    for name in ["SP500", "BRENT", "GOLD", "DXY"]:
        df[f"{name}_RET5"] = 100 * df[name].pct_change(config.HORIZON)
    return df


def load_dataset():
    """Processed dataset from notebook 01 plus the engineered features."""
    df = pd.read_csv(config.PROCESSED_FILE, index_col="date", parse_dates=True)
    return add_features(df)


def make_splits(df, features, mode):
    """
    Build X and y for the train, validation and test sets.

    mode = "level"  : y is the yield 5 trading days later (as in the book)
    mode = "change" : y is how much the yield moves over the next 5 days

    Each set also keeps today's yield ("today") and the real future yield
    ("future") so predictions can always be turned back into yield levels.
    """
    df = df.dropna(subset=features)
    today = df[config.TARGETS]
    future = today.shift(-config.HORIZON)
    y = future if mode == "level" else future - today

    keep = y.notna().all(axis=1)          # the last 5 days have no target yet
    df, today, future, y = df[keep], today[keep], future[keep], y[keep]

    dates = df.index
    train = np.asarray(dates <= config.TRAIN_END)
    val = np.asarray((dates > config.TRAIN_END) & (dates <= config.VAL_END))
    test = np.asarray(dates > config.VAL_END)
    # gap: drop the last 5 rows of train and val so that no target
    # window overlaps with the next set
    train[np.where(train)[0][-config.HORIZON:]] = False
    val[np.where(val)[0][-config.HORIZON:]] = False

    splits = {}
    for name, mask in [("train", train), ("val", val), ("test", test)]:
        splits[name] = {"X": df.loc[mask, features], "y": y[mask],
                        "today": today[mask], "future": future[mask]}
    return splits


def scale(data):
    """
    Standardise X and y using training statistics only (neural nets train much
    better when inputs and outputs are around 0 with similar spread).
    Returns the scaled arrays, the fitted X scaler and a function to undo the y scaling.
    """
    x_scaler = StandardScaler().fit(data["train"]["X"])
    y_mean = data["train"]["y"].mean().values
    y_std = data["train"]["y"].std().values
    scaled = {}
    for part in ["train", "val", "test"]:
        X = x_scaler.transform(data[part]["X"]).astype("float32")
        y = ((data[part]["y"].values - y_mean) / y_std).astype("float32")
        scaled[part] = (X, y)

    def unscale_y(pred):
        return np.asarray(pred).reshape(-1, len(y_mean)) * y_std + y_mean

    return scaled, x_scaler, unscale_y


def get_experiment(df, experiment):
    """Splits for experiment "A", "B" or "C" (see config.EXPERIMENTS), plus its target mode."""
    settings = config.EXPERIMENTS[experiment]
    feature_list = FEATURE_SETS[settings["features"]]
    return make_splits(df, feature_list, settings["mode"]), settings["mode"]
