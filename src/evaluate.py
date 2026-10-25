"""Metrics and saving results, shared by all model notebooks."""
import inspect
import json

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src import config


def to_level(pred, today, mode):
    """Turn model output back into a yield level (in %)."""
    pred = np.asarray(pred, dtype=float).reshape(len(today), -1)
    return pred + today.values if mode == "change" else pred


def evaluate(future, pred_level, today):
    """Test metrics for each target. All inputs are yield levels in %."""
    pred_level = np.asarray(pred_level)
    rows = []
    for i, t in enumerate(config.TARGETS):
        actual, pred, base = future[t].values, pred_level[:, i], today[t].values
        rmse = np.sqrt(mean_squared_error(actual, pred))
        naive_rmse = np.sqrt(mean_squared_error(actual, base))  # "no change" forecast
        actual_chg, pred_chg = actual - base, pred - base
        moved = actual_chg != 0
        if np.allclose(pred_chg, 0):
            direction = np.nan            # the naive model never predicts a direction
        else:
            direction = np.mean(np.sign(pred_chg[moved]) == np.sign(actual_chg[moved]))
        rows.append({
            "target": config.TARGET_NAMES[t],
            "RMSE": rmse,
            "MAE": mean_absolute_error(actual, pred),
            "R2": r2_score(actual, pred),
            "R2_change": r2_score(actual_chg, pred_chg),
            "skill_vs_naive": 1 - rmse / naive_rmse,
            "direction_acc": direction,
        })
    return pd.DataFrame(rows).set_index("target")


def average_rmse(future, pred_level):
    return np.mean(np.sqrt(((future.values - np.asarray(pred_level)) ** 2).mean(axis=0)))


def save_predictions(model_name, experiment, seed, split, pred_level):
    """Save test predictions so the comparison notebook only has to read CSVs."""
    config.PRED_DIR.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(index=split["future"].index)
    for i, t in enumerate(config.TARGETS):
        out[f"{t}_actual"] = split["future"][t]
        out[f"{t}_pred"] = np.asarray(pred_level)[:, i]
        out[f"{t}_today"] = split["today"][t]
    out.to_csv(config.PRED_DIR / f"{model_name}_{experiment}_seed{seed}.csv")


def save_run_info(model_name, info):
    """Save training time, lines of code etc. (used for the Simplicity score)."""
    folder = config.RESULTS_DIR / "run_info"
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / f"{model_name}.json", "w") as f:
        json.dump(info, f, indent=2)


def save_importance(model_name, importance, method):
    folder = config.RESULTS_DIR / "importance"
    folder.mkdir(parents=True, exist_ok=True)
    importance.rename(method).to_csv(folder / f"{model_name}.csv")


def count_lines(func):
    """Lines of code in a function, not counting blank lines and comments."""
    lines = inspect.getsource(func).splitlines()
    return sum(1 for line in lines if line.strip() and not line.strip().startswith("#"))


def permutation_importance(predict_level, X, future, n_repeats=5, seed=config.SEED):
    """
    Shuffle one feature at a time and see how much the test RMSE gets worse.
    Works for any model, so it's used for the neural nets that have no
    built in feature importance.
    """
    rng = np.random.default_rng(seed)
    base_rmse = average_rmse(future, predict_level(X))
    result = {}
    for col in X.columns:
        increase = []
        for _ in range(n_repeats):
            X_shuffled = X.copy()
            X_shuffled[col] = rng.permutation(X_shuffled[col].values)
            increase.append(average_rmse(future, predict_level(X_shuffled)) - base_rmse)
        result[col] = np.mean(increase)
    return pd.Series(result).sort_values(ascending=False)
