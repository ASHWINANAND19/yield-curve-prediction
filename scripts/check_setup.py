"""
Phase 0 check: make sure every library in the main environment works
before starting on the real data.

Each framework fits the same small fake regression problem with
12 features and 3 targets (same shape as our yield problem: 1M, 5Y, 30Y).
It also checks that FRED and Yahoo Finance can be reached.

Each framework runs in its own process. On macOS, XGBoost and PyTorch
each ship their own OpenMP library, and if both are loaded in one
process PyTorch hangs. (In the project every model has its own
notebook, so this never comes up there.)

Run:  .venv/bin/python scripts/check_setup.py
"""
import io
import subprocess
import sys
import time
import warnings

import numpy as np
import pandas as pd
import requests
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")
SEED = 42

# fake data: 1000 rows, 12 features, 3 targets
X, y = make_regression(n_samples=1000, n_features=12, n_informative=8,
                       n_targets=3, noise=10, random_state=SEED)
X = X.astype("float32")
# standardise the targets so the neural nets train quickly
y = ((y - y.mean(axis=0)) / y.std(axis=0)).astype("float32")

# train / validation / test (validation is only used for early stopping)
X_train, y_train = X[:600], y[:600]
X_val, y_val = X[600:800], y[600:800]
X_test, y_test = X[800:], y[800:]

feat_cols = [f"x{i}" for i in range(X.shape[1])]
target_cols = ["y1", "y2", "y3"]


def check_xgboost():
    from xgboost import XGBRegressor
    model = XGBRegressor(n_estimators=500, max_depth=3, learning_rate=0.05,
                         early_stopping_rounds=30, random_state=SEED)
    # xgboost accepts a 2D y directly (it builds trees for each target)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model.predict(X_test)


def check_keras():
    import keras
    keras.utils.set_random_seed(SEED)
    model = keras.Sequential([
        keras.Input(shape=(X.shape[1],)),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(3),
    ])
    model.compile(optimizer="adam", loss="mse")
    stop = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=200, batch_size=32, callbacks=[stop], verbose=0)
    return model.predict(X_test, verbose=0)


def check_fastai():
    import torch
    from fastai.tabular.all import (TabularPandas, Normalize, RegressionBlock,
                                    tabular_learner, set_seed)
    set_seed(SEED)
    df = pd.DataFrame(np.hstack([X, y]), columns=feat_cols + target_cols)
    splits = (list(range(600)), list(range(600, 800)))
    to = TabularPandas(df.iloc[:800], procs=[Normalize], cont_names=feat_cols,
                       y_names=target_cols, y_block=RegressionBlock(), splits=splits)
    # force CPU so Apple's MPS backend doesn't cause surprises
    dls = to.dataloaders(bs=64, device=torch.device("cpu"))
    learn = tabular_learner(dls, layers=[200, 100], n_out=3)
    with learn.no_bar(), learn.no_logging():
        learn.fit_one_cycle(20, 1e-2)
        test_dl = learn.dls.test_dl(df.iloc[800:])
        preds, _ = learn.get_preds(dl=test_dl)
    return preds.numpy()


def check_tabnet():
    from pytorch_tabnet.tab_model import TabNetRegressor
    model = TabNetRegressor(n_d=8, n_a=8, n_steps=3, seed=SEED,
                            verbose=0, device_name="cpu")
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
              max_epochs=200, patience=20, batch_size=256, virtual_batch_size=64)
    return model.predict(X_test)


def check_fred():
    # requests is used instead of pd.read_csv(url) because the python.org
    # build of Python on macOS can't verify SSL certificates on its own
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10&cosd=2026-09-01"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    fred = pd.read_csv(io.StringIO(response.text), na_values=".").dropna()
    print(f"  FRED  DGS10: {fred.iloc[-1, 1]:.2f}% (on {fred.iloc[-1, 0]})")


def check_yahoo():
    import yfinance as yf
    tnx = yf.Ticker("^TNX").history(period="1mo")["Close"]
    print(f"  Yahoo ^TNX : {tnx.iloc[-1]:.2f}% (on {tnx.index[-1].date()})")


CHECKS = {"XGBoost": check_xgboost, "Keras": check_keras,
          "fastai": check_fastai, "TabNet": check_tabnet}


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # child process: run a single framework and print its test R2
        pred = CHECKS[sys.argv[1]]()
        print(f"R2={r2_score(y_test, pred):.3f}")
        sys.exit(0)

    from importlib.metadata import version
    print("Python", sys.version.split()[0])
    for lib in ["numpy", "pandas", "scikit-learn", "xgboost", "tensorflow", "keras",
                "torch", "fastai", "pytorch-tabnet", "yfinance"]:
        print(f"  {lib:<15}{version(lib)}")

    print("\nData sources (latest 10Y yield from each):")
    for check in [check_fred, check_yahoo]:
        try:
            check()
        except Exception as e:
            print(f"  {check.__name__} FAILED: {e}")

    print("\nModels (R2 on the fake test set, should be well above 0.5):")
    results = []
    for name in CHECKS:
        start = time.time()
        run = subprocess.run([sys.executable, __file__, name],
                             capture_output=True, text=True, timeout=600)
        lines = run.stdout.strip().splitlines()
        if run.returncode == 0 and lines and lines[-1].startswith("R2="):
            results.append([name, "OK", float(lines[-1][3:]), round(time.time() - start, 1)])
        else:
            error = run.stderr.strip().splitlines()[-1] if run.stderr.strip() else "no output"
            results.append([name, f"FAILED: {error}", None, None])
    print(pd.DataFrame(results, columns=["framework", "status", "test_r2", "seconds"]).to_string(index=False))
