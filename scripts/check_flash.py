"""
Phase 0 check for the Lightning Flash environment (.venv-flash, Python 3.10).

Fits Flash's TabularRegressor on the same fake data as check_setup.py,
one model per target.

Run:  .venv-flash/bin/python scripts/check_flash.py
"""
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score

import torch
import pytorch_lightning as pl
import flash
from flash.tabular import TabularRegressionData, TabularRegressor

warnings.filterwarnings("ignore")
SEED = 42
BACKBONE = "category_embedding"   # plain MLP, so it isn't a repeat of TabNet

print("Python", sys.version.split()[0])
for lib in [np, pd, torch, pl, flash]:
    print(f"  {lib.__name__:<18}{lib.__version__}")
print("Available backbones:", TabularRegressor.available_backbones())

# same fake data as check_setup.py
X, y = make_regression(n_samples=1000, n_features=12, n_informative=8,
                       n_targets=3, noise=10, random_state=SEED)
y = (y - y.mean(axis=0)) / y.std(axis=0)

feat_cols = [f"x{i}" for i in range(X.shape[1])]
target_cols = ["y1", "y2", "y3"]
df = pd.DataFrame(np.hstack([X, y]), columns=feat_cols + target_cols)
train_df, val_df, test_df = df.iloc[:600], df.iloc[600:800], df.iloc[800:]

start = time.time()
test_preds = []
for target in target_cols:
    pl.seed_everything(SEED, workers=True)
    datamodule = TabularRegressionData.from_data_frame(
        numerical_fields=feat_cols,
        target_field=target,
        train_data_frame=train_df,
        val_data_frame=val_df,
        predict_data_frame=test_df[feat_cols],
        batch_size=64,
    )
    model = TabularRegressor.from_data(datamodule, backbone=BACKBONE, learning_rate=1e-3)
    trainer = flash.Trainer(max_epochs=30, accelerator="cpu", logger=False,
                            enable_progress_bar=False, enable_model_summary=False,
                            enable_checkpointing=False)
    trainer.fit(model, datamodule=datamodule)
    batches = trainer.predict(model, datamodule=datamodule)
    test_preds.append(np.concatenate([np.ravel(b) for b in batches]))

pred = np.column_stack(test_preds)
r2 = r2_score(test_df[target_cols].values, pred)
print(f"\nLightning Flash ({BACKBONE}): test R2 = {r2:.3f}, {time.time() - start:.1f}s")
