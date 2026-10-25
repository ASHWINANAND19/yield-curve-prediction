"""Settings shared by every notebook, so all models use exactly the same data."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_FILE = ROOT / "data" / "processed" / "dataset.csv"
RESULTS_DIR = ROOT / "results"
PRED_DIR = RESULTS_DIR / "predictions"
FIG_DIR = RESULTS_DIR / "figures"

# data is frozen at END_DATE so the results can be reproduced later.
# Downloads start earlier so the lagged quarterly debt data covers 2010.
DOWNLOAD_START = "2008-01-01"
START_DATE = "2010-01-01"
END_DATE = "2026-09-30"

# time based split (never random, otherwise the model sees the future)
TRAIN_END = "2021-12-31"
VAL_END = "2022-12-31"

HORIZON = 5     # predict 5 trading days (one week) ahead

# Every model is trained in three experiments:
#   A: predict the yield level (as in the original case study), all 23 features
#   B: predict the 5-day change, all 23 features
#   C: predict the 5-day change, without the 8 raw yield levels as inputs
#      (only spreads, changes, ratios and market indicators, which don't trend)
EXPERIMENTS = {
    "A": {"mode": "level", "features": "all"},
    "B": {"mode": "change", "features": "all"},
    "C": {"mode": "change", "features": "no_levels"},
}
EXPERIMENT_NAMES = {"A": "A: level, all features",
                    "B": "B: change, all features",
                    "C": "C: change, no yield levels"}
SEED = 42
SEEDS = [42, 7, 2024]   # neural nets are run with 3 seeds

# ---- FRED series ----
TENORS = ["DGS1MO", "DGS3MO", "DGS1", "DGS2", "DGS5", "DGS7", "DGS10", "DGS30"]
TARGETS = ["DGS1MO", "DGS5", "DGS30"]
TARGET_NAMES = {"DGS1MO": "1M", "DGS5": "5Y", "DGS30": "30Y"}
TENOR_LABELS = {"DGS1MO": "1M", "DGS3MO": "3M", "DGS1": "1Y", "DGS2": "2Y",
                "DGS5": "5Y", "DGS7": "7Y", "DGS10": "10Y", "DGS30": "30Y"}
DEBT_SERIES = ["TREAST", "FYGFDPUN", "FDHBFIN", "GFDEBTN"]
FRED_SERIES = TENORS + ["BAA10Y"] + DEBT_SERIES

# Quarterly debt numbers are published months after the quarter they describe.
# FRED dates them at the start of the quarter, so we shift them forward by
# roughly the publication delay to avoid using numbers before they existed.
PUBLICATION_LAG_QUARTERS = {"FYGFDPUN": 2, "GFDEBTN": 2, "FDHBFIN": 3}
# weekly Fed holdings (Wednesday level) come out on Thursday afternoon
TREAST_LAG_DAYS = 2

# ---- Yahoo Finance tickers ----
# Treasury yield indices (only used to cross check against FRED)
YAHOO_YIELDS = {"^IRX": "YH_3M", "^FVX": "YH_5Y", "^TNX": "YH_10Y", "^TYX": "YH_30Y"}
# market sentiment features (Brent instead of WTI because WTI went negative in 2020)
YAHOO_MARKET = {"^VIX": "VIX", "^GSPC": "SP500", "BZ=F": "BRENT",
                "GC=F": "GOLD", "DX-Y.NYB": "DXY"}
