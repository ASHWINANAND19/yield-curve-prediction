"""Downloading the raw data from FRED and Yahoo Finance and building one daily table."""
import io
import time
import warnings

import pandas as pd
import requests
import yfinance as yf

from src import config

# yfinance still uses some pandas options that pandas 3 warns about (harmless)
warnings.filterwarnings("ignore", module="yfinance")


def download_fred(series_id, start=config.DOWNLOAD_START, end=config.END_DATE):
    """Download one FRED series using the public CSV link (no API key needed)."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd={start}&coed={end}"
    # requests instead of pd.read_csv(url) because the python.org build of
    # Python on macOS can't verify the SSL certificate by itself
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    df = pd.read_csv(io.StringIO(response.text), na_values=".", index_col=0, parse_dates=True)
    df.index.name = "date"
    return df[series_id]


def download_yahoo(ticker, start=config.DOWNLOAD_START, end=config.END_DATE):
    """Download daily closing prices of one Yahoo Finance ticker."""
    # the end date is exclusive in yfinance, so add one day
    end_plus_one = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    history = yf.Ticker(ticker).history(start=start, end=end_plus_one, auto_adjust=False)
    close = history["Close"]
    close.index = close.index.tz_localize(None).normalize()
    close.index.name = "date"
    return close


def download_all():
    """Download every series and save it to data/raw/ as its own CSV."""
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    for series_id in config.FRED_SERIES:
        s = download_fred(series_id)
        s.to_csv(config.RAW_DIR / f"fred_{series_id}.csv")
        print(f"FRED  {series_id:<9} {s.notna().sum():>5} values  {s.index.min().date()} to {s.index.max().date()}")

    yahoo_tickers = {**config.YAHOO_YIELDS, **config.YAHOO_MARKET}
    for ticker, name in yahoo_tickers.items():
        for attempt in range(3):
            try:
                s = download_yahoo(ticker).rename(name)
                break
            except Exception as e:   # Yahoo sometimes rate limits, so wait and retry
                print(f"  retrying {ticker}: {e}")
                time.sleep(10)
        else:
            raise RuntimeError(f"could not download {ticker} from Yahoo Finance")
        s.to_csv(config.RAW_DIR / f"yahoo_{name}.csv")
        print(f"Yahoo {ticker:<9} {len(s):>5} values  {s.index.min().date()} to {s.index.max().date()}")
        time.sleep(1)


def load_raw():
    """Read the saved raw CSVs back into two DataFrames (FRED and Yahoo)."""
    fred = {s: pd.read_csv(config.RAW_DIR / f"fred_{s}.csv", index_col=0, parse_dates=True)[s]
            for s in config.FRED_SERIES}
    names = list(config.YAHOO_YIELDS.values()) + list(config.YAHOO_MARKET.values())
    yahoo = {n: pd.read_csv(config.RAW_DIR / f"yahoo_{n}.csv", index_col=0, parse_dates=True)[n]
             for n in names}
    return fred, yahoo


def align(series, calendar):
    """Put a series on the daily calendar, carrying the last known value forward."""
    series = series.dropna()
    combined = series.reindex(calendar.union(series.index)).ffill()
    return combined.reindex(calendar)


def build_dataset(fred, yahoo):
    """Combine everything into one table with one row per Treasury trading day."""
    # master calendar = days on which Treasury yields were published
    yields = pd.DataFrame({t: fred[t] for t in config.TENORS})
    yields = yields.dropna(how="all").ffill()
    calendar = yields.index

    df = yields.copy()
    df["BAA10Y"] = align(fred["BAA10Y"], calendar)

    # weekly Fed holdings: only known a couple of days after the Wednesday they describe
    treast = fred["TREAST"].dropna()
    treast.index = treast.index + pd.Timedelta(days=config.TREAST_LAG_DAYS)
    df["TREAST"] = align(treast, calendar)

    # quarterly debt data: shift by the publication delay before forward filling
    for s, lag in config.PUBLICATION_LAG_QUARTERS.items():
        quarterly = fred[s].dropna()
        quarterly.index = quarterly.index + pd.DateOffset(months=3 * lag)
        df[s] = align(quarterly, calendar)

    # share of total public debt (in %). FDHBFIN is in billions, the rest in millions.
    df["GOV_PCT"] = 100 * df["TREAST"] / df["GFDEBTN"]
    df["HOM_PCT"] = 100 * df["FYGFDPUN"] / df["GFDEBTN"]
    df["FOR_PCT"] = 100 * df["FDHBFIN"] * 1000 / df["GFDEBTN"]

    for name, s in yahoo.items():
        df[name] = align(s, calendar)

    df = df.loc[config.START_DATE:config.END_DATE]
    return df.drop(columns=config.DEBT_SERIES)
