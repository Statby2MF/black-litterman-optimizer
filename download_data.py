"""
Script de préparation des données pour Black-Litterman.

Stratégie :
    - Crypto (BTC, ETH) : téléchargés via yfinance
    - US (SPY) et BRVM (SNTS, BOAB, ECOC) : lus depuis data/brvm/*.csv

Usage :
    python download_data.py
    python download_data.py --force

Auteur : Statby2Mf
Projet : Black-Litterman Portfolio Optimization (M2 Statistique, UGB Saint-Louis)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yfinance as yf

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
BRVM_DIR = DATA_DIR / "brvm"
BRVM_DIR.mkdir(exist_ok=True)

PRICES_FILE = DATA_DIR / "prices.csv"

# Crypto via yfinance
CRYPTO_TICKERS = {"BTC-USD": "Bitcoin", "ETH-USD": "Ethereum"}

# Fichiers CSV locaux
LOCAL_FILES = {
    "US_SPY": "US_SPY.csv",
    "BRVM_SNTS": "BRVM_SNTS.csv",
    "BRVM_BOAB": "BRVM_BOAB.csv",
    "BRVM_ECOC": "BRVM_ECOC.csv",
}

START_DATE = "2018-01-01"


def download_crypto(ticker: str, start: str = START_DATE) -> pd.Series:
    print(f"   → {ticker}… ", end="", flush=True)
    df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if df.empty:
        raise RuntimeError(f"Aucune donnée pour {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        prices = df["Close"][ticker]
    else:
        prices = df["Close"]
    prices = prices.dropna().sort_index()
    prices.name = ticker
    print(f"{len(prices)} obs ({prices.index[0].date()} → {prices.index[-1].date()})")
    return prices


def load_csv_file(filename: str, col_name: str) -> pd.Series | None:
    path = BRVM_DIR / filename
    if not path.exists():
        print(f"   ⚠️ {filename} introuvable → ignoré")
        return None
    df = pd.read_csv(path)
    date_col = next((c for c in ["Date", "date", "DATE"] if c in df.columns), df.columns[0])
    price_col = next((c for c in ["Close", "close", "CLOSE"] if c in df.columns), df.columns[1])
    df[date_col] = pd.to_datetime(df[date_col], utc=True).dt.tz_localize(None)
    df = df.set_index(date_col)[price_col]
    df = df.sort_index()
    df.name = col_name
    df = df[~df.index.duplicated(keep="last")]
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if PRICES_FILE.exists() and not args.force:
        print(f"ℹ️  {PRICES_FILE} existe déjà. Utilise --force.")
        return

    print(f"📥 Téléchargement crypto ({len(CRYPTO_TICKERS)} actifs)…")
    series_list = [download_crypto(t) for t in CRYPTO_TICKERS.keys()]

    print(f"\n📥 Chargement CSV locaux ({len(LOCAL_FILES)} actifs)…")
    for col_name, filename in LOCAL_FILES.items():
        print(f"   → {col_name}… ", end="", flush=True)
        s = load_csv_file(filename, col_name)
        if s is not None:
            print(f"{len(s)} obs ({s.index[0].date()} → {s.index[-1].date()})")
            series_list.append(s)

    prices = pd.concat(series_list, axis=1, join="outer", sort=False).sort_index()
    prices.to_csv(PRICES_FILE)
    print(f"\n💾 Sauvegardé : {PRICES_FILE}")
    print(f"   Shape : {prices.shape}")
    print(f"   Colonnes : {list(prices.columns)}")

    print("\n" + "=" * 70)
    print("📊 RAPPORT DE QUALITÉ")
    print("=" * 70)
    print(f"Période : {prices.index[0].date()} → {prices.index[-1].date()}")
    print(f"Observations : {len(prices)}")
    print(f"Actifs : {prices.shape[1]}")
    print("\nNaN par colonne :")
    print(prices.isna().sum().to_string())
    print("=" * 70)


if __name__ == "__main__":
    main()
