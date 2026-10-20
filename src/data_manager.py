"""
Data manager du projet Black-Litterman.

Fournit :
    - Chargement des prix (crypto + US + BRVM)
    - Calcul des rendements
    - Market caps (pour le prior BL)
    - Rapport de qualité

Univers :
    - Crypto      : BTC-USD, ETH-USD
    - US          : US_SPY
    - BRVM        : BRVM_SNTS, BRVM_BOAB, BRVM_ECOC

Auteur : Statby2Mf
Projet : Black-Litterman Portfolio Optimization (M2 Statistique, UGB Saint-Louis)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
PRICES_FILE = DATA_DIR / "prices.csv"

#: Univers final
ASSETS: dict[str, str] = {
    "BTC-USD": "Bitcoin",
    "ETH-USD": "Ethereum",
    "US_SPY": "S&P 500 ETF",
    "BRVM_SNTS": "Sonatel",
    "BRVM_BOAB": "Bank of Africa",
    "BRVM_ECOC": "Ecobank",
}

#: Market caps approximatives (en milliards USD) — pour le prior BL
#: ⚠️ À mettre à jour manuellement (approximations 2026)
MARKET_CAPS: dict[str, float] = {
    "BTC-USD": 1200.0,      # ~1.2T USD
    "ETH-USD": 400.0,       # ~400B USD
    "US_SPY": 500.0,        # ETF SPY ~500B AUM
    "BRVM_SNTS": 2.5,       # ~2.5B USD (Sonatel)
    "BRVM_BOAB": 0.8,       # ~800M USD
    "BRVM_ECOC": 0.5,       # ~500M USD
}


# ---------------------------------------------------------------------------
# Structure de qualité
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DataQualityReport:
    n_obs: int
    n_assets: int
    start: pd.Timestamp
    end: pd.Timestamp
    missing_before_ffill: int
    missing_after_ffill: int

    def __str__(self) -> str:
        return (
            f"DataQualityReport(\n"
            f"  Période      : {self.start.date()} → {self.end.date()}\n"
            f"  Observations : {self.n_obs} jours × {self.n_assets} actifs\n"
            f"  NaN avant    : {self.missing_before_ffill}\n"
            f"  NaN après    : {self.missing_after_ffill}\n"
            f")"
        )


# ---------------------------------------------------------------------------
# Chargement
# ---------------------------------------------------------------------------

def load_prices() -> pd.DataFrame:
    """Charge les prix depuis data/prices.csv."""
    if not PRICES_FILE.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {PRICES_FILE}\n"
            f"→ Lance d'abord : python download_data.py"
        )
    return pd.read_csv(PRICES_FILE, index_col=0, parse_dates=True)


# ---------------------------------------------------------------------------
# Nettoyage
# ---------------------------------------------------------------------------

def clean_prices(prices: pd.DataFrame) -> tuple[pd.DataFrame, DataQualityReport]:
    """Nettoie les prix et renvoie un rapport."""
    n_before = int(prices.isna().sum().sum())
    prices = prices.dropna(how="all")
    prices = prices.ffill(limit=1)
    prices = prices.dropna()
    n_after = int(prices.isna().sum().sum())

    report = DataQualityReport(
        n_obs=len(prices),
        n_assets=prices.shape[1],
        start=prices.index[0],
        end=prices.index[-1],
        missing_before_ffill=n_before,
        missing_after_ffill=n_after,
    )
    return prices, report


# ---------------------------------------------------------------------------
# Rendements
# ---------------------------------------------------------------------------

def compute_log_returns(prices: pd.DataFrame, scale: float = 100.0) -> pd.DataFrame:
    """Calcule les log-rendements (%) — stables numériquement."""
    prices_clean, _ = clean_prices(prices)
    returns = scale * np.log(prices_clean / prices_clean.shift(1))
    return returns.dropna()


# ---------------------------------------------------------------------------
# Market caps
# ---------------------------------------------------------------------------

def get_market_caps() -> pd.Series:
    """
    Renvoie les market caps (en milliards USD) pour les actifs de l'univers.

    Utilisé pour le prior Black-Litterman (market_implied_prior_returns).
    """
    prices = load_prices()
    available = [t for t in prices.columns if t in MARKET_CAPS]
    caps = pd.Series({t: MARKET_CAPS[t] for t in available}, name="market_cap")
    return caps


# ---------------------------------------------------------------------------
# API haut-niveau
# ---------------------------------------------------------------------------

def get_all_returns() -> pd.DataFrame:
    """Renvoie les log-rendements (%) de tous les actifs."""
    prices = load_prices()
    return compute_log_returns(prices)


def get_quality_report() -> DataQualityReport:
    """Renvoie le rapport de qualité."""
    prices = load_prices()
    _, report = clean_prices(prices)
    return report


# ---------------------------------------------------------------------------
# Test rapide
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("🔍 Test du data_manager (Black-Litterman)\n")

    report = get_quality_report()
    print(report)
    print()

    print("📊 Aperçu des rendements (%) :")
    returns = get_all_returns()
    print(f"Shape : {returns.shape}")
    print(f"Période : {returns.index[0].date()} → {returns.index[-1].date()}")
    print()
    print(returns.describe().round(3).to_string())
    print()

    print("📊 Market caps (milliards USD) :")
    caps = get_market_caps()
    print(caps.to_string())
