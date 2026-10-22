"""
Calcul du prior Black-Litterman (rendements implicites du marché).

Le prior BL repose sur l'hypothèse que le marché est en équilibre :
les rendements attendus sont "implicites" dans les prix actuels.

Formule :
    Π = δ · Σ · w_mkt

où :
    δ (delta) = coefficient d'aversion au risque du marché
    Σ (Sigma) = matrice de covariance des rendements
    w_mkt     = poids de marché (proportionnels aux market caps)

Référence :
    Black, F., & Litterman, R. (1992). Global Portfolio Optimization.
    Idzorek, T. M. (2005). A Step-by-Step Guide to the Black-Litterman Model.

Auteur : Statby2Mf
Projet : Black-Litterman Portfolio Optimization (M2 Statistique, UGB Saint-Louis)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data_manager import get_all_returns, get_market_caps


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#: Taux sans risque annuel (approximation 2021-2026)
RISK_FREE_RATE = 0.02

#: Jours de trading par an
TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------

@dataclass
class MarketEquilibrium:
    """
    Résultat du calcul du prior Black-Litterman.

    Attributes
    ----------
    weights_market : pd.Series
        Poids de marché (normalisés à 100%).
    covariance : pd.DataFrame
        Matrice de covariance annualisée des rendements.
    delta : float
        Coefficient d'aversion au risque (typiquement 2-4).
    prior_returns : pd.Series
        Rendements implicites annuels (%).
    """
    weights_market: pd.Series
    covariance: pd.DataFrame
    delta: float
    prior_returns: pd.Series

    def summary(self) -> str:
        lines = [
            "═══ Market Equilibrium (Prior Black-Litterman) ═══",
            f"  δ (aversion au risque) : {self.delta:.4f}",
            "",
            "  📊 Poids de marché :",
        ]
        for asset, w in self.weights_market.items():
            lines.append(f"    {asset:15s} : {w:6.2%}")

        lines.append("")
        lines.append("  📈 Rendements implicites (annualisés) :")
        for asset, r in self.prior_returns.items():
            lines.append(f"    {asset:15s} : {r:+7.3f}%")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Calcul des poids de marché
# ---------------------------------------------------------------------------

def compute_market_weights(market_caps: pd.Series) -> pd.Series:
    """
    Calcule les poids de marché à partir des market caps.

    Poids_i = market_cap_i / somme(market_caps)

    Parameters
    ----------
    market_caps : pd.Series
        Market caps (milliards USD).

    Returns
    -------
    pd.Series
        Poids normalisés (somme = 1).
    """
    if len(market_caps) == 0:
        raise ValueError("Market caps vides")
    return market_caps / market_caps.sum()


# ---------------------------------------------------------------------------
# Matrice de covariance
# ---------------------------------------------------------------------------

def compute_covariance(
    returns: pd.DataFrame,
    annualize: bool = True,
) -> pd.DataFrame:
    """
    Calcule la matrice de covariance des rendements.

    Parameters
    ----------
    returns : pd.DataFrame
        Log-rendements (%) journaliers.
    annualize : bool
        Si True, annualise (× 252).

    Returns
    -------
    pd.DataFrame
        Matrice de covariance (K×K).
    """
    cov = returns.cov()
    if annualize:
        cov = cov * TRADING_DAYS
    return cov


# ---------------------------------------------------------------------------
# Coefficient delta
# ---------------------------------------------------------------------------

def compute_delta(
    weights_market: pd.Series,
    covariance: pd.DataFrame,
    market_return: float = 0.10,
    risk_free: float = RISK_FREE_RATE,
) -> float:
    """
    Calcule le coefficient δ d'aversion au risque.

    Formule (Idzorek 2005) :
        δ = (R_marché - r_f) / σ²_marché

    ⚠️ ATTENTION AUX UNITÉS :
        - market_return, risk_free : en décimal (0.10 = 10%)
        - covariance : en %² (car rendements × 100)
        - Il faut diviser la covariance par 10 000 pour la mettre en décimal²
    """
    w = weights_market.values
    Sigma = covariance.values

    # ⚠️ Conversion : de %² à décimal² (÷ 100² = ÷ 10 000)
    Sigma_decimal = Sigma / 10_000.0

    sigma_market_sq = float(w @ Sigma_decimal @ w)

    if sigma_market_sq <= 0:
        raise ValueError("Variance de marché nulle")

    delta = (market_return - risk_free) / sigma_market_sq
    return float(delta)


# ---------------------------------------------------------------------------
# Prior Black-Litterman
# ---------------------------------------------------------------------------

def compute_prior_returns(
    delta: float,
    covariance: pd.DataFrame,
    weights_market: pd.Series,
) -> pd.Series:
    """
    Calcule les rendements implicites du marché.

    Formule (Black-Litterman 1992) :
        Π = δ · Σ · w_mkt

    Parameters
    ----------
    delta : float
        Aversion au risque.
    covariance : pd.DataFrame
        Matrice de covariance annualisée.
    weights_market : pd.Series
        Poids de marché.

    Returns
    -------
    pd.Series
        Rendements implicites annualisés (en %).
    """
        # ⚠️ Formule : Π = δ · Σ · w_mkt
    # Comme Sigma est en %² et w_mkt en décimal, Pi sera en %².
    # Il faut multiplier par 100 pour revenir en %.
    # MAIS attention : Σ en %² × w en décimal = %²
    # Et on veut Π en % (rendement).
    # Π = δ · (Σ_en_%² / 100) · w → en % (car Σ/100 est en % de rendement)

    # Version correcte : garder Σ en %², multiplier par w (décimal), puis
    # diviser par 100 pour obtenir un rendement en %.

    Pi = delta * (covariance.values @ weights_market.values) / 100.0
    Pi_series = pd.Series(Pi, index=covariance.index, name="prior_return")
    return Pi_series


# ---------------------------------------------------------------------------
# Pipeline complet
# ---------------------------------------------------------------------------

def compute_market_equilibrium(
    market_return: float = 0.10,
) -> MarketEquilibrium:
    """
    Pipeline complet : données → poids marché → covariance → delta → prior.

    Parameters
    ----------
    market_return : float
        Rendement annuel attendu du marché (ex : 0.10 = +10%).
        Typiquement 8-12% pour un portefeuille multi-actifs.

    Returns
    -------
    MarketEquilibrium
    """
    # 1. Données
    returns = get_all_returns()
    caps = get_market_caps()

    # 2. Poids de marché
    weights = compute_market_weights(caps)

    # 3. Covariance
    cov = compute_covariance(returns, annualize=True)

    # 4. Delta
    delta = compute_delta(weights, cov, market_return=market_return)

    # 5. Prior
    prior = compute_prior_returns(delta, cov, weights)

    return MarketEquilibrium(
        weights_market=weights,
        covariance=cov,
        delta=delta,
        prior_returns=prior,
    )


# ---------------------------------------------------------------------------
# Test rapide
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("🔍 Test du module market_equilibrium\n")

    eq = compute_market_equilibrium()
    print(eq.summary())

    print()
    print("📊 Matrice de covariance (annualisée, en %²) :")
    print(eq.covariance.round(2).to_string())
