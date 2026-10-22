"""
Modèle Black-Litterman — Fusion bayésienne prior + vues.

Ce module implémente le cœur du modèle Black-Litterman :
    - Combine le prior (rendements implicites du marché)
    - Avec des vues (analyste + ML)
    - Pour produire des rendements "posterior" corrigés
    - Et optimiser le portefeuille

Formule :
    E[R]_BL = [(τΣ)⁻¹ + PᵀΩ⁻¹P]⁻¹ [(τΣ)⁻¹Π + PᵀΩ⁻¹Q]

où :
    τ (tau) = incertitude sur le prior
    Π (Pi)  = rendements implicites du marché
    P, Q    = vues
    Ω       = incertitude des vues (Idzorek)

Références :
    - Black, F., & Litterman, R. (1992).
    - Idzorek, T. M. (2005).

Auteur : Statby2Mf
Projet : Black-Litterman Portfolio Optimization (M2 Statistique, UGB Saint-Louis)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier, black_litterman
from pypfopt.black_litterman import BlackLittermanModel

from src.market_equilibrium import compute_market_equilibrium, MarketEquilibrium
from src.views_generator import ViewsSet, build_default_views


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#: Tau : incertitude sur le prior (standard PyPortfolioOpt)
DEFAULT_TAU = 0.05

#: Contraintes de poids
MAX_WEIGHT = 1.0   # Pas de short
MIN_WEIGHT = 0.0   # Pas de short


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------

@dataclass
class BlackLittermanResult:
    """
    Résultat du modèle Black-Litterman.

    Attributes
    ----------
    assets : list[str]
        Liste des actifs.
    prior_returns : pd.Series
        Rendements implicites du marché (%).
    posterior_returns : pd.Series
        Rendements postérieurs après fusion (%).
    posterior_cov : pd.DataFrame
        Covariance postérieure.
    weights : pd.Series
        Poids optimaux du portefeuille.
    tau : float
        Paramètre τ utilisé.
    n_views : int
        Nombre de vues.
    """
    assets: list[str]
    prior_returns: pd.Series
    posterior_returns: pd.Series
    posterior_cov: pd.DataFrame
    weights: pd.Series
    tau: float
    n_views: int

    def summary(self) -> str:
        lines = [
            "═══ Black-Litterman Result ═══",
            f"  τ (tau)        : {self.tau}",
            f"  Nombre de vues : {self.n_views}",
            "",
            "  📊 Rendements prior → posterior :",
        ]
        for asset in self.assets:
            prior = self.prior_returns[asset]
            post = self.posterior_returns[asset]
            delta = post - prior
            lines.append(
                f"    {asset:15s} : "
                f"{prior:+7.2f}% → {post:+7.2f}% "
                f"(Δ {delta:+6.2f}%)"
            )

        lines.append("")
        lines.append("  💼 Poids optimaux :")
        for asset in self.assets:
            w = self.weights.get(asset, 0.0)
            bar = "█" * int(w * 50)
            lines.append(f"    {asset:15s} : {w:6.2%}  {bar}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Construction des vues pour PyPortfolioOpt
# ---------------------------------------------------------------------------

def build_bl_views(
    views_set: ViewsSet,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Convertit un ViewsSet en format PyPortfolioOpt (v1.5.5+).

    Returns
    -------
    (P, Q, confidence) : (pd.DataFrame, np.ndarray, np.ndarray)
        P : DataFrame (n_views × n_assets) avec les poids de sélection
        Q : np.ndarray des rendements attendus (en décimal)
        confidence : np.ndarray des confiances (0-1)
    """
    P_matrix, Q_series, conf_series = views_set.to_matrices()

    if len(P_matrix) == 0:
        return pd.DataFrame(), np.array([]), np.array([])

    # P en DataFrame (PyPortfolioOpt v1.5.5+ exige DataFrame ou array)
    P_df = P_matrix.astype(float)

    # Q en numpy array (décimal)
    Q_array = (Q_series / 100.0).values.astype(float)

    # Confidence en numpy array
    conf_array = conf_series.values.astype(float)

    return P_df, Q_array, conf_array

# ---------------------------------------------------------------------------
# Optimisation manuelle (contourne PyPortfolioOpt)
# ---------------------------------------------------------------------------

def optimize_max_sharpe(
    posterior_returns: pd.Series,
    posterior_cov: pd.DataFrame,
    risk_free: float = 0.02,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
) -> pd.Series:
    """
    Optimisation Max Sharpe via scipy.optimize.

    Maximise : (w'μ - r_f) / sqrt(w'Σw)
    Sujet à : Σw_i = 1, min_weight ≤ w_i ≤ max_weight

    Parameters
    ----------
    posterior_returns : pd.Series
        Rendements postérieurs (en décimal).
    posterior_cov : pd.DataFrame
        Covariance postérieure (en décimal²).
    risk_free : float
        Taux sans risque (décimal).
    min_weight, max_weight : float
        Bornes des poids.

    Returns
    -------
    pd.Series
        Poids optimaux (somme = 1).
    """
    from scipy.optimize import minimize

    assets = list(posterior_returns.index)
    n = len(assets)

    mu = np.asarray(posterior_returns.values, dtype=float)
    Sigma = np.asarray(posterior_cov.values, dtype=float)

    # Nettoyage : enlève NaN/inf
    mu = np.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)
    Sigma = np.nan_to_num(Sigma, nan=0.0, posinf=0.0, neginf=0.0)
    # Symétrise
    Sigma = (Sigma + Sigma.T) / 2
    # Régularise (assure définie positive)
    Sigma = Sigma + np.eye(n) * 1e-8

    def neg_sharpe(w):
        port_ret = w @ mu
        port_vol = np.sqrt(max(w @ Sigma @ w, 1e-12))
        return -(port_ret - risk_free) / port_vol

    # Contraintes
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(min_weight, max_weight)] * n

    # Point de départ : équipondéré
    w0 = np.ones(n) / n

    try:
        result = minimize(
            neg_sharpe, w0, method="SLSQP",
            bounds=bounds, constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-9},
        )
        if not result.success:
            # Fallback : équipondéré
            return pd.Series(1.0 / n, index=assets)
        weights = np.clip(result.x, min_weight, max_weight)
        weights = weights / weights.sum()  # renormalise
        return pd.Series(weights, index=assets)
    except Exception:
        return pd.Series(1.0 / n, index=assets)
# ---------------------------------------------------------------------------
# Modèle Black-Litterman
# ---------------------------------------------------------------------------

def run_black_litterman(
    views_set: ViewsSet | None = None,
    tau: float = DEFAULT_TAU,
    risk_free: float = 0.02,
    verbose: bool = True,
) -> BlackLittermanResult:
    """
    Pipeline complet Black-Litterman.
    """
    # 1. Prior (market equilibrium)
    if verbose:
        print("  [1/5] Calcul du prior (market equilibrium)…")
    eq = compute_market_equilibrium(delta=3.0)

    assets = list(eq.covariance.columns)
    n_assets = len(assets)

    # ⚠️ Convertit la covariance en décimal² pour PyPortfolioOpt
    cov_decimal = eq.covariance / 10_000.0

    # Prior en décimal
    prior_decimal = eq.prior_returns / 100.0

    # 2. Vues
    if views_set is None:
        if verbose:
            print("  [2/5] Génération des vues par défaut…")
        views_set = build_default_views(assets, include_ml=True)

    if verbose:
        print(f"  [3/5] Préparation des {len(views_set.views)} vues…")
    P_df, Q, confidence = build_bl_views(views_set)

    # 3. Modèle BL
    if verbose:
        print("  [4/5] Construction du modèle Black-Litterman…")
    bl = BlackLittermanModel(
        cov_matrix=cov_decimal,
        pi=prior_decimal,
        Q=Q,
        P=P_df,
        omega="idzorek",
        view_confidences=confidence,
        tau=tau,
    )

    posterior_returns = bl.bl_returns()
    posterior_cov = bl.bl_cov()

    # ⚠️ CORRECTION CRUCIALE : forcer DataFrame avec les bons index/columns
    posterior_cov = pd.DataFrame(
        np.asarray(posterior_cov),
        index=assets,
        columns=assets,
    )
    posterior_returns = pd.Series(
        np.asarray(posterior_returns),
        index=assets,
    )

    # 4. Optimisation du portefeuille (via scipy, contourne PyPortfolioOpt)
    if verbose:
        print("  [5/5] Optimisation du portefeuille (max Sharpe via scipy)…")

    weights = optimize_max_sharpe(
        posterior_returns=posterior_returns,
        posterior_cov=posterior_cov,
        risk_free=risk_free,
        min_weight=MIN_WEIGHT,
        max_weight=MAX_WEIGHT,
    )

    if verbose:
        print(f"     ✅ Optimisation OK (somme des poids = {weights.sum():.4f})")

    # Récap
    prior_pct = prior_decimal * 100
    posterior_pct = posterior_returns * 100

    return BlackLittermanResult(
        assets=assets,
        prior_returns=prior_pct,
        posterior_returns=posterior_pct,
        posterior_cov=posterior_cov,
        weights=weights,
        tau=tau,
        n_views=len(views_set.views),
    )
# ---------------------------------------------------------------------------
# Test rapide
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("🔍 Test du module Black-Litterman")
    print("=" * 70)

    result = run_black_litterman(tau=0.05)
    print()
    print(result.summary())
