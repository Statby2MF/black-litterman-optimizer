"""
Génération de vues pour Black-Litterman.

Ce module fournit 3 types de vues :
    1. Vues ABSOLUES      : "BTC fera +X%"
    2. Vues RELATIVES     : "BRVM sur-performera US de +X%"
    3. Vues ML            : Prédictions d'un modèle simple (momentum)

Chaque vue est accompagnée d'un niveau de confiance (0-100%),
utilisé par l'approche Idzorek pour calibrer Ω.

Référence :
    Idzorek, T. M. (2005). A Step-by-Step Guide to the
    Black-Litterman Model.

Auteur : Statby2Mf
Projet : Black-Litterman Portfolio Optimization (M2 Statistique, UGB Saint-Louis)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from src.data_manager import get_all_returns, ASSETS


# ---------------------------------------------------------------------------
# Structure d'une vue
# ---------------------------------------------------------------------------

@dataclass
class View:
    """
    Une vue Black-Litterman.

    Attributes
    ----------
    name : str
        Nom lisible de la vue.
    P : pd.Series
        Vecteur de sélection (taille = nombre d'actifs).
        Les valeurs peuvent être négatives (pour les vues relatives).
    Q : float
        Rendement attendu de la vue (en %).
    confidence : float
        Confiance dans la vue (0 à 1).
    view_type : str
        'absolute', 'relative', ou 'ml'.
    """
    name: str
    P: pd.Series
    Q: float
    confidence: float
    view_type: str

    def __post_init__(self):
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"confidence doit être dans [0, 1], reçu {self.confidence}")
        if self.view_type not in ("absolute", "relative", "ml"):
            raise ValueError(f"view_type invalide : {self.view_type}")


@dataclass
class ViewsSet:
    """
    Ensemble de vues pour Black-Litterman.

    Attributes
    ----------
    assets : list[str]
        Liste des actifs (ordre important pour P).
    views : list[View]
        Liste des vues.
    """
    assets: list[str]
    views: list[View] = field(default_factory=list)

    def add_view(self, view: View):
        """Ajoute une vue après vérification des dimensions."""
        if len(view.P) != len(self.assets):
            raise ValueError(
                f"P a {len(view.P)} éléments, attendu {len(self.assets)}"
            )
        self.views.append(view)

    def to_matrices(self) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Convertit les vues en matrices (P, Q, confidence).

        Returns
        -------
        (P, Q, confidence) : (DataFrame, Series, Series)
        """
        if len(self.views) == 0:
            return (
                pd.DataFrame(columns=self.assets),
                pd.Series(dtype=float),
                pd.Series(dtype=float),
            )

        P_matrix = pd.DataFrame(
            [v.P.values for v in self.views],
            columns=self.assets,
            index=[v.name for v in self.views],
        )
        Q_vector = pd.Series(
            [v.Q for v in self.views],
            index=[v.name for v in self.views],
        )
        conf_vector = pd.Series(
            [v.confidence for v in self.views],
            index=[v.name for v in self.views],
        )
        return P_matrix, Q_vector, conf_vector

    def summary(self) -> str:
        lines = [f"═══ Views Set ({len(self.views)} vues) ═══"]
        for v in self.views:
            if v.view_type == "absolute":
                asset_idx = v.P.idxmax()
                lines.append(
                    f"  [{v.view_type:8s}] {v.name:25s} : "
                    f"{asset_idx} → {v.Q:+.2f}% (conf {v.confidence:.0%})"
                )
            elif v.view_type == "relative":
                long_assets = v.P[v.P > 0].index.tolist()
                short_assets = v.P[v.P < 0].index.tolist()
                lines.append(
                    f"  [{v.view_type:8s}] {v.name:25s} : "
                    f"long {long_assets} / short {short_assets} "
                    f"→ {v.Q:+.2f}% (conf {v.confidence:.0%})"
                )
            else:
                lines.append(
                    f"  [{v.view_type:8s}] {v.name:25s} : "
                    f"→ {v.Q:+.2f}% (conf {v.confidence:.0%})"
                )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Constructeur de vue absolue
# ---------------------------------------------------------------------------

def make_absolute_view(
    assets: list[str],
    asset: str,
    expected_return: float,
    confidence: float = 0.5,
    name: str | None = None,
) -> View:
    """
    Crée une vue absolue : "L'actif X fera Y%".

    Parameters
    ----------
    assets : list[str]
    asset : str
        L'actif concerné.
    expected_return : float
        Rendement attendu (en %).
    confidence : float
        Confiance (0-1).
    name : str, optional
    """
    if asset not in assets:
        raise ValueError(f"Actif '{asset}' inconnu. Disponibles : {assets}")

    P = pd.Series(0.0, index=assets)
    P[asset] = 1.0

    return View(
        name=name or f"Abs: {asset} = {expected_return:+.1f}%",
        P=P,
        Q=float(expected_return),
        confidence=confidence,
        view_type="absolute",
    )


# ---------------------------------------------------------------------------
# Constructeur de vue relative
# ---------------------------------------------------------------------------

def make_relative_view(
    assets: list[str],
    long_assets: list[str],
    short_assets: list[str],
    expected_spread: float,
    confidence: float = 0.5,
    name: str | None = None,
) -> View:
    """
    Crée une vue relative : "Les long_assets sur-performeront les short_assets de X%".

    Parameters
    ----------
    assets : list[str]
    long_assets : list[str]
        Actifs qui vont sur-performer.
    short_assets : list[str]
        Actifs qui vont sous-performer.
    expected_spread : float
        Écart de rendement attendu (en %).
    confidence : float
    """
    for a in long_assets + short_assets:
        if a not in assets:
            raise ValueError(f"Actif '{a}' inconnu")

    P = pd.Series(0.0, index=assets)
    # Poids équipondérés pour long et short
    for a in long_assets:
        P[a] = 1.0 / len(long_assets)
    for a in short_assets:
        P[a] = -1.0 / len(short_assets)

    return View(
        name=name or f"Rel: {long_assets} vs {short_assets}",
        P=P,
        Q=float(expected_spread),
        confidence=confidence,
        view_type="relative",
    )


# ---------------------------------------------------------------------------
# Vues basées sur ML (momentum)
# ---------------------------------------------------------------------------

def generate_ml_views(
    assets: list[str],
    lookback: int = 60,
    confidence: float = 0.3,
) -> list[View]:
    """
    Génère des vues basées sur un modèle "momentum" (ML simple).

    Principe : si un actif a eu un rendement positif sur les N derniers
    jours, on prédit qu'il continuera à monter.

    ⚠️ Confiance faible (30%) car momentum est un signal simple.

    Parameters
    ----------
    assets : list[str]
    lookback : int
        Fenêtre de lookback (60 jours par défaut).
    confidence : float
        Confiance (0.3 = faible, car momentum est basique).

    Returns
    -------
    list[View]
    """
    returns = get_all_returns()
    available = [a for a in assets if a in returns.columns]
    if len(available) < 1:
        return []

    # Rendement cumulé sur la fenêtre
    recent_returns = returns[available].tail(lookback)
    momentum = (1 + recent_returns / 100).prod() - 1  # en décimal
    momentum_pct = momentum * 100  # en %

    views = []
    for asset in available:
        # Prédit un rendement proportionnel au momentum (mais amorti)
        predicted = float(momentum_pct[asset]) * 0.5  # amorti
        if abs(predicted) < 1.0:  # ignore les prédictions trop faibles
            continue

        view = make_absolute_view(
            assets=assets,
            asset=asset,
            expected_return=predicted,
            confidence=confidence,
            name=f"ML Momentum: {asset}",
        )
        # Override view_type
        view.view_type = "ml"
        views.append(view)

    return views


# ---------------------------------------------------------------------------
# Set de vues par défaut (analyste)
# ---------------------------------------------------------------------------

def build_default_views(
    assets: list[str],
    include_ml: bool = True,
) -> ViewsSet:
    """
    Construit un set de vues par défaut combinant :
        - Vues absolues (analyste) : BTC +25%, ETH +20%, US_SPY +8%
        - Vues relatives : BRVM sur-performera US de +3%
        - Vues ML (momentum) : si include_ml=True

    Parameters
    ----------
    assets : list[str]
    include_ml : bool

    Returns
    -------
    ViewsSet
    """
    vs = ViewsSet(assets=assets)

    # --- Vues absolues analyste ---
    # (plus prudentes que le prior BL)
    if "BTC-USD" in assets:
        vs.add_view(make_absolute_view(
            assets, "BTC-USD", expected_return=25.0, confidence=0.6,
            name="Analyste: BTC = +25%",
        ))

    if "ETH-USD" in assets:
        vs.add_view(make_absolute_view(
            assets, "ETH-USD", expected_return=20.0, confidence=0.5,
            name="Analyste: ETH = +20%",
        ))

    if "US_SPY" in assets:
        vs.add_view(make_absolute_view(
            assets, "US_SPY", expected_return=8.0, confidence=0.7,
            name="Analyste: US_SPY = +8%",
        ))

    # --- Vue relative : BRVM > US ---
    brvm_assets = [a for a in assets if a.startswith("BRVM_")]
    if brvm_assets and "US_SPY" in assets:
        vs.add_view(make_relative_view(
            assets,
            long_assets=brvm_assets,
            short_assets=["US_SPY"],
            expected_spread=3.0,
            confidence=0.4,
            name=f"Analyste: {brvm_assets} > US_SPY",
        ))

    # --- Vues ML ---
    if include_ml:
        ml_views = generate_ml_views(assets, confidence=0.3)
        for v in ml_views:
            vs.add_view(v)

    return vs


# ---------------------------------------------------------------------------
# Test rapide
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("🔍 Test du module views_generator\n")

    # Récupère les actifs disponibles
    returns = get_all_returns()
    assets = list(returns.columns)

    print(f"📊 Actifs disponibles ({len(assets)}) :")
    for a in assets:
        print(f"   - {a}")

    # Construit les vues par défaut
    print("\n" + "=" * 70)
    print("🔄 Construction des vues par défaut")
    print("=" * 70)
    vs = build_default_views(assets, include_ml=True)
    print(vs.summary())

    # Matrices
    print("\n" + "=" * 70)
    print("📊 Matrices P, Q, confidence")
    print("=" * 70)
    P, Q, conf = vs.to_matrices()

    print("\nMatrice P (sélection des actifs) :")
    print(P.round(3).to_string())

    print("\nVecteur Q (rendements attendus des vues, en %) :")
    print(Q.round(2).to_string())

    print("\nVecteur confidence (0-1) :")
    print(conf.round(2).to_string())
