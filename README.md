# 📊 Black-Litterman Portfolio Optimization

> **Optimisation de portefeuille par fusion bayésienne** — Prior de marché + vues analyste & ML, sur un univers multi-actifs (Crypto + US + BRVM).

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPortfolioOpt](https://img.shields.io/badge/PyPortfolioOpt-1.5.5+-green.svg)](https://pyportfolioopt.readthedocs.io/)

---

## 🎯 Objectifs

Ce projet implémente le modèle **Black-Litterman** (Black & Litterman, 1992), qui corrige le principal défaut de Markowitz : son instabilité due aux erreurs d'estimation des rendements.

Le projet :
1. Calcule le **prior** (rendements implicites du marché via les market caps).
2. Génère des **vues** (analyste + ML momentum).
3. Combine les deux par **fusion bayésienne** (formule Idzorek).
4. Optimise le portefeuille (Max Sharpe via SLSQP).
5. **Compare** BL vs Markowitz vs 1/N (in-sample / out-of-sample).

---

## 📸 Aperçu du rapport

![Rapport - Couverture](docs/screenshots/report_cover.png)
![Rapport - Analyse critique](docs/screenshots/report_critical.png)

---

## 🏆 Résultats clés (2021-2026)

### 1. Le portefeuille Black-Litterman

| Actif | Poids | Justification |
|-------|-------|---------------|
| US_SPY | **35.6%** | Rendement posterior solide, faible vol |
| BTC-USD | 28.8% | Prior très élevé, tempéré par les vues |
| BRVM_SNTS | 22.8% | **Vue relative "BRVM > US"** |
| BRVM_BOAB | 7.6% | Vue relative |
| ETH-USD | 3.6% | Volatilité élevée pénalisée |
| BRVM_ECOC | 1.8% | - |

*La BRVM, qui ne pèse que 0.18% du marché mondial, obtient **32%** grâce aux vues relatives.*

### 2. Comparaison out-of-sample (2025-2026)

| Portefeuille | Sharpe test | Rendement | Max DD |
|--------------|-------------|-----------|--------|
| Markowitz | **1.687** 🥇 | +20.2% | **-7.5%** |
| 1/N | 0.935 | +16.3% | -14.2% |
| **Black-Litterman** | **0.692** | +11.0% | -17.3% |

---

## 📝 Analyse critique — Un résultat honnête

**Sur notre échantillon 2021-2026, Markowitz surperforme BL.** C'est un résultat que nous assumons et documentons.

**Pourquoi ?**
1. **Prior déséquilibré** : dominé par la crypto (market caps énormes), ce qui a pénalisé BL quand la crypto a sous-performé.
2. **Période atypique** : 2025-2026 = bull BRVM (+54%) + bear crypto. Markowitz, concentré à 84% sur la BRVM, a eu "raison" par chance.
3. **Vues ML trop agressives** : le momentum a renforcé le prior au lieu de le corriger.

**Ce que ça enseigne** : BL n'est pas une baguette magique. Sa robustesse théorique dépend d'un prior bien calibré, de vues de qualité, et d'une période de test couvrant plusieurs régimes. Sur 15-20 ans et plusieurs régimes, BL démontre sa supériorité (voir Platanakis & Urquhart, 2019).

**En entretien** : *"J'ai construit un projet Black-Litterman complet. Résultat inattendu : BL sous-performe Markowitz sur mon échantillon. J'ai documenté honnêtement pourquoi. Cette analyse critique m'a permis de comprendre les limites du modèle et les pistes d'amélioration."*

---

## 🏗️ Architecture
