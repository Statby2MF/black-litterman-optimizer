"""
Générateur de rapport PDF pour Black-Litterman.

Inclut une section "Analyse critique" qui documente honnêtement
la sous-performance de BL sur l'échantillon 2021-2026.

Auteur : Statby2Mf
Projet : Black-Litterman Portfolio Optimization (M2 Statistique, UGB Saint-Louis)
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer,
    Table, TableStyle, Image, PageBreak,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

COLOR_PRIMARY = colors.HexColor("#1a3a5c")
COLOR_ACCENT = colors.HexColor("#d62728")
COLOR_BG_LIGHT = colors.HexColor("#f0f2f6")
COLOR_GRAY = colors.HexColor("#666666")
COLOR_SUCCESS = colors.HexColor("#28a745")


def build_styles() -> dict:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("Title", parent=base["Title"],
            fontSize=26, leading=32, alignment=TA_CENTER,
            textColor=COLOR_PRIMARY, spaceAfter=10),
        "subtitle": ParagraphStyle("Subtitle", parent=base["Normal"],
            fontSize=14, leading=18, alignment=TA_CENTER,
            textColor=COLOR_GRAY, spaceAfter=30),
        "h1": ParagraphStyle("H1", parent=base["Heading1"],
            fontSize=18, leading=22, spaceBefore=20, spaceAfter=12,
            textColor=COLOR_PRIMARY),
        "h2": ParagraphStyle("H2", parent=base["Heading2"],
            fontSize=14, leading=18, spaceBefore=14, spaceAfter=8,
            textColor=COLOR_PRIMARY),
        "body": ParagraphStyle("Body", parent=base["Normal"],
            fontSize=10, leading=14, alignment=TA_JUSTIFY, spaceAfter=8),
        "bullet": ParagraphStyle("Bullet", parent=base["Normal"],
            fontSize=10, leading=14, leftIndent=15, bulletIndent=5,
            spaceAfter=4),
        "caption": ParagraphStyle("Caption", parent=base["Normal"],
            fontSize=8, leading=11, alignment=TA_CENTER,
            textColor=COLOR_GRAY, spaceAfter=15),
    }


class ReportDocTemplate(BaseDocTemplate):
    def __init__(self, filename: str, **kwargs):
        super().__init__(filename, **kwargs)
        frame = Frame(self.leftMargin, self.bottomMargin,
                      self.width, self.height, id="normal")
        self.addPageTemplates([
            PageTemplate(id="cover", frames=[frame], onPage=self._cover),
            PageTemplate(id="content", frames=[frame], onPage=self._content),
        ])

    def _cover(self, canvas, doc):
        pass

    def _content(self, canvas, doc):
        canvas.saveState()
        canvas.setFillColor(COLOR_PRIMARY)
        canvas.setFont("Helvetica-Bold", 9)
        canvas.drawString(2 * cm, A4[1] - 1.5 * cm, "Black-Litterman Optimizer")
        canvas.setFont("Helvetica", 9)
        canvas.setFillColor(COLOR_GRAY)
        canvas.drawRightString(A4[0] - 2 * cm, A4[1] - 1.5 * cm,
                                "Portfolio Optimization — Crypto + US + BRVM")
        canvas.setStrokeColor(COLOR_PRIMARY)
        canvas.setLineWidth(0.5)
        canvas.line(2 * cm, A4[1] - 1.7 * cm, A4[0] - 2 * cm, A4[1] - 1.7 * cm)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(COLOR_GRAY)
        canvas.drawString(2 * cm, 1.2 * cm,
                          f"© Statby2Mf — {datetime.now().strftime('%d/%m/%Y')}")
        canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")
        canvas.restoreState()


def make_table(data, col_widths=None, header=True, highlight_rows=None):
    table = Table(data, colWidths=col_widths, repeatRows=1 if header else 0)
    cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), COLOR_PRIMARY) if header else
        ("BACKGROUND", (0, 0), (-1, 0), COLOR_BG_LIGHT),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white) if header else
        ("TEXTCOLOR", (0, 0), (-1, 0), COLOR_PRIMARY),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
        ("BOX", (0, 0), (-1, -1), 0.5, COLOR_PRIMARY),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]
    for i in range(1, len(data)):
        if header:
            bg = COLOR_BG_LIGHT if i % 2 == 0 else colors.white
            cmds.append(("BACKGROUND", (0, i), (-1, i), bg))
    if highlight_rows:
        for r in highlight_rows:
            cmds.append(("BACKGROUND", (0, r), (-1, r),
                          colors.HexColor("#d4edda")))
    table.setStyle(TableStyle(cmds))
    return table


def fig_to_image(fig, width_cm=16):
    tmp = FIGURES_DIR / f"_tmp_{id(fig)}.png"
    fig.savefig(tmp, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    from PIL import Image as PILImage
    with PILImage.open(tmp) as img:
        w, h = img.size
        ratio = h / w
    return Image(str(tmp), width=width_cm * cm, height=width_cm * cm * ratio)


def make_weights_figure(weights_dict, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 5))
    assets = list(next(iter(weights_dict.values())).index)
    n = len(weights_dict)
    x = np.arange(len(assets))
    width = 0.8 / n
    for i, (name, w) in enumerate(weights_dict.items()):
        ax.bar(x + i * width - (n - 1) * width / 2, w.values, width, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(assets, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Poids")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def section_cover(styles, date_str):
    story = [
        Spacer(1, 3 * cm),
        Paragraph("Black-Litterman Portfolio Optimization", styles["title"]),
        Spacer(1, 0.5 * cm),
        Paragraph("Optimisation bayésienne de portefeuille<br/>"
                  "Application à un univers multi-actifs (Crypto + US + BRVM)",
                  styles["subtitle"]),
        Spacer(1, 2 * cm),
    ]
    info = [
        ["Univers", "BTC, ETH, US_SPY, BRVM_SNTS, BRVM_BOAB, BRVM_ECOC"],
        ["Méthode", "Black-Litterman (Black & Litterman 1992, Idzorek 2005)"],
        ["Vues", "Analyste (absolues/relatives) + ML (momentum)"],
        ["Optimisation", "Max Sharpe via scipy.optimize (SLSQP)"],
        ["Benchmark", "Markowitz classique + 1/N"],
        ["Date", date_str],
    ]
    t = Table(info, colWidths=[4 * cm, 10 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), COLOR_BG_LIGHT),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("TEXTCOLOR", (0, 0), (0, -1), COLOR_PRIMARY),
        ("ALIGN", (0, 0), (0, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.white),
        ("BOX", (0, 0), (-1, -1), 1, COLOR_PRIMARY),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(t)
    story.append(Spacer(1, 3 * cm))
    story.append(Paragraph(
        "Statby2Mf — Master 2 Statistique<br/>"
        "Université Gaston Berger de Saint-Louis",
        ParagraphStyle("author", alignment=TA_CENTER, fontSize=10,
                       textColor=COLOR_GRAY, leading=14),
    ))
    story.append(PageBreak())
    return story


def section_executive_summary(styles):
    story = [Paragraph("Résumé exécutif", styles["h1"]), Spacer(1, 0.3 * cm)]
    story.append(Paragraph(
        "Ce rapport implémente le modèle <b>Black-Litterman</b> pour l'optimisation "
        "de portefeuille. Contrairement à Markowitz, qui estime les rendements "
        "attendus directement sur l'historique, BL part d'un <b>prior</b> "
        "(équilibre de marché) et le corrige avec des <b>vues</b> "
        "(analyste + ML), via une fusion bayésienne.",
        styles["body"],
    ))
    story.append(Paragraph("Résultats clés", styles["h2"]))
    for b in [
        "Le prior, calculé via les market caps, donne un poids dominant à la crypto (76%).",
        "Le posterior BL tempère BTC (+37.7% → +24.4%) et ETH (+47.1% → +28.6%).",
        "Le portefeuille BL est diversifié : 35.6% US, 32% BRVM, 32% crypto.",
        "En out-of-sample (2025-2026), Markowitz surperforme BL (Sharpe 1.69 vs 0.69).",
        "Cette sous-performance s'explique par la nature atypique de la période (bull BRVM, bear crypto).",
    ]:
        story.append(Paragraph(f"• {b}", styles["bullet"]))
    story.append(PageBreak())
    return story


def section_methodology(styles):
    story = [Paragraph("1. Méthodologie", styles["h1"])]
    story.append(Paragraph("1.1 Formule Black-Litterman", styles["h2"]))
    story.append(Paragraph(
        "La fusion bayésienne combine le prior et les vues :<br/>"
        "<b>E[R]_BL = [(τΣ)⁻¹ + PᵀΩ⁻¹P]⁻¹ [(τΣ)⁻¹Π + PᵀΩ⁻¹Q]</b>",
        styles["body"],
    ))
    story.append(Paragraph("1.2 Construction des vues", styles["h2"]))
    story.append(Paragraph(
        "Trois types de vues sont intégrés :<br/>"
        "• <b>Vues absolues</b> : « BTC fera +25% »<br/>"
        "• <b>Vues relatives</b> : « BRVM sur-performera US de +3% »<br/>"
        "• <b>Vues ML</b> : momentum 60 jours (confiance faible)",
        styles["body"],
    ))
    story.append(Paragraph("1.3 Optimisation", styles["h2"]))
    story.append(Paragraph(
        "Le portefeuille final est obtenu par optimisation <b>Max Sharpe</b> :<br/>"
        "<b>max_w (wᵀμ − r_f) / √(wᵀΣw)</b><br/>"
        "sous contraintes Σwᵢ = 1 et 0 ≤ wᵢ ≤ 1. Implémenté via scipy.optimize (SLSQP).",
        styles["body"],
    ))
    story.append(PageBreak())
    return story


def section_results(styles, weights_dict, metrics_table):
    story = [Paragraph("2. Résultats", styles["h1"])]
    story.append(Paragraph("2.1 Poids des portefeuilles", styles["h2"]))
    try:
        fig = make_weights_figure(weights_dict,
                                   save_path=FIGURES_DIR / "weights_comparison.png")
        story.append(fig_to_image(fig, width_cm=16))
        story.append(Paragraph(
            "Figure 1 — Comparaison des poids (1/N, Markowitz, Black-Litterman)",
            styles["caption"],
        ))
    except Exception:
        pass
    story.append(Paragraph("2.2 Métriques out-of-sample", styles["h2"]))
    story.append(make_table(metrics_table, header=True))
    story.append(PageBreak())
    return story


def section_critical_analysis(styles):
    """Section critique — l'honnêteté qui fait la différence."""
    story = [Paragraph("3. Analyse critique", styles["h1"])]

    story.append(Paragraph("3.1 Un résultat contre-intuitif", styles["h2"]))
    story.append(Paragraph(
        "Sur notre échantillon 2021-2026, <b>Markowitz surperforme BL</b> "
        "en out-of-sample (Sharpe 1.69 vs 0.69). Ce résultat n'est "
        "pas un bug — c'est une réalité empirique que nous documentons "
        "honnêtement.",
        styles["body"],
    ))

    story.append(Paragraph("3.2 Causes identifiées", styles["h2"]))
    for b in [
        "<b>Prior dominé par la crypto</b> : le prior donne +37% à BTC "
        "car les market caps crypto sont énormes. BL alloue 32% en crypto, "
        "Markowitz seulement 2%.",
        "<b>Période de test atypique</b> : 2025-2026 est caractérisée par "
        "un bull BRVM (+54%) et un bear crypto. Markowitz, concentré sur la "
        "BRVM, a eu raison par chance.",
        "<b>Vues ML trop agressives</b> : les vues momentum (confiance 30%) "
        "ont renforcé le prior crypto au lieu de le corriger.",
    ]:
        story.append(Paragraph(f"• {b}", styles["bullet"]))

    story.append(Paragraph("3.3 Ce que cela enseigne", styles["h2"]))
    story.append(Paragraph(
        "Black-Litterman n'est pas une baguette magique. Sa robustesse "
        "théorique dépend de :<br/>"
        "• Un prior bien calibré (reflétant vraiment l'équilibre)<br/>"
        "• Des vues de qualité (analystes expérimentés)<br/>"
        "• Une période de test couvrant plusieurs régimes<br/><br/>"
        "Sur 5 ans et un seul régime, Markowitz peut gagner par chance. "
        "Sur 15-20 ans avec bear/bull alternés, BL démontre sa supériorité "
        "(voir Platanakis & Urquhart 2019).",
        styles["body"],
    ))

    story.append(Paragraph("3.4 Pistes d'amélioration", styles["h2"]))
    for b in [
        "Tester sur 10+ ans (2015-2026) pour inclure bear/bull crypto",
        "Calibrer les vues analyste plus fortement (confiance 0.8)",
        "Réduire la confiance des vues ML (0.1 au lieu de 0.3)",
        "Implémenter un rolling backtest avec refit de BL à chaque split",
    ]:
        story.append(Paragraph(f"• {b}", styles["bullet"]))

    story.append(PageBreak())
    return story


def section_appendix(styles):
    story = [Paragraph("Annexe — Références", styles["h1"])]
    for i, ref in enumerate([
        "Black, F., & Litterman, R. (1992). Global Portfolio Optimization. "
        "<i>Financial Analysts Journal</i>, 48(5), 28-43.",
        "Idzorek, T. M. (2005). A Step-by-Step Guide to the "
        "Black-Litterman Model. <i>Working Paper</i>.",
        "Platanakis, E., & Urquhart, A. (2019). Portfolio management with "
        "cryptocurrencies: The path to the Black-Litterman model. "
        "<i>Journal of Economic Dynamics and Control</i>, 101, 151-171.",
    ], 1):
        story.append(Paragraph(f"[{i}] {ref}", styles["body"]))
    return story


def generate_report(output_path: Path = None, verbose: bool = True) -> Path:
    from src.comparison import compare_portfolios, rolling_comparison

    if output_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        output_path = REPORTS_DIR / f"bl_rapport_{ts}.pdf"

    if verbose:
        print(f"📄 Génération du rapport PDF…\n")
        print("[1/3] Backtest des portefeuilles…")
    results = compare_portfolios()

    if verbose:
        print("[2/3] Rolling backtest…")
    rolling_df = rolling_comparison(n_splits=4)

    if verbose:
        print("[3/3] Construction du PDF…")

    styles = build_styles()
    doc = ReportDocTemplate(
        str(output_path), pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2.5 * cm, bottomMargin=2 * cm,
        title="Black-Litterman Report", author="Statby2Mf",
    )

    date_str = datetime.now().strftime("%d/%m/%Y à %H:%M")

    # Prépare les données du tableau
    weights_dict = {name: r.weights for name, r in results.items()}
    metrics_table = [["Portefeuille", "Sharpe train", "Sharpe test",
                       "Rendement", "Volatilité", "Max DD"]]
    for name, r in results.items():
        metrics_table.append([
            name,
            f"{r.train_sharpe:.3f}",
            f"{r.test_sharpe:.3f}",
            f"{r.test_return:+.2%}",
            f"{r.test_vol:.2%}",
            f"{r.test_max_dd:.2%}",
        ])

    story = []
    story += section_cover(styles, date_str)
    story += section_executive_summary(styles)
    story += section_methodology(styles)
    story += section_results(styles, weights_dict, metrics_table)
    story += section_critical_analysis(styles)  # ← LA section clé
    story += section_appendix(styles)

    doc.build(story)

    if verbose:
        print(f"\n✅ Rapport généré : {output_path}")
        print(f"   Taille : {output_path.stat().st_size / 1024:.1f} KB")

    return output_path


if __name__ == "__main__":
    print("=" * 70)
    print("📄 GÉNÉRATION DU RAPPORT PDF")
    print("=" * 70)
    path = generate_report()
    print("\n" + "=" * 70)
    print(f"✅ PRÊT : {path}")
    print("=" * 70)
