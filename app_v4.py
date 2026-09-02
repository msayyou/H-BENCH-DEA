# app.py — DEA-H v3.9
# REIV Hospitality · Asset Management Hôtelier

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import warnings

from dea_model import HotelDEAAnalyzer, QUADRANT_LABELS

# ── Multi-Module DEA-H (v3.2) ────────────────────────────────────────────────
from modules_config import MODULES, check_module_feasibility
from dea_model import run_multi_module
from synthesis_tab import render_synthesis_tab, render_module_selector
# --- pdf_fiche_actif inline ---
from io import BytesIO, StringIO
from datetime import datetime
import numpy as np

# ── ReportLab & Kaleido — imports globaux ────────────────────────────────────
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                    Table, TableStyle, HRFlowable, Image, PageBreak)
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

try:
    import kaleido  # noqa: F401
    KALEIDO_AVAILABLE = True
except ImportError:
    KALEIDO_AVAILABLE = False


def plotly_to_png_bytes(fig, width=700, height=400, scale=2):
    if not KALEIDO_AVAILABLE:
        return None
    try:
        return fig.to_image(format="png", width=width, height=height, scale=scale)
    except Exception:
        return None


def generate_fiche_actif_pdf(
    hotel: str,
    dea,
    quadrant_labels: dict,
    avg_salary: float = 35_000,
    revpar_value: float = 1,
    jours_exploit: int = 365,
) -> bytes:
    """
    Génère un PDF Fiche Actif pour un hôtel.
    Retourne bytes prêts pour st.download_button.
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.lib import colors
        from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                        Table, TableStyle, HRFlowable)
        from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    except ImportError:
        return b""

    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=1.8*cm, rightMargin=1.8*cm,
                            topMargin=1.5*cm, bottomMargin=1.5*cm)

    # ── Couleurs REIV ────────────────────────────────────────────────────────
    NAVY   = colors.HexColor("#1a3a5c")
    BLUE   = colors.HexColor("#2e6da4")
    GREEN  = colors.HexColor("#27ae60")
    ORANGE = colors.HexColor("#f39c12")
    RED    = colors.HexColor("#e74c3c")
    LGRAY  = colors.HexColor("#f5f5f5")
    WHITE  = colors.white

    styles = getSampleStyleSheet()

    def S(name, **kw):
        return ParagraphStyle(name, parent=styles["Normal"], **kw)

    title_s  = S("T",  fontSize=16, textColor=NAVY, fontName="Helvetica-Bold", spaceAfter=2)
    sub_s    = S("S",  fontSize=9,  textColor=BLUE, fontName="Helvetica",      spaceAfter=4)
    h2_s     = S("H2", fontSize=10, textColor=NAVY, fontName="Helvetica-Bold", spaceBefore=8, spaceAfter=3)
    body_s   = S("B",  fontSize=8,  textColor=colors.black, fontName="Helvetica")
    small_s  = S("SM", fontSize=7,  textColor=colors.grey,  fontName="Helvetica-Oblique")
    alert_s  = S("AL", fontSize=8,  textColor=RED,  fontName="Helvetica-Bold")

    # ── Data ────────────────────────────────────────────────────────────────
    bcc   = dea.bcc_scores.get(hotel, 0)
    ccr   = dea.ccr_scores.get(hotel, 0)
    scale = dea.scale_efficiency.get(hotel, 0)
    q     = dea.quadrants.get(hotel, "—")
    qlbl  = quadrant_labels.get(q, q)
    trank = dea.topsis_ranks.get(hotel, "—")
    tscore= dea.topsis_scores.get(hotel, 0)
    seg   = dea.kmeans_labels.get(hotel, "—")
    n     = dea.n

    raw   = dea.df.loc[hotel]
    _has_ch = 'nb_chambres' in raw.index and pd.notna(raw['nb_chambres'])
    lits  = float(raw["nb_chambres"]) if _has_ch else None
    emp   = float(raw["nb_employes"])
    costs = float(raw["couts_op_ex"])
    rvp   = float(raw["revpar"])
    sat   = float(raw["satisfaction"])
    occ   = float(raw["taux_occupation"])
    nights = (lits * jours_exploit * occ / 100) if _has_ch else None
    ca_est = (rvp * lits * jours_exploit) if _has_ch else None

    slk_emp = dea.slacks.get(hotel, {}).get("inputs", {}).get("nb_employes", 0)
    slk_rvp = dea.slacks.get(hotel, {}).get("outputs", {}).get("revpar", 0)
    up_fte  = round(slk_emp * avg_salary / 1000)
    # Chambres disponibles x jours — le RevPAR intègre déjà l'occupation
    up_rev  = (round(slk_rvp * lits * jours_exploit * revpar_value / 1_000_000, 2)
               if _has_ch else None)

    peers   = dea.peers.get(hotel, {})
    targets = dea.targets.get(hotel, {})

    bcc_color = GREEN if bcc >= 0.90 else ORANGE if bcc >= 0.80 else RED

    def score_color(v):
        return GREEN if v >= 0.90 else ORANGE if v >= 0.80 else RED

    # ── Story ────────────────────────────────────────────────────────────────
    story = []

    # Header
    story.append(Paragraph(f"FICHE ACTIF DEA-H — {hotel}", title_s))
    story.append(Paragraph(
        f"REIV Hospitality · DEA-H v3.5 · Généré le {datetime.now().strftime('%d/%m/%Y %H:%M')} · "
        f"Jours exploitation : {jours_exploit}j",
        small_s
    ))
    story.append(HRFlowable(width="100%", thickness=2, color=NAVY, spaceAfter=6))

    # KPIs principaux
    kpi_data = [
        ["Score BCC", "Score CCR", "Eff. Échelle", "Rang TOPSIS", "Quadrant", "Segment"],
        [f"{bcc:.1%}", f"{ccr:.1%}", f"{scale:.1%}", f"#{trank}/{n}", qlbl[:15], seg[:20]],
    ]
    kpi_table = Table(kpi_data, colWidths=[2.8*cm]*6)
    kpi_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), NAVY),
        ("TEXTCOLOR",  (0,0), (-1,0), WHITE),
        ("BACKGROUND", (0,1), (-1,1), LGRAY),
        ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTNAME",   (0,1), (-1,1), "Helvetica-Bold"),
        ("FONTSIZE",   (0,0), (-1,-1), 8),
        ("ALIGN",      (0,0), (-1,-1), "CENTER"),
        ("VALIGN",     (0,0), (-1,-1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [LGRAY]),
        ("GRID",       (0,0), (-1,-1), 0.5, colors.white),
        ("ROUNDEDCORNERS", [3]),
    ]))
    story.append(kpi_table)
    story.append(Spacer(1, 6))

    # Données brutes
    story.append(Paragraph("Données opérationnelles", h2_s))
    raw_data = [
        ["Indicateur",       "Valeur",              "Indicateur",           "Valeur"],
        ["ETP",              f"{emp:.0f}",           "RevPAR",              f"{rvp:.0f} €"],
        ["Taux occupation",  f"{occ:.1f}%",          "Satisfaction",        f"{sat:.1f}/10"],
        ["Charges op.",      f"{costs:.2f} M€",      "Chambres" if _has_ch else "Chambres",
         f"{int(lits)}" if _has_ch else "n/d"],
    ]
    if _has_ch:
        raw_data.append(["CA estimé",        f"{ca_est/1e6:.2f} M€", "Nuitées estimées",    f"{nights:,.0f}"])
    else:
        raw_data.append(["CA estimé", "n/d — chambres", "Nuitées estimées", "n/d — chambres"])
    raw_table = Table(raw_data, colWidths=[3.8*cm, 2.8*cm, 4.0*cm, 2.8*cm])
    raw_table.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,0), BLUE),
        ("TEXTCOLOR",   (0,0), (-1,0), WHITE),
        ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTNAME",    (0,1), (-1,-1), "Helvetica"),
        ("FONTSIZE",    (0,0), (-1,-1), 8),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [WHITE, LGRAY]),
        ("GRID",        (0,0), (-1,-1), 0.3, colors.lightgrey),
        ("ALIGN",       (1,0), (1,-1), "RIGHT"),
        ("ALIGN",       (3,0), (3,-1), "RIGHT"),
    ]))
    story.append(raw_table)
    story.append(Spacer(1, 6))

    # Slacks & upside
    story.append(Paragraph("Plan d'action — Upside estimé", h2_s))
    slk_in  = dea.slacks.get(hotel, {}).get("inputs",  {})
    slk_out = dea.slacks.get(hotel, {}).get("outputs", {})
    tgt_in  = targets.get("inputs",  {})
    tgt_out = targets.get("outputs", {})

    plan_data = [["Variable", "Actuel", "Cible", "Slack", "Amélioration"]]
    for col in dea.input_cols:
        cur = float(raw[col])
        tgt = tgt_in.get(col, cur)
        slk = slk_in.get(col, 0)
        pct = f"{(tgt-cur)/cur*100:+.1f}%" if cur > 0 else "—"
        plan_data.append([f"↓ {col}", f"{cur:.1f}", f"{tgt:.1f}", f"{slk:.2f}", pct])
    for col in ["revpar", "satisfaction", "taux_occupation"]:
        cur = float(raw[col])
        tgt = tgt_out.get(col, cur)
        slk = slk_out.get(col, 0)
        pct = f"{(tgt-cur)/cur*100:+.1f}%" if cur > 0 else "—"
        plan_data.append([f"↑ {col}", f"{cur:.1f}", f"{tgt:.1f}", f"{slk:.2f}", pct])

    plan_table = Table(plan_data, colWidths=[4.2*cm, 2.4*cm, 2.4*cm, 2.0*cm, 2.6*cm])
    plan_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), NAVY),
        ("TEXTCOLOR",  (0,0), (-1,0), WHITE),
        ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",   (0,0), (-1,-1), 7.5),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [WHITE, LGRAY]),
        ("GRID",       (0,0), (-1,-1), 0.3, colors.lightgrey),
        ("ALIGN",      (1,0), (-1,-1), "RIGHT"),
        ("ALIGN",      (0,0), (0,-1), "LEFT"),
    ]))
    story.append(plan_table)

    # Upside financier
    _up_rev_txt = f"{up_rev:.2f} M€/an" if up_rev is not None else "n/d — chambres"
    upside_txt = f"Upside ETP : {up_fte} k€/an  |  Upside RevPAR : {_up_rev_txt}"
    story.append(Spacer(1, 4))
    if up_fte > 0 or (up_rev or 0) > 0:
        story.append(Paragraph(f"→ {upside_txt}", alert_s))

    # Peers
    story.append(Spacer(1, 6))
    story.append(Paragraph("Hôtels de référence (Peers)", h2_s))
    if peers:
        peer_data = [["Peer", "Poids λ", "BCC Peer"]]
        for p, w in sorted(peers.items(), key=lambda x: -x[1])[:5]:
            peer_data.append([p, f"{w:.3f}", f"{dea.bcc_scores.get(p,0):.1%}"])
        peer_table = Table(peer_data, colWidths=[7*cm, 3*cm, 3.5*cm])
        peer_table.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), BLUE),
            ("TEXTCOLOR",  (0,0), (-1,0), WHITE),
            ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE",   (0,0), (-1,-1), 8),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [WHITE, LGRAY]),
            ("GRID",       (0,0), (-1,-1), 0.3, colors.lightgrey),
        ]))
        story.append(peer_table)
    else:
        story.append(Paragraph("✅ Cet hôtel est lui-même un peer de référence.", body_s))

    # Footer
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=0.5, color=NAVY))
    story.append(Paragraph(
        "DEA-H v3.5 · REIV Hospitality · "
        "Charnes et al. (1978), Banker et al. (1984), Barros (2005) · "
        "Confidentiel — usage interne asset manager",
        small_s
    ))

    doc.build(story)
    return buf.getvalue()


def generate_portfolio_report_pdf(dea, quadrant_labels: dict, top_n: int = 10,
                                  cap_input=None, sw_results=None,
                                  jours_exploit: int = 365,
                                  ft_pct: float = 0.50,
                                  avg_salary: float = 35_000,
                                  portfolio_name: str = "",
                                  thresholds: dict = None,
                                  module_results: dict = None) -> bytes:
    """
    Rapport PDF portfolio — document de comité d'investissement.

    Contenu : couverture, note de lecture, dashboard, quadrants, TOPSIS,
    SBM (Tone 2001), Super-Efficience (Andersen & Petersen 1993),
    Slacks (Barros 2005), Métafrontière (O'Donnell et al. 2008),
    Simar-Wilson (2007), Efficience Capital, DEA Capital vs Opérationnel,
    Expense Flex & Flow Through (Russo & Legel), Diagnostic croisé
    Opérationnel × Financier × Commercial × RH avec plan d'action.
    """
    if not REPORTLAB_AVAILABLE:
        return b""

    # Seuils d'interprétation — conventions sectorielles par défaut,
    # recalibrables depuis la barre latérale de l'application
    _TH = {'crit': 0.85, 'ft_norm': 0.50, 'ft_low': 0.40, 'sbm': 0.15,
           'tgr': 0.15, 'goppam': 2.5, 'rgi': 100}
    if thresholds:
        _TH.update({k: v for k, v in thresholds.items() if v is not None})
    _TH_CUSTOM = bool(thresholds) and any(
        abs(_TH[k] - d) > 1e-9 for k, d in
        {'crit': .85, 'ft_norm': .50, 'ft_low': .40, 'sbm': .15,
         'tgr': .15, 'goppam': 2.5, 'rgi': 100}.items())

    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=1.6*cm, rightMargin=1.6*cm,
                            topMargin=1.4*cm, bottomMargin=1.4*cm,
                            title="Rapport Portfolio DEA-H",
                            author="REIV Hospitality")

    NAVY  = colors.HexColor("#1a3a5c"); BLUE  = colors.HexColor("#2e6da4")
    RED   = colors.HexColor("#c0392b"); GREEN = colors.HexColor("#1e8449")
    AMBER = colors.HexColor("#b9770e"); LGRAY = colors.HexColor("#f4f6f8")
    BOX   = colors.HexColor("#eef3f8"); WHITE = colors.white

    styles = getSampleStyleSheet()
    def S(name, **kw): return ParagraphStyle(name, parent=styles["Normal"], **kw)

    cover_title = S("CT", fontSize=24, textColor=NAVY,  fontName="Helvetica-Bold",
                    alignment=TA_CENTER, spaceAfter=14, leading=28)
    cover_sub   = S("CS", fontSize=12, textColor=BLUE,  fontName="Helvetica",
                    alignment=TA_CENTER, spaceAfter=7)
    h1_s   = S("H1", fontSize=13,  textColor=NAVY, fontName="Helvetica-Bold",
               spaceBefore=14, spaceAfter=3)
    h2_s   = S("H2", fontSize=10,  textColor=NAVY, fontName="Helvetica-Bold",
               spaceBefore=8,  spaceAfter=3)
    meth_s = S("ME", fontSize=7.6, textColor=colors.HexColor("#4a5568"),
               fontName="Helvetica-Oblique", spaceAfter=6, leading=10)
    body_s = S("B",  fontSize=8.3, textColor=colors.black, fontName="Helvetica",
               spaceAfter=5, leading=11.5)
    read_s = S("RD", fontSize=8.3, textColor=NAVY, fontName="Helvetica",
               leading=11.5, leftIndent=6, rightIndent=6,
               spaceBefore=4, spaceAfter=4)
    small_s= S("SM", fontSize=7,   textColor=colors.grey, fontName="Helvetica-Oblique")

    def _v(hotel, col, default=0.0):
        """Accès sûr à une colonne — le jeu de variables varie selon le portefeuille."""
        try:
            if col not in dea.df.columns:
                return default
            val = dea.df.loc[hotel, col]
            return default if pd.isna(val) else float(val)
        except Exception:
            return default

    def _clean(x, n=None):
        """Retire les emojis (non rendus par Helvetica) et tronque au besoin."""
        t = "".join(ch for ch in str(x) if ord(ch) < 0x2190 or ch in "€²★—–’·")
        t = " ".join(t.split())
        return t[:n] if n else t

    def _tbl(data, widths, hdr=NAVY, fs=7.4, align_right=None):
        t = Table(data, colWidths=widths, repeatRows=1)
        style = [
            ("BACKGROUND",   (0,0), (-1,0), hdr),
            ("TEXTCOLOR",    (0,0), (-1,0), WHITE),
            ("FONTNAME",     (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE",     (0,0), (-1,-1), fs),
            ("ROWBACKGROUNDS",(0,1),(-1,-1), [WHITE, LGRAY]),
            ("GRID",         (0,0), (-1,-1), 0.3, colors.lightgrey),
            ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
            ("TOPPADDING",   (0,0), (-1,-1), 3),
            ("BOTTOMPADDING",(0,0), (-1,-1), 3),
        ]
        for c in (align_right or []):
            style.append(("ALIGN", (c,1), (c,-1), "RIGHT"))
        t.setStyle(TableStyle(style))
        return t

    def _lecture(txt):
        """Encadré 'Lecture investisseur'."""
        p = Paragraph(f"<b>Lecture investisseur —</b> {txt}", read_s)
        box = Table([[p]], colWidths=[17.4*cm])
        box.setStyle(TableStyle([
            ("BACKGROUND",  (0,0), (-1,-1), BOX),
            ("BOX",         (0,0), (-1,-1), 0.6, BLUE),
            ("LEFTPADDING", (0,0), (-1,-1), 8),
            ("RIGHTPADDING",(0,0), (-1,-1), 8),
            ("TOPPADDING",  (0,0), (-1,-1), 6),
            ("BOTTOMPADDING",(0,0),(-1,-1), 6),
        ]))
        return box

    # ── Registre de signaux : alimenté par chaque section au fil du rapport ──
    # Permet à la synthèse finale de croiser les diagnostics, au lieu de laisser
    # chaque section raisonner isolément.
    _sig_reg = {h: set() for h in dea.hotels}
    _SIGLAB = {
        'critique'    : "Efficience critique",
        'double_peine': "Gestion + taille",
        'sbm_masque'  : "Inefficience masquée",
        'capital_bas' : "Capital sous-productif",
        'ft_faible'   : "Conversion faible",
        'recul'       : "En recul vs N-1",
    }
    def _sig(h, k):
        if h in _sig_reg:
            _sig_reg[h].add(k)

    for _h in dea.hotels:
        if dea.bcc_scores.get(_h, 1) < _TH['crit']:
            _sig(_h, 'critique')
        if dea.quadrants.get(_h) == 'Q4':
            _sig(_h, 'double_peine')

    story = []

    # ══════════════════ COUVERTURE ══════════════════
    story.append(Spacer(1, 4.5*cm))
    story.append(Paragraph("RAPPORT PORTFOLIO", cover_title))
    story.append(Paragraph("Analyse d'efficience DEA multi-dimensionnelle", cover_sub))
    story.append(HRFlowable(width="55%", thickness=2, color=NAVY, spaceAfter=18))
    story.append(Paragraph(f"{dea.n} hôtels analysés", cover_sub))
    story.append(Paragraph(f"Généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')}", cover_sub))
    if illustrative_data:
        story.append(Spacer(1, 0.6*cm))
        story.append(Paragraph(
            "⚠️ DONNÉES ILLUSTRATIVES — tout ou partie des chiffres de ce rapport sont "
            "des estimations, pas des données vérifiées par le client. Les scores "
            "présentés ne sont pas des résultats définitifs.",
            S("illustrative_warn", fontSize=10, textColor=colors.HexColor("#B00020"),
              alignment=TA_CENTER, fontName="Helvetica-Bold")))
    story.append(Spacer(1, 2.5*cm))
    story.append(Paragraph(
        "REIV Hospitality &#183; DEA-H v4",
        S("ft", fontSize=10, textColor=colors.grey, alignment=TA_CENTER)))
    story.append(Spacer(1, 0.4*cm))
    story.append(Paragraph(
        "Document confidentiel — usage restreint au destinataire",
        S("ft2", fontSize=8, textColor=colors.grey, alignment=TA_CENTER)))
    story.append(PageBreak())

    # ══════════════════ NOTE DE LECTURE ══════════════════
    story.append(Paragraph("Note de lecture", h1_s))
    story.append(Paragraph(
        "La Data Envelopment Analysis (DEA) mesure l'efficience <b>relative</b> de chaque "
        "actif face aux meilleures pratiques observées dans le portefeuille — sans "
        "hypothèse sur la forme de la fonction de production. Un score de 70 % signifie "
        "que l'hôtel pourrait produire le même résultat avec 30 % de ressources en moins, "
        "au vu de ce que ses pairs réalisent effectivement.",
        body_s))
    story.append(Paragraph(
        "Trois principes de lecture. <b>Premièrement</b>, tous les scores sont relatifs : "
        "un portefeuille homogène et performant produira des scores plus sévères qu'un "
        "portefeuille hétérogène. <b>Deuxièmement</b>, un score de 100 % ne signifie pas "
        "l'excellence absolue mais l'absence de pair démontrant qu'on peut faire mieux. "
        "<b>Troisièmement</b>, la valeur opérationnelle réside moins dans le classement "
        "que dans les <i>slacks</i> — les écarts chiffrés poste par poste.",
        body_s))

    _n_var = len(getattr(dea, 'input_cols', [])) + len(getattr(dea, 'output_cols', []))
    _seuil = 3 * _n_var
    if dea.n < _seuil:
        story.append(Spacer(1, 4))
        story.append(_lecture(
            f"Le portefeuille compte {dea.n} unités pour {_n_var} variables. La règle "
            f"empirique de Cooper, Seiford &amp; Tone (2007) recommande n &gt;= 3(m+s) = "
            f"{_seuil}. En deçà, le pouvoir discriminant de la DEA se réduit et les scores "
            "doivent être lus comme des ordres de grandeur, non comme des mesures fines. "
            "L'analyse par métafrontière ci-après atténue partiellement cette limite."))

    # ══════════════════ DASHBOARD ══════════════════
    story.append(Paragraph("1. Dashboard portefeuille", h1_s))
    story.append(Paragraph(
        "Modèle BCC orienté input, rendements d'échelle variables — Banker, Charnes &amp; "
        "Cooper (1984). L'efficacité d'échelle (CCR/BCC) isole la part de l'inefficience "
        "imputable à une taille inadaptée au marché.", meth_s))

    avg_bcc   = sum(dea.bcc_scores.values()) / len(dea.bcc_scores)
    n_eff     = sum(1 for s in dea.bcc_scores.values() if s >= 0.999)
    n_crit    = sum(1 for s in dea.bcc_scores.values() if s < _TH['crit'])
    avg_scale = sum(dea.scale_efficiency.values()) / len(dea.scale_efficiency)
    _pct_eff  = n_eff / dea.n

    kpi = [["Indicateur", "Valeur", "Ce que cela signifie"],
           ["Efficience BCC moyenne", f"{avg_bcc:.1%}",
            "Marge de progression moyenne sur la gestion courante"],
           ["Hôtels sur la frontière", f"{n_eff} / {dea.n}",
            "Actifs servant de référence — aucun pair ne fait mieux"],
           [f"Hôtels critiques (BCC < {_TH['crit']:.0%})", f"{n_crit} / {dea.n}",
            "Plan d'action prioritaire"],
           ["Efficacité d'échelle moyenne", f"{avg_scale:.1%}",
            "Adéquation de la taille au marché desservi"]]
    story.append(_tbl(kpi, [5.4*cm, 2.6*cm, 9.4*cm], fs=7.8, align_right=[1]))

    if _pct_eff < 0.15:
        story.append(_lecture(
            f"Seuls {_pct_eff:.0%} des actifs atteignent la frontière, ce qui est "
            "inhabituellement bas. Le cas typique est celui d'un actif atypique — souvent "
            "de petite taille et fort RevPAR — qui domine seul l'ensemble et écrase "
            "mécaniquement les scores des autres. Avant d'en tirer des conclusions "
            "managériales, il faut vérifier la comparabilité technologique du portefeuille "
            "(section Métafrontière)."))
    elif _pct_eff > 0.50:
        story.append(_lecture(
            f"{_pct_eff:.0%} des actifs sont efficients — le modèle discrimine peu. "
            "Réduire le nombre de variables ou élargir l'échantillon renforcerait le "
            "pouvoir de séparation (Vía et al., 2013)."))
    else:
        _gap_moy = 1 - avg_bcc
        story.append(_lecture(
            f"{_pct_eff:.0%} des actifs définissent la frontière — le modèle discrimine "
            "correctement, ni concentré sur un actif atypique ni trop permissif. "
            f"L'écart moyen à la frontière s'établit à {_gap_moy:.0%} : c'est la marge "
            "de progression théorique du portefeuille si chaque actif rejoignait les "
            "pratiques de ses pairs les mieux placés. "
            + ("L'efficacité d'échelle moyenne, plus faible que l'efficience de gestion, "
               "indique que le dimensionnement pèse davantage que l'exploitation — un "
               "levier d'arbitrage plus que de management."
               if avg_scale < avg_bcc - 0.05 else
               "L'efficience de gestion est le facteur limitant, davantage que le "
               "dimensionnement des actifs — les leviers sont opérationnels."
               if avg_bcc < avg_scale - 0.05 else
               "Gestion et dimensionnement contribuent à parts comparables aux écarts "
               "observés.")))

    # ══════════════════ QUADRANTS ══════════════════
    story.append(Paragraph("2. Quadrants — gestion pure x efficacité d'échelle", h1_s))
    story.append(Paragraph(
        "Croisement de deux diagnostics distincts. L'axe <b>gestion pure</b> (BCC) mesure "
        "la qualité d'exploitation à taille donnée ; l'axe <b>échelle</b> mesure si la "
        "taille elle-même est pertinente. Les leviers d'action diffèrent radicalement : "
        "l'un relève du management, l'autre de l'arbitrage capitalistique.", meth_s))

    _qmean = ("Q1 : bien géré et bien dimensionné — actif de référence, à protéger. | "
              "Q2 : bien géré mais mal dimensionné — le problème est la taille, pas "
              "l'équipe ; extension, réduction ou cession. | "
              "Q3 : bien dimensionné mais mal géré — levier managérial actionnable à "
              "court terme, sans capital. | "
              "Q4 : double handicap — restructuration ou arbitrage.")
    story.append(Paragraph(_qmean, body_s))

    # Construction directe depuis dea.quadrants (les libellés de
    # get_quadrant_summary contiennent des emojis non rendus par Helvetica)
    _qmeta = {
        'Q1': ("Efficient",          "Bien géré et bien dimensionné",
               "Préserver — modèle interne"),
        'Q2': ("Problème d'échelle", "Bien géré, taille inadaptée",
               "Arbitrage capitalistique"),
        'Q3': ("Problème de gestion","Bien dimensionné, gestion perfectible",
               "Plan managérial — effet rapide"),
        'Q4': ("Double peine",       "Gestion et taille déficientes",
               "Restructuration ou cession"),
    }
    qd = [["Quadrant", "Nb", "Part", "BCC moyen", "Diagnostic", "Levier prioritaire"]]
    for _qk, (_lab, _diag, _lev) in _qmeta.items():
        _hs = [h for h in dea.hotels if dea.quadrants.get(h) == _qk]
        if not _hs:
            continue
        _bm = sum(dea.bcc_scores.get(h, 0) for h in _hs) / len(_hs)
        qd.append([_lab, str(len(_hs)), f"{len(_hs)/dea.n:.0%}",
                   f"{_bm:.1%}", _diag, _lev])
    if len(qd) > 1:
        story.append(_tbl(qd, [3.3*cm, 1.0*cm, 1.2*cm, 2.0*cm, 5.0*cm, 4.9*cm],
                          align_right=[1, 2, 3]))
    else:
        story.append(Paragraph("Synthèse par quadrant indisponible.", body_s))

    story.append(PageBreak())

    # ══════════════════ TOPSIS ══════════════════
    story.append(Paragraph("3. Classement multicritère TOPSIS", h1_s))
    story.append(Paragraph(
        "Hwang &amp; Yoon (1981). Le score DEA seul ne suffit pas à hiérarchiser : deux "
        "hôtels à 100 % ne sont pas équivalents pour un investisseur. TOPSIS agrège "
        "efficience, échelle, RevPAR et occupation en mesurant la distance à la solution "
        "idéale et à la solution anti-idéale.", meth_s))

    top_data = [["Rg", "Hôtel", "TOPSIS", "BCC", "Échelle", "Quadrant"]]
    for _, r in dea.get_topsis_ranking().head(top_n).iterrows():
        h = r["Hôtel"]; q = dea.quadrants.get(h, "")
        top_data.append([f"{int(r['Rang'])}", _clean(h, 40), f"{r['Score TOPSIS']:.3f}",
                         f"{dea.bcc_scores.get(h,0):.1%}",
                         f"{dea.scale_efficiency.get(h,0):.1%}",
                         _clean(quadrant_labels.get(q, q), 20)])
    story.append(_tbl(top_data, [1.0*cm, 6.6*cm, 2.0*cm, 2.0*cm, 2.0*cm, 3.8*cm],
                      align_right=[2, 3, 4]))

    # ══════════════════ SUPER-EFFICIENCE ══════════════════
    story.append(Paragraph("4. Super-efficience — départager les actifs à 100 %", h1_s))
    story.append(Paragraph(
        "Andersen &amp; Petersen (1993). La DEA classique plafonne à 1 : tous les actifs "
        "efficients sont ex aequo. La super-efficience réévalue chaque unité en l'excluant "
        "de sa propre référence. Un score de 1,45 indique que l'hôtel pourrait consommer "
        "45 % de ressources en plus tout en restant sur la frontière — c'est une mesure de "
        "la <b>robustesse</b> de son avantage, non de sa performance courante.", meth_s))
    try:
        se_df = dea.compute_super_efficiency()
        _n_se_total = len(se_df)
        sd = [["Hôtel", "BCC", "Super-eff.", "Lecture"]]
        for _, r in se_df.head(min(12, len(se_df))).iterrows():
            _se  = r['Super-Efficience']
            _bccv = dea.bcc_scores.get(r['Hôtel'], 0)
            if isinstance(_se, (int, float)) and _bccv >= 0.999:
                _lec = ("Avantage robuste — marge de sécurité"
                        if _se >= 1.15 else
                        "Efficient mais fragile — seul sur son segment")
            else:
                _lec = "Inefficient — score inchangé par construction"
            sd.append([_clean(r['Hôtel'], 42), _clean(r['BCC']),
                       f"{_se:.3f}" if isinstance(_se, (int, float)) else _clean(_se),
                       _lec])
        if _n_se_total > 12:
            story.append(Paragraph(
                f"Tableau limité aux 12 premiers actifs (sur {_n_se_total} au total, triés par "
                f"super-efficience) — les lectures ci-dessous portent sur l'ensemble du portefeuille, "
                f"pas uniquement sur les lignes affichées.", small_s))
        story.append(_tbl(sd, [6.6*cm, 1.9*cm, 2.2*cm, 6.7*cm],
                          align_right=[1, 2]))

        _lead = se_df.iloc[0]
        _ls   = _lead['Super-Efficience']
        if isinstance(_ls, (int, float)) and _ls > 1.5:
            story.append(_lecture(
                f"{_clean(_lead['Hôtel'])} affiche une super-efficience de {_ls:.2f}, très "
                "au-dessus de 1. Deux lectures opposées coexistent et il faut trancher : "
                "soit l'actif dispose d'un avantage structurel réel et durable, soit il est "
                "atypique au point de ne plus être comparable au reste du portefeuille — "
                "auquel cas il fausse tous les autres scores. Andersen &amp; Petersen "
                "recommandent d'examiner ces cas comme des <i>outliers</i> potentiels."))
        else:
            _fr = [r for _, r in se_df.iterrows()
                   if isinstance(r['Super-Efficience'], (int, float))
                   and dea.bcc_scores.get(r['Hôtel'], 0) >= 0.999
                   and r['Super-Efficience'] < 1.05]
            story.append(_lecture(
                (f"Aucun actif ne dépasse 1,5 : les références du portefeuille ont des "
                 "avantages mesurés, sans domination excessive. Les scores des autres "
                 "actifs ne sont donc pas distordus par un cas atypique. "
                 if isinstance(_ls, (int, float)) else "")
                + (f"En revanche, {len(_fr)} actif(s) efficient(s) affichent une "
                   "super-efficience proche de 1 : leur position sur la frontière tient "
                   "à peu de chose et peut basculer à la première variation "
                   "d'exploitation. Ce sont des références fragiles, à ne pas ériger en "
                   "modèle interne sans vérification."
                   if _fr else
                   "Les actifs efficients disposent tous d'une marge de sécurité — leur "
                   "position sur la frontière est robuste et ils peuvent servir de "
                   "référence interne.")))
    except Exception as _e:
        story.append(Paragraph(f"Super-efficience non calculable ({_clean(_e, 90)}).", body_s))

    story.append(PageBreak())

    # ══════════════════ SBM ══════════════════
    story.append(Paragraph("5. SBM — efficience fondée sur les écarts", h1_s))
    story.append(Paragraph(
        "Tone (2001). Le modèle radial BCC réduit tous les inputs dans la même proportion "
        "et ignore les écarts résiduels. Un hôtel peut ainsi afficher 100 % tout en "
        "gaspillant sur un poste précis. Le SBM intègre directement ces écarts : il est "
        "systématiquement plus sévère, et c'est précisément son intérêt — il révèle les "
        "inefficiences que le score radial masque.", meth_s))
    try:
        sbm = dea.compute_sbm()
        rows_sbm, gaps = [], []
        for h in dea.hotels:
            r_ = sbm.get(h, {})
            rho = r_.get('score'); bcc = dea.bcc_scores.get(h, 0)
            if rho is None:
                continue
            gap = bcc - rho
            if gap >= _TH['sbm']:
                _sig(h, 'sbm_masque')
            gaps.append((h, gap, bcc, rho))
        gaps.sort(key=lambda x: -x[1])
        sd2 = [["Hôtel", "BCC (radial)", "SBM (non radial)", "Écart", "Signal"]]
        for h, gap, bcc, rho in gaps[:12]:
            sig = ("Inefficience masquée" if gap >= _TH['sbm'] else
                   "Écart modéré"          if gap >= 0.05 else
                   "Diagnostics cohérents")
            sd2.append([_clean(h, 38), f"{bcc:.1%}", f"{rho:.1%}", f"{gap:+.1%}", sig])
        story.append(_tbl(sd2, [6.2*cm, 2.5*cm, 2.9*cm, 1.9*cm, 3.9*cm],
                          align_right=[1, 2, 3]))
        _n_masq = sum(1 for _, g, _, _ in gaps if g >= _TH['sbm'])
        if gaps and gaps[0][1] >= _TH['sbm']:
            story.append(_lecture(
                f"{_clean(gaps[0][0])} perd {gaps[0][1]:.0%} entre le score radial et le "
                "score SBM. L'écart signale une inefficience concentrée sur un ou deux "
                "postes spécifiques plutôt qu'une sous-performance générale — donc un "
                "levier ciblé, identifiable dans la section Slacks."
                + (f" {_n_masq} actifs au total dépassent le seuil de "
                   f"{_TH['sbm']*100:.0f} points : le score radial flatte "
                   "systématiquement ce portefeuille, et les décisions fondées sur lui "
                   "seul sous-estiment le potentiel d'amélioration."
                   if _n_masq >= 3 else "")))
        elif gaps:
            _gm = sum(g for _, g, _, _ in gaps) / len(gaps)
            story.append(_lecture(
                f"Écart moyen BCC-SBM de {_gm*100:.1f} points, sous le seuil de "
                f"{_TH['sbm']*100:.0f} retenu. Les deux mesures convergent : les scores "
                "radiaux ne masquent pas d'inefficience localisée, et les écarts observés "
                "sont bien répartis sur l'ensemble des postes plutôt que concentrés sur "
                "un poste unique. Les plans d'action peuvent s'appuyer sur les scores BCC "
                "sans correction."))
    except Exception as _e:
        story.append(Paragraph(f"SBM non calculable ({_clean(_e, 90)}).", body_s))

    # ══════════════════ SLACKS ══════════════════
    story.append(Paragraph("6. Slacks — gaspillages et potentiels chiffrés", h1_s))
    story.append(Paragraph(
        "Barros (2005). C'est la section la plus opérationnelle du rapport. Le slack "
        "traduit le score d'efficience en unités physiques : ETP en excès, euros de RevPAR "
        "non captés, points d'occupation manquants. Les cibles sont issues du comportement "
        "réel des pairs, non d'un objectif budgétaire théorique.", meth_s))

    # Libellés génériques — le portefeuille peut être analysé avec n'importe
    # quel jeu de variables (mode standard / compact / minimal)
    _VARLAB = {
        'nb_employes'       : "ETP",
        'nb_chambres'           : "Chambres",
        'couts_op_ex'       : "OpEx",
        'payroll_total'     : "Masse salariale",
        'energy_kwh'        : "Énergie",
        'surface_m2'        : "Surface m²",
        'revpar'            : "RevPAR",
        'taux_occupation'   : "Occupation",
        'satisfaction'      : "Satisfaction",
        'satisfaction_score': "Satisfaction",
        'gop'               : "GOP",
        'total_revenue'     : "CA total",
    }
    def _vl(c):  return _VARLAB.get(c, str(c).replace('_', ' ').capitalize())

    crit = sorted([h for h in dea.hotels if dea.bcc_scores[h] < _TH['crit']],
                  key=lambda x: dea.bcc_scores[x])
    _in_cols  = list(getattr(dea, 'input_cols',  []))
    _out_cols = list(getattr(dea, 'output_cols', []))

    if crit and (_in_cols or _out_cols):
        # Ne retenir que les variables présentant effectivement du slack
        _act_in  = [c for c in _in_cols
                    if any((dea.slacks.get(h, {}).get('inputs', {}).get(c, 0) or 0) > 1e-6
                           for h in crit)]
        _act_out = [c for c in _out_cols
                    if any((dea.slacks.get(h, {}).get('outputs', {}).get(c, 0) or 0) > 1e-6
                           for h in crit)]
        # Au-delà de 4 colonnes de variables les en-têtes se chevauchent :
        # on privilégie les postes où le gisement total est le plus élevé
        def _tot(kind, c):
            return sum((dea.slacks.get(h, {}).get(kind, {}).get(c, 0) or 0) for h in crit)
        _act_in  = sorted(_act_in,  key=lambda c: -_tot('inputs',  c))[:2]
        _act_out = sorted(_act_out, key=lambda c: -_tot('outputs', c))[:2]

        # En-têtes courts (- = input à réduire, + = output à gagner) + légende
        _hdr = (["Hôtel", "BCC", "Quadrant"]
                + [f"{_vl(c)} (-)" for c in _act_in]
                + [f"{_vl(c)} (+)" for c in _act_out])
        cd = [_hdr]
        _qlab = {'Q1': "Efficient", 'Q2': "Problème d'échelle",
                 'Q3': "Problème de gestion", 'Q4': "Double peine"}
        _tot_sal = 0.0
        for h in crit[:14]:
            _row = [_clean(h, 34), f"{dea.bcc_scores[h]:.1%}",
                    _qlab.get(dea.quadrants.get(h, ""), "—")]
            for c in _act_in:
                v = dea.slacks.get(h, {}).get('inputs', {}).get(c, 0) or 0
                _row.append(f"{v:,.1f}".replace(",", " ") if v > 1e-6 else "—")
                if c == 'nb_employes':
                    _tot_sal += v * avg_salary / 1000
            for c in _act_out:
                v = dea.slacks.get(h, {}).get('outputs', {}).get(c, 0) or 0
                _row.append(f"{v:,.1f}".replace(",", " ") if v > 1e-6 else "—")
            cd.append(_row)

        _ncol = len(_hdr)
        _nvar = _ncol - 3
        _w0   = 5.6*cm if _nvar <= 2 else 5.0*cm if _nvar == 3 else 4.6*cm
        _rest = (17.4*cm - _w0 - 1.6*cm - 3.2*cm) / max(_nvar, 1)
        story.append(_tbl(cd, [_w0, 1.6*cm, 3.2*cm] + [_rest] * _nvar,
                          hdr=RED, fs=7.0, align_right=list(range(1, _ncol))))
        story.append(Paragraph(
            "(-) quantité d'input à retirer pour rejoindre la frontière &#183; "
            "(+) quantité d'output à gagner, à ressources constantes."
            + ("  Seuls les deux postes au gisement le plus élevé sont affichés ; "
               "le détail complet par actif figure dans l'onglet Fiche Actif."
               if (len(_in_cols) > 2 or len(_out_cols) > 2) else ""),
            small_s))

        if _tot_sal > 0:
            story.append(_lecture(
                "Le potentiel d'économie sur la seule masse salariale des actifs critiques "
                f"atteint {_tot_sal:,.0f} k€/an".replace(",", " ") +
                f" (hypothèse {avg_salary/1000:,.0f} k€ chargés par ETP, paramétrable dans "
                "l'application). Ce chiffre est un plafond théorique : il suppose une "
                "convergence intégrale vers les pratiques des pairs, sans coût de "
                "transition ni contrainte sociale. À pondérer par un facteur de "
                "réalisation de 30 à 50 % pour un plan à 24 mois.".replace(",", " ")))
        elif _act_in or _act_out:
            story.append(_lecture(
                "Les écarts se concentrent sur "
                + ", ".join(_vl(c) for c in (_act_in + _act_out))
                + ". Chaque valeur indique la quantité à retirer (input) ou à gagner "
                "(output) pour rejoindre la frontière, à production constante par "
                "ailleurs. Ce sont des cibles observées chez des pairs réels, pas des "
                "objectifs budgétaires."))
    else:
        story.append(Paragraph(
            f"Aucun actif sous le seuil de {_TH['crit']:.0%} — le portefeuille est homogène "
            "en gestion courante.", body_s))

    story.append(PageBreak())

    # ══════════════════ METAFRONTIERE ══════════════════
    story.append(Paragraph("7. Métafrontière — GTE, MTE et écart technologique", h1_s))
    story.append(Paragraph(
        "O'Donnell, Rao &amp; Battese (2008). Comparer un resort familial de 1 100 clés à "
        "une boutique urbaine de 90 clés n'a pas de sens : ces actifs n'opèrent pas sous la "
        "même technologie de production. La métafrontière décompose l'efficience en deux "
        "composantes distinctes.", meth_s))
    story.append(Paragraph(
        "<b>MTE</b> (efficience intra-groupe) — qualité de gestion face aux pairs "
        "directs ; c'est la responsabilité de l'exploitant. "
        "<b>TGR</b> (technology gap ratio) — écart entre la frontière du segment et la "
        "meilleure technologie disponible ; c'est le potentiel structurel du segment, qui "
        "relève de l'arbitrage d'allocation. "
        "<b>GTE = MTE x TGR</b> — efficience globale. Un exploitant excellent (MTE élevé) "
        "dans un segment structurellement faible (TGR bas) restera globalement médiocre : "
        "le problème n'est pas l'équipe, c'est le positionnement.", body_s))
    try:
        grp    = dea.get_auto_size_groups()
        meta   = dea.compute_metafrontier(grp)
        msum   = dea.get_metafrontier_summary(meta)
        md = [["Groupe", "N", "GTE moyen", "MTE moyen", "TGR moyen", "Meilleur TGR", "Pire TGR"]]
        for _, r in msum.iterrows():
            md.append([_clean(r['Groupe'], 22), str(int(r['N hôtels'])),
                       _clean(r['GTE moyen']), _clean(r['MTE moyen']),
                       _clean(r['TGR moyen']), _clean(r['Meilleur TGR']),
                       _clean(r['Pire TGR'])])
        story.append(_tbl(md, [4.0*cm, 1.0*cm, 2.4*cm, 2.4*cm, 2.4*cm, 2.6*cm, 2.4*cm],
                          align_right=[1, 2, 3, 4, 5, 6]))

        # Un groupe d'un seul actif ne constitue pas une frontière estimable
        _sizes  = meta.groupby('Groupe').size()
        _thin   = [g for g, k in _sizes.items() if k < 3]
        if _thin:
            story.append(Paragraph(
                "Attention : le(s) groupe(s) " +
                ", ".join(f"« {_clean(g)} » ({int(_sizes[g])} actif"
                          + ("s" if _sizes[g] > 1 else "") + ")" for g in _thin) +
                " comptent moins de trois unités. Une frontière estimée sur si peu "
                "d'observations est mécaniquement atteinte par ses propres membres : "
                "leur TGR de 100 % traduit l'absence de comparaison, non une supériorité "
                "technologique démontrée.", body_s))

        _tgr_num = meta[~meta['Groupe'].isin(_thin)].groupby('Groupe')['TGR'].mean()
        if len(_tgr_num) > 1:
            _best, _worst = _tgr_num.idxmax(), _tgr_num.idxmin()
            _spread = (_tgr_num.max() - _tgr_num.min()) * 100
            story.append(_lecture(
                f"L'écart technologique entre « {_clean(_best)} » ({_tgr_num.max():.0%}) et "
                f"« {_clean(_worst)} » ({_tgr_num.min():.0%}) atteint {_spread:.1f} points. "
                + ("Cet écart est substantiel : il indique que le segment le plus faible "
                   "souffre d'une contrainte structurelle — positionnement, marché ou "
                   "configuration d'actif — que le management seul ne peut compenser. "
                   "L'arbitrage relève du comité d'investissement, pas de l'exploitation."
                   if _spread >= _TH['tgr']*100 else
                   "Cet écart reste contenu : les segments partagent une technologie de "
                   "production proche, et les écarts observés relèvent principalement de "
                   "la qualité d'exploitation — donc actionnables sans capital.")))
        elif len(_sizes) <= 1:
            story.append(Paragraph(
                "Un seul groupe de taille suffisante : la décomposition métafrontière "
                "n'apporte pas d'information — le portefeuille est technologiquement "
                "homogène au sens de la segmentation retenue.", body_s))
    except Exception as _e:
        story.append(Paragraph(
            f"Métafrontière non calculable ({_clean(_e, 90)}). Un minimum de deux groupes "
            "suffisamment peuplés est requis.", body_s))

    # ══════════════════ SIMAR-WILSON ══════════════════
    story.append(Paragraph("8. Déterminants de l'efficience — régression tronquée bootstrappée", h1_s))
    story.append(Paragraph(
        "Simar &amp; Wilson (2007), Algorithme 1. Les scores DEA sont mécaniquement "
        "corrélés entre eux — chaque score dépend de l'échantillon entier — ce qui invalide "
        "les tests statistiques usuels. La procédure exclut les unités efficientes "
        "(troncature) puis reconstruit la distribution des coefficients par bootstrap "
        "paramétrique. Contrairement au Tobit censuré, elle produit des intervalles de "
        "confiance valides.", meth_s))

    # ── Auto-calcul si l'onglet Stage 2 n'a pas été lancé ────────────────────
    # Le rapport doit être complet sans manipulation préalable de l'utilisateur.
    _sw_auto = False
    if not (sw_results and not sw_results.get('error')
            and sw_results.get('coef_df') is not None):
        try:
            build_stage2_vars(dea)                     # crée les variables dérivées
            # Ordre de priorité : variables structurelles interprétables d'abord.
            # Une seule variable par famille — log_nb_chambres et nb_chambres mesurent la
            # même chose et créeraient une colinéarité quasi parfaite.
            _cands = ['classement_etoiles', 'saison_dummy', 'log_nb_chambres',
                      'ltv_proxy', 'gop_margin_pct', 'capex_per_room',
                      'asset_yield', 'surface_m2', 'energy_kwh', 'couts_op_ex']
            _FAM = {'nb_chambres': 'taille', 'log_nb_chambres': 'taille',
                    'surface_m2': 'taille', 'capex_annuel': 'capex',
                    'capex_per_room': 'capex', 'ltv_proxy': 'capex'}
            _n_ineff = sum(1 for s in dea.bcc_scores.values() if s < 1 - 1e-8)

            # Une variable du modèle DEA ne peut pas expliquer le score qu'elle
            # a servi à produire — endogénéité (Simar & Wilson 2007, §2).
            _dea_vars = set(getattr(dea, 'input_cols', [])) | set(getattr(dea, 'output_cols', []))

            _env, _fams = [], set()
            for c in _cands:
                if c not in dea.df.columns or c in _dea_vars:
                    continue
                _col = pd.to_numeric(dea.df[c], errors='coerce')
                if _col.notna().sum() < dea.n or _col.nunique() < 2:
                    continue
                _f = _FAM.get(c)
                if _f and _f in _fams:          # une seule variable par famille
                    continue
                # Écarter toute variable trop corrélée à une variable déjà retenue
                if any(abs(_col.corr(pd.to_numeric(dea.df[p], errors='coerce'))) > 0.85
                       for p in _env):
                    continue
                _env.append(c)
                if _f:
                    _fams.add(_f)
            # Contrainte de degrés de liberté : n_inefficients >= k + 2
            _env = _env[:max(0, min(3, _n_ineff - 2))]
            if len(_env) >= 2:
                # Le coût du bootstrap croît avec n et k — on l'ajuste pour que la
                # génération du rapport reste sous la minute. Simar & Wilson (2007)
                # retiennent B = 100 pour l'algorithme 1 ; l'onglet Stage 2 permet
                # de monter jusqu'à 500 pour une estimation de publication.
                _B = 100 if dea.n <= 15 else 60 if dea.n <= 30 else 40
                sw_results = compute_simar_wilson(dea, _env, n_bootstrap=_B)
                _sw_auto   = True
        except Exception:
            sw_results = None

    if sw_results and not sw_results.get('error') and sw_results.get('coef_df') is not None:
        cdf = sw_results['coef_df']
        _SWLAB = {
            'classement_etoiles': "Classement (étoiles)",
            'saison_dummy'      : "Saisonnalité (resort = 1)",
            'log_nb_chambres'       : "Taille — log(chambres)",
            'nb_chambres'           : "Nombre de chambres",
            'surface_m2'        : "Surface totale (m²)",
            'ltv_proxy'         : "Intensité capital / CA",
            'gop_margin_pct'    : "Marge GOP (%)",
            'capex_per_room'    : "CAPEX par chambre",
            'asset_yield'       : "Rendement des actifs",
            'energy_kwh'        : "Consommation énergie",
            'payroll_total'     : "Masse salariale",
            'couts_op_ex'       : "Charges opérationnelles",
        }
        def _swl(v):
            v = str(v)
            return _SWLAB.get(v, v.replace('_', ' ').capitalize())

        wd = [["Variable", "Coefficient", "IC 95 %", "p-value", "Effet"]]
        for _, r in cdf.iterrows():
            _lo, _hi = r.get('IC95% Lo'), r.get('IC95% Hi')
            _ic = (f"[{_lo:.3f} ; {_hi:.3f}]"
                   if isinstance(_lo, (int, float)) and isinstance(_hi, (int, float)) else "—")
            _pv = r.get('p-value')
            wd.append([_clean(_swl(r.get('Variable', '')), 34),
                       f"{r.get('Coeff.', 0):+.4f}" if isinstance(r.get('Coeff.'), (int, float)) else "—",
                       _ic,
                       f"{_pv:.3f}" if isinstance(_pv, (int, float)) else "—",
                       _clean(r.get('Effet', ''), 24)])
        story.append(_tbl(wd, [5.4*cm, 2.6*cm, 4.0*cm, 2.0*cm, 3.4*cm],
                          align_right=[1, 3]))
        story.append(Paragraph(
            f"Estimation sur {sw_results.get('n_inefficients','—')} unités inefficientes "
            f"(sur {sw_results.get('n','—')}), {sw_results.get('n_bootstrap','—')} "
            "réplications bootstrap."
            + ("  Variables sélectionnées automatiquement parmi celles disponibles dans "
               "le fichier ; l'onglet Stage 2 permet de choisir un autre jeu."
               if _sw_auto else ""), small_s))
        _sw_sig_rows = [r for _, r in cdf.iterrows()
                if isinstance(r.get('p-value'), (int, float)) and r['p-value'] < 0.05
                and str(r.get('Variable', '')).lower() != 'constante']
        if _sw_sig_rows:
            _desc = "; ".join(
                f"{_clean(_swl(r['Variable']))} ({'+' if r.get('Coeff.', 0) > 0 else '-'})"
                for r in _sw_sig_rows[:3])
            story.append(_lecture(
                f"{len(_sw_sig_rows)} déterminant(s) ressortent au seuil de 5 % : {_desc}. "
                "Un coefficient est exploitable lorsque son intervalle de confiance à "
                "95 % exclut zéro ; signe positif, la variable améliore l'efficience. "
                "Ce sont des <b>corrélations conditionnelles</b>, pas des relations "
                "causales — elles orientent la due diligence, elles ne la remplacent pas."))
        else:
            story.append(_lecture(
                "Aucun déterminant ne ressort au seuil de 5 %. Sur un échantillon de "
                f"{sw_results.get('n_inefficients','—')} unités inefficientes, la "
                "puissance statistique est faible : l'absence de significativité ne "
                "démontre pas l'absence d'effet. Le résultat utile est ici négatif — "
                "aucune des variables structurelles testées n'explique les écarts "
                "d'efficience, qui relèvent donc de la qualité d'exploitation."))
    else:
        _why = ("le portefeuille compte trop peu d'unités inefficientes pour estimer une "
                "régression tronquée (il en faut au moins quatre)"
                if sum(1 for s in dea.bcc_scores.values() if s < 1 - 1e-8) < 4 else
                "aucune variable environnementale exploitable n'a été trouvée dans le "
                "fichier source")
        story.append(Paragraph(
            f"Analyse non réalisable : {_why}. Enrichir le fichier avec des variables "
            "structurelles — classement, surface, CAPEX, masse salariale, consommation "
            "énergétique, indicateur de saisonnalité — active cette section.", body_s))

    story.append(PageBreak())

    # ══════════════════ CAPITAL ══════════════════
    story.append(Paragraph("9. Efficience du capital", h1_s))
    story.append(Paragraph(
        "L'efficience opérationnelle ignore le capital immobilisé. Un hôtel peut être "
        "parfaitement géré au quotidien tout en détruisant de la valeur si son actif est "
        "surdimensionné ou sur-capitalisé. Le GOPPAM (GOP par m² disponible) et le "
        "rendement CAPEX rapportent la performance à la base d'actifs.", meth_s))

    def _cap_ok(t):
        try:
            return (t is not None and 'gop (k€)' in getattr(t, 'columns', [])
                    and float(t[['surface_m2', 'capex_annuel (k€)',
                                 'gop (k€)']].sum().sum()) > 0)
        except Exception:
            return False

    _has_cap  = _cap_ok(cap_input)
    _cap_auto = False

    # ── Repli : reconstruire le tableau capital depuis le fichier source ─────
    # Le rapport ne doit pas exiger un passage préalable par l'onglet Capital.
    if not _has_cap:
        _src = {'surface_m2': 'surface_m2', 'capex_annuel (k€)': 'capex_annuel',
                'gop (k€)': 'gop', 'classement (★)': 'classement_etoiles'}
        if all(c in dea.df.columns for c in ('surface_m2', 'capex_annuel', 'gop')):
            try:
                _rows_auto = {}
                for h in dea.hotels:
                    _rows_auto[h] = {
                        'surface_m2'        : _v(h, 'surface_m2'),
                        # le fichier source stocke CAPEX et GOP en €, le tableau en k€
                        'capex_annuel (k€)' : _v(h, 'capex_annuel') / 1000,
                        'gop (k€)'          : _v(h, 'gop') / 1000,
                        'classement (★)'    : int(_v(h, 'classement_etoiles', 3) or 3),
                    }
                _auto_df = pd.DataFrame(_rows_auto).T.loc[dea.hotels]
                if dea.has_chambres:
                    # Détection d'unité : si CAPEX/chambre < 0,05 k€, les données
                    # étaient déjà en k€ dans le fichier — on annule la division
                    _med = (_auto_df['capex_annuel (k€)'] /
                            pd.Series({h: max(_v(h, 'nb_chambres'), 1) for h in dea.hotels})).median()
                    if _med < 0.05:
                        _auto_df['capex_annuel (k€)'] *= 1000
                        _auto_df['gop (k€)']          *= 1000
                if _cap_ok(_auto_df):
                    cap_input, _has_cap, _cap_auto = _auto_df, True, True
            except Exception:
                pass

    if _cap_auto:
        story.append(Paragraph(
            "Données capital lues directement dans le fichier source "
            "(colonnes surface_m2, capex_annuel, gop). L'onglet « Capital &amp; Flow "
            "Through » permet de les ajuster manuellement avant de régénérer le rapport.",
            small_s))

    if _has_cap:
        cap_rows_pdf = []
        for h in dea.hotels:
            if h not in cap_input.index:
                continue
            surf  = float(cap_input.loc[h, 'surface_m2'] or 0)
            capex = float(cap_input.loc[h, 'capex_annuel (k€)'] or 0)
            gop   = float(cap_input.loc[h, 'gop (k€)'] or 0)
            # CAPEX/chambre et Rendement exigent un vrai nombre de chambres —
            # plus de repli RevPAR x lits x jours (proxy non fiable, cf. audit
            # portefeuille Ibis Atream : rapport chambres/lits de 1:1 à 7:1
            # selon l'hôtel, aucune conversion fiable possible).
            if dea.has_chambres:
                lits = _v(h, 'nb_chambres')
            else:
                lits = None
            if dea.has_trevpar:
                ca = _v(h, 'total_revenue') / 1000  # k€
            else:
                ca = None
            cap_rows_pdf.append({
                'h': h,
                'goppam' : gop * 1000 / surf if surf > 0 else None,
                'capexch': (capex / lits) if (lits and lits > 0) else None,
                'rend'   : (ca / capex) if (ca is not None and capex > 0) else None,
                'marge'  : (gop / ca * 100) if (ca is not None and ca > 0) else None,
                'bcc'    : dea.bcc_scores.get(h, 0),
            })
        cap_rows_pdf.sort(key=lambda r: (r['goppam'] is None, -(r['goppam'] or 0)))

        cpd = [["Hôtel", "BCC", "GOPPAM (€/m²)", "CAPEX/chambre (k€)", "Rend. CAPEX (x)", "Marge GOP"]]
        for r in cap_rows_pdf[:14]:
            cpd.append([_clean(r['h'], 36), f"{r['bcc']:.1%}",
                        f"{r['goppam']:.0f}"  if r['goppam']  else "—",
                        f"{r['capexch']:.2f}" if r['capexch'] else "—",
                        f"{r['rend']:.1f}"    if r['rend']    else "—",
                        f"{r['marge']:.1f}%"  if r['marge']   else "—"])
        story.append(_tbl(cpd, [5.6*cm, 1.8*cm, 2.8*cm, 2.7*cm, 2.7*cm, 2.0*cm],
                          align_right=[1, 2, 3, 4, 5]))

        _gv = [r['goppam'] for r in cap_rows_pdf if r['goppam']]
        _mv = [r['marge']  for r in cap_rows_pdf if r['marge']]
        if _gv:
            story.append(Paragraph(
                f"GOPPAM moyen : {sum(_gv)/len(_gv):.0f} €/m² — amplitude de "
                f"{min(_gv):.0f} à {max(_gv):.0f} €/m²."
                + (f" Marge GOP moyenne : {sum(_mv)/len(_mv):.1f} %." if _mv else ""),
                body_s))
            _ampl = max(_gv) / max(min(_gv), 1e-9)
            if _ampl > _TH['goppam']:
                story.append(_lecture(
                    f"L'amplitude du GOPPAM atteint un facteur {_ampl:.1f}x au sein du "
                    f"portefeuille, au-delà du seuil de {_TH['goppam']:.1f}x retenu. "
                    "Un tel écart tient rarement à la seule qualité d'exploitation : il "
                    "reflète des positionnements et des configurations d'actifs "
                    "hétérogènes. Les actifs du bas de tableau immobilisent une surface "
                    "que leur GOP ne rentabilise pas — piste de reconversion partielle ou "
                    "de cession."))
            else:
                story.append(_lecture(
                    f"L'amplitude du GOPPAM reste contenue à {_ampl:.1f}x, sous le seuil "
                    f"de {_TH['goppam']:.1f}x. Les actifs valorisent leur surface de "
                    "manière homogène : les écarts de performance relèvent de "
                    "l'exploitation plutôt que de la configuration des actifs. "
                    + ("La marge GOP moyenne "
                       f"de {sum(_mv)/len(_mv):.1f} % situe le portefeuille "
                       + ("au-dessus des standards du secteur — la structure de coûts "
                          "est maîtrisée."
                          if sum(_mv)/len(_mv) >= 30 else
                          "dans la fourchette usuelle du secteur."
                          if sum(_mv)/len(_mv) >= 20 else
                          "sous les standards du secteur — la structure de coûts mérite "
                          "un examen, indépendamment de l'efficience DEA.")
                       if _mv else "")))

        # ── DEA Capital vs DEA Opérationnel ──
        story.append(Paragraph("10. DEA Capital vs DEA Opérationnel", h1_s))
        story.append(Paragraph(
            "Second modèle DEA avec surface, CAPEX et capacité en inputs, chiffre "
            "d'affaires et GOP en outputs. La comparaison des deux scores sépare deux "
            "questions que les ratios classiques confondent : l'exploitation est-elle "
            "performante, et le capital immobilisé est-il justifié ?", meth_s))
        try:
            _bk = dea.df.copy()
            _tmp = dea.df.copy()
            for h in cap_input.index:
                if h in _tmp.index:
                    _tmp.loc[h, 'surface_m2']   = float(cap_input.loc[h, 'surface_m2'] or 0)
                    _tmp.loc[h, 'capex_annuel'] = float(cap_input.loc[h, 'capex_annuel (k€)'] or 0) * 1000
                    _tmp.loc[h, 'gop']          = float(cap_input.loc[h, 'gop (k€)'] or 0) * 1000
            dea.df = _tmp
            cdea = dea.compute_capital_dea()
            dea.df = _bk

            if cdea is not None and not cdea.empty:
                dd = [["Hôtel", "DEA Opér.", "DEA Capital", "Écart", "Lecture"]]
                for _, r in cdea.head(14).iterrows():
                    try:
                        _dv = float(str(r['Δ (Capital-Opérat.)']).replace('%','').replace('+',''))
                        if _dv <= -10:
                            _sig(r['Hôtel'], 'capital_bas')
                    except Exception:
                        pass
                    dd.append([_clean(r['Hôtel'], 42), _clean(r['DEA Opérationnel']),
                               _clean(r['DEA Capital']), _clean(r['Δ (Capital-Opérat.)']),
                               _clean(r['Lecture'], 30)])
                story.append(_tbl(dd, [6.6*cm, 2.0*cm, 2.2*cm, 1.8*cm, 4.8*cm],
                                  align_right=[1, 2, 3]))
                story.append(_lecture(
                    "Écart positif : le capital est mieux employé que l'exploitation — "
                    "l'actif est bien dimensionné, le levier est managérial. Écart "
                    "négatif marqué : l'actif immobilise plus de capital qu'il n'en "
                    "rentabilise, indépendamment de la qualité de gestion. C'est un signal "
                    "d'arbitrage, pas un signal d'exploitation."))
            else:
                story.append(Paragraph(
                    "DEA Capital non calculable — surface, CAPEX et revenus doivent être "
                    "renseignés pour l'ensemble des actifs.", body_s))
        except Exception as _e:
            dea.df = _bk
            story.append(Paragraph(f"DEA Capital indisponible ({_clean(_e, 90)}).", body_s))

        # ── Expense Flex & Flow Through ──
        story.append(Paragraph("11. Expense Flex &amp; Flow Through", h1_s))
        story.append(Paragraph(
            "Flow Through = &#916;GOP / &#916;CA — part de chaque euro de revenu "
            "supplémentaire qui atteint le résultat. Expense Flex = 1 - FT lorsque le "
            "revenu recule ; il mesure la capacité à flexibiliser les charges en phase de "
            f"repli. Norme retenue : {_TH['ft_norm']:.0%}. Réf. : Russo &amp; Legel, hospitality "
            "management accounting.", meth_s))

        _hn1 = (dea.has_chambres
                and 'revpar_n1' in dea.df.columns and 'gop_n1' in dea.df.columns
                and dea.df['revpar_n1'].fillna(0).sum() > 0
                and dea.df['gop_n1'].fillna(0).sum() > 0)
        if not dea.has_chambres:
            story.append(Paragraph(
                "Non calculable : le CA N et N-1 se reconstruisent par RevPAR × chambres "
                "× jours — aucun repli fiable sans un vrai nombre de chambres.", body_s))
        elif _hn1:
            fd = [["Hôtel", "CA N (k€)", "CA N-1 (k€)", "Var. CA", "Flow Through", "Qualité"]]
            _fts = []
            for h in dea.hotels:
                if h not in cap_input.index:
                    continue
                lits = _v(h, 'nb_chambres')
                ca_n  = _v(h, 'revpar')    * lits * jours_exploit / 1000
                ca_p  = _v(h, 'revpar_n1') * lits * jours_exploit / 1000
                gop_n = float(cap_input.loc[h, 'gop (k€)'] or 0)
                gop_p = _v(h, 'gop_n1')
                if ca_p <= 0 or gop_n <= 0 or abs(ca_n - ca_p) < 1e-6:
                    continue
                ft = (gop_n - gop_p) / (ca_n - ca_p)
                if ft < _TH['ft_low']:
                    _sig(h, 'ft_faible')
                _fts.append((h, ft))
                q = ("Excellent" if ft >= _TH['ft_norm'] * 1.2
                     else "Correct"   if ft >= _TH['ft_norm'] * 0.9
                     else "Faible"    if ft >= _TH['ft_low']
                     else "Très faible")
                fd.append([_clean(h, 42), f"{ca_n:,.0f}".replace(",", " "),
                           f"{ca_p:,.0f}".replace(",", " "),
                           f"{(ca_n/ca_p - 1):+.1%}", f"{ft:.1%}", q])
            if len(fd) > 1:
                story.append(_tbl(fd, [6.4*cm, 2.3*cm, 2.3*cm, 1.9*cm, 2.3*cm, 2.2*cm],
                                  align_right=[1, 2, 3, 4]))
                _avg = sum(f for _, f in _fts) / len(_fts)
                _low = [h for h, f in _fts if f < _TH['ft_low']]
                story.append(_lecture(
                    f"Flow Through moyen du portefeuille : {_avg:.0%} contre une norme "
                    f"retenue de {_TH['ft_norm']:.0%}. "
                    + (f"{len(_low)} actif(s) restent sous {_TH['ft_low']:.0%} — leur structure de coûts "
                       "absorbe la croissance du revenu au lieu de la convertir en "
                       "résultat. C'est le symptôme d'une base de charges fixes trop "
                       "lourde ou d'une croissance obtenue par le volume plutôt que par le "
                       "prix : un point de RevPAR gagné en occupation coûte en charges "
                       "variables, un point gagné en ADR ne coûte rien."
                       if _low else
                       "L'ensemble du portefeuille convertit correctement la croissance en "
                       "résultat.")))
        else:
            story.append(Paragraph(
                "Flow Through non calculable : les colonnes revpar_n1 et gop_n1 sont "
                "requises pour établir la baseline N-1 par actif. Une baseline unique "
                "appliquée à l'ensemble du portefeuille produirait des résultats non "
                "significatifs dès lors que les RevPAR sont hétérogènes.", body_s))
    else:
        story.append(Paragraph(
            "Le fichier source ne contient aucune donnée de capital. Les sections 9 à 11 "
            "(Efficience du capital, DEA Capital, Flow Through) exigent trois colonnes par "
            "actif : <b>surface_m2</b>, <b>capex_annuel</b> et <b>gop</b>. Deux voies pour "
            "les fournir — les ajouter au fichier source, ou les saisir dans l'onglet "
            "« Capital &amp; Flow Through », qui propose un modèle Excel pré-rempli aux "
            "noms de vos actifs.", body_s))
        story.append(_lecture(
            "Sans ces trois colonnes, l'analyse reste purement opérationnelle : elle "
            "mesure la qualité de gestion mais reste muette sur le capital immobilisé. "
            "Pour un investisseur, c'est précisément la moitié manquante — un actif "
            "irréprochable en exploitation peut détruire de la valeur s'il mobilise une "
            "base d'actifs que son GOP ne rentabilise pas."))

    # ══════════════════ MALMQUIST ══════════════════
    story.append(PageBreak())
    story.append(Paragraph("12. Malmquist — évolution de la productivité N-1 vers N", h1_s))
    story.append(Paragraph(
        "Caves, Christensen &amp; Diewert (1982) ; Färe et al. (1994). Toutes les sections "
        "précédentes photographient un instant. Celle-ci mesure le mouvement. Un hôtel "
        "peut afficher un score médiocre tout en progressant fortement — c'est une "
        "information d'investissement différente de celle d'un actif stable et efficient.",
        meth_s))
    story.append(Paragraph(
        "L'indice se décompose en deux effets que les ratios classiques confondent. "
        "Le <b>catch-up</b> (rattrapage) mesure si l'hôtel s'est rapproché de la frontière "
        "de son époque : c'est la performance propre de l'équipe. Le <b>frontier shift</b> "
        "(déplacement de frontière) mesure si la frontière elle-même a progressé : c'est "
        "l'effet du marché, dont personne dans l'hôtel n'est responsable. "
        "<b>TFP = catch-up x frontier shift.</b> Au-dessus de 1, la productivité totale "
        "progresse ; en dessous, elle recule.", body_s))
    story.append(Paragraph(
        "La distinction est décisive pour un investisseur. Un TFP de 1,21 obtenu avec un "
        "catch-up de 1,00 et un frontier shift de 1,21 signifie que l'hôtel n'a rien "
        "amélioré : il a été porté par son marché. Le jour où le marché se retourne, la "
        "progression disparaît. À l'inverse, un catch-up supérieur à 1 dans un marché "
        "atone est une vraie création de valeur managériale.", body_s))
    try:
        _mq = compute_malmquist(dea)
        if _mq is None or _mq.empty:
            story.append(Paragraph(
                "Analyse non disponible : les colonnes de l'exercice précédent "
                "(nb_chambres_n1, nb_employes_n1, couts_op_ex_n1, revpar_n1, "
                "taux_occupation_n1) sont requises dans le fichier source.", body_s))
        else:
            _calc = _mq[_mq['Malmquist TFP'] != '—'].copy()
            md2 = [["Hôtel", "BCC N-1", "BCC N", "Catch-up", "Frontier shift", "TFP", "Lecture"]]
            for _, r in _mq.iterrows():
                try:
                    if float(r['Catch-up']) < 1.0:
                        _sig(r['Hôtel'], 'recul')
                except Exception:
                    pass
                _cat = r.get('Catégorie')
                _cat = "Non calculable" if (not isinstance(_cat, str) or not _cat) else _cat
                md2.append([_clean(r['Hôtel'], 42), _clean(r['BCC N-1']), _clean(r['BCC N']),
                            _clean(r['Catch-up']), _clean(r['Frontier Shift']),
                            _clean(r['Malmquist TFP']), _clean(_cat, 18)])
            story.append(_tbl(md2, [6.2*cm, 1.5*cm, 1.4*cm, 1.7*cm, 2.1*cm, 1.5*cm, 3.0*cm],
                              align_right=[1, 2, 3, 4, 5]))

            _nc = len(_mq) - len(_calc)
            if _nc > 0:
                story.append(Paragraph(
                    f"{_nc} actif(s) non calculables : l'indice exige que l'hôtel soit "
                    "évaluable sur les deux frontières (N-1 et N). Les unités situées aux "
                    "bornes de l'échantillon sortent parfois du domaine de faisabilité du "
                    "programme linéaire croisé — limite connue de la méthode, sans "
                    "conséquence sur les autres résultats.", small_s))

            if len(_calc) > 0:
                _cu = pd.to_numeric(_calc['Catch-up'], errors='coerce')
                _fs = pd.to_numeric(_calc['Frontier Shift'], errors='coerce')
                _tf = pd.to_numeric(_calc['Malmquist TFP'], errors='coerce')
                _n_maree = int(((_cu <= 1.0) & (_tf > 1.0)).sum())
                _n_alpha = int(((_cu > 1.0) & (_tf > 1.0)).sum())
                _txt = (f"Frontier shift moyen de {_fs.mean():.3f} : la frontière du "
                        f"portefeuille s'est {'déplacée vers le haut' if _fs.mean() > 1 else 'contractée'} "
                        f"de {abs(_fs.mean()-1)*100:.1f} % entre les deux exercices. ")
                if _n_maree > 0:
                    _txt += (f"{_n_maree} actif(s) affichent une productivité en hausse sans "
                             "progrès de gestion propre — leur amélioration vient "
                             "intégralement du marché. C'est une performance empruntée, pas "
                             "acquise : elle ne survivra pas à un retournement de cycle. ")
                if _n_alpha > 0:
                    _txt += (f"{_n_alpha} actif(s) combinent rattrapage et marché porteur — "
                             "c'est là que se trouve la création de valeur réelle.")
                story.append(_lecture(_txt))
    except Exception as _e:
        story.append(Paragraph(f"Malmquist non calculable ({_clean(_e, 90)}).", body_s))

    # ══════════════════ DIAGNOSTIC CROISÉ MULTI-DIMENSIONS ══════════════════
    story.append(PageBreak())
    story.append(Paragraph("13. Diagnostic croisé — Opérationnel × Financier × Commercial × RH", h1_s))
    story.append(Paragraph(
        "Le score BCC (sections 1 à 12) compare chaque hôtel à ses pairs sur chambres, "
        "ETP et charges opérationnelles — il indique QUI sous-performe, pas POURQUOI. "
        "Cette section croise le diagnostic opérationnel avec les modules Financier, "
        "Commercial/Revenue Management et RH quand des données réelles sont disponibles, "
        "pour distinguer un problème managérial ciblé d'un problème structurel touchant "
        "plusieurs fonctions à la fois.", meth_s))

    _dim_modules_pdf = {
        'financial_usali'   : ('💰 Financier', "Renégocier la structure de coûts départementaux "
                                "(Rooms/F&B), réduire les charges non distribuées, revoir les "
                                "contrats fournisseurs sur le poste identifié comme faible."),
        'revenue_management': ('📈 Commercial / Revenue Mgmt', "Revoir la stratégie de "
                                "distribution (dépendance OTA/GDS), renégocier les commissions "
                                "canal, ajuster l'équilibre tarif/occupation (ADR vs TO)."),
        'workforce'          : ('👥 Ressources Humaines', "Revoir la productivité du travail "
                                "(ratio ETP/activité), la formation, l'organisation des équipes "
                                "sur les postes identifiés comme faibles."),
    }
    _mr = module_results or {}
    _dim_scores_pdf = {}
    for _mid, (_lbl, _sol) in _dim_modules_pdf.items():
        _res = _mr.get(_mid)
        if _res is not None and not getattr(_res, 'error', None) and not _res.scores.empty:
            _dim_scores_pdf[_lbl] = (dict(zip(_res.scores['dmu_name'], _res.scores['efficiency'])), _sol)

    if not _dim_scores_pdf:
        story.append(Paragraph(
            "Non calculable : aucun module Financier / Commercial / RH actif avec de "
            "vraies données au moment de la génération. Le diagnostic reste limité à "
            "la dimension opérationnelle (sections 1 à 12) tant que ces données ne "
            "sont pas fournies via l'enrichissement.", body_s))
    else:
        _diag_rows_pdf = []
        for h in dea.hotels:
            _bcc_h = dea.bcc_scores.get(h, 1.0)
            if _bcc_h >= _TH['crit']:
                continue
            _weak = [(lbl, sol) for lbl, (sc, sol) in _dim_scores_pdf.items()
                     if sc.get(h) is not None and sc[h] < _TH['crit']]
            if _weak:
                _diag_rows_pdf.append((h, _bcc_h, _weak))

        if not _diag_rows_pdf:
            story.append(Paragraph(
                "Les hôtels critiques en BCC ne montrent pas de faiblesse identifiable "
                "sur les autres dimensions actives — la sous-performance semble "
                "d'origine purement opérationnelle (voir Slacks, section 6).", body_s))
        else:
            _n_multi_pdf = sum(1 for _, _, w in _diag_rows_pdf if len(w) >= 2)
            dd = [["Hôtel", "BCC", "Dimensions en cause", "Plan d'action"]]
            for h, bcc_h, weak in _diag_rows_pdf[:12]:
                dd.append([_clean(h, 34), f"{bcc_h:.1%}",
                          _clean(", ".join(l.split(" ", 1)[-1] for l, _ in weak), 30),
                          _clean(" | ".join(s for _, s in weak), 70)])
            story.append(_tbl(dd, [4.3*cm, 1.4*cm, 4.0*cm, 7.6*cm]))
            if len(_diag_rows_pdf) > 12:
                story.append(Paragraph(
                    f"Tableau limité aux 12 premiers actifs (sur {len(_diag_rows_pdf)} au total).",
                    small_s))
            story.append(_lecture(
                f"{len(_diag_rows_pdf)} hôtel(s) critique(s) en BCC montrent aussi une "
                f"faiblesse identifiable sur au moins une autre dimension"
                + (f", dont {_n_multi_pdf} sur plusieurs dimensions à la fois — signal de "
                   f"problème structurel : un plan managérial seul n'y suffira pas, "
                   f"l'arbitrage doit couvrir les fonctions concernées simultanément."
                   if _n_multi_pdf else
                   " — dans chaque cas une seule dimension supplémentaire est en cause, "
                   "un levier ciblé sur cette fonction devrait suffire.")
            ))

    # ══════════════════ SIGNAUX CUMULÉS ══════════════════
    story.append(PageBreak())
    story.append(Paragraph("Signaux cumulés — lecture transversale", h1_s))
    story.append(Paragraph(
        "Chaque section précédente pose un diagnostic isolé. Cette lecture les croise : "
        "un actif signalé une seule fois relève d'un ajustement ciblé, un actif cumulant "
        "trois signaux indépendants pose une question de détention. La convergence de "
        "diagnostics méthodologiquement distincts est un indice plus solide qu'un score "
        "unique, quel qu'il soit.", meth_s))

    _cum = sorted(((h, s) for h, s in _sig_reg.items() if len(s) >= 2),
                  key=lambda x: (-len(x[1]), dea.bcc_scores.get(x[0], 0)))
    if _cum:
        _cd = [["Hôtel", "BCC", "Signaux", "Diagnostics convergents"]]
        for h, sgs in _cum[:12]:
            _cd.append([_clean(h, 38), f"{dea.bcc_scores.get(h,0):.1%}", str(len(sgs)),
                        _clean(" · ".join(_SIGLAB[k] for k in sorted(sgs)), 92)])
        story.append(_tbl(_cd, [5.2*cm, 1.5*cm, 1.4*cm, 9.3*cm],
                          hdr=AMBER, fs=6.8, align_right=[1, 2]))

        _max  = max(len(s) for _, s in _cum)
        _lourd = [h for h, s in _cum if len(s) >= 3]
        _txt = (f"{len(_cum)} actif(s) cumulent au moins deux signaux, "
                f"jusqu'à {_max} pour le plus exposé. ")
        if _lourd:
            _txt += (f"Les {len(_lourd)} actif(s) à trois signaux ou plus — "
                     + ", ".join(_clean(h) for h in _lourd[:3])
                     + (" …" if len(_lourd) > 3 else "")
                     + " — méritent un examen dédié avant le prochain cycle "
                     "budgétaire : le cumul suggère une cause structurelle commune "
                     "plutôt qu'une série de dysfonctionnements indépendants. ")
        _txt += ("À l'inverse, un actif absent de ce tableau ne présente aucune "
                 "convergence de signaux : ses écarts éventuels relèvent d'ajustements "
                 "d'exploitation.")
        story.append(_lecture(_txt))

        # Combinaisons remarquables — la lecture croisée proprement dite
        _combi = []
        for h, s in _sig_reg.items():
            if {'capital_bas', 'ft_faible'} <= s:
                _combi.append((h, "Capital sous-productif et conversion faible",
                               "L'actif immobilise du capital que son exploitation ne "
                               "rentabilise pas, et sa structure de coûts absorbe la "
                               "croissance. Arbitrage à instruire — un plan managérial "
                               "seul ne corrigera pas les deux."))
            elif {'sbm_masque', 'recul'} <= s:
                _combi.append((h, "Inefficience masquée et dégradation",
                               "Le score radial flatte la situation réelle et la "
                               "trajectoire est baissière. Le diagnostic apparent "
                               "sous-estime le problème ; auditer les postes identifiés "
                               "en section Slacks."))

        if _combi:
            story.append(Spacer(1, 6))
            story.append(Paragraph("Combinaisons appelant une décision", h2_s))
            # Les chaînes ne se coupent pas dans un tableau ReportLab : le texte
            # long doit passer par un Paragraph pour être retourné à la ligne.
            _cell = S("CEL", fontSize=6.8, leading=8.4, fontName="Helvetica")
            _cellb= S("CLB", fontSize=6.8, leading=8.4, fontName="Helvetica-Bold")
            _kd = [["Hôtel", "Combinaison", "Ce qu'elle implique"]]
            for h, lab, txt in _combi[:8]:
                _kd.append([Paragraph(_clean(h), _cellb),
                            Paragraph(lab, _cell),
                            Paragraph(txt, _cell)])
            story.append(_tbl(_kd, [4.0*cm, 3.9*cm, 9.5*cm], hdr=RED, fs=7.0))
    else:
        story.append(Paragraph(
            "Aucun actif ne cumule plusieurs signaux de vigilance. Les écarts observés "
            "dans les sections précédentes sont isolés et relèvent chacun d'un levier "
            "propre — c'est le profil d'un portefeuille sain, dont les marges de "
            "progression sont opérationnelles et non structurelles.", body_s))

    # ══════════════════ SYNTHESE ══════════════════
    story.append(Spacer(1, 10))
    story.append(Paragraph("Synthèse — priorités d'action", h1_s))

    _q3 = [h for h in dea.hotels if dea.quadrants.get(h) == 'Q3']
    _q2 = [h for h in dea.hotels if dea.quadrants.get(h) == 'Q2']
    _q4 = [h for h in dea.hotels if dea.quadrants.get(h) == 'Q4']
    _q1 = [h for h in dea.hotels if dea.quadrants.get(h) == 'Q1']

    sy = [["Horizon", "Cible", "Action", "Nb actifs"]]
    if _q3:
        sy.append(["0 - 6 mois", "Problème de gestion (Q3)",
                   "Plan managérial — taille adaptée, levier immédiat sans capital",
                   str(len(_q3))])
    if _q4:
        sy.append(["6 - 18 mois", "Double handicap (Q4)",
                   "Restructuration opérationnelle puis réexamen de la détention",
                   str(len(_q4))])
    if _q2:
        sy.append(["12 - 24 mois", "Problème d'échelle (Q2)",
                   "Arbitrage capitalistique — extension, reconfiguration ou cession",
                   str(len(_q2))])
    if _q1:
        sy.append(["Continu", "Actifs de référence (Q1)",
                   "Documenter les pratiques et les diffuser au portefeuille",
                   str(len(_q1))])
    if len(sy) > 1:
        story.append(_tbl(sy, [2.6*cm, 4.4*cm, 8.4*cm, 2.0*cm], align_right=[3]))

    story.append(Spacer(1, 8))
    story.append(Paragraph("Limites méthodologiques", h1_s))
    story.append(Paragraph(
        "La DEA est une méthode déterministe : elle attribue tout écart à la frontière à "
        "de l'inefficience, sans distinguer la part imputable au bruit statistique ou à "
        "des facteurs exogènes non modélisés. Les scores sont sensibles au choix des "
        "variables et à la composition de l'échantillon — ajouter ou retirer un actif "
        "modifie les scores de tous les autres. Ce rapport constitue un outil de "
        "priorisation et de dialogue avec les exploitants ; il ne se substitue ni à une "
        "due diligence, ni à une analyse de marché, ni au jugement du comité "
        "d'investissement.", body_s))

    story.append(Paragraph(
        "Les commentaires « lecture investisseur » se déclenchent sur des seuils "
        + ("<b>ajustés pour ce portefeuille</b>" if _TH_CUSTOM
           else "correspondant aux conventions du secteur")
        + f" : actif critique sous {_TH['crit']:.0%} de BCC, Flow Through de référence "
        f"{_TH['ft_norm']:.0%} (alerte sous {_TH['ft_low']:.0%}), écart BCC-SBM signalé "
        f"au-delà de {_TH['sbm']*100:.0f} points, écart technologique au-delà de "
        f"{_TH['tgr']*100:.0f} points, amplitude GOPPAM au-delà de {_TH['goppam']:.1f}x."
        + ("  Ces valeurs ont été modifiées par rapport aux réglages par défaut ; "
           "elles doivent être cohérentes avec le positionnement du portefeuille."
           if _TH_CUSTOM else ""), small_s))

    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=0.5, color=NAVY))
    story.append(Paragraph(
        "DEA-H v4 &#183; REIV Hospitality &#183; "
        "Charnes, Cooper &amp; Rhodes (1978) &#183; Banker, Charnes &amp; Cooper (1984) &#183; "
        "Andersen &amp; Petersen (1993) &#183; Tone (2001) &#183; Barros (2005) &#183; "
        "Simar &amp; Wilson (2007) &#183; O'Donnell et al. (2008) &#183; "
        "Pulina &amp; Santoni (2018) &#183; Confidentiel",
        small_s))

    doc.build(story)
    return buf.getvalue()


# ── Malmquist · Tobit · MDEA Room/F&B · Mann-Whitney ─────────────────────
from malmquist_tobit import (
    compute_malmquist, has_n1_cols, compute_tobit,
    compute_mdea_room_fb, mann_whitney_groups, MDEA_COL_MAP,
    compute_simar_wilson, build_stage2_vars,
)

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
#  Config page
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="DEA-H — Asset Manager Benchmarking",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Police & Base ─────────────────────────────── */
html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; }

/* ── Header principal ──────────────────────────── */
.main-header {
    font-size: 2.4rem; font-weight: 900; letter-spacing: -0.5px;
    background: linear-gradient(135deg, #1a3a5c 0%, #2e6da4 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    text-align: center; margin-bottom: 0.3rem; line-height: 1.1;
}
.sub-header {
    text-align: center; color: #6b7280; font-size: 0.95rem;
    margin-bottom: 0.3rem; letter-spacing: 0.3px;
}
.reiv-badge {
    text-align: center; margin-bottom: 1.5rem;
}
.reiv-badge span {
    background: #1a3a5c; color: white; font-size: 0.72rem;
    font-weight: 600; padding: 3px 10px; border-radius: 20px;
    letter-spacing: 0.5px; text-transform: uppercase;
}

/* ── Section titles ────────────────────────────── */
.section-title {
    font-size: 1.15rem; font-weight: 700; color: #1a3a5c;
    border-left: 4px solid #2e6da4; padding-left: 0.7rem;
    margin: 1.4rem 0 0.8rem 0;
}

/* ── KPI boxes ─────────────────────────────────── */
.kpi-box {
    background: linear-gradient(135deg, #1a3a5c, #2e6da4);
    padding: 1rem 1.2rem; border-radius: 0.8rem;
    color: white; text-align: center;
}
.kpi-warn {
    background: linear-gradient(135deg, #c0392b, #e74c3c);
    padding: 1rem 1.2rem; border-radius: 0.8rem;
    color: white; text-align: center;
}
.kpi-ok {
    background: linear-gradient(135deg, #1e8449, #27ae60);
    padding: 1rem 1.2rem; border-radius: 0.8rem;
    color: white; text-align: center;
}

/* ── Metric cards (native Streamlit) ───────────── */
[data-testid="metric-container"] {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 0.8rem 1rem;
    box-shadow: 0 1px 4px rgba(26,58,92,0.06);
}
[data-testid="metric-container"] label {
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    color: #64748b !important;
    text-transform: uppercase;
    letter-spacing: 0.4px;
}
[data-testid="stMetricValue"] {
    font-size: 1.55rem !important;
    font-weight: 800 !important;
    color: #1a3a5c !important;
}

/* ── Sidebar ────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a3a5c 0%, #0f2238 100%);
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stNumberInput label,
[data-testid="stSidebar"] .stRadio label { color: #cbd5e1 !important; font-size: 0.82rem !important; }
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: #93c5fd !important; font-size: 0.9rem !important;
    text-transform: uppercase; letter-spacing: 0.8px; font-weight: 700;
}
[data-testid="stSidebar"] hr { border-color: #2e6da4 !important; }

/* ── Tabs ───────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px; background: #f1f5f9;
    padding: 4px; border-radius: 10px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 7px; padding: 6px 14px;
    font-size: 0.82rem; font-weight: 600; color: #64748b;
}
.stTabs [aria-selected="true"] {
    background: white !important; color: #1a3a5c !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.12);
}

/* ── Dataframes ─────────────────────────────────── */
[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }

/* ── Buttons ────────────────────────────────────── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #1a3a5c, #2e6da4) !important;
    border: none !important; border-radius: 8px !important;
    font-weight: 700 !important; letter-spacing: 0.3px;
    box-shadow: 0 2px 8px rgba(26,58,92,0.25) !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 14px rgba(26,58,92,0.35) !important;
}

/* ── Divider ────────────────────────────────────── */
hr { border-color: #e2e8f0 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  Titre
# ─────────────────────────────────────────────
st.markdown('<h1 class="main-header">DEA-H — Asset Manager Benchmarking</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Analyse BCC/CCR &middot; TOPSIS &middot; K-means &middot; Metafrontière &middot; Multi-Module DEA</p>', unsafe_allow_html=True)
st.markdown('<div class="reiv-badge"><span>REIV Hospitality · v3.9</span></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  Session state init (multi-module)
# ─────────────────────────────────────────────
if "se_results" not in st.session_state:
    st.session_state["se_results"] = None
if "ce_results" not in st.session_state:
    st.session_state["ce_results"] = None
if "malmquist_results" not in st.session_state:
    st.session_state["malmquist_results"] = None
if "tobit_results" not in st.session_state:
    st.session_state["tobit_results"] = None
if "mdea_rf_results" not in st.session_state:
    st.session_state["mdea_rf_results"] = None
if "mw_group_results" not in st.session_state:
    st.session_state["mw_group_results"] = None
if "module_results" not in st.session_state:
    st.session_state["module_results"] = {}
if "active_modules_mm" not in st.session_state:
    st.session_state["active_modules_mm"] = []
if "variable_overrides_mm" not in st.session_state:
    st.session_state["variable_overrides_mm"] = {}
if "_csv_hash" not in st.session_state:
    st.session_state["_csv_hash"] = None

# ─────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    st.subheader("💼 Paramètres économiques")
    avg_salary   = st.number_input("Coût ETP moyen (€/an)", help="Salaire chargé moyen par ETP. Appliquer la même valeur pour tous les actifs du compset.",           value=35_000, step=5_000, format="%i")
    # Défaut = 1 : dans l'immense majorité des fichiers, RevPAR est déjà en €.
    # Un défaut à 1000 multipliait silencieusement tous les upsides par 1000.
    revpar_value = st.number_input(
        "Multiplicateur RevPAR (1 = €)",
        help="Laisser à 1 si RevPAR est en € réels dans le CSV — cas normal. "
             "Ajuster uniquement si RevPAR est exprimé en indice ou en k€.",
        value=1, step=1, min_value=1, format="%i")
    jours_exploit = st.number_input(
        "Jours d'exploitation / an",
        value=365, min_value=30, max_value=365, step=1, format="%i",
        help="Resort saisonnier : 180-240 | Urban : 340-365. Corrige PAR, GOPPAM, TREVPAR."
    )
    st.markdown("---")
    illustrative_data = st.checkbox(
        "⚠️ Données illustratives (pas encore vérifiées)",
        value=False,
        help="À cocher si tout ou partie des données saisies sont des estimations/"
             "approximations plutôt que des chiffres confirmés par le client. Affiche "
             "un bandeau d'avertissement dans l'app et sur le rapport PDF, pour éviter "
             "qu'un résultat basé sur des données illustratives soit lu comme un "
             "résultat vérifié.",
    )

    _ft_int = st.slider(
        "Flow Through % (GOP/ΔRevenu)",
        min_value=20, max_value=80, value=45, step=1,
        format="%d%%",
        help="Part du revenu marginal convertie en GOP. Benchmark Europe : hôtellerie urbaine ~45%. "
             "Appliquer la même valeur pour tous les actifs du compset."
    )
    ft_pct = _ft_int / 100  # Conversion en décimal pour les calculs

    st.markdown("---")
    st.caption(
        "💡 Si votre compset mélange des catégories (3★/5★, urban/resort), utilisez l'onglet "
        "**🌐 Metafrontière** pour corriger le biais inter-segments via GTE/TGR (Assaf et al. 2010)."
    )

    st.markdown("---")
    st.subheader("📐 Seuils Quadrants")
    bcc_threshold   = st.slider("Seuil BCC efficience",   0.70, 0.99, 0.90, 0.01)
    scale_threshold = st.slider("Seuil Scale Efficiency", 0.70, 0.99, 0.90, 0.01)

    # ── Seuils de lecture du rapport ─────────────────────────────────────────
    # Les valeurs par défaut sont des conventions sectorielles hôtellerie.
    # Elles peuvent être mal calibrées sur un portefeuille très haut de gamme
    # ou très économique : ces curseurs permettent de les recaler.
    with st.expander("📄 Seuils d'interprétation du rapport"):
        st.caption(
            "Ces seuils déclenchent les commentaires « lecture investisseur » du "
            "rapport PDF. Ils ne modifient aucun calcul d'efficience."
        )
        _th_crit   = st.slider("Actif critique — BCC sous", 0.60, 0.95, 0.85, 0.01,
                               key='th_crit',
                               help="En dessous de ce score, l'actif entre dans le plan d'action prioritaire.")
        _th_ft     = st.slider("Flow Through — norme sectorielle", 0.30, 0.70, 0.50, 0.05,
                               key='th_ft',
                               help="Part du revenu supplémentaire convertie en GOP. 50 % est la norme usuelle.")
        _th_ft_low = st.slider("Flow Through — seuil d'alerte", 0.15, 0.50, 0.40, 0.05,
                               key='th_ftlow',
                               help="En dessous, la structure de coûts absorbe la croissance.")
        _th_sbm    = st.slider("Écart BCC-SBM signalé (pts)", 5, 40, 15, 1,
                               key='th_sbm',
                               help="Au-delà, le score radial masque une inefficience localisée.")
        _th_tgr    = st.slider("Écart technologique TGR signalé (pts)", 5, 40, 15, 1,
                               key='th_tgr',
                               help="Au-delà, l'écart entre segments relève de l'arbitrage, pas du management.")
        _th_goppam = st.slider("Amplitude GOPPAM signalée (×)", 1.5, 5.0, 2.5, 0.1,
                               key='th_goppam',
                               help="Rapport max/min du GOP par m² au-delà duquel l'hétérogénéité est commentée.")

    REPORT_THRESHOLDS = {
        'crit'  : _th_crit,   'ft_norm': _th_ft,     'ft_low': _th_ft_low,
        'sbm'   : _th_sbm / 100.0,  'tgr': _th_tgr / 100.0,
        'goppam': _th_goppam,
    }

    st.markdown("---")
    st.subheader("📐 Mode variables (Raab & Lichty, 2002)")
    _n_hotels = len(df) if 'df' in dir() else 0
    _has_ch_col = ('df' in dir()) and ('nb_chambres' in df.columns)
    _std_min = 18 if _has_ch_col else 15
    if _n_hotels > 0 and _n_hotels < _std_min:
        st.warning(f"⚠️ {_n_hotels} hôtels — en dessous du seuil standard ({_std_min}). Voir mode ci-dessous.")

    dea_mode = st.radio(
        "Nombre de variables",
        options=['standard', 'compact', 'minimal'],
        format_func=lambda x: {
            'standard': (f"Standard ({'3+3' if _has_ch_col else '2+3'}) — min "
                         f"{'18' if _has_ch_col else '15'} hôtels · "
                         f"{'Chambres + ETP + OpEx' if _has_ch_col else 'ETP + OpEx'} → RevPAR + Sat + TO"),
            'compact' : f"Compact (2+2) — min 12 hôtels · ETP + OpEx → RevPAR + Satisfaction",
            'minimal' : f"Minimal (2+1) — min 9 hôtels  · ETP + OpEx → RevPAR",
        }[x],
        help="Réduire les variables si votre compset est petit. Standard utilise nb_chambres "
             "automatiquement s'il est présent dans le fichier (aucune vérification de fiabilité "
             "— si la colonne existe, elle est utilisée). "
             "Règle : n_hôtels ≥ 3 × (n_inputs + n_outputs). Réf. : Raab & Lichty (2002).",
        index=0,
    )

    if dea_mode != 'standard':
        st.caption(
            "💡 En mode réduit, la **cross-efficience (Tab 3)** reste recommandée — "
            "elle compense partiellement la sur-définition avec peu de DMUs."
        )

    st.markdown("---")
    st.subheader("🔄 Orientation du modèle")
    orientation = st.radio(
        "Modèle DEA",
        options=['input', 'output'],
        format_func=lambda x: "📥 Input-Oriented (plan restructuration)" if x == 'input'
                               else "📤 Output-Oriented (plan croissance)",
        help="Input : 'De combien réduire les ressources ?' / Output : 'De combien augmenter les revenus ?' (Barros, 2005)",
    )

    st.markdown("---")
    with st.expander("⚖️ Poids TOPSIS (Cornell methodology)", expanded=False):
        st.caption("Charnes et al. (1978) — Somme des poids = 1. Défaut Cornell : BCC 35% · Scale 25% · RevPAR 25% · TO 15%")
        _w_bcc    = st.slider("BCC (Gestion pure)",   0.0, 1.0, 0.35, 0.05, key='w_bcc')
        _w_scale  = st.slider("Scale Efficiency",     0.0, 1.0, 0.25, 0.05, key='w_scale')
        _w_revpar = st.slider("RevPAR",               0.0, 1.0, 0.25, 0.05, key='w_revpar')
        _w_to     = st.slider("Taux d'occupation",   0.0, 1.0, 0.15, 0.05, key='w_to')
        _w_sum    = _w_bcc + _w_scale + _w_revpar + _w_to
        _w_bcc    = _w_bcc / _w_sum if _w_sum > 0 else 0.35
        _w_scale  = _w_scale / _w_sum if _w_sum > 0 else 0.25
        _w_revpar = _w_revpar / _w_sum if _w_sum > 0 else 0.25
        _w_to     = _w_to / _w_sum if _w_sum > 0 else 0.15
        st.caption(f"Poids normalisés : BCC {_w_bcc:.0%} · Scale {_w_scale:.0%} · RevPAR {_w_revpar:.0%} · TO {_w_to:.0%}")
    topsis_weights = [_w_bcc, _w_scale, _w_revpar, _w_to]

    st.markdown("---")
    uploaded_file = st.file_uploader(
        "Excel ou CSV : hotel_name (1re colonne), nb_chambres, nb_employes, "
        "couts_op_ex, revpar, satisfaction, taux_occupation",
        type=['xlsx', 'xls', 'csv'],
        help="Excel recommandé — évite les erreurs de séparateur et les virgules "
             "dans les noms d'hôtels.",
    )
    use_sample = st.checkbox("📋 Données d'exemple — Meliá Group Espagne (24 hôtels)", value=(uploaded_file is None))

# ─────────────────────────────────────────────
#  Chargement des données
# ─────────────────────────────────────────────
NUMERIC_COLS = ['nb_employes', 'couts_op_ex', 'revpar', 'satisfaction', 'taux_occupation']
OPTIONAL_NUMERIC_COLS = ['nb_chambres']  # jamais requis — non fiable sans vraie donnée

def load_sample() -> pd.DataFrame:
    """
    Portefeuille de référence — Meliá Group Espagne (24 hôtels)
    DEA-H v4 · REIV Hospitality
    Satisfaction /5→/10 · couts_op_ex M€ · financiers k€

    Ne contient QUE les colonnes de base DEA-H (celles qu'un vrai client
    fournirait) + les colonnes N-1 pour Malmquist. Les colonnes multi-module
    (rooms_revenue, fb_revenue, total_revenue, energy_kwh, water_m3,
    co2_tonnes, adr, trevpar, revenue_per_fte, book_value_assets, ebitda...)
    ont été retirées : elles étaient des reformulations à ratio fixe de
    revpar/nb_chambres/surface_m2 (ex. energy_kwh = 228,125 × surface_m2 pour
    les 24 hôtels, sans exception), pas de vraies données Meliá — leur
    présence faisait passer artificiellement les modules Financier USALI,
    Capital & Actifs, Main-d'œuvre, Revenue Management et ESG en "disponible"
    sans qu'aucune vraie donnée indépendante ne les alimente. Avec ce jeu de
    données, seuls Opérationnel Global et Qualité & Satisfaction sont
    honnêtement calculables — comme sur un vrai fichier client non enrichi.
    """
    d = {
        'hotel_name': ['Meliá South Beach', 'Meliá Calviá Beach', 'Sol Wave House All Suites', 'Sol Barbados', 'Sol House The Studio', 'Sol Guadalupe', 'Innside by Meliá Calviá Beach', 'ME Marbella', 'Meliá Madrid Princesa', 'Meliá Galgos (Madrid)', 'Meliá Castilla (Madrid - Part)', 'Tryp Madrid Gran Vía', 'Tryp Madrid Chamartín', 'Tryp Madrid Atocha', 'Meliá Atlanterra (Cadix)', 'Meliá Sol y Nieve (Sierra Nevada)', 'Sol Pelicanos Ocas (Benidorm)', 'Meliá Puerto de la Cruz (Ténérife)', 'Sol Lanzarote', 'Meliá Sierra Nevada', 'Meliá María Pita (La Corogne)', 'Meliá San Sebastián Orly', 'Meliá Alicante', 'Sol Príncipe (Torremolinos)'],
        'nb_chambres': [240, 316, 184, 342, 290, 303, 272, 180, 274, 356, 400, 175, 199, 149, 288, 258, 794, 300, 343, 221, 183, 102, 545, 799],
        'nb_employes': [85, 110, 65, 120, 95, 100, 90, 140, 115, 130, 160, 55, 60, 50, 95, 85, 210, 95, 110, 75, 60, 40, 170, 220],
        'couts_op_ex': [3.84, 4.92, 2.88, 5.52, 4.56, 4.68, 4.2, 6.24, 5.16, 5.88, 7.32, 2.52, 2.76, 2.28, 4.32, 4.08, 9.36, 3.96, 4.8, 3.48, 2.64, 1.92, 8.16, 9.72],
        'revpar': [145, 138, 120, 115, 125, 98, 130, 260, 155, 140, 165, 110, 95, 118, 142, 135, 85, 90, 105, 128, 88, 150, 122, 95],
        'satisfaction': [9.0, 8.6, 8.2, 8.4, 8.8, 8.0, 9.2, 9.4, 8.8, 8.4, 8.6, 7.8, 8.0, 8.2, 9.0, 8.6, 8.0, 8.2, 8.6, 8.4, 8.8, 9.2, 8.4, 8.2],
        'taux_occupation': [82, 79, 85, 88, 81, 76, 84, 73, 78, 75, 80, 83, 74, 81, 86, 68, 92, 84, 87, 65, 72, 79, 83, 89],
        'surface_m2': [9600, 12640, 7360, 13680, 11600, 12120, 10880, 7200, 10960, 14240, 16000, 7000, 7960, 5960, 11520, 10320, 31760, 12000, 13720, 8840, 7320, 4080, 21800, 31960],
        'gop': [4740.6, 5727.3, 3122.5, 5731.0, 4851.0, 3730.7, 4923.9, 5654.9, 5480.3, 6190.7, 8768.8, 2645.5, 2316.5, 2363.9, 5838.8, 3927.0, 10298.6, 3758.2, 5205.4, 3049.9, 1925.8, 2004.3, 9163.3, 11211.9],
        # CAPEX différenciés par catégorie (k€/an): 3★=0.90, 4★=1.75, 4★Sup=2.40, 5★=3.80 k€/ch
        # (taux benchmark secteur — CAPEX réels par hôtel non disponibles ; même
        # réserve que ci-dessus, conservé car colonne DEA-H de base préexistante,
        # pas ajoutée pour le système multi-module)
        'capex_annuel': [576.0, 553.0, 322.0, 598.5, 507.5, 530.2, 476.0, 684.0, 1041.2, 623.0, 700.0, 157.5, 179.1, 260.8, 504.0, 451.5, 714.6, 525.0, 600.2, 386.8, 320.2, 178.5, 953.8, 1398.2],
        'classement_etoiles': [5, 4, 4, 4, 4, 3, 5, 5, 5, 4, 5, 3, 3, 4, 5, 4, 3, 3, 4, 4, 3, 5, 4, 3],
        'categorie': ['4★ Sup','4★','4★','4★','4★','3★','4★','5★','5★','4★','4★','3★','3★','4★','4★','4★','3★','4★','4★','4★','3★','4★','4★','4★'],
        # Saisonnalité : 1=resort/côtier/montagne (haute saison), 0=urbain/année ronde
        # Réf. : Pulina & Santoni (2018) ; Cracolici et al. (2008)
        'saison_dummy': [1,1,1,1,1,1,1,1,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,1],
        'nb_chambres_n1': [240, 316, 184, 342, 290, 303, 272, 180, 274, 356, 400, 175, 199, 149, 288, 258, 794, 300, 343, 221, 183, 102, 545, 799],
        'nb_employes_n1': [86, 108, 65, 121, 94, 101, 87, 143, 115, 126, 163, 55, 58, 49, 92, 82, 208, 92, 110, 72, 61, 40, 164, 212],
        'couts_op_ex_n1': [3.728, 5.037, 2.825, 5.564, 4.566, 4.593, 4.227, 6.33, 5.123, 6.011, 7.184, 2.48, 2.695, 2.265, 4.406, 4.004, 9.331, 4.049, 4.796, 3.501, 2.641, 1.929, 7.978, 9.832],
        'revpar_n1': [136.0, 131.5, 115.7, 106.6, 114.0, 89.9, 124.8, 241.7, 142.3, 131.5, 154.3, 103.9, 86.8, 109.4, 135.2, 123.2, 80.2, 86.0, 96.6, 116.7, 83.2, 140.4, 117.1, 91.8],
        'satisfaction_n1': [9.0, 8.5, 8.2, 8.4, 8.7, 8.0, 9.2, 9.4, 8.6, 8.4, 8.5, 7.8, 7.9, 8.0, 9.0, 8.4, 7.8, 8.0, 8.5, 8.2, 8.7, 9.2, 8.2, 8.0],
        'taux_occupation_n1': [79.2, 75.1, 80.2, 82.7, 75.8, 71.1, 82.9, 70.5, 76.3, 73.3, 74.9, 79.0, 72.6, 78.4, 83.0, 64.6, 87.0, 81.1, 84.5, 64.1, 67.2, 77.0, 77.4, 85.9],
    }
    return pd.DataFrame(d).set_index('hotel_name')

if uploaded_file is not None:
    import io, csv as _csv
    _fname   = (uploaded_file.name or '').lower()
    _is_xlsx = _fname.endswith(('.xlsx', '.xls'))

    if _is_xlsx:
        # ── Excel : pas de séparateur, pas de BOM, pas d'ambiguïté virgule ──
        try:
            df = pd.read_excel(uploaded_file, index_col=0)
        except Exception as _e:
            st.sidebar.error(f"❌ Lecture Excel impossible : {_e}")
            st.sidebar.caption(
                "Vérifiez que la 1re feuille contient les données, "
                "l'en-tête en ligne 1 et hotel_name en colonne A."
            )
            st.stop()
        sep = 'Excel'
    else:
        # utf-8-sig retire le BOM Excel (sinon 1re colonne = '﻿hotel_name')
        raw = uploaded_file.read().decode('utf-8-sig', errors='replace')
        raw = raw.replace('\r\n', '\n').replace('\r', '\n')   # normalise CRLF Windows
        sep = ';' if raw.count(';') > raw.count(',') else ','

        # ── Contrôle d'intégrité : lignes au nombre de champs incohérent ─────
        # Cause fréquente : nom d'hôtel contenant une virgule sans guillemets
        # ex. The Hoxton, Paris  →  doit s'écrire  "The Hoxton, Paris"
        _rows      = list(_csv.reader(io.StringIO(raw), delimiter=sep))
        _rows      = [r for r in _rows if any(str(c).strip() for c in r)]
        _n_head    = len(_rows[0]) if _rows else 0
        _bad_lines = [(i + 2, len(r), r[0] if r else '')
                      for i, r in enumerate(_rows[1:]) if len(r) != _n_head]
        if _bad_lines:
            st.sidebar.error(
                f"❌ {len(_bad_lines)} ligne(s) mal formée(s) — "
                f"l'en-tête a {_n_head} colonnes."
            )
            for _ln, _nf, _first in _bad_lines[:5]:
                st.sidebar.caption(f"• Ligne {_ln} : {_nf} champs — commence par « {_first} »")
            st.sidebar.warning(
                "Cause la plus fréquente : un nom d'hôtel contient une virgule "
                "non protégée. Entourez-le de guillemets, ex. \"The Hoxton, Paris\", "
                "utilisez « ; » comme séparateur — ou déposez un fichier Excel."
            )
            st.stop()

        df = pd.read_csv(io.StringIO(raw), index_col=0, sep=sep, skipinitialspace=True)

    df.index = df.index.astype(str).str.strip()
    df.columns = [str(c).strip() for c in df.columns]

    missing_cols = [c for c in NUMERIC_COLS if c not in df.columns]
    if missing_cols:
        st.sidebar.error(f"❌ Colonnes manquantes : {', '.join(missing_cols)}")
        st.sidebar.caption("Colonnes requises : " + ", ".join(NUMERIC_COLS))
        st.stop()
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    if 'nb_chambres' in df.columns:
        df['nb_chambres'] = pd.to_numeric(df['nb_chambres'], errors='coerce')

    _n_before = len(df)
    df = df.dropna(subset=NUMERIC_COLS)
    _n_dropped = _n_before - len(df)

    if len(df) < 3:
        st.sidebar.error(
            f"❌ Seulement {len(df)} hôtel(s) exploitable(s) sur {_n_before} lignes lues. "
            "Le DEA exige au minimum 3 unités."
        )
        st.sidebar.caption(
            "Vérifiez que les colonnes numériques ne contiennent ni texte, "
            "ni cellule vide, ni séparateur décimal « , » (utilisez « . »)."
        )
        st.stop()

    st.sidebar.success(
        f"✅ {len(df)} hôtels chargés"
        + ("  ·  format Excel" if _is_xlsx else f"  ·  séparateur '{sep}'")
    )
    if _n_dropped > 0:
        st.sidebar.warning(
            f"⚠️ {_n_dropped} ligne(s) écartée(s) — valeur non numérique ou manquante "
            f"dans : {', '.join(NUMERIC_COLS)}"
        )

    # ── Détection valeurs négatives ou nulles (Pastor 1996 / Tone 2001) ──────
    _neg_issues = []
    for _col in ['revpar', 'taux_occupation', 'nb_chambres', 'nb_employes']:
        if _col in df.columns:
            _neg = (df[_col] <= 0).sum()
            if _neg > 0:
                _neg_issues.append(f"{_col} ({_neg} valeur{'s' if _neg>1 else ''} ≤ 0)")
    if 'gop' in df.columns:
        _neg_gop = (df['gop'] < 0).sum()
        if _neg_gop > 0:
            _neg_issues.append(f"gop ({_neg_gop} valeur{'s' if _neg_gop>1 else ''} négative{'s' if _neg_gop>1 else ''})")
    if _neg_issues:
        st.sidebar.warning(
            f"⚠️ **Valeurs non strictement positives détectées :** {', '.join(_neg_issues)}. "
            "Les modèles BCC/CCR exigent x > 0 et y > 0. "
            "Les actifs concernés risquent de biaiser la frontière DEA. "
            "Recommandation : les exclure du compset et les analyser séparément via l'onglet Fiche Actif (What-If)."
        )
    # Invalidation cache si CSV change
    _new_hash = hash(raw[:2000] + str(len(df)))
    if st.session_state.get('_csv_hash') != _new_hash:
        st.session_state['module_results'] = {}
        st.session_state['_csv_hash'] = _new_hash
elif use_sample:
    df = load_sample()
    st.sidebar.info("📋 Données d'exemple chargées")
else:
    st.sidebar.warning("👆 Importez un CSV ou activez les données d'exemple")
    st.stop()

with st.sidebar.expander("👁️ Aperçu données", expanded=False):
    st.dataframe(df, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR — Sélection modules Multi-Module DEA (v3.2)
# ─────────────────────────────────────────────────────────────────────────────
# Préparer df_mm : réinitialiser l'index pour avoir hotel_name en colonne
# + mapper les colonnes standard DEA-H vers les noms attendus par modules_config
_df_mm = df.reset_index().rename(columns={"hotel_name": "hotel_name"})

# ── Enrichissement optionnel : vraies données pour débloquer les modules ────
# check_module_feasibility bloque désormais un module tant que ses colonnes
# "required" ne sont pas de la vraie donnée (cf. _proxy_cols plus bas). La
# seule façon légitime de débloquer Financier USALI / Capital & Actifs /
# Main-d'œuvre / Revenue Management / ESG est de fournir les colonnes
# réelles ci-dessous — pas de contourner le contrôle.
with st.sidebar.expander("📂 Enrichir avec données réelles (débloquer des modules)", expanded=False):
    st.caption(
        "Colonnes attendues (ordre libre, uniquement celles que vous avez) : "
        "**hotel_name** puis parmi — "
        "`nb_rooms` (Opérationnel) · "
        "`rooms_revenue`/`rooms_cost`/`fb_cost`/`fb_revenue`/`other_dept_cost`/`other_dept_revenue`/`undistributed_expenses`/`fixed_charges`/`ebitda` (Financier USALI) · "
        "`total_revenue`/`book_value_assets` (Capital & Actifs) · "
        "`payroll_total`/`hours_worked`/`training_cost` (Main-d'œuvre) · "
        "`marketing_cost`/`ota_gds_cost`/`sales_fte`/`promo_budget`/`trevpar`/`adr` (Revenue Management) · "
        "`energy_kwh`/`water_m3`/`co2_tonnes`/`waste_tonnes` (ESG).  \n"
        "La colonne `hotel_name` doit correspondre exactement aux noms du fichier DEA-H principal."
    )
    _enrich_upload = st.file_uploader("Fichier d'enrichissement (CSV ou Excel)", type=["csv", "xlsx", "xls"], key="mm_enrich_upload")
    if _enrich_upload is not None:
        try:
            if _enrich_upload.name.endswith((".xlsx", ".xls")):
                _enrich_df = pd.read_excel(_enrich_upload)
            else:
                _enrich_raw = _enrich_upload.getvalue().decode("utf-8-sig")
                _enrich_df = pd.read_csv(StringIO(_enrich_raw))
            if "hotel_name" not in _enrich_df.columns:
                st.error("Colonne `hotel_name` introuvable dans le fichier importé.")
            else:
                _matched = _enrich_df["hotel_name"].isin(_df_mm["hotel_name"]).sum()
                _new_cols = [c for c in _enrich_df.columns if c != "hotel_name"]
                _df_mm = _df_mm.merge(_enrich_df, on="hotel_name", how="left", suffixes=("", "_enrichi"))
                st.success(f"{_matched}/{len(_enrich_df)} hôtels appariés · colonnes ajoutées : {', '.join(_new_cols)}")
                if _matched < len(_enrich_df):
                    st.warning("Certains noms d'hôtel du fichier importé ne correspondent à aucun hôtel du fichier DEA-H principal — vérifiez l'orthographe exacte.")
        except Exception as _e:
            st.error(f"Import impossible : {_e}")

# Mapping automatique colonnes DEA-H v3 → noms standard modules_config
_COL_ALIAS = {
    "nb_chambres"         : "nb_rooms",
    "nb_employes"      : "fte_total",
    "couts_op_ex"      : "opex_total",
    "revpar"           : "revpar",
    "satisfaction"     : "satisfaction_score",
    "taux_occupation"  : "occupancy_rate",
    "total_revenue"    : "total_revenue",
    "surface_m2"       : "surface_m2",
    "capex_annuel"     : "capex",
    "gop"              : "gop",
    "classement_etoiles": "star_achievement",
}
for old, new in _COL_ALIAS.items():
    if old in _df_mm.columns and new not in _df_mm.columns:
        _df_mm[new] = _df_mm[old]

# ── Colonnes calculées automatiquement ──────────────────────────────────────
# Chaque colonne créée ici est une reformulation arithmétique (RevPAR ×
# chambres × 365...) d'une colonne déjà utilisée ailleurs — pas une donnée
# nouvelle. Trackée dans _proxy_cols pour ne PAS pouvoir satisfaire, seule, un
# champ "required" (cf. check_module_feasibility) : un module ne doit pas se
# déclarer calculable sur une reformulation, mais uniquement sur une vraie
# donnée.
#
# nb_chambres → nb_rooms N'EST PAS trackée ici : depuis le renommage de la
# colonne source (ex-nb_lits), c'est un renommage 1:1 légitime — nb_chambres
# EST le nombre de chambres saisi par l'utilisateur, donc nb_rooms l'est aussi.
_proxy_cols = set()

# ADR = RevPAR / (TO/100)  — toujours calculable depuis le dataset DEA-H base
if "revpar" in _df_mm.columns and "occupancy_rate" in _df_mm.columns:
    if "adr" not in _df_mm.columns:
        _df_mm["adr"] = (
            _df_mm["revpar"] / (_df_mm["occupancy_rate"] / 100).replace(0, np.nan)
        ).round(2)
        _proxy_cols.add("adr")

# TRevPAR = CA total / nb_rooms (si total_revenue disponible)
if "total_revenue" in _df_mm.columns and "nb_rooms" in _df_mm.columns:
    if "trevpar" not in _df_mm.columns:
        _df_mm["trevpar"] = (
            _df_mm["total_revenue"] / (_df_mm["nb_rooms"] * 365)
        ).round(2)
        _proxy_cols.add("trevpar")

# revenue_per_fte = CA / ETP
if "total_revenue" in _df_mm.columns and "fte_total" in _df_mm.columns:
    if "revenue_per_fte" not in _df_mm.columns:
        _df_mm["revenue_per_fte"] = (
            _df_mm["total_revenue"] / _df_mm["fte_total"].replace(0, np.nan)
        ).round(0)
        _proxy_cols.add("revenue_per_fte")
elif "revpar" in _df_mm.columns and "nb_rooms" in _df_mm.columns and "fte_total" in _df_mm.columns:
    if "revenue_per_fte" not in _df_mm.columns:
        _df_mm["revenue_per_fte"] = (
            _df_mm["revpar"] * 365 * _df_mm["nb_rooms"] / _df_mm["fte_total"].replace(0, np.nan)
        ).round(0)
        _proxy_cols.add("revenue_per_fte")

# rooms_revenue estimé = revpar × nb_rooms × 365 (proxy si absent)
if "revpar" in _df_mm.columns and "nb_rooms" in _df_mm.columns:
    if "rooms_revenue" not in _df_mm.columns:
        _df_mm["rooms_revenue"] = (
            _df_mm["revpar"] * (_df_mm["occupancy_rate"] / 100 if "occupancy_rate" in _df_mm.columns else 1)
            * _df_mm["nb_rooms"] * 365
        ).round(0)
        _proxy_cols.add("rooms_revenue")

# total_revenue estimé = rooms_revenue (proxy minimal si absent)
if "total_revenue" not in _df_mm.columns and "rooms_revenue" in _df_mm.columns:
    _df_mm["total_revenue"] = _df_mm["rooms_revenue"]
    _proxy_cols.add("total_revenue")

# Injecter les colonnes calculees dans dea.df si analyse deja lancee
# pour que les onglets STR / Capital utilisent les vraies valeurs
if 'dea' in st.session_state:
    for _inject_col in ['adr', 'total_revenue', 'trevpar', 'revenue_per_fte', 'rooms_revenue']:
        if _inject_col in _df_mm.columns and _inject_col not in st.session_state['dea'].df.columns:
            _mm_vals = _df_mm.set_index('hotel_name')[_inject_col]
            try:
                st.session_state['dea'].df[_inject_col] = [
                    _mm_vals.get(h, np.nan) for h in st.session_state['dea'].hotels
                ]
            except Exception:
                pass

_available_mm = [c for c in _df_mm.columns if c != "hotel_name"]

# Sélecteur sidebar
_active_modules, _variable_overrides = render_module_selector(_available_mm, proxy_cols=_proxy_cols)
st.session_state["active_modules_mm"]    = _active_modules
st.session_state["variable_overrides_mm"] = _variable_overrides

# ─────────────────────────────────────────────
#  Bouton d'analyse
# ─────────────────────────────────────────────
if st.button("🚀 LANCER L'ANALYSE DEA COMPLÈTE", type="primary", use_container_width=True):
    with st.spinner("Calcul BCC · CCR · Slacks · TOPSIS · K-means · Quadrants…"):
        dea = HotelDEAAnalyzer.__new__(HotelDEAAnalyzer)
        dea.df              = df.copy()
        dea.hotels          = df.index.tolist()
        dea.n               = len(dea.hotels)
        dea.orientation     = orientation
        dea.mode            = dea_mode if 'dea_mode' in dir() else 'standard'
        dea.has_chambres    = 'nb_chambres'   in df.columns
        # nb_chambres est un input DEA standard dans la littérature (Barros
        # 2005) — utilisé quand une vraie donnée est disponible, ignoré
        # sinon. Ce n'est pas la variable qui pose problème, c'est une
        # colonne source non fiable qui la remplace parfois (cf. audit
        # portefeuille Ibis Atream) : la solution est de fournir un vrai
        # nombre de chambres, pas de l'exclure définitivement du modèle.
        # Compact/minimal restent volontairement à variables réduites,
        # indépendamment de la disponibilité de nb_chambres — c'est leur
        # raison d'être pour les petits échantillons (Raab & Lichty 2002).
        if dea_mode == 'standard' and dea.has_chambres:
            dea.input_cols  = ['nb_chambres', 'nb_employes', 'couts_op_ex']
        else:
            dea.input_cols  = ['nb_employes', 'couts_op_ex']
        dea.output_cols     = (['revpar'] if dea_mode == 'minimal' else ['revpar','satisfaction'])
        if dea_mode == 'standard': dea.output_cols = ['revpar','satisfaction','taux_occupation']
        _n_vars = len(dea.input_cols) + len(dea.output_cols)
        dea._dmu_ratio_warning = dea.n < 3 * _n_vars
        dea._dmu_ratio_info    = f"{dea.n} DMUs < 3×{_n_vars} = {3*_n_vars} [mode {dea_mode}]"
        dea.has_trevpar  = 'total_revenue'    in df.columns
        dea.has_surface  = 'surface_m2'       in df.columns
        dea.has_capex    = 'capex_annuel'     in df.columns
        dea.has_gop      = 'gop'              in df.columns
        dea.has_goppam   = 'surface_m2'       in df.columns and 'gop' in df.columns
        dea.has_stars    = 'classement_etoiles' in df.columns
        dea.has_flow     = 'gop_n1'           in df.columns and 'revenu_n1' in df.columns
        dea.inputs  = dea.df[dea.input_cols].values.astype(float)
        dea.outputs = dea.df[dea.output_cols].values.astype(float)
        dea.bcc_scores      = {}; dea.ccr_scores      = {}; dea.scale_efficiency = {}
        dea.slacks          = {}; dea.peers            = {}; dea.targets          = {}
        dea.topsis_scores   = {}; dea.topsis_ranks     = {}
        dea.kmeans_clusters = {}; dea.kmeans_labels    = {}
        dea.quadrants       = {}; dea.trevpar          = {}; dea.goppam           = {}
        dea.capital_metrics = {}
        dea._topsis_weights = topsis_weights
        dea._run_analysis()
        dea._compute_quadrants(bcc_threshold=bcc_threshold, scale_threshold=scale_threshold)
        board = dea.generate_board_report(avg_salary=avg_salary, revpar_value=revpar_value)
        st.session_state['dea']   = dea
        st.session_state['board'] = board

    # ── Multi-Module DEA (v3.2) ─────────────────────────────────────────────
    _active = st.session_state.get("active_modules_mm", [])
    if _active:
        with st.spinner(f"Calcul Multi-Module DEA ({len(_active)} modules actifs)…"):
            _overrides = st.session_state.get("variable_overrides_mm", {})
            _module_results = run_multi_module(
                df=_df_mm,
                dmu_col="hotel_name",
                active_modules=_active,
                variable_overrides=_overrides if _overrides else None,
                proxy_cols=_proxy_cols,
            )
            st.session_state["module_results"] = _module_results
        _ok = sum(1 for r in _module_results.values() if not r.error and not r.scores.empty)
        st.success(f"✅ Analyse terminée ! ({_ok}/{len(_active)} modules multi-dim OK)")
    else:
        st.success("✅ Analyse terminée !")

        # Signal Pastor (1996) si translation appliquée
        if getattr(dea, 'translation_applied', {}):
            _trans = dea.translation_applied
            _detail = ', '.join(f'{k} (décalage +{v})' for k, v in _trans.items())
            st.info(
                f"ℹ️ **Translation invariance appliquée — Pastor (1996)** : {_detail}. "
                "Valeurs non strictement positives détectées et corrigées automatiquement. "
                "BCC-VRS est translation-invariant pour les outputs — scores valides. "
                "Scores CCR calculés sur données originales (CCR non translation-invariant)."
            )

if 'dea' not in st.session_state:
    st.info("👆 Cliquez sur **LANCER L'ANALYSE** pour démarrer.")
    st.stop()

dea: HotelDEAAnalyzer = st.session_state['dea']
board: pd.DataFrame   = st.session_state['board']

_v31_attrs = ['has_surface', 'has_capex', 'has_gop', 'has_flow', 'capital_metrics']
if any(not hasattr(dea, attr) for attr in _v31_attrs):
    del st.session_state['dea']
    del st.session_state['board']
    st.warning("⚠️ Mise à jour détectée — veuillez **relancer l'analyse** pour actualiser.")
    st.stop()

# ─────────────────────────────────────────────
#  KPIs globaux
# ─────────────────────────────────────────────
avg_bcc      = np.mean(list(dea.bcc_scores.values()))
n_efficient  = sum(1 for s in dea.bcc_scores.values() if s >= 0.999)
n_critical   = sum(1 for s in dea.bcc_scores.values() if s < 0.85)
avg_scale    = np.mean(list(dea.scale_efficiency.values()))

best_topsis = min(dea.topsis_ranks, key=dea.topsis_ranks.get)

# ─────────────────────────────────────────────
#  11 ONGLETS
# ─────────────────────────────────────────────
tab_board, tab_kpi, tab_topsis, tab_quad, tab_slacks, tab_fiche, tab_meta, tab_malm, tab_capital, tab_budget, tab_synth = st.tabs([
    "📋 Rapport Board",
    "📈 Dashboard KPIs",
    "🏆 Classement TOPSIS",
    "📐 Quadrants & Échelle",
    "🔥 Slacks & Gaspillages",
    "🔍 Fiche Actif",
    "🌐 Metafrontière",
    "📈 Malmquist & Dynamique Temporelle",
    "💰 Capital & Flow Through",
    "📁 Variance Budget",
    "📥 Synthèse & Rapport Final",
])

# ══════════════════════════════════════════════
# TAB 1 — RAPPORT BOARD
# ══════════════════════════════════════════════
with tab_board:

    # ── Alerte ratio DMUs/variables ──────────────────────────────────────────
    if getattr(dea, '_dmu_ratio_warning', False):
        st.warning(f"⚠️ **Ratio DMUs/variables insuffisant** : {dea._dmu_ratio_info} — Réf. : Raab & Lichty (2002).")

    # ── Header contextuel ────────────────────────────────────────────────────
    _orient_lbl = "📥 Input-Oriented" if getattr(dea, 'orientation', 'input') == 'input' else "📤 Output-Oriented"
    st.markdown(
        f"<div style='background:#1a3a5c;color:white;padding:10px 16px;border-radius:8px;margin-bottom:12px;'>"
        f"<b>DEA-H v4 — Rapport Comité d'Investissement</b> &nbsp;·&nbsp; "
        f"{dea.n} hôtels analysés &nbsp;·&nbsp; {_orient_lbl} &nbsp;·&nbsp; "
        f"{datetime.now().strftime('%d/%m/%Y')}</div>",
        unsafe_allow_html=True,
    )

    # ── KPI Cards ────────────────────────────────────────────────────────────
    _avg_bcc   = np.mean(list(dea.bcc_scores.values()))
    _n_eff     = sum(1 for s in dea.bcc_scores.values() if s >= 0.999)
    _n_crit    = sum(1 for s in dea.bcc_scores.values() if s < 0.85)
    _avg_scale = np.mean(list(dea.scale_efficiency.values()))
    _best_tp   = min(dea.topsis_ranks, key=dea.topsis_ranks.get)
    _kc1, _kc2, _kc3, _kc4, _kc5 = st.columns(5)
    _kc1.metric("📊 BCC moyen",        f"{_avg_bcc:.1%}", delta=f"{_avg_bcc-0.85:+.1%} vs seuil 85%")
    _kc2.metric("🏆 Hôtels efficaces", f"{_n_eff}/{dea.n}", delta=f"{_n_eff/dea.n:.0%}", delta_color="off")
    _kc3.metric("⚙️ Scale Eff. moy.", f"{_avg_scale:.1%}")
    _kc4.metric("🔴 Critiques <85%",  _n_crit, delta=f"{_n_crit/dea.n:.0%}", delta_color="inverse")
    _kc5.metric("🥇 Leader TOPSIS",   (_best_tp[:18] if len(_best_tp) > 18 else _best_tp))

    # ── Upside financier total ───────────────────────────────────────────────
    if not dea.has_chambres:
        st.info(
            "💼 Upside ETP/RevPAR/GOP en € — non calculable : le CA se reconstruit "
            "par RevPAR × chambres × jours, aucun repli fiable sans un vrai nombre "
            "de chambres. Fournissez-le via l'enrichissement pour débloquer ces KPIs."
        )
    else:
        _up_fte = _up_rev = _up_gop = _ca_tot = 0.0
        for _h in dea.hotels:
            _lits  = float(dea.df.loc[_h, 'nb_chambres'])
            _rvp   = float(dea.df.loc[_h, 'revpar'])
            _se    = dea.slacks.get(_h,{}).get('inputs',{}).get('nb_employes', 0)
            _sr    = dea.slacks.get(_h,{}).get('outputs',{}).get('revpar', 0)
            _up_fte += _se * avg_salary / 1000
            # CA = RevPAR × lits × jours. Ne PAS multiplier par le taux
            # d'occupation : RevPAR = ADR × OCC l'intègre déjà (USALI / Kimes 1989).
            _ca_tot += _rvp * _lits * jours_exploit * revpar_value / 1_000_000
            _up_rev += _sr  * _lits * jours_exploit * revpar_value / 1_000_000
            _up_gop += _sr  * _lits * jours_exploit * revpar_value * ft_pct / 1_000_000

        _pct_rev = (_up_rev / _ca_tot * 100) if _ca_tot > 0 else 0

        st.markdown("---")
        _uc1, _uc2, _uc3 = st.columns(3)
        _uc1.metric("💼 Upside ETP total", f"{_up_fte:,.0f} k€/an",
                    help="Réduction de masse salariale si alignement sur la frontière DEA. "
                         f"Hypothèse : {avg_salary:,.0f} € chargés par ETP.".replace(",", " "))
        _uc2.metric("🏨 Upside RevPAR total", f"{_up_rev:,.1f} M€/an".replace(",", " "),
                    delta=f"{_pct_rev:.0f}% du CA", delta_color="off",
                    help=f"Revenu additionnel théorique si tous les actifs atteignaient leur "
                         f"RevPAR cible. CA actuel du portefeuille : {_ca_tot:,.0f} M€/an.".replace(",", " "))
        _uc3.metric("💰 Upside GOP /FT estimé", f"{_up_gop:,.1f} M€/an".replace(",", " "),
                    help=f"Upside RevPAR converti au Flow Through de {ft_pct:.0%}.")

        # ── Contrôle de plausibilité ────────────────────────────────────────────
        if _pct_rev > 40:
            st.warning(
                f"⚠️ **Upside de {_pct_rev:.0f}% du chiffre d'affaires — chiffre à ne pas "
                "communiquer tel quel.** Les slacks d'output mesurent la distance à la "
                "frontière *toutes choses égales par ailleurs* : c'est un plafond "
                "mathématique, pas un potentiel commercial. Un écart de cette ampleur "
                "signale généralement un modèle sous-spécifié — trop peu de variables pour "
                "le nombre d'actifs, ou un compset hétérogène. "
                + ("**Vous êtes en mode réduit** : repassez en mode Standard si votre "
                   "fichier contient les six variables, ou consultez la Métafrontière pour "
                   "corriger l'hétérogénéité entre segments."
                   if dea_mode != 'standard' else
                   "Vérifiez l'homogénéité du compset via la Métafrontière.")
            )
        elif _pct_rev > 0:
            st.caption(
                f"Upside RevPAR = {_pct_rev:.0f}% du CA actuel ({_ca_tot:,.0f} M€/an). "
                "Il s'agit d'un plafond théorique supposant une convergence intégrale vers "
                "les pratiques des pairs — à pondérer par un facteur de réalisation de 30 à "
                "50 % pour un plan à 24 mois.".replace(",", " ")
            )

    # ── Verdict stratégique automatique ─────────────────────────────────────
    st.markdown("---")
    _qc = {}
    for _h in dea.hotels:
        _q = dea.quadrants.get(_h, 'Q4'); _qc[_q] = _qc.get(_q, 0) + 1
    _q1, _q2, _q3, _q4 = _qc.get('Q1',0), _qc.get('Q2',0), _qc.get('Q3',0), _qc.get('Q4',0)
    if _avg_bcc >= 0.90 and _q1 >= dea.n * 0.6:
        _vrd = "✅ Portefeuille mature — efficience élevée. Stratégie de rétention et benchmarking opérationnel."
        _vc  = "#1e8449"
    elif _q3 >= dea.n * 0.4:
        _vrd = "🧠 Problème de gestion dominant (Q3) — actifs bien dimensionnés mais sous-exploités. Plan opérationnel prioritaire."
        _vc  = "#e67e22"
    elif _q4 >= dea.n * 0.3:
        _vrd = "🔴 Portefeuille sous pression — actifs en Q4 (double inefficience). Arbitrage et restructuration recommandés."
        _vc  = "#c0392b"
    elif _q2 >= dea.n * 0.4:
        _vrd = "⚙️ Problème d'échelle structurel (Q2) — gestion saine, taille inadaptée. Croissance ou cession sélective."
        _vc  = "#2e6da4"
    else:
        _vrd = f"📊 Portefeuille mixte ({_q1} Q1 · {_q2} Q2 · {_q3} Q3 · {_q4} Q4) — approche différenciée par quadrant."
        _vc  = "#555555"
    st.markdown(
        f"<div style='background:{_vc}18;border-left:4px solid {_vc};"
        f"padding:10px 16px;border-radius:0 6px 6px 0;margin:4px 0 12px 0;'>"
        f"<b>Verdict Stratégique</b> — {_vrd}</div>", unsafe_allow_html=True)

    # ── Graphiques côte à côte ───────────────────────────────────────────────
    _gc1, _gc2 = st.columns(2)
    with _gc1:
        _bvals = [dea.bcc_scores[h] for h in dea.hotels]
        _fig_h = go.Figure(go.Histogram(x=_bvals, nbinsx=10, marker_color='#2e6da4', opacity=0.8))
        _fig_h.add_vline(x=_avg_bcc, line_dash="dash", line_color="red", annotation_text=f"Moy. {_avg_bcc:.1%}")
        _fig_h.update_layout(title="Distribution BCC", xaxis=dict(title="Score BCC", tickformat='.0%'),
                             yaxis_title="Nb hôtels", height=300, paper_bgcolor='rgba(0,0,0,0)',
                             margin=dict(l=40,r=20,t=40,b=40))
        st.plotly_chart(_fig_h, use_container_width=True)
    with _gc2:
        _qcols = {'Q1':'#1e8449','Q2':'#2e6da4','Q3':'#e67e22','Q4':'#c0392b'}
        _fig_q = go.Figure()
        for _h in dea.hotels:
            _qq = dea.quadrants.get(_h,'Q4')
            _fig_q.add_trace(go.Scatter(x=[dea.scale_efficiency[_h]], y=[dea.bcc_scores[_h]],
                mode='markers', text=[_h],
                hovertemplate="<b>%{text}</b><br>Scale : %{x:.1%}<br>BCC : %{y:.1%}<extra></extra>",
                marker=dict(size=9, color=_qcols.get(_qq,'gray')), showlegend=False))
        _fig_q.add_hline(y=bcc_threshold, line_dash='dash', line_color='lightgray', line_width=1)
        _fig_q.add_vline(x=scale_threshold, line_dash='dash', line_color='lightgray', line_width=1)
        _fig_q.update_layout(title="Carte Quadrants BCC × Scale",
            xaxis=dict(title="Scale Efficiency", range=[0.5,1.05], tickformat='.0%'),
            yaxis=dict(title="Score BCC", range=[0.3,1.05], tickformat='.0%'),
            height=300, paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=40,r=20,t=40,b=40))
        st.plotly_chart(_fig_q, use_container_width=True)

    # ── Top 3 / Bottom 3 ────────────────────────────────────────────────────
    st.markdown("---")
    _tc1, _tc2 = st.columns(2)
    with _tc1:
        st.markdown('<p class="section-title">🏆 Top 3 — Leaders TOPSIS</p>', unsafe_allow_html=True)
        for _i, _h in enumerate(sorted(dea.hotels, key=lambda h: dea.topsis_ranks[h])[:3], 1):
            _bcc_h = dea.bcc_scores[_h]; _c = '#1e8449' if _bcc_h >= 0.90 else '#f39c12'
            st.markdown(
                f"<div style='padding:8px 12px;background:#f8f9fa;border-left:3px solid {_c};"
                f"border-radius:0 4px 4px 0;margin-bottom:6px;'>"
                f"<b>#{_i} {_h}</b><br/><small>BCC {_bcc_h:.1%} · TOPSIS {dea.topsis_scores[_h]:.3f} · "
                f"{QUADRANT_LABELS.get(dea.quadrants.get(_h,''),'')}</small></div>", unsafe_allow_html=True)
    with _tc2:
        st.markdown('<p class="section-title">🔴 Bottom 3 — Priorités</p>', unsafe_allow_html=True)
        for _h in sorted(dea.hotels, key=lambda h: dea.bcc_scores[h])[:3]:
            _bcc_h = dea.bcc_scores[_h]
            _uf = round(dea.slacks.get(_h,{}).get('inputs',{}).get('nb_employes',0) * avg_salary / 1000)
            st.markdown(
                f"<div style='padding:8px 12px;background:#fff5f5;border-left:3px solid #c0392b;"
                f"border-radius:0 4px 4px 0;margin-bottom:6px;'>"
                f"<b>{_h}</b><br/><small>BCC {_bcc_h:.1%} · "
                f"{QUADRANT_LABELS.get(dea.quadrants.get(_h,''),'')}"
                f"{f' · Upside ETP : {_uf} k€/an' if _uf > 0 else ''}</small></div>", unsafe_allow_html=True)

    # ── TOPSIS Composite Score Pi (Tab 10 → résumé Board) ────────────────────
    st.markdown("---")
    st.markdown('<p class="section-title">🎯 Classement Composite — Score Pi (Multi-critères)</p>', unsafe_allow_html=True)
    st.caption(
        "TOPSIS composite : BCC · RevPAR · Satisfaction · TO"
        + (" · ETP/chambre · Coût/chambre" if dea.has_chambres else "")
        + ". Pondération Shannon entropy (α=0.6). Voir Tab 10 pour le détail complet."
    )
    try:
        _bcc_s  = pd.Series(dea.bcc_scores)
        _dm_b   = pd.DataFrame({
            'DEA BCC'   : _bcc_s,
            'RevPAR'    : dea.df['revpar'],
            'Satisfaction': dea.df['satisfaction'],
            'TO%'       : dea.df['taux_occupation'],
        }, index=dea.hotels).astype(float)
        if dea.has_chambres:
            _etp_r  = dea.df['nb_employes'] / dea.df['nb_chambres']
            _cpor   = dea.df['couts_op_ex'] / dea.df['nb_chambres']
            _dm_b['ETP_inv']  = (1 / _etp_r).replace([np.inf], 0)
            _dm_b['CPOR_inv'] = (1 / _cpor).replace([np.inf], 0)
        _X = _dm_b.values; _nc = _X.shape[1]
        _Xn = np.zeros_like(_X)
        for _j in range(_nc):
            _xmn, _xmx = _X[:,_j].min(), _X[:,_j].max()
            _Xn[:,_j] = (_X[:,_j]-_xmn)/(_xmx-_xmn) if _xmx > _xmn else 0.5
        _ej = np.zeros(_nc)
        for _j in range(_nc):
            _col = _Xn[:,_j]; _s = _col.sum()
            if _s > 0:
                _p = _col / _s
                with np.errstate(divide='ignore', invalid='ignore'):
                    _lp = np.where(_p>0, np.log(_p), 0)
                _ej[_j] = -np.sum(_p*_lp)/np.log(dea.n) if dea.n > 1 else 0
        _dj = 1 - _ej; _alpha = 0.6
        _w  = (1-_alpha)*(_dj/_dj.sum() if _dj.sum()>0 else np.ones(_nc)/_nc) + _alpha*(np.ones(_nc)/_nc)
        _V  = _Xn * _w; _Vp = _V.max(axis=0); _Vm = _V.min(axis=0)
        _Sp = np.sqrt(((_V-_Vp)**2).sum(axis=1)); _Sm = np.sqrt(((_V-_Vm)**2).sum(axis=1))
        _Pi = _Sm / (_Sp + _Sm + 1e-10)
        _q25pi, _q75pi = np.percentile(_Pi, [25, 75])
        _pi_rows = [{'Rang': 0, 'Hôtel': h,
                     'Score Pi': round(_Pi[i], 4),
                     'DEA BCC': f"{dea.bcc_scores.get(h,0):.1%}",
                     'Quadrant': QUADRANT_LABELS.get(dea.quadrants.get(h,''), ''),
                     'Signal': ('🟢 Top 25%' if _Pi[i]>=_q75pi else '🔴 Bottom 25%' if _Pi[i]<_q25pi else '🟡 Médian')}
                    for i, h in enumerate(dea.hotels)]
        _pi_df = pd.DataFrame(_pi_rows).sort_values('Score Pi', ascending=False).reset_index(drop=True)
        _pi_df['Rang'] = range(1, len(_pi_df)+1)
        _pic1, _pic2 = st.columns([2, 1])
        with _pic1:
            st.dataframe(_pi_df, use_container_width=True, hide_index=True)
            st.download_button("⬇️ Exporter Score Pi (CSV)",
                data=_pi_df.to_csv(index=False, sep=';', encoding='utf-8-sig'),
                file_name=f"deah_score_pi_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv",
                key='dl_pi_board')
        with _pic2:
            _best_pi  = _pi_df.iloc[0]['Hôtel']
            _worst_pi = _pi_df.iloc[-1]['Hôtel']
            st.metric("🥇 Leader composite", _best_pi[:20])
            st.metric("Score Pi leader", f"{_pi_df.iloc[0]['Score Pi']:.4f}")
            st.metric("⚠️ Priorité composite", _worst_pi[:20])
            st.metric("Score Pi priorité", f"{_pi_df.iloc[-1]['Score Pi']:.4f}")
        # Stocker pour réutilisation Tab 10 (Pi + poids critères)
        st.session_state['score_pi_df'] = _pi_df
        st.session_state['score_pi_crits'] = ['DEA BCC','RevPAR','Satisfaction','TO%','ETP_inv','CPOR_inv']
        st.session_state['score_pi_w'] = _w.tolist()
    except Exception as _e_pi:
        st.info(f"Score Pi : données insuffisantes pour le calcul composite ({_e_pi})")

    # ── Tableau Board + Quadrants + Mahalanobis ──────────────────────────────
    st.markdown("---")
    st.markdown('<p class="section-title">📋 Tableau Consolidé — Tous les actifs</p>', unsafe_allow_html=True)
    st.dataframe(board, use_container_width=True, hide_index=True)
    st.markdown('<p class="section-title">Répartition par Quadrant</p>', unsafe_allow_html=True)
    q_summary = dea.get_quadrant_summary()
    st.dataframe(q_summary.drop(columns=['Hôtels'], errors='ignore'), use_container_width=True, hide_index=True)
    st.markdown('<p class="section-title">Détection Outliers — Distance de Mahalanobis</p>', unsafe_allow_html=True)
    st.caption("Poldrugovac et al. (2016) — D² ∼ χ²(k). Outlier si p < 0.01.")
    _mah_df = dea.detect_outliers_mahalanobis(threshold_p=0.01)
    _n_out  = _mah_df['Outlier'].sum()
    if _n_out > 0:
        st.error(f"**{_n_out} outlier(s) :** {', '.join(_mah_df[_mah_df['Outlier']]['Hôtel'].tolist())} — vérifier le compset.")
    else:
        st.success("✅ Aucun outlier (p > 0.01) — compset homogène.")
    st.dataframe(_mah_df, use_container_width=True, hide_index=True)

    st.caption("📚 *Réf. : Charnes, Cooper & Rhodes (1978) · Banker et al. (1984) · "
               "Poldrugovac et al. (2016) Mahalanobis · Hwang & Yoon (1981) TOPSIS.*")


# ── Recommandations SBM par variable ─────────────────────────────────────────
_SBM_RECO = {
    'nb_employes'       : "Optimiser ratio ETP/chambre — masse salariale au-dessus de la norme compset. Piste : réorganisation, polyvalence, externalisation.",
    'couts_op_ex'       : "Coûts opérationnels excessifs — RevPAR/OpEx sous la norme. Réviser contrats fournisseurs, énergie, maintenance.",
    'nb_chambres'           : "Capacité installée sous-productive — TO insuffisant vs taille. Repositionnement ou reconfiguration de la capacité à évaluer.",
    'revpar'            : "Levier RevPAR identifié — yield management non optimisé. Comparer ADR et TO aux benchmarks du compset.",
    'taux_occupation'   : "Taux d'occupation sous-optimal — renforcer distribution et action commerciale. Revoir mix canal et contrats corporate.",
    'satisfaction_score': "Satisfaction sous le potentiel — impact pricing power et fidélisation. Plan d'action qualité de service prioritaire.",
    'gop'               : "Marge GOP insuffisante — revoir structure de coûts ou repositionner l'offre tarifaire.",
}
_SBM_RECO_DEFAULT = "Analyser les données opérationnelles détaillées pour identifier le levier prioritaire."

def _sbm_dominant_analysis(sbm_hotel: dict, dea, hotel: str) -> dict:
    """
    Identifie le slack SBM dominant (contribution normalisée max)
    et retourne le levier + recommandation ciblée.
    """
    si = sbm_hotel.get('slacks_in',  {})
    so = sbm_hotel.get('slacks_out', {})

    # Normalisation par range — même logique que compute_sbm()
    x_range = {col: float(dea.inputs[:, i].max() - dea.inputs[:, i].min())
               for i, col in enumerate(dea.input_cols)}
    y_range = {col: float(dea.outputs[:, r].max() - dea.outputs[:, r].min())
               for r, col in enumerate(dea.output_cols)}

    contributions = {}
    for col, val in si.items():
        if val is not None and x_range.get(col, 0) > 1e-9:
            contributions[(col, 'Input ↓')] = val / x_range[col]
    for col, val in so.items():
        if val is not None and y_range.get(col, 0) > 1e-9:
            contributions[(col, 'Output ↑')] = val / y_range[col]

    if not contributions:
        return {'col': '—', 'type': '—', 'contrib': 0, 'reco': _SBM_RECO_DEFAULT}

    (dom_col, dom_type), dom_contrib = max(contributions.items(), key=lambda x: x[1])

    # % amélioration possible
    try:
        cur_val = float(dea.df.loc[hotel, dom_col]) if dom_col in dea.df.columns else None
        raw_slack = (si if dom_type == 'Input ↓' else so).get(dom_col, 0) or 0
        pct = f"{raw_slack / abs(cur_val):.1%}" if cur_val and abs(cur_val) > 1e-6 else "N/A"
    except Exception:
        pct = "N/A"

    return {
        'col'    : dom_col,
        'type'   : dom_type,
        'contrib': round(dom_contrib, 4),
        'pct'    : pct,
        'reco'   : _SBM_RECO.get(dom_col, _SBM_RECO_DEFAULT),
    }

# TAB 2 — DASHBOARD KPIs
# ══════════════════════════════════════════════
with tab_kpi:
    st.markdown('<p class="section-title">Distribution des scores d\'efficacité</p>', unsafe_allow_html=True)

    # ── Signal benchmark % hôtels efficients (Poldrugovac et al. 2016 ; papier Via@ 2013) ──
    _n_eff_t2  = sum(1 for s in dea.bcc_scores.values() if s >= 0.999)
    _pct_eff   = _n_eff_t2 / dea.n

    if _pct_eff < 0.12:
        st.warning(
            f"⚠️ **Seulement {_n_eff_t2}/{dea.n} hôtels ({_pct_eff:.0%}) sur la frontière** — "
            "taux inhabituellement bas. Deux causes possibles : inefficiences réelles importantes, "
            "ou compset trop hétérogène. Vérifier la cohérence du compset (Tab 1 — Mahalanobis)."
        )
    elif _pct_eff > 0.50:
        st.warning(
            f"⚠️ **{_n_eff_t2}/{dea.n} hôtels ({_pct_eff:.0%}) à BCC = 1** — taux inhabituellement élevé. "
            "Le compset est probablement trop petit ou trop homogène pour être discriminant. "
            f"Rappel : minimum {3 * 6} hôtels pour 6 variables (règle 3×)."
        )
    else:
        st.success(
            f"✅ **{_n_eff_t2}/{dea.n} hôtels ({_pct_eff:.0%}) sur la frontière** — "
            "taux conforme aux benchmarks empiriques multi-études (15–25%). "
            "Le compset est suffisamment discriminant."
        )

    hotels_list = dea.hotels
    bcc_vals    = [dea.bcc_scores[h] for h in hotels_list]
    col1, col2 = st.columns(2)
    with col1:
        fig_hist = go.Figure(go.Histogram(
            x=bcc_vals, nbinsx=10, marker_color='#2e6da4', opacity=0.8,
            xbins=dict(start=0, end=1.05, size=0.1),
        ))
        fig_hist.update_layout(
            title="Distribution scores BCC", xaxis_title="Score BCC", yaxis_title="Nombre d'hôtels",
            bargap=0.05, height=380, paper_bgcolor='rgba(0,0,0,0)',
        )
        fig_hist.add_vline(x=avg_bcc, line_dash="dash", line_color="red",
                           annotation_text=f"Moy. {avg_bcc:.1%}")
        st.plotly_chart(fig_hist, use_container_width=True)
    with col2:
        sorted_hotels = sorted(hotels_list, key=lambda h: dea.bcc_scores[h])
        # NB : ne pas nommer cette variable `colors` — elle écraserait
        # reportlab.lib.colors au niveau module et casserait le rapport PDF
        _bar_colors_bcc = ['#27ae60' if dea.bcc_scores[h] >= 0.95
                           else '#f39c12' if dea.bcc_scores[h] >= 0.85 else '#e74c3c'
                           for h in sorted_hotels]
        fig_bars = go.Figure(go.Bar(
            x=[dea.bcc_scores[h] for h in sorted_hotels], y=sorted_hotels,
            orientation='h', marker_color=_bar_colors_bcc, opacity=0.85,
        ))
        fig_bars.update_layout(
            title="Scores BCC par hôtel", xaxis=dict(range=[0, 1.05], tickformat='.0%'),
            height=420, paper_bgcolor='rgba(0,0,0,0)',
        )
        fig_bars.add_vline(x=0.90, line_dash="dot", line_color="#c0392b", annotation_text="Seuil 90%")
        st.plotly_chart(fig_bars, use_container_width=True)

    st.markdown('<p class="section-title">BCC vs CCR — Efficacité Pure vs Globale</p>', unsafe_allow_html=True)
    fig_scatter = go.Figure()
    for h in hotels_list:
        fig_scatter.add_trace(go.Scatter(
            x=[dea.ccr_scores[h]], y=[dea.bcc_scores[h]], mode='markers',
            text=[h], hovertemplate="<b>%{text}</b><br>CCR : %{x:.1%}<br>BCC : %{y:.1%}<extra></extra>",
            marker=dict(size=10, color='#2e6da4'), name=h, showlegend=False,
        ))
    fig_scatter.add_shape(type='line', x0=0, y0=0, x1=1, y1=1, line=dict(dash='dash', color='gray'))
    fig_scatter.update_layout(
        title="BCC vs CCR (points au-dessus de la diagonale = problème d'échelle)",
        xaxis=dict(title="Score CCR", range=[0, 1.05], tickformat='.0%'),
        yaxis=dict(title="Score BCC", range=[0, 1.05], tickformat='.0%'),
        height=420, paper_bgcolor='rgba(0,0,0,0)',
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # ── SBM — Score non-radial (Tone 2001) ───────────────────────────────────
    st.markdown('<p class="section-title">📐 SBM — Score non-radial (Tone 2001)</p>', unsafe_allow_html=True)
    st.caption(
        "SBM (Slack-Based Measure) mesure l'inefficience directement via les slacks inputs ET outputs. "
        "Contrairement à BCC/CCR (modèles radiaux), SBM gère nativement les GOP négatifs, "
        "RevPAR nuls et variables d'environnement non-positives — idéal pour les actifs en difficulté "
        "ou en repositionnement. ⚠️ Score NON comparable au score BCC. "
        "Réf. : Tone (2001) EJOR ; Tone & Tsutsui (2010)."
    )
    if st.button("🔄 Calculer SBM (tous les hôtels)", key="sbm_btn_t2"):
        with st.spinner("Calcul SBM en cours…"):
            try:
                _sbm_res = dea.compute_sbm()
                st.session_state["sbm_results"] = _sbm_res
                st.success(f"✅ SBM calculé — {len(_sbm_res)} DMUs")
            except Exception as _e:
                st.error(f"Erreur SBM : {_e}")

    if st.session_state.get("sbm_results"):
        _sbm = st.session_state["sbm_results"]
        _sbm_rows = []
        for h in dea.hotels:
            _r = _sbm.get(h, {})
            _score = _r.get('score')
            _bcc   = dea.bcc_scores.get(h, 0)
            _delta = round(_score - _bcc, 4) if _score is not None else None
            _label = ('✅ SBM-efficient' if _score is not None and _score >= 0.999
                      else '🟡 Inefficience modérée' if _score is not None and _score >= 0.80
                      else '🔴 Inefficience forte' if _score is not None else '⚠️ Non résolu')
            _dom = _sbm_dominant_analysis(_r, dea, h) if _r.get('feasible') else {}
            _sbm_rows.append({
                'Hôtel'          : h,
                'BCC'            : f"{_bcc:.1%}",
                'SBM ρ*'         : f"{_score:.4f}" if _score is not None else '—',
                'Δ SBM−BCC'      : f"{_delta:+.4f}" if _delta is not None else '—',
                'Signal'         : _label,
                'Levier dominant': _dom.get('col', '—'),
                'Type'           : _dom.get('type', '—'),
                '% amélioration' : _dom.get('pct', '—'),
            })
        st.dataframe(pd.DataFrame(_sbm_rows), use_container_width=True, hide_index=True)
        st.info(
            "**SBM < BCC** : l'hôtel présente des inefficiences sur des dimensions "
            "non capturées par le modèle radial BCC — vérifier les slacks détaillés dans "
            "la **Fiche Actif (Tab 7)**. "
            "**SBM ≈ BCC** : les deux modèles convergent — résultat robuste."
        )

# ══════════════════════════════════════════════
# TAB 3 — CLASSEMENT TOPSIS
# ══════════════════════════════════════════════
with tab_topsis:
    st.markdown('<p class="section-title">🏆 Classement TOPSIS — Multi-critères pondérés</p>', unsafe_allow_html=True)
    st.caption("Critères : BCC 35% · Scale Efficiency 25% · RevPAR 25% · Taux d'Occupation 15%")
    ranking = dea.get_topsis_ranking()
    st.dataframe(ranking, use_container_width=True, hide_index=True)

    fig_topsis = go.Figure(go.Bar(
        x=ranking['Score TOPSIS'], y=ranking['Hôtel'], orientation='h',
        marker=dict(color=ranking['Score TOPSIS'], colorscale='Blues', showscale=True, colorbar=dict(title="Score")),
        text=[f"{s:.3f}" for s in ranking['Score TOPSIS']], textposition='outside',
    ))
    fig_topsis.update_layout(
        title="Classement TOPSIS (1 = meilleur)",
        xaxis=dict(title="Score TOPSIS", range=[0, 1.1]),
        height=max(350, len(dea.hotels) * 28), paper_bgcolor='rgba(0,0,0,0)',
    )
    st.plotly_chart(fig_topsis, use_container_width=True)

    st.markdown('<p class="section-title">Profil Top 3 TOPSIS</p>', unsafe_allow_html=True)
    top3 = ranking.head(3)['Hôtel'].tolist()
    categories = ['BCC', 'Scale Eff.', 'RevPAR norm.', 'TO norm.']
    revpar_max = max(dea.df['revpar']); to_max = max(dea.df['taux_occupation'])
    fig_radar = go.Figure()
    colors_radar = ['#2e6da4', '#27ae60', '#e67e22']
    for i, h in enumerate(top3):
        vals = [dea.bcc_scores[h], dea.scale_efficiency[h],
                float(dea.df.loc[h, 'revpar']) / revpar_max,
                float(dea.df.loc[h, 'taux_occupation']) / to_max]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals + [vals[0]], theta=categories + [categories[0]],
            fill='toself', name=h, line_color=colors_radar[i], opacity=0.6,
        ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(range=[0, 1])), title="Profil multi-critères — Top 3",
        height=380, paper_bgcolor='rgba(0,0,0,0)',
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # ── Super-Efficience & Cross-Efficience ──────────────────────────────────
    st.markdown("---")
    st.markdown('<p class="section-title">Super-Efficience & Cross-Efficience — Discrimination avancée</p>', unsafe_allow_html=True)

    _adv_col1, _adv_col2 = st.columns(2)

    with _adv_col1:
        st.markdown("**Super-Efficience — Andersen & Petersen (1993)**")
        st.caption(
            "Les DMUs efficients (BCC=1) reçoivent un score > 1 : "
            "ils pourraient consommer plus d'inputs tout en restant hors de la frontière "
            "construite sans eux. Permet de classer le **top portfolio**."
        )
        if st.button("Calculer Super-Efficience", key="se_btn"):
            with st.spinner("Calcul super-efficience..."):
                try:
                    _se_df = dea.compute_super_efficiency()
                    st.session_state["se_results"] = _se_df
                except Exception as _e:
                    st.error(f"Erreur : {_e}")

        if st.session_state.get("se_results") is not None:
            _se = st.session_state["se_results"]
            st.dataframe(_se, use_container_width=True, hide_index=True)

            # Bar chart super-efficience
            _se_num = _se.copy()
            _se_num["_se_val"] = pd.to_numeric(_se_num["Super-Efficience"], errors="coerce")
            # Couleurs par seuil (colorscale normalisé 0→1 obligatoire en Plotly)
            _se_colors = [
                "#1a8a4a" if v >= 1.5 else
                "#27ae60" if v >= 1.1 else
                "#f1c40f" if v >= 1.0 else
                "#f39c12" if v >= 0.85 else "#e74c3c"
                for v in _se_num["_se_val"]
            ]
            fig_se = go.Figure(go.Bar(
                x=_se_num["_se_val"], y=_se_num["Hôtel"], orientation="h",
                marker_color=_se_colors, opacity=0.85,
                text=[f"{v:.3f}" for v in _se_num["_se_val"]],
                textposition="outside",
            ))
            fig_se.add_vline(x=1.0, line_dash="dash", line_color="#27ae60",
                             annotation_text="Frontière BCC", annotation_font_color="#27ae60")
            fig_se.update_layout(
                title="Super-Efficience (> 1.0 = au-delà de la frontière)",
                xaxis=dict(range=[0.2, max(_se_num["_se_val"])*1.15]),
                height=max(350, dea.n*28), paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_se, use_container_width=True)

    with _adv_col2:
        st.markdown("**Cross-Efficience — Doyle & Green (1994)**")
        st.caption(
            "Chaque DMU est évalué par les poids optimaux de TOUS ses pairs. "
            "Élimine le choix arbitraire des poids. "
            "Un score élevé signifie une efficience **robuste** indépendamment du système de poids choisi."
        )
        if st.button("Calculer Cross-Efficience", key="ce_btn"):
            with st.spinner("Calcul cross-efficience (forme multiplicatrice BCC)..."):
                try:
                    _ce_df, _ce_matrix = dea.compute_cross_efficiency()
                    st.session_state["ce_results"] = (_ce_df, _ce_matrix)
                except Exception as _e:
                    st.error(f"Erreur : {_e}")

        if st.session_state.get("ce_results") is not None:
            _ce_df, _ce_matrix = st.session_state["ce_results"]
            st.dataframe(_ce_df[["Rang CE","Hôtel","BCC","Cross-Efficience","Δ BCC-CE","Lecture"]],
                         use_container_width=True, hide_index=True)

            # Scatter BCC vs Cross-Efficience
            _ce_df_num = _ce_df.copy()
            _ce_df_num["_bcc_f"] = _ce_df_num["BCC"].str.replace("%","").astype(float)/100
            _ce_df_num["_ce_f"]  = _ce_df_num["Cross-Efficience"].astype(float)

            fig_ce = go.Figure()
            for _, r in _ce_df_num.iterrows():
                _col = "#27ae60" if r["_bcc_f"]>=0.95 and r["_ce_f"]>=0.85 else                        "#e74c3c" if r["_bcc_f"]>=0.95 and r["_ce_f"]<0.75 else                        "#3498db"
                fig_ce.add_trace(go.Scatter(
                    x=[r["_bcc_f"]], y=[r["_ce_f"]],
                    mode="markers", text=[r["Hôtel"]],
                    marker=dict(size=12, color=_col), showlegend=False,
                    hovertemplate=f"<b>{r['Hôtel']}</b><br>BCC: {r['_bcc_f']:.1%}<br>CE: {r['_ce_f']:.3f}<extra></extra>",
                ))
            # Diagonale
            fig_ce.add_shape(type="line", x0=0.3, y0=0.3, x1=1.0, y1=1.0,
                             line=dict(dash="dot", color="gray", width=1))
            # Zones
            fig_ce.add_annotation(x=0.98, y=0.95, text="Robuste ✅",
                                   showarrow=False, font=dict(size=9, color="#27ae60"))
            fig_ce.add_annotation(x=0.98, y=0.65, text="Fragile ⚠️",
                                   showarrow=False, font=dict(size=9, color="#e74c3c"))
            fig_ce.update_layout(
                title="BCC vs Cross-Efficience — Robustesse des scores",
                xaxis=dict(title="Score BCC", tickformat=".0%", range=[0.3,1.05]),
                yaxis=dict(title="Cross-Efficience", range=[0.2,1.05]),
                height=360, paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_ce, use_container_width=True)

            # Heatmap de la matrice CE
            with st.expander("🗺️ Matrice cross-efficience complète", expanded=False):
                st.caption("Ligne j = poids du DMU j évaluant les colonnes k. Diagonale = auto-évaluation (score BCC).")
                fig_hm = go.Figure(go.Heatmap(
                    z=_ce_matrix,
                    x=dea.hotels,
                    y=dea.hotels,
                    colorscale="RdYlGn", zmin=0, zmax=1,
                    text=[[f"{v:.2f}" for v in row] for row in _ce_matrix],
                    texttemplate="%{text}", textfont=dict(size=8),
                    colorbar=dict(title="CE Score"),
                ))
                fig_hm.update_layout(
                    height=max(300, dea.n*30+80),
                    margin=dict(l=120, r=20, t=30, b=80),
                    paper_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(tickangle=-45),
                )
                st.plotly_chart(fig_hm, use_container_width=True)

    # Tableau comparatif consolidé
    if st.session_state.get("se_results") is not None and st.session_state.get("ce_results") is not None:
        st.markdown("---")
        st.markdown('<p class="section-title">Tableau de Décision Consolidé — BCC · Super-Eff. · Cross-Eff. · TOPSIS</p>', unsafe_allow_html=True)
        st.caption("Doyle & Green (1994) · Andersen & Petersen (1993) — Un actif robuste est performant sur les 4 dimensions.")
        with st.expander("ℹ️ Limites de la cross-efficience — Assurance Regions (Thompson et al., 1990)"):
            st.markdown(
                """
La cross-efficience traite les poids arbitraires **ex-post** (après optimisation).
Les **Assurance Regions** (Thompson et al., 1990) les contraignent **ex-ante** dans le LP :

> Exemple : `0.10 ≤ u_satisfaction / u_RevPAR ≤ 0.50`

Cela interdit d'annuler la satisfaction pour atteindre BCC = 1 artificiellement.

**Statut DEA-H :** non implémenté — évolution identifiée. Garde-fou actuel : ce tableau.
                """
            )

        _se_d = st.session_state.get("se_results")
        _ce_d = st.session_state.get("ce_results")
        if _se_d is None or _ce_d is None:
            st.info("Calculez SE et CE d'abord.")
        else:
          _se_d = _se_d.set_index("Hôtel")
          _ce_d = _ce_d[0].set_index("Hôtel")
        _consol = []
        for h in dea.hotels:
            _consol.append({
                "Hôtel"            : h,
                "BCC"              : f"{dea.bcc_scores[h]:.1%}",
                "Super-Eff."       : _se_d.loc[h, "Super-Efficience"] if h in _se_d.index else "—",
                "Cross-Eff."       : _ce_d.loc[h, "Cross-Efficience"] if h in _ce_d.index else "—",
                "Rang TOPSIS"      : f"#{dea.topsis_ranks.get(h,'—')}",
                "Rang CE"          : f"#{int(_ce_d.loc[h, 'Rang CE'])}" if h in _ce_d.index else "—",
                "Rang SE"          : f"#{int(_se_d.loc[h, 'Rang SE'])}" if h in _se_d.index else "—",
                "Verdict"          : (
                    "🟢 Leader confirmé"
                    if dea.bcc_scores[h] >= 0.90
                    and (float(_ce_d.loc[h,"Cross-Efficience"]) if h in _ce_d.index else 0) >= 0.80
                    else "🟡 Efficient fragile"
                    if dea.bcc_scores[h] >= 0.90
                    else "🔴 Plan d'action requis"
                ),
            })
        st.dataframe(pd.DataFrame(_consol), use_container_width=True, hide_index=True)

        _csv_consol = pd.DataFrame(_consol).to_csv(index=False, sep=";", decimal=",")
        st.download_button("⬇️ Exporter tableau consolidé (CSV)", data=_csv_consol.encode("utf-8-sig"),
                           file_name="deah_ranking_consolide.csv", mime="text/csv")

    st.markdown("---")
    st.markdown('<p class="section-title">Profil Compset - Grille de coherence (Exhibit 8)</p>', unsafe_allow_html=True)
    st.caption('Valider que les DMUs sont comparables avant interpretation DEA. RGI doit rester entre 80 et 130% pour valider le compset.')
    st.info(
        "**Localisation** pré-remplie à partir du nom de l'hôtel (aéroport pour "
        "CDG/Orly/Nice Aéroport, périphérie pour Labège/Saint-Grégoire — vrais lieux "
        "identifiables, base géographique réelle) — à corriger si besoin. "
        "**Affiliation** (Franchise / Contrat de gestion / Owner-operated — terme "
        "standard du secteur) pré-remplie en alternance simple (aucune base réelle, "
        "aucun lien avec la performance des hôtels — juste une répartition arbitraire "
        "pour que la variable soit testable) : ne pas présenter comme une vraie "
        "donnée sans l'avoir remplacée par les vraies structures Atream."
    )

    def _guess_localisation(hotel_name: str) -> str:
        _n = hotel_name.lower()
        if 'roissy' in _n or 'orly' in _n or 'aéroport' in _n or 'aeroport' in _n:
            return 'Airport'
        if 'labège' in _n or 'labege' in _n or 'saint-grégoire' in _n or 'saint-gregoire' in _n:
            return 'Suburban'
        return 'Centre-ville'

    _compset_init = pd.DataFrame({
        'Hôtel'           : dea.hotels,
        'Annee ouv.'      : [0]*dea.n,
        'Dern. renov.'    : [0]*dea.n,
        'Classement (e)'  : [3]*dea.n,
        # Alternance simple par position dans la liste — pas de lien avec le score
        # BCC ou tout autre résultat du modèle (endogénéité). Purement arbitraire,
        # affiché comme tel, à remplacer par les vraies données Atream.
        # Colonne unique pour la structure de gestion — fusionne l'ancienne
        # "Gestion" (3rd party/Brand-managed/Owner-operated) dans "Affiliation",
        # le terme standard du secteur (Franchise / Contrat de gestion / Owner-
        # operated), pour ne garder qu'une seule variable testable au lieu de
        # deux taxonomies qui se recouvraient.
        # Alternance 3 catégories par position — pas de lien avec le score BCC.
        # Franchise = franchiseur et exploitant liés/mêmes (ex. Ibis + AccorInvest
        # pour Atream) · Mgmt contract = exploitant opérateur professionnel,
        # indépendamment de la marque · Owner-operated = propriétaire seul, ni
        # franchise ni contrat de gestion.
        'Affiliation'     : [['Franchise', 'Mgmt contract', 'Owner-operated'][i % 3] for i in range(dea.n)],
        'Meeting (m2)'    : [0]*dea.n,
        'Localisation'    : [_guess_localisation(h) for h in dea.hotels],
    }).set_index('Hôtel')
    if dea.has_chambres:
        _compset_init.insert(0, 'Capacité (chambres)',
                              [int(dea.df.loc[h, 'nb_chambres']) for h in dea.hotels])
    if 'compset_profile' not in st.session_state or set(st.session_state.get('compset_profile', pd.DataFrame()).index) != set(dea.hotels):
        st.session_state['compset_profile'] = _compset_init

    _col_cfg = {
        'Annee ouv.'    : st.column_config.NumberColumn('Annee ouv.', min_value=1800, max_value=2030, format='%d'),
        'Dern. renov.'  : st.column_config.NumberColumn('Dern. renov.', min_value=1800, max_value=2030, format='%d'),
        'Classement (e)': st.column_config.SelectboxColumn('Classement', options=[1,2,3,4,5]),
        'Affiliation'   : st.column_config.SelectboxColumn(
            'Affiliation', options=['Franchise', 'Mgmt contract', 'Owner-operated'],
            help="Franchise = franchiseur et exploitant liés ou identiques (ex. marque Ibis "
                 "+ opérateur AccorInvest) · Mgmt contract = exploité par un opérateur "
                 "professionnel, la marque n'est pas le sujet · Owner-operated = "
                 "propriétaire seul, ni franchise ni contrat de gestion."),
        'Localisation'  : st.column_config.SelectboxColumn('Localisation', options=['Centre-ville','Suburban','Airport','Resort','Route']),
        'Meeting (m2)'  : st.column_config.NumberColumn('Meeting m2', min_value=0, format='%d'),
    }
    if dea.has_chambres:
        _col_cfg['Capacité (chambres)'] = st.column_config.NumberColumn(
            'Chambres', min_value=0, format='%d',
            help="Nombre de chambres réel, fourni via l'enrichissement.")
    _cs = st.data_editor(
        st.session_state['compset_profile'], use_container_width=True,
        column_config=_col_cfg, key='compset_editor',
    )
    st.session_state['compset_profile'] = _cs
    if dea.has_chambres:
        _ch_vals = _cs['Capacité (chambres)'].values
        if _ch_vals.max() > 0 and _ch_vals.min() > 0:
            _ratio = _ch_vals.max() / _ch_vals.min()
            if _ratio > 3: st.warning(f'Ratio taille max/min = {_ratio:.1f}x - compset heterogene. Segmenter ou affiner.')
            else: st.success(f'Homogeneite capacite OK : ratio max/min = {_ratio:.1f}x')
    else:
        st.caption("Homogénéité de taille non vérifiable — nombre de chambres réel non fourni.")

    st.markdown("---")
    kpis_def = {
        'RevPAR (€)': (dea.df['revpar'], 'benefit'), 'Satisfaction': (dea.df['satisfaction'], 'benefit'),
        'TO (%)': (dea.df['taux_occupation'], 'benefit'),
        'Score BCC': (pd.Series(dea.bcc_scores), 'benefit'),
    }
    if dea.has_chambres:
        kpis_def['ETP / chambre'] = (dea.df['nb_employes'] / dea.df['nb_chambres'], 'cost')
        kpis_def['CPOR (k€/ch)']  = (dea.df['couts_op_ex'] / dea.df['nb_chambres'], 'cost')
    else:
        kpis_def['ETP (total)']   = (dea.df['nb_employes'], 'cost')
        kpis_def['OpEx (k€)']     = (dea.df['couts_op_ex'], 'cost')
    q_stats = []
    for kpi_name, (series, direction) in kpis_def.items():
        s = series.dropna()
        q_stats.append({'KPI': kpi_name, 'Direction': '↑ max' if direction=='benefit' else '↓ min',
                        'Min': round(s.min(),2), 'P25': round(s.quantile(0.25),2),
                        'Médiane': round(s.median(),2), 'P75': round(s.quantile(0.75),2),
                        'Max': round(s.max(),2), 'Moy.': round(s.mean(),2)})
    _q_stats_df = pd.DataFrame(q_stats)
    st.dataframe(_q_stats_df, use_container_width=True, hide_index=True)
    st.download_button(
        "⬇️ Télécharger quartiles KPIs (CSV)", data=_q_stats_df.to_csv(index=False, sep=';', encoding='utf-8-sig'),
        file_name=f"deah_quartiles_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv",
        key='dl_quartiles',
    )

    selected_bm = st.selectbox("Hôtel à positionner", dea.hotels, key='bm_hotel')
    pos_rows = []
    for kpi_name, (series, direction) in kpis_def.items():
        s = series.dropna()
        val = float(series.loc[selected_bm]) if selected_bm in series.index else None
        if val is None: continue
        q25, q50, q75 = s.quantile(0.25), s.median(), s.quantile(0.75)
        if direction == 'benefit':
            pos = ('🟢 Top 25%' if val >= q75 else '🟡 Q3 (50-75%)' if val >= q50
                   else '🟠 Q2 (25-50%)' if val >= q25 else '🔴 Bottom 25%')
        else:
            pos = ('🟢 Top 25%' if val <= q25 else '🟡 Q3 (50-75%)' if val <= q50
                   else '🟠 Q2 (25-50%)' if val <= q75 else '🔴 Bottom 25%')
        pos_rows.append({'KPI': kpi_name, 'Valeur': round(val,2), 'P25': round(q25,2),
                         'Médiane': round(q50,2), 'P75': round(q75,2), 'Position': pos})
    _pos_df = pd.DataFrame(pos_rows)
    st.dataframe(_pos_df, use_container_width=True, hide_index=True)
    st.download_button(
        f"⬇️ Télécharger positionnement {selected_bm} (CSV)",
        data=_pos_df.to_csv(index=False, sep=';', encoding='utf-8-sig'),
        file_name=f"deah_position_{selected_bm}_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv",
        key='dl_position',
    )

    st.markdown("---")
    st.markdown('<p class="section-title">🎯 TOPSIS Composite — Score Pi</p>', unsafe_allow_html=True)
    st.caption("Pondération hybride Shannon entropy (α=0.6) — calculé en Tab 1 (Rapport Board). Voir Tab 1 pour l'export CSV complet.")

    # Réutiliser le Score Pi calculé en Tab 1 si disponible
    _pi_cached = st.session_state.get('score_pi_df', None)
    if _pi_cached is not None:
        Pi = np.array([_pi_cached.loc[_pi_cached['Hôtel']==h, 'Score Pi'].values[0]
                       if h in _pi_cached['Hôtel'].values else 0.0
                       for h in dea.hotels])
    else:
        # Recalcul si Tab 1 non encore visité
        bcc_s = pd.Series(dea.bcc_scores)
        dm = pd.DataFrame({
            'DEA BCC': bcc_s, 'RevPAR': dea.df['revpar'], 'Satisfaction': dea.df['satisfaction'],
            'TO%': dea.df['taux_occupation'],
        }, index=dea.hotels).astype(float)
        if dea.has_chambres:
            etp_r = dea.df['nb_employes'] / dea.df['nb_chambres']
            cpor  = dea.df['couts_op_ex'] / dea.df['nb_chambres']
            dm['ETP_inv']  = (1 / etp_r).replace([np.inf], 0)
            dm['CPOR_inv'] = (1 / cpor).replace([np.inf], 0)
        cap_inp_tab10 = st.session_state.get('capital_input', None)
        if (cap_inp_tab10 is not None and 'gop (k€)' in cap_inp_tab10.columns
                and dea.has_trevpar):
            rev_est = dea.df['total_revenue']
            gop_vals = cap_inp_tab10['gop (k€)'].astype(float) * 1000
            dm['GOP Margin'] = (gop_vals / rev_est.replace(0, np.nan)).fillna(0)
        crit_labels = list(dm.columns); X = dm.values.astype(float); n_dmu, n_crit = X.shape
        Xn = np.zeros_like(X)
        for j in range(n_crit):
            xmin, xmax = X[:,j].min(), X[:,j].max()
            Xn[:,j] = (X[:,j]-xmin)/(xmax-xmin) if xmax > xmin else 0.5
        ej = np.zeros(n_crit)
        for j in range(n_crit):
            col = Xn[:,j]; s = col.sum()
            if s > 0:
                p = col / s
                with np.errstate(divide='ignore', invalid='ignore'):
                    lp = np.where(p>0, np.log(p), 0)
                ej[j] = -np.sum(p*lp) / np.log(n_dmu) if n_dmu > 1 else 0
        alpha = 0.6; dj = 1 - ej
        w = (1-alpha)*(dj/dj.sum() if dj.sum()>0 else np.ones(n_crit)/n_crit) + alpha*(np.ones(n_crit)/n_crit)
        V = Xn * w; V_plus = V.max(axis=0); V_minus = V.min(axis=0)
        S_plus = np.sqrt(((V-V_plus)**2).sum(axis=1)); S_minus = np.sqrt(((V-V_minus)**2).sum(axis=1))
        Pi = S_minus / (S_plus + S_minus + 1e-10)
    q25t, q50t, q75t = np.percentile(Pi, [25,50,75])
    def _rgi_for(h):
        """RGI depuis le tableau par hôtel (str_benchmarks) — None si référence absente."""
        try:
            _ref = float(st.session_state['str_benchmarks'].loc[h, 'RevPAR marché (€)'])
            _rvp = float(dea.df.loc[h, 'revpar'])
            return round(_rvp / _ref * 100, 1) if _ref > 0 else None
        except Exception:
            return None

    topsis_bm_rows = [{'Rang': 0, 'Hôtel': hotel, 'Score Pi': round(Pi[i], 4),
                       'DEA BCC': f"{dea.bcc_scores.get(hotel,0):.1%}",
                       'RGI': _rgi_for(hotel) or '—',
                       'Quartile': ('Q4 — Top 25%' if Pi[i]>=q75t else 'Q3' if Pi[i]>=q50t
                                    else 'Q2' if Pi[i]>=q25t else 'Q1 — Bottom 25%')}
                      for i, hotel in enumerate(dea.hotels)]
    topsis_bm_df = (pd.DataFrame(topsis_bm_rows).sort_values('Score Pi', ascending=False).reset_index(drop=True))
    topsis_bm_df['Rang'] = range(1, len(topsis_bm_df)+1)
    col_x, col_y = st.columns([2,1])
    with col_x:
        st.dataframe(topsis_bm_df, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Télécharger TOPSIS composite (CSV)",
            data=topsis_bm_df.to_csv(index=False, sep=';', encoding='utf-8-sig'),
            file_name=f"deah_topsis_composite_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv",
            key='dl_topsis_bm',
        )
    with col_y:
        st.markdown("**Poids des critères**")
        _pi_crits = st.session_state.get('score_pi_crits', None)
        _pi_w     = st.session_state.get('score_pi_w', None)
        if _pi_crits and _pi_w:
            st.dataframe(pd.DataFrame({'Critère': _pi_crits, 'Poids': [f"{v:.1%}" for v in _pi_w]}),
                         use_container_width=True, hide_index=True)
        else:
            st.caption("Visitez d'abord Tab 1 (Rapport Board) pour calculer et afficher les poids.")



# ══════════════════════════════════════════════
# TAB 4 — SEGMENTATION K-MEANS
# ══════════════════════════════════════════════
with tab_quad:
    st.markdown('<p class="section-title">📐 4 Quadrants — Gestion Pure × Efficacité d\'Échelle</p>', unsafe_allow_html=True)
    quadrant_colors = {'Q1': '#1e8449', 'Q2': '#2e6da4', 'Q3': '#f39c12', 'Q4': '#c0392b'}
    fig_q = go.Figure()
    for q, label in QUADRANT_LABELS.items():
        hotels_q = [h for h in dea.hotels if dea.quadrants.get(h) == q]
        if hotels_q:
            fig_q.add_trace(go.Scatter(
                x=[dea.scale_efficiency[h] for h in hotels_q],
                y=[dea.bcc_scores[h] for h in hotels_q],
                mode='markers', text=hotels_q,
                hovertemplate="<b>%{text}</b><br>Scale Eff. : %{x:.1%}<br>BCC : %{y:.1%}<extra></extra>",
                marker=dict(size=13, color=quadrant_colors[q], symbol='circle',
                            line=dict(width=1, color='white')),
                name=label,
            ))
    fig_q.add_hline(y=bcc_threshold, line_dash='dash', line_color='gray', opacity=0.5,
                    annotation_text=f"Seuil BCC {bcc_threshold:.0%}")
    fig_q.add_vline(x=scale_threshold, line_dash='dash', line_color='gray', opacity=0.5,
                    annotation_text=f"Seuil Scale {scale_threshold:.0%}")
    fig_q.update_layout(
        title="Carte des Quadrants (BCC × Scale Efficiency)",
        xaxis=dict(title="Scale Efficiency (CCR/BCC)", range=[0.6, 1.05], tickformat='.0%'),
        yaxis=dict(title="Score BCC (Gestion Pure)", range=[0.3, 1.05], tickformat='.0%'),
        height=520, paper_bgcolor='rgba(0,0,0,0)', legend_title="Quadrant",
    )
    st.plotly_chart(fig_q, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.info("""
**Lecture des quadrants**
- **Q1 🏆 Efficient** : BCC ≥ seuil ET Scale ≥ seuil → Conserver, benchmark
- **Q2 ⚙️ Échelle** : BCC ≥ seuil MAIS Scale < seuil → Problème de taille structurel → lire le **TGR (Tab 8)**
- **Q3 🧠 Gestion** : BCC < seuil MAIS Scale ≥ seuil → Problème de gestion pure → plan opérationnel
- **Q4 🔴 Double** : BCC < seuil ET Scale < seuil → Intervention prioritaire
        """)
        with st.expander("ℹ️ Hôtels Q2 — Comment les interpréter et quoi faire"):
            st.markdown(
                """
**Q2 = BCC ≥ seuil ET Scale Efficiency < seuil**

L'hôtel est **bien géré** (gestion pure efficiente) mais **mal dimensionné** pour son marché :
trop petit pour bénéficier des économies d'échelle, ou trop grand pour remplir sa capacité
au niveau de prix du segment.

**Pourquoi ne pas relancer en Output-Oriented ?**
Changer d'orientation pose la question *"avec ces ressources, que devrait-il produire ?"*
Mais si le problème est la taille (80 chambres dans un marché 200+ chambres, ou l'inverse),
la réponse ne changera pas le diagnostic. Le score sera différent, la prescription identique.

**Ce qu'il faut lire : TGR (Metafrontière, Tab 8)**

| TGR | Lecture | Décision |
|---|---|---|
| TGR faible | Segment structurellement défavorable | Cession · Repositionnement · Extension |
| TGR élevé | Bon segment, problème d'échelle conjoncturel | Croissance · Mix produit |

Réf. : Assaf, Barros & Josiassen (2010) — metafrontière GTE/MTE/TGR.
                """
            )
    with col2:
        q_df = dea.get_quadrant_summary()
        st.dataframe(q_df[['Quadrant','N hôtels','BCC moyen','Eff. Éch.']], use_container_width=True, hide_index=True)

    hotels_sorted = sorted(dea.hotels, key=lambda h: dea.scale_efficiency[h])
    fig_scale = go.Figure()
    fig_scale.add_trace(go.Bar(name='BCC (Gestion pure)', x=hotels_sorted,
                               y=[dea.bcc_scores[h] for h in hotels_sorted],
                               marker_color='#2e6da4', opacity=0.85))
    fig_scale.add_trace(go.Bar(name='CCR (Global)', x=hotels_sorted,
                               y=[dea.ccr_scores[h] for h in hotels_sorted],
                               marker_color='#e74c3c', opacity=0.65))
    fig_scale.update_layout(barmode='group', title="BCC vs CCR — Écart = Impact d'Échelle",
                            yaxis=dict(tickformat='.0%', range=[0, 1.1]),
                            height=380, paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_scale, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 6 — SLACKS & GASPILLAGES
# ══════════════════════════════════════════════
with tab_slacks:
    st.markdown('<p class="section-title">🔥 Slacks — Gaspillages et Potentiels d\'Amélioration</p>', unsafe_allow_html=True)
    input_names  = dea.input_cols
    output_names = ['revpar', 'satisfaction', 'taux_occupation']
    _input_labels = {'nb_chambres': 'Chambres', 'nb_employes': 'Employés', 'couts_op_ex': 'Coûts Op.'}
    hotels_inefficient = [h for h in dea.hotels if dea.bcc_scores[h] < 0.999]

    if not hotels_inefficient:
        st.success("✅ Tous les hôtels sont sur la frontière d'efficacité — aucun gaspillage détecté.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Slacks Inputs (% de réduction possible)**")
            slack_matrix_in = []
            for h in hotels_inefficient:
                row = []
                for col_name in input_names:
                    s = dea.slacks[h]['inputs'].get(col_name, 0)
                    cur = float(dea.df.loc[h, col_name])
                    row.append(round(s / cur * 100, 1) if cur > 0 else 0)
                slack_matrix_in.append(row)
            fig_heat_in = go.Figure(go.Heatmap(
                z=slack_matrix_in, x=[_input_labels.get(c, c) for c in input_names], y=hotels_inefficient,
                colorscale='Reds',
                text=[[f"{v:.1f}%" for v in row] for row in slack_matrix_in],
                texttemplate="%{text}", colorbar=dict(title="%"),
            ))
            fig_heat_in.update_layout(title="Gaspillages Inputs (%)",
                                      height=max(300, len(hotels_inefficient)*30+80),
                                      paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_heat_in, use_container_width=True)
        with col2:
            st.markdown("**Slacks Outputs (% d'amélioration possible)**")
            slack_matrix_out = []
            for h in hotels_inefficient:
                row = []
                for col_name in output_names:
                    s = dea.slacks[h]['outputs'].get(col_name, 0)
                    cur = float(dea.df.loc[h, col_name])
                    row.append(round(s / cur * 100, 1) if cur > 0 else 0)
                slack_matrix_out.append(row)
            fig_heat_out = go.Figure(go.Heatmap(
                z=slack_matrix_out, x=['RevPAR', 'Satisfaction', 'TO'], y=hotels_inefficient,
                colorscale='Blues',
                text=[[f"{v:.1f}%" for v in row] for row in slack_matrix_out],
                texttemplate="%{text}", colorbar=dict(title="%"),
            ))
            fig_heat_out.update_layout(title="Potentiel Amélioration Outputs (%)",
                                       height=max(300, len(hotels_inefficient)*30+80),
                                       paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_heat_out, use_container_width=True)

        st.markdown('<p class="section-title">Upside Financier Estimé</p>', unsafe_allow_html=True)
        if not dea.has_chambres:
            st.info(
                "Upside RevPAR non calculable — nombre de chambres réel non fourni. "
                "Seul l'upside ETP reste affiché."
            )
            upside_data = []
            for h in hotels_inefficient:
                slack_emp = dea.slacks[h]['inputs'].get('nb_employes', 0)
                upside_data.append({
                    'Hôtel': h, 'Score BCC': f"{dea.bcc_scores[h]:.1%}",
                    'Slack ETP (nb)': round(slack_emp, 1),
                    'Upside ETP (k€/an)': round(slack_emp * avg_salary / 1000),
                })
            st.dataframe(pd.DataFrame(upside_data), use_container_width=True, hide_index=True)
        else:
            upside_data = []
            for h in hotels_inefficient:
                slack_emp    = dea.slacks[h]['inputs'].get('nb_employes', 0)
                slack_revpar = dea.slacks[h]['outputs'].get('revpar', 0)
                lits         = float(dea.df.loc[h, 'nb_chambres'])
                upside_fte   = round(slack_emp * avg_salary / 1000)
                upside_rev   = round(slack_revpar * lits * jours_exploit * revpar_value / 1_000_000, 1)
                upside_data.append({
                    'Hôtel': h, 'Score BCC': f"{dea.bcc_scores[h]:.1%}",
                    'Slack ETP (nb)': round(slack_emp, 1),
                    'Upside ETP (k€/an)': upside_fte,
                    'Slack RevPAR (€)': round(slack_revpar, 1),
                    'Upside RevPAR (M€/an)': upside_rev,
                })
            st.dataframe(pd.DataFrame(upside_data), use_container_width=True, hide_index=True)

        # PAR / POR / % CA sur les slacks (Russo & Legel, Exhibit 3)
        st.markdown('<p class="section-title">Slacks en metriques USALI (PAR / POR / % CA)</p>', unsafe_allow_html=True)
        if not dea.has_chambres:
            st.info(
                "Non calculable : PAR (per available room) et % CA exigent un vrai "
                "nombre de chambres — non fourni."
            )
        else:
            st.caption('PAR = Per Available Room-Night | POR = Per Occupied Room | % CA = % Chiffre Affaires total')

            par_por_rows = []
            for h in hotels_inefficient:
                lits_h   = float(dea.df.loc[h, 'nb_chambres'])
                revpar_h = float(dea.df.loc[h, 'revpar'])
                occ_h    = float(dea.df.loc[h, 'taux_occupation']) / 100
                nights_h = lits_h * jours_exploit * occ_h
                ca_h     = revpar_h * lits_h * jours_exploit
                par_h    = lits_h * jours_exploit

                s_emp  = dea.slacks[h]['inputs'].get('nb_employes', 0)
                s_cost = dea.slacks[h]['inputs'].get('couts_op_ex', 0) * 1_000_000
                s_rev  = dea.slacks[h]['outputs'].get('revpar', 0)
                val_emp_eur = s_emp * avg_salary
                val_rev_h   = s_rev * nights_h

                par_por_rows.append({
                    'Hôtel'                    : h,
                    'BCC'                      : f"{dea.bcc_scores[h]:.1%}",
                    'Slack ETP (nb)'           : round(s_emp, 1) if s_emp > 0 else '--',
                    'ETP PAR (e/ch dispo)'     : round(val_emp_eur / par_h, 2) if par_h > 0 and s_emp > 0 else '--',
                    'ETP % CA'                 : f"{val_emp_eur / ca_h * 100:.1f}%" if ca_h > 0 and s_emp > 0 else '--',
                    'Couts Op PAR (e)'         : round(s_cost / par_h, 2) if par_h > 0 and s_cost > 0 else '--',
                    'Couts Op % CA'            : f"{s_cost / ca_h * 100:.1f}%" if ca_h > 0 and s_cost > 0 else '--',
                    'Slack RevPAR POR (e/nuit)': round(s_rev, 2) if s_rev > 0 else '--',
                    'Upside Rev brut (ke)'     : round(val_rev_h / 1000, 1) if val_rev_h > 0 else '--',
                })

            if par_por_rows:
                st.dataframe(pd.DataFrame(par_por_rows), use_container_width=True, hide_index=True)
                c1p, c2p, c3p = st.columns(3)
                c1p.info('**PAR** charges fixes, A&G, Maintenance, Utilities')
                c2p.info('**POR** Rooms dept, F&B, couts variables')
                c3p.info('**% CA** Management fees, Marketing, Franchise')

# ══════════════════════════════════════════════
# TAB 7 — FICHE ACTIF DRILL-DOWN
# ══════════════════════════════════════════════
with tab_fiche:
    st.markdown('<p class="section-title">🔍 Fiche Actif — Analyse Détaillée</p>', unsafe_allow_html=True)
    selected = st.selectbox("Sélectionner un hôtel", options=dea.hotels, index=0)
    bcc   = dea.bcc_scores[selected]; ccr   = dea.ccr_scores[selected]
    scale = dea.scale_efficiency[selected]; q = dea.quadrants.get(selected, '—')
    t_rank = dea.topsis_ranks.get(selected, '—'); t_score = dea.topsis_scores.get(selected, 0)
    cluster = dea.kmeans_labels.get(selected, '—')

    st.markdown(f"### {selected}")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("BCC", f"{bcc:.1%}"); m2.metric("CCR", f"{ccr:.1%}")
    m3.metric("Scale Eff.", f"{scale:.1%}"); m4.metric("Quadrant", QUADRANT_LABELS.get(q, q))
    m5.metric("Rang TOPSIS", f"#{t_rank}/{dea.n}"); m6.metric("Segment", cluster)

    st.markdown("---")
    col_left, col_right = st.columns([1, 1])
    with col_left:
        st.markdown("**📊 Données brutes**")
        raw = dea.df.loc[selected]
        _indics, _vals = ['Employés', 'Coûts Op. (M€)', 'RevPAR (€)', 'Satisfaction', 'Taux Occup. (%)'], \
                          [raw['nb_employes'], raw['couts_op_ex'], raw['revpar'], raw['satisfaction'], raw['taux_occupation']]
        if dea.has_chambres:
            _indics.insert(0, 'Nombre de chambres'); _vals.insert(0, raw['nb_chambres'])
        raw_df = pd.DataFrame({'Indicateur': _indics, 'Valeur': _vals})
        st.dataframe(raw_df, use_container_width=True, hide_index=True)
        st.markdown("**🤝 Hôtels de référence (Peers)**")
        peers = dea.peers.get(selected, {})
        if peers:
            peers_df = pd.DataFrame([
                {'Peer': p, 'Poids λ': f"{w:.3f}", 'BCC Peer': f"{dea.bcc_scores.get(p, 0):.1%}"}
                for p, w in sorted(peers.items(), key=lambda x: -x[1])
            ])
            st.dataframe(peers_df, use_container_width=True, hide_index=True)
        else:
            st.success("Cet hôtel est lui-même un peer de référence.")
    with col_right:
        st.markdown(f"**🎯 Plan d'action — {dea.orientation_label}**")
        barros_df = dea.get_barros_table(selected)
        st.dataframe(barros_df, use_container_width=True, hide_index=True)

    rev_max = max(dea.df['revpar']); to_max = max(dea.df['taux_occupation'])
    sat_max = max(dea.df['satisfaction']); emp_max = max(dea.df['nb_employes'])
    cost_max = max(dea.df['couts_op_ex'])
    def norm_hotel(h):
        return [dea.bcc_scores[h], dea.scale_efficiency[h],
                float(dea.df.loc[h, 'revpar']) / rev_max,
                float(dea.df.loc[h, 'taux_occupation']) / to_max,
                float(dea.df.loc[h, 'satisfaction']) / sat_max,
                1 - float(dea.df.loc[h, 'nb_employes']) / emp_max]
    cats = ['BCC', 'Scale Eff.', 'RevPAR', 'TO', 'Satisfaction', 'Efficience ETP']
    vals_sel = norm_hotel(selected)
    vals_avg = [np.mean([norm_hotel(h)[i] for h in dea.hotels]) for i in range(len(cats))]
    fig_radar2 = go.Figure()
    for vals, name, color in [(vals_sel, selected, '#2e6da4'), (vals_avg, 'Moyenne portefeuille', '#e74c3c')]:
        fig_radar2.add_trace(go.Scatterpolar(
            r=vals + [vals[0]], theta=cats + [cats[0]], fill='toself', name=name,
            line_color=color, opacity=0.55,
        ))
    fig_radar2.update_layout(polar=dict(radialaxis=dict(range=[0, 1])),
                             title=f"Profil {selected} vs Portefeuille",
                             height=420, paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_radar2, use_container_width=True)

    fig_pos = go.Figure()
    for h in dea.hotels:
        is_sel = (h == selected)
        color = '#e74c3c' if is_sel else '#aec6e8'
        size  = 16        if is_sel else 9
        fig_pos.add_trace(go.Scatter(
            x=[dea.scale_efficiency[h]], y=[dea.bcc_scores[h]],
            mode='markers+text' if is_sel else 'markers',
            text=[h], textposition='top center',
            textfont=dict(size=11, color='red'),
            hovertemplate="<b>%{text}</b><br>Scale : %{x:.1%}<br>BCC : %{y:.1%}<extra></extra>",
            marker=dict(size=size, color=color), showlegend=False,
        ))
    fig_pos.add_hline(y=bcc_threshold, line_dash='dash', line_color='lightgray')
    fig_pos.add_vline(x=scale_threshold, line_dash='dash', line_color='lightgray')
    fig_pos.update_layout(
        title=f"Position de {selected} dans le portefeuille",
        xaxis=dict(title="Scale Efficiency", range=[0.6, 1.05], tickformat='.0%'),
        yaxis=dict(title="Score BCC", range=[0.3, 1.05], tickformat='.0%'),
        height=420, paper_bgcolor='rgba(0,0,0,0)',
    )
    st.plotly_chart(fig_pos, use_container_width=True)

    # ── SBM — Analyse non-radiale Fiche Actif ────────────────────────────────
    st.markdown("---")
    st.markdown('<p class="section-title">📐 SBM — Analyse non-radiale (Tone 2001)</p>', unsafe_allow_html=True)
    st.caption(
        "SBM mesure l'inefficience via les slacks directement — sans contraction radiale. "
        "Recommandé pour les actifs avec données non-positives (GOP < 0, RevPAR nul). "
        "Score NON comparable au BCC."
    )

    _sbm_hotel = st.session_state.get("sbm_results", {}).get(selected)
    if _sbm_hotel is None:
        st.info("Cliquer **Calculer SBM** dans l'onglet 2 pour activer cette section.")
    else:
        _sbm_score = _sbm_hotel.get('score')
        _bcc_score = dea.bcc_scores.get(selected, 0)

        # Métriques comparatives BCC vs SBM
        _sc1, _sc2, _sc3 = st.columns(3)
        _sc1.metric("Score BCC (radial)",   f"{_bcc_score:.1%}")
        _sc2.metric("Score SBM ρ* (non-radial)", f"{_sbm_score:.4f}" if _sbm_score else "—",
                    delta=f"{(_sbm_score - _bcc_score):+.4f}" if _sbm_score else None)
        _gap = abs(_sbm_score - _bcc_score) if _sbm_score else None
        _sc3.metric("Écart |SBM − BCC|", f"{_gap:.4f}" if _gap is not None else "—",
                    help="Écart > 0.05 : inefficiences non-radiales significatives à investiguer")

        if _sbm_score is not None and _gap is not None:
            if _gap > 0.10:
                st.warning(
                    f"⚠️ Écart SBM−BCC = {_gap:.4f} — inefficiences non-radiales importantes. "
                    "L'hôtel présente des gaspillages sur des dimensions spécifiques non capturées "
                    "par le modèle radial. Analyser les slacks ci-dessous."
                )
            elif _gap > 0.05:
                st.info(f"ℹ️ Écart modéré ({_gap:.4f}) — vérifier les slacks pour identifier les leviers.")
            else:
                st.success(f"✅ BCC et SBM convergent (écart {_gap:.4f}) — résultat robuste.")

        # ── Analyse automatique — Levier dominant ───────────────────────────
        if _sbm_hotel.get('feasible') and _sbm_score is not None and _sbm_score < 0.999:
            _dom = _sbm_dominant_analysis(_sbm_hotel, dea, selected)
            st.markdown("---")
            st.markdown("**🎯 Levier prioritaire identifié par SBM**")
            _dl1, _dl2, _dl3 = st.columns(3)
            _dl1.metric("Variable dominante", _dom['col'])
            _dl2.metric("Direction",          _dom['type'])
            _dl3.metric("Amélioration possible", _dom['pct'])
            st.info(f"💡 **Recommandation :** {_dom['reco']}")

            # Confrontation BCC vs SBM sur le levier dominant
            _dom_col = _dom['col']
            if _dom_col in dea.slacks.get(selected, {}).get('inputs', {}):
                _bcc_slack = dea.slacks[selected]['inputs'].get(_dom_col, 0)
                _sbm_slack = _sbm_hotel['slacks_in'].get(_dom_col, 0)
                if _bcc_slack is not None and _sbm_slack is not None:
                    st.markdown(
                        f"**BCC slack `{_dom_col}` = {_bcc_slack:.4f}** vs "
                        f"**SBM slack = {_sbm_slack:.4f}** — "
                        + ("SBM détecte un gaspillage supplémentaire non-radial." if _sbm_slack > _bcc_slack
                           else "Les deux modèles convergent sur ce levier.")
                    )
            elif _dom_col in dea.slacks.get(selected, {}).get('outputs', {}):
                _bcc_slack = dea.slacks[selected]['outputs'].get(_dom_col, 0)
                _sbm_slack = _sbm_hotel['slacks_out'].get(_dom_col, 0)
                if _bcc_slack is not None and _sbm_slack is not None:
                    st.markdown(
                        f"**BCC slack `{_dom_col}` = {_bcc_slack:.4f}** vs "
                        f"**SBM slack = {_sbm_slack:.4f}** — "
                        + ("SBM détecte un potentiel d'output supérieur." if _sbm_slack > _bcc_slack
                           else "Les deux modèles convergent sur ce levier.")
                    )

        # Slacks SBM détaillés
        _si = _sbm_hotel.get('slacks_in', {})
        _so = _sbm_hotel.get('slacks_out', {})
        if _si or _so:
            st.markdown("**Slacks SBM — Gaspillages résiduels (toutes dimensions)**")
            _slack_rows = (
                [{'Dimension': k, 'Type': 'Input ↓', 'Slack SBM': f"{v:.4f}" if v else '0'}
                 for k, v in _si.items()]
                + [{'Dimension': k, 'Type': 'Output ↑', 'Slack SBM': f"{v:.4f}" if v else '0'}
                   for k, v in _so.items()]
            )
            st.dataframe(pd.DataFrame(_slack_rows), use_container_width=True, hide_index=True)

        # Signal données non-positives
        _has_neg = any(
            dea.df.loc[selected, c] <= 0
            for c in dea.input_cols + dea.output_cols
            if c in dea.df.columns
        )
        if _has_neg:
            st.info(
                "🔢 Données non-positives détectées sur cet actif — "
                "le score SBM est calculé via range normalization (Tone & Tsutsui 2010) "
                "et reste valide. Le score BCC a également bénéficié de la translation invariance Pastor (1996)."
            )

    # ── Plan B — Inputs non-discrétionnaires (Banker & Morey 1986) ──────────
    st.markdown("---")
    st.markdown('<p class="section-title">🔒 Plan B — Variables structurellement fixes</p>', unsafe_allow_html=True)
    st.caption(
        "Certains inputs ne peuvent pas être réduits à court terme (masse salariale fixe, "
        "charges contractuelles). Verrouillez-les : le modèle recalcule les cibles de "
        "rattrapage sur les variables discrétionnaires restantes. "
        "Réf. : Banker & Morey (1986) Management Science."
    )

    _nd_fixed = st.multiselect(
        "Variables à verrouiller (non-modifiables)",
        options=dea.input_cols,
        default=[],
        key="nd_fixed_cols",
        help="Ex : couts_op_ex si une part est contractuellement fixe à court terme."
    )

    if st.button("🔄 Calculer Plan B", key="nd_btn"):
        if not _nd_fixed:
            st.warning("Sélectionner au moins un input à verrouiller.")
        elif len(_nd_fixed) >= len(dea.input_cols):
            st.error("Impossible — au moins un input doit rester discrétionnaire.")
        else:
            with st.spinner("Calcul DEA non-discrétionnaire…"):
                try:
                    _nd_res = dea.compute_nondiscretionary_targets(selected, _nd_fixed)
                    st.session_state["nd_results"] = _nd_res
                except Exception as _nd_e:
                    st.error(f"Erreur : {_nd_e}")

    _nd = st.session_state.get("nd_results")
    if _nd and _nd.get('hotel') == selected:
        if _nd.get('infeasible'):
            st.error(
                "⚠️ Aucune solution trouvée avec ces contraintes. "
                "Le compset ne contient pas d'hôtel de référence avec les mêmes inputs fixes — "
                "essayer de déverrouiller une variable."
            )
        else:
            _th_nd = _nd['theta_nd']
            _th_bcc = dea.bcc_scores.get(selected, 0)

            _pb1, _pb2, _pb3 = st.columns(3)
            _pb1.metric("Score BCC original",       f"{_th_bcc:.1%}")
            _pb2.metric("θ Plan B (inputs fixes)",  f"{_th_nd:.1%}",
                        help="Efficience atteignable en verrouillant les inputs sélectionnés")
            _pb3.metric("Variables verrouillées",   ", ".join(_nd_fixed))

            if _th_nd > _th_bcc + 0.005:
                st.warning(
                    f"⚠️ Le Plan B requiert θ = {_th_nd:.1%} au lieu de {_th_bcc:.1%} — "
                    "la frontière est plus difficile à atteindre sans réduire les inputs fixes. "
                    "Un effort plus important est requis sur les variables discrétionnaires."
                )
            else:
                st.success("✅ La frontière reste atteignable avec les inputs verrouillés.")

            # Tableau de rattrapage
            _disc = _nd['targets_disc']
            _cur  = _nd['current_vals']
            _slk  = _nd['slacks_disc']
            _plan_rows = []
            for col, tgt in _disc.items():
                cur = _cur.get(col, 0)
                slk = _slk.get(col, 0)
                pct = f"{(tgt - cur) / cur:.1%}" if cur and abs(cur) > 1e-6 else "—"
                _plan_rows.append({
                    'Variable discrétionnaire' : col,
                    'Valeur actuelle'           : round(cur, 2),
                    'Cible Plan B'              : round(tgt, 2),
                    'Effort requis'             : pct,
                    'Slack résiduel'            : round(slk, 4),
                })

            st.markdown("**Cibles de rattrapage — variables discrétionnaires**")
            st.dataframe(pd.DataFrame(_plan_rows), use_container_width=True, hide_index=True)

            # Cibles outputs
            _out_rows = [
                {'Output': col, 'Valeur actuelle': round(float(dea.df.loc[selected, col]), 2)
                 if col in dea.df.columns else '—', 'Cible minimum': tgt}
                for col, tgt in _nd['targets_out'].items()
            ]
            if _out_rows:
                st.markdown("**Outputs — niveau minimum à maintenir**")
                st.dataframe(pd.DataFrame(_out_rows), use_container_width=True, hide_index=True)

            st.info(
                "Compenser sur les variables discrétionnaires ci-dessus pour atteindre la frontière "
                "sans modifier les inputs verrouillés."
            )

    # Export PDF Fiche Actif (P1.1)
    st.markdown("---")
    st.markdown('<p class="section-title">📥 Export PDF</p>', unsafe_allow_html=True)
    if st.button('📄 Générer PDF Fiche Actif', key='pdf_btn'):
        with st.spinner('Génération PDF...'):
            try:
                _pdf_bytes = generate_fiche_actif_pdf(
                    hotel=selected, dea=dea,
                    quadrant_labels=QUADRANT_LABELS,
                    avg_salary=avg_salary,
                    revpar_value=revpar_value,
                    jours_exploit=jours_exploit,
                )
                if _pdf_bytes:
                    st.download_button(
                        '⬇️ Télécharger PDF Fiche Actif',
                        data=_pdf_bytes,
                        file_name=f'fiche_actif_{selected}_{datetime.now().strftime("%Y%m%d")}.pdf',
                        mime='application/pdf',
                    )
                else:
                    st.error('ReportLab non disponible — ajouter reportlab dans requirements.txt')
            except Exception as _pdf_e:
                st.error(f'Erreur PDF : {_pdf_e}')


    # ══════════════════════════════════════════════════════════════════════════
    # SIMULATEUR WHAT-IF — Recalcul DEA réel (Option B + C)
    # Réf. : Charnes et al. (1978), Banker et al. (1984)
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown('<p class="section-title">🔬 Simulateur What-If — Impact sur le Score BCC</p>', unsafe_allow_html=True)
    st.caption(
        "Modifiez les paramètres de l'hôtel et recalculez son score BCC réel (recalcul DEA complet, "
        "pas d'interpolation). La frontière d'efficience est recalculée sur l'ensemble du compset modifié. "
        "Réf. : Charnes, Cooper & Rhodes (1978), Banker, Charnes & Cooper (1984)."
    )

    _wi_raw  = dea.df.loc[selected]
    _wi_has_ch = dea.has_chambres
    _wi_lits = float(_wi_raw['nb_chambres']) if _wi_has_ch else None
    _wi_emp  = float(_wi_raw['nb_employes'])
    _wi_opex = float(_wi_raw['couts_op_ex'])
    _wi_rvp  = float(_wi_raw['revpar'])
    _wi_sat  = float(_wi_raw['satisfaction'])
    _wi_occ  = float(_wi_raw['taux_occupation'])

    # Sliders — inputs
    _wi_col1, _wi_col2 = st.columns(2)
    with _wi_col1:
        st.markdown("**📥 Inputs (ressources)**")
        if _wi_has_ch:
            _wi_new_lits = st.slider(
                "Nombre de chambres", 
                min_value=max(10,  int(_wi_lits * 0.5)),
                max_value=int(_wi_lits * 1.5),
                value=int(_wi_lits),
                step=5, key="wi_lits"
            )
        else:
            _wi_new_lits = None
        _wi_new_emp = st.slider(
            "Nombre d'ETP",
            min_value=max(5,   int(_wi_emp  * 0.5)),
            max_value=int(_wi_emp  * 1.5),
            value=int(_wi_emp),
            step=1, key="wi_emp"
        )
        _wi_new_opex = st.slider(
            "Charges opérationnelles (M€)",
            min_value=round(_wi_opex * 0.5, 2),
            max_value=round(_wi_opex * 1.5, 2),
            value=round(_wi_opex, 2),
            step=0.05, key="wi_opex"
        )

    with _wi_col2:
        st.markdown("**📤 Outputs (performances)**")
        _wi_new_rvp = st.slider(
            "RevPAR (€)",
            min_value=max(10,  int(_wi_rvp  * 0.5)),
            max_value=int(_wi_rvp  * 2.0),
            value=int(_wi_rvp),
            step=1, key="wi_rvp"
        )
        _wi_new_sat = st.slider(
            "Satisfaction (/10)",
            min_value=round(_wi_sat * 0.7, 1),
            max_value=10.0,
            value=round(_wi_sat, 1),
            step=0.1, key="wi_sat"
        )
        _wi_new_occ = st.slider(
            "Taux d'occupation (%)",
            min_value=max(20, int(_wi_occ * 0.5)),
            max_value=98,
            value=int(_wi_occ),
            step=1, key="wi_occ"
        )

    # Visualisation immédiate des écarts vs valeurs actuelles (Option C)
    _wi_changes = {
        "ETP"    : (_wi_emp,  _wi_new_emp,   "↓ moins = mieux"),
        "OpEx"   : (_wi_opex, _wi_new_opex,  "↓ moins = mieux"),
        "RevPAR" : (_wi_rvp,  _wi_new_rvp,   "↑ plus = mieux"),
        "Sat."   : (_wi_sat,  _wi_new_sat,   "↑ plus = mieux"),
        "TO"     : (_wi_occ,  _wi_new_occ,   "↑ plus = mieux"),
    }
    if _wi_has_ch:
        _wi_changes = {"Lits": (_wi_lits, _wi_new_lits, "↓ moins = mieux"), **_wi_changes}
    _wi_deltas = []
    for _k, (old, new, _) in _wi_changes.items():
        if old > 0:
            _pct = (new - old) / old * 100
            _wi_deltas.append({"Paramètre": _k, "Actuel": old, "Simulé": new,
                                "Δ": f"{_pct:+.1f}%",
                                "Direction": "input" if _k in ("Lits","ETP","OpEx") else "output"})

    _changed = any(abs(d["Actuel"] - d["Simulé"]) > 0.001 for d in _wi_deltas)

    if _changed:
        _dc1, _dc2 = st.columns(2)
        _inputs_d  = [d for d in _wi_deltas if d["Direction"] == "input"]
        _outputs_d = [d for d in _wi_deltas if d["Direction"] == "output"]

        with _dc1:
            st.markdown("**Variation des Inputs**")
            for d in _inputs_d:
                _ico = "🟢" if d["Actuel"] > d["Simulé"] else "🔴" if d["Actuel"] < d["Simulé"] else "⚪"
                st.markdown(
                    f"<div style='padding:6px 10px;background:#f8f9fa;border-radius:4px;margin-bottom:4px;'>"
                    f"{_ico} <b>{d['Paramètre']}</b> : {d['Actuel']} → {d['Simulé']} "
                    + ("<span style='color:#1e8449'>" if d['Actuel'] > d['Simulé'] else "<span style='color:#c0392b'>")
                    + f"{d['Δ']}</span>"
                    f"</div>", unsafe_allow_html=True)
        with _dc2:
            st.markdown("**Variation des Outputs**")
            for d in _outputs_d:
                _ico = "🟢" if d["Simulé"] > d["Actuel"] else "🔴" if d["Simulé"] < d["Actuel"] else "⚪"
                st.markdown(
                    f"<div style='padding:6px 10px;background:#f8f9fa;border-radius:4px;margin-bottom:4px;'>"
                    f"{_ico} <b>{d['Paramètre']}</b> : {d['Actuel']} → {d['Simulé']} "
                    + ("<span style='color:#1e8449'>" if d['Simulé'] > d['Actuel'] else "<span style='color:#c0392b'>")
                    + f"{d['Δ']}</span>"
                    f"</div>", unsafe_allow_html=True)

    # Bouton recalcul DEA réel (Option B)
    st.markdown("")
    _wi_btn_col, _wi_res_col = st.columns([1, 2])
    with _wi_btn_col:
        _wi_run = st.button(
            "🔄 Recalculer le score BCC",
            key="wi_run_btn",
            type="primary",
            disabled=not _changed,
            help="Relance le calcul DEA complet avec les paramètres modifiés. "
                 "Recalcul réel, pas d'interpolation — Charnes et al. (1978)."
        )

    if _wi_run or st.session_state.get("wi_last_result") and not _changed:
        if _wi_run:
            with st.spinner("Recalcul DEA en cours..."):
                # ── Recalcul BCC réel ────────────────────────────────────────
                # Modifier le DataFrame uniquement pour l'hôtel sélectionné
                import pulp as _pulp
                _df_mod = dea.df.copy()
                if _wi_has_ch:
                    _df_mod.loc[selected, 'nb_chambres']   = _wi_new_lits
                _df_mod.loc[selected, 'nb_employes']   = _wi_new_emp
                _df_mod.loc[selected, 'couts_op_ex']   = _wi_new_opex
                _df_mod.loc[selected, 'revpar']        = _wi_new_rvp
                _df_mod.loc[selected, 'satisfaction']  = _wi_new_sat
                _df_mod.loc[selected, 'taux_occupation'] = _wi_new_occ

                _wi_inputs_mat  = _df_mod[dea.input_cols].values.astype(float)
                _wi_outputs_mat = _df_mod[dea.output_cols].values.astype(float)
                _wi_hotels      = _df_mod.index.tolist()
                _wi_n           = len(_wi_hotels)
                _wi_idx         = _wi_hotels.index(selected)

                # Résoudre le LP BCC pour l'hôtel sélectionné uniquement
                _wi_model  = _pulp.LpProblem("BCC_WhatIf", _pulp.LpMinimize)
                _wi_theta  = _pulp.LpVariable("theta", lowBound=0)
                _wi_lambdas= _pulp.LpVariable.dicts("lam", range(_wi_n), lowBound=0)

                _wi_model += _wi_theta

                for _j in range(_wi_inputs_mat.shape[1]):
                    _wi_model += (
                        _pulp.lpSum(_wi_lambdas[_k] * _wi_inputs_mat[_k, _j] for _k in range(_wi_n))
                        <= _wi_theta * _wi_inputs_mat[_wi_idx, _j]
                    )
                for _j in range(_wi_outputs_mat.shape[1]):
                    _wi_model += (
                        _pulp.lpSum(_wi_lambdas[_k] * _wi_outputs_mat[_k, _j] for _k in range(_wi_n))
                        >= _wi_outputs_mat[_wi_idx, _j]
                    )
                _wi_model += _pulp.lpSum(_wi_lambdas.values()) == 1  # BCC

                _wi_model.solve(_pulp.PULP_CBC_CMD(msg=False))
                _wi_bcc_new = round(min(_pulp.value(_wi_theta) or 1.0, 1.0), 4)

                st.session_state["wi_last_result"] = {
                    "hotel"    : selected,
                    "bcc_old"  : dea.bcc_scores[selected],
                    "bcc_new"  : _wi_bcc_new,
                    "params"   : {
                        "lits":_wi_new_lits,"emp":_wi_new_emp,"opex":_wi_new_opex,
                        "rvp":_wi_new_rvp,"sat":_wi_new_sat,"occ":_wi_new_occ
                    }
                }

        # ── Affichage résultat ───────────────────────────────────────────────
        _wr = st.session_state.get("wi_last_result", {})
        if _wr and _wr.get("hotel") == selected:
            _bcc_old = _wr["bcc_old"]
            _bcc_new = _wr["bcc_new"]
            _delta   = _bcc_new - _bcc_old

            with _wi_res_col:
                if abs(_delta) < 0.001:
                    _color = "#555"; _icon = "⚪"
                elif _delta > 0:
                    _color = "#1e8449"; _icon = "📈"
                else:
                    _color = "#c0392b"; _icon = "📉"

                st.markdown(
                    f"<div style='background:{_color}18;border:2px solid {_color};"
                    f"border-radius:8px;padding:12px 18px;text-align:center;'>"
                    f"<div style='font-size:1.8rem;font-weight:800;color:{_color};'>"
                    f"{_icon} BCC : {_bcc_old:.1%} → {_bcc_new:.1%}</div>"
                    f"<div style='font-size:1.1rem;color:{_color};margin-top:4px;'>"
                    f"Δ = {_delta:+.1%} ({_delta*100:+.1f} pts)</div>"
                    f"<div style='font-size:0.8rem;color:#555;margin-top:6px;'>"
                    f"Recalcul DEA réel · Charnes et al. (1978) BCC</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            # Interprétation automatique
            st.markdown("")
            if _bcc_new >= 0.999:
                st.success(f"✅ **Sur la frontière d'efficience** — avec ces paramètres, {selected} atteint 100% BCC. Ce scénario est un objectif de référence réaliste.")
            elif _bcc_new > _bcc_old + 0.05:
                st.info(f"📈 **Gain significatif** (+{_delta:.1%}) — ce scénario améliore substantiellement l'efficience. Vérifiez la faisabilité opérationnelle des changements.")
            elif _bcc_new < _bcc_old - 0.02:
                st.warning(f"📉 **Détérioration** ({_delta:.1%}) — ce scénario dégrade l'efficience. Revoir la combinaison inputs/outputs.")
            else:
                st.info(f"➡️ **Impact limité** ({_delta:+.1%}) — les modifications ont peu d'effet sur le score BCC. L'inefficience vient d'autres leviers.")

    st.caption(
        "📚 *Réf. : Charnes, Cooper & Rhodes (1978) CCR · Banker, Charnes & Cooper (1984) BCC · "
        "Recalcul LP réel via PuLP/CBC — pas d'interpolation. "
        "Résultat valide uniquement pour l'hôtel sélectionné, frontière définie par le compset inchangé.*"
    )

# ══════════════════════════════════════════════
# TAB 8 — METAFRONTIÈRE
# ══════════════════════════════════════════════
with tab_meta:
    # ══════════════════════════════════════════════════════════════════════
    #  SEGMENTATION K-MEANS — fusionnée depuis son ancien onglet autonome.
    #  Sa place naturelle est ici : la segmentation détermine les groupes sur
    #  lesquels la métafrontière estime les frontières technologiques.
    # ══════════════════════════════════════════════════════════════════════
    with st.expander('🗂️ Segmentation K-means — base des groupes technologiques',
                     expanded=False):
        st.markdown('<p class="section-title">🗂️ Segmentation K-means — 4 Clusters</p>', unsafe_allow_html=True)

        # Silhouette score + méthode du coude
        _sil = getattr(dea, 'kmeans_silhouette', None)
        _inertias = getattr(dea, 'kmeans_inertias', {})
        _sil_col1, _sil_col2 = st.columns(2)
        with _sil_col1:
            if _sil is not None:
                _sil_label = 'excellent' if _sil > 0.7 else ('bon' if _sil > 0.5 else ('moyen' if _sil > 0.3 else 'faible'))
                st.metric('Silhouette Score (k=4)', f'{_sil:.3f}',
                          help='Rousseeuw (1987). >0.7=excellent | 0.5-0.7=bon | 0.3-0.5=moyen | <0.3=faible — remettre k en question')
                st.caption(f'Qualité clustering : **{_sil_label}**')
        with _sil_col2:
            if _inertias:
                # go already imported at top
                _fig_elbow = go.Figure(go.Scatter(x=list(_inertias.keys()), y=list(_inertias.values()),
                    mode='lines+markers', marker=dict(color='#2e6da4')))
                _fig_elbow.add_vline(x=4, line_dash='dash', line_color='red', annotation_text='k=4 actuel')
                _fig_elbow.update_layout(title='Méthode du coude', xaxis_title='k', yaxis_title='Inertie',
                    height=200, margin=dict(t=30,b=20,l=40,r=20), paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(_fig_elbow, use_container_width=True)

        cluster_summary = dea.get_cluster_summary()
        st.dataframe(cluster_summary, use_container_width=True, hide_index=True)

        cluster_colors = {
            '🏆 Leaders': '#1e8449', '📈 Intermédiaires': '#2e6da4',
            '⚠️ Sous-performants': '#f39c12', '🔴 Critiques': '#c0392b',
        }
        fig_km = go.Figure()
        for lbl, color in cluster_colors.items():
            hotels_c = [h for h in dea.hotels if dea.kmeans_labels.get(h) == lbl]
            if hotels_c:
                fig_km.add_trace(go.Scatter(
                    x=[dea.scale_efficiency[h] for h in hotels_c],
                    y=[dea.bcc_scores[h] for h in hotels_c],
                    mode='markers', text=hotels_c,
                    hovertemplate="<b>%{text}</b><br>Scale : %{x:.1%}<br>BCC : %{y:.1%}<extra></extra>",
                    marker=dict(size=12, color=color, line=dict(width=1, color='white')), name=lbl,
                ))
        fig_km.update_layout(
            title="Segmentation K-means : BCC vs Efficacité d'Échelle",
            xaxis=dict(title="Scale Efficiency", range=[0, 1.05], tickformat='.0%'),
            yaxis=dict(title="Score BCC", range=[0, 1.05], tickformat='.0%'),
            height=480, paper_bgcolor='rgba(0,0,0,0)', legend_title="Segment",
        )
        st.plotly_chart(fig_km, use_container_width=True)

        fig_box = go.Figure()
        for lbl, color in cluster_colors.items():
            hotels_c = [h for h in dea.hotels if dea.kmeans_labels.get(h) == lbl]
            if hotels_c:
                fig_box.add_trace(go.Box(
                    y=[dea.bcc_scores[h] for h in hotels_c], name=lbl,
                    marker_color=color, boxpoints='all',
                ))
        fig_box.update_layout(
            title="Distribution BCC par segment",
            yaxis=dict(title="Score BCC", tickformat='.0%'),
            height=380, paper_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig_box, use_container_width=True)

    # ══════════════════════════════════════════════
    # TAB 5 — QUADRANTS & ÉCHELLE
    # ══════════════════════════════════════════════

    st.markdown('<p class="section-title">🌐 Metafrontière — Analyse GTE / MTE / TGR</p>', unsafe_allow_html=True)
    st.info("""
**Principe (Assaf et al., 2010) :**
- **GTE** — efficience relative au meilleur modèle opératoire de son groupe (pairs du même segment)
- **MTE** — efficience relative au meilleur modèle opératoire observé dans tout le compset
- **TGR** (MTE/GTE) — écart entre le modèle opératoire du groupe et le meilleur modèle connu

*"Technologie" = façon de combiner les inputs pour produire les outputs (jargon économique, pas IT).*
Un hôtel GTE élevé + TGR faible = bien géré dans un segment structurellement défavorable → repositionnement stratégique, pas plan opérationnel.
    """)
    st.caption(
        "**Recommandation méthodologique :** si votre portefeuille mélange des catégories "
        "(urban/resort, 3★/5★), utilisez la segmentation ci-dessous pour corriger le biais "
        "inter-groupes (Assaf et al. 2010). Si votre compset est homogène, tous les hôtels "
        "appartiennent au même groupe — GTE = MTE pour tous, sans impact sur les scores."
    )
    seg_options = {}
    if dea.has_chambres:
        seg_options['🏠 Taille (auto depuis nb_chambres)'] = 'taille'
    if 'type_gestion' in dea.df.columns:
        seg_options['⚙️ Type de gestion (indépendant/chaîne)'] = 'type_gestion'
    if 'classement_etoiles' in dea.df.columns:
        seg_options['⭐ Classement étoiles (1★ → 5★)'] = 'classement_etoiles'

    if not seg_options:
        st.warning(
            "Aucune dimension de segmentation disponible — ni nombre de chambres "
            "réel, ni type de gestion, ni classement étoiles dans le fichier. "
            "La métafrontière ne peut pas être calculée sans au moins une de "
            "ces colonnes."
        )
        groups = meta_df = meta_sum = None
    else:
        seg_label = st.selectbox(
            "Dimension de segmentation",
            options=list(seg_options.keys()),
            help="Choisir la dimension qui définit les groupes homogènes dans votre compset. "
                 "Réf. : Assaf, Barros & Josiassen (2010) — metafrontière bootstrappée."
        )
        seg_key = seg_options[seg_label]
        groups = dea.get_auto_size_groups() if seg_key == 'taille' else dea.df[seg_key].rename('groupe')
        with st.spinner("Calcul GTE / MTE / TGR…"):
            meta_df  = dea.compute_metafrontier(groups)
            meta_sum = dea.get_metafrontier_summary(meta_df)
        warns = [w for w in meta_df['_warn'].dropna().unique() if w]
        for w in warns:
            st.warning(w)
    if meta_df is not None:
        st.dataframe(meta_sum, use_container_width=True, hide_index=True)
        display_cols = ['Hôtel', 'Groupe', 'GTE', 'MTE', 'TGR', 'Interprétation']
        st.dataframe(meta_df[display_cols].sort_values('TGR'), use_container_width=True, hide_index=True)

        # Convertir groupes en string (classement_etoiles = entiers → crash Plotly)
        meta_df['Groupe'] = meta_df['Groupe'].astype(str)
        grp_colors  = px.colors.qualitative.Set2
        unique_grps = meta_df['Groupe'].unique()
        color_map   = {g: grp_colors[i % len(grp_colors)] for i, g in enumerate(unique_grps)}
        fig_meta = go.Figure()
        for grp in unique_grps:
            sub = meta_df[meta_df['Groupe'] == grp]
            fig_meta.add_trace(go.Scatter(
                x=sub['GTE'], y=sub['TGR'], mode='markers',
                text=sub['Hôtel'],
                hovertemplate="<b>%{text}</b><br>GTE : %{x:.1%}<br>TGR : %{y:.1%}<extra></extra>",
                marker=dict(size=12, color=color_map[grp], line=dict(width=1, color='white')), name=str(grp),
            ))
        fig_meta.add_hline(y=0.90, line_dash='dash', line_color='gray', opacity=0.4)
        fig_meta.add_vline(x=0.90, line_dash='dash', line_color='gray', opacity=0.4)
        fig_meta.update_layout(
            title="Positionnement GTE × TGR par groupe",
            xaxis=dict(title="GTE", range=[0.3, 1.08], tickformat='.0%'),
            yaxis=dict(title="TGR", range=[0.3, 1.08], tickformat='.0%'),
            height=520, paper_bgcolor='rgba(0,0,0,0)', legend_title="Groupe",
        )
        st.plotly_chart(fig_meta, use_container_width=True)

        meta_sorted = meta_df.sort_values('TGR')
        fig_tgr = go.Figure(go.Bar(
            x=meta_sorted['TGR'], y=meta_sorted['Hôtel'], orientation='h',
            marker=dict(color=meta_sorted['TGR'], colorscale='RdYlGn', cmin=0.4, cmax=1.0,
                        showscale=True, colorbar=dict(title="TGR", tickformat='.0%')),
            text=[f"{v:.1%}" for v in meta_sorted['TGR']], textposition='outside',
            customdata=meta_sorted[['GTE', 'MTE', 'Groupe']].values,
            hovertemplate="<b>%{y}</b><br>TGR : %{x:.1%}<br>GTE : %{customdata[0]:.1%}<br>MTE : %{customdata[1]:.1%}<br>Groupe : %{customdata[2]}<extra></extra>",
        ))
        fig_tgr.update_layout(
            title="Technology Gap Ratio", xaxis=dict(title="TGR", range=[0, 1.15], tickformat='.0%'),
            height=max(380, len(dea.hotels) * 28), paper_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig_tgr, use_container_width=True)


        col1, col2 = st.columns(2)
        with col1:
            st.success("**✅ GTE élevé + TGR élevé** — Leader absolu. Conserver, benchmark.")
            st.warning("**⚙️ GTE élevé + TGR faible** — Bon gestionnaire, mauvais segment → arbitrage.")
        with col2:
            st.info("**🧠 GTE faible + TGR élevé** — Potentiel là, gestion insuffisante → plan opérationnel.")
            st.error("**🔴 GTE faible + TGR faible** — Double gap. Cession ou restructuration.")
        st.caption(
            "« Élevé » = quartile supérieur de GTE/TGR **dans ce portefeuille** (plancher 75% en "
            "absolu), pas un seuil fixe universel — sur un compset homogène, un seuil fixe "
            "classerait presque tout le monde en \"Leader absolu\" sans que ce soit discriminant."
        )

        st.info(
            "💡 **Peers restreints au segment pertinent via GTE** — la frontière intra-groupe "
            "garantit que chaque hôtel est évalué contre ses vrais comparables (même segment). "
            "Restreindre manuellement les peers est redondant avec cette mécanique. "
            "Réf. : Assaf, Barros & Josiassen (2010) Table 3.",
        )

        # ── ANOVA second stage (Poldrugovac 2016 ; Assaf 2009) ────────────────────────
        st.markdown("---")
        st.markdown('<p class="section-title">ANOVA Second Stage -- Differences efficience entre groupes</p>', unsafe_allow_html=True)
        st.caption('Poldrugovac et al. (2016), Färe et al. (1994), Yu (2012), Tobin (1958) Table 6 ; Assaf et al. (2009) Table 3 -- Test H0 : pas de difference efficience BCC entre groupes. Welch si variances heterogenes (Levene p < 0.05).')

        try:
            # Mann-Whitney (Yu 2012) + ANOVA classique
            _mw_res = mann_whitney_groups(dea, groups)
            if 'error' not in _mw_res:
                col_a1, col_a2 = st.columns([3, 1])
                with col_a1:
                    st.markdown('**Mann-Whitney U par groupe (Yu 2012, Table 4-5)**')
                    st.dataframe(_mw_res['summary'], use_container_width=True, hide_index=True)
                    st.dataframe(_mw_res['pairs'][['Groupe A','Groupe B','Moy. BCC A','Moy. BCC B','p-value','Sig.*','Verdict']],
                                 use_container_width=True, hide_index=True)
                with col_a2:
                    st.metric('Groupes testés', _mw_res['n_groupes'])
                    st.metric('Alpha Bonferroni', f"{_mw_res['alpha_bonf']:.4f}")
                    st.caption(_mw_res['test'])
                st.info('*** p<0.01  ** p<0.05  * p<0.10  ns = non significatif. '
                        'Test non-paramétrique recommandé pour scores DEA censurés en 1.0 '
                        '(Simar & Wilson 2007 ; Yu 2012).')
            else:
                _anova_summ, _anova_res = dea.anova_efficiency_by_groups(groups)
                st.dataframe(_anova_summ, use_container_width=True, hide_index=True)
                st.dataframe(_anova_res, use_container_width=True, hide_index=True)
        except Exception as _e:
            st.info(f"ANOVA non disponible : {_e}")

        # ── Bootstrap Metafrontière (Assaf 2009 ; Simar & Wilson 2007) ─────────────
        st.markdown("---")
        st.markdown('<p class="section-title">Bootstrap Metafrontière — Intervalles de confiance IC 95%</p>', unsafe_allow_html=True)
        st.caption("Assaf, Barros & Josiassen (2009) ; Simar & Wilson (2007) — 2000 itérations dans le papier original. Ici 200 itérations pour performance. IC 95% sur GTE, MTE, TGR.")

        _n_boot = st.slider("Nombre d'itérations bootstrap", min_value=50, max_value=500, value=100, step=50, key="meta_bootstrap_n")
        if st.button("🔄 Lancer Bootstrap Metafrontière", key="meta_boot_btn"):
            with st.spinner(f"Bootstrap {_n_boot} itérations en cours…"):
                try:
                    _boot_df = dea.compute_metafrontier_bootstrap(groups, n_bootstrap=_n_boot)
                    st.session_state['meta_bootstrap'] = _boot_df
                    st.success(f"✅ Bootstrap terminé — {_n_boot} itérations")
                except Exception as _e:
                    st.error(f"Erreur bootstrap : {_e}")

        if st.session_state.get('meta_bootstrap') is not None:
            _bdf = st.session_state['meta_bootstrap']
            _boot_cols = ['Hôtel', 'Groupe', 'GTE', 'GTE IC bas', 'GTE IC haut',
                          'MTE', 'MTE IC bas', 'MTE IC haut',
                          'TGR', 'TGR IC bas', 'TGR IC haut', 'Interprétation']
            _boot_cols_avail = [c for c in _boot_cols if c in _bdf.columns]
            st.dataframe(_bdf[_boot_cols_avail].sort_values('TGR'), use_container_width=True, hide_index=True)
            st.caption("IC bas / IC haut = quantiles 2.5% et 97.5% des scores bootstrap. Un IC large indique une incertitude statistique élevée sur le score.")

    # ── KPIs enrichis — Market share + Guests/ETP (Assaf 2009) ───────────
    st.markdown("---")
    st.markdown('<p class="section-title">KPIs Enrichis — Market Share & Productivité du Travail</p>', unsafe_allow_html=True)
    st.caption("Assaf et al. (2009) Table 2 — Market share intra-compset | Guests/ETP = nb nuitées / ETP (productivité travail).")

    _enr_df = dea.compute_enriched_kpis()
    if _enr_df is None:
        st.warning(
            "Non calculable : nuitées, CA estimé, market share et Guests/ETP exigent un "
            "vrai nombre de chambres — absent ou non fourni via l'enrichissement."
        )
    else:
        st.dataframe(_enr_df, use_container_width=True, hide_index=True)

        # Scatter market share vs BCC
        _enr_bcc = [dea.bcc_scores.get(h, 0) for h in _enr_df['Hôtel']]
        fig_ms = go.Figure()
        for i, h in enumerate(_enr_df['Hôtel']):
            _ms = _enr_df.iloc[i]['Market share (%)']
            _bcc = _enr_bcc[i]
            _color = '#27ae60' if _bcc >= 0.90 else '#f39c12' if _bcc >= 0.80 else '#e74c3c'
            fig_ms.add_trace(go.Scatter(
                x=[_ms], y=[_bcc], mode='markers', text=[h],
                hovertemplate="<b>%{text}</b><br>Market share : %{x:.1f}%<br>BCC : %{y:.1%}<extra></extra>",
                marker=dict(size=10, color=_color), showlegend=False,
            ))
        fig_ms.update_layout(
            title="Market Share intra-compset vs Efficience BCC (Assaf et al. 2009)",
            xaxis=dict(title="Market Share (%)"),
            yaxis=dict(title="Score BCC", tickformat='.0%', range=[0.3, 1.08]),
            height=400, paper_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig_ms, use_container_width=True)



# ══════════════════════════════════════════════
# TAB 9 — CAPITAL & FLOW THROUGH
# ══════════════════════════════════════════════
with tab_capital:
    st.markdown('<p class="section-title">💰 Efficience Capital & Flow Through</p>', unsafe_allow_html=True)
    # Flow Through global — défini dans la sidebar (pas de différenciation par étoiles)
    # FT% global depuis sidebar

    init_data = {}
    for hotel in dea.hotels:
        row = {
            'surface_m2'        : float(dea.df.loc[hotel, 'surface_m2'])        if (getattr(dea,'has_surface',False) and 'surface_m2'         in dea.df.columns) else 0.0,
            'capex_annuel (k€)' : float(dea.df.loc[hotel, 'capex_annuel'])      if (getattr(dea,'has_capex',  False) and 'capex_annuel'         in dea.df.columns) else 0.0,
            'gop (k€)'          : float(dea.df.loc[hotel, 'gop'])               if (getattr(dea,'has_gop',    False) and 'gop'                  in dea.df.columns) else 0.0,
            'classement (★)'    : int(dea.df.loc[hotel, 'classement_etoiles'])  if (getattr(dea,'has_stars',  False) and 'classement_etoiles'   in dea.df.columns) else 3,
        }
        init_data[hotel] = row
    # Conserver l'ordre stable des hôtels
    df_init = pd.DataFrame(init_data).T.loc[dea.hotels]

    if ('capital_input' not in st.session_state or
            set(st.session_state['capital_input'].index) != set(dea.hotels)):
        # Première initialisation OU changement de portefeuille → reset complet
        st.session_state['capital_input'] = df_init.copy()
    else:
        # Ne pré-remplir depuis le CSV que les colonnes ENCORE à zéro dans la saisie
        # (évite d'écraser les valeurs déjà saisies par l'utilisateur)
        cur = st.session_state['capital_input']
        for col in ['surface_m2','capex_annuel (k€)','gop (k€)','classement (★)']:
            if df_init[col].sum() > 0 and cur[col].sum() == 0:
                st.session_state['capital_input'][col] = df_init[col]
        # Garantir l'ordre stable
        st.session_state['capital_input'] = st.session_state['capital_input'].loc[dea.hotels]

    csv_cols = []
    if getattr(dea,'has_surface',False): csv_cols.append('surface_m2')
    if getattr(dea,'has_capex',  False): csv_cols.append('capex_annuel')
    if getattr(dea,'has_gop',    False): csv_cols.append('gop')
    if getattr(dea,'has_stars',  False): csv_cols.append('classement_etoiles')
    if csv_cols:
        st.success(f"✅ Données lues depuis le CSV : {', '.join(csv_cols)}")
    else:
        st.info("Aucune colonne capital dans le CSV — saisissez les valeurs ci-dessous, "
                "ou utilisez le modèle Excel/CSV pré-rempli.")

    # ── Modèle à télécharger / import de fichier rempli ──────────────────────
    with st.expander("📄 Modèle Excel/CSV — remplir hors application", expanded=(not csv_cols)):
        st.caption(
            "Téléchargez le modèle pré-rempli avec vos hôtels, complétez-le dans "
            "Excel, puis réimportez-le ici. Les valeurs remplaceront le tableau."
        )

        _tpl = st.session_state['capital_input'].copy()
        _tpl.index.name = 'hotel_name'

        _dl1, _dl2, _dl3 = st.columns([1, 1, 2])

        with _dl1:
            _tpl_xlsx = BytesIO()
            with pd.ExcelWriter(_tpl_xlsx, engine='openpyxl') as _w:
                _tpl.to_excel(_w, sheet_name='Capital')
                _ws = _w.sheets['Capital']
                _ws.column_dimensions['A'].width = 38
                for _c in 'BCDE':
                    _ws.column_dimensions[_c].width = 20
                _ws.freeze_panes = 'B2'
            st.download_button(
                "⬇️ Modèle Excel", data=_tpl_xlsx.getvalue(),
                file_name="deah_capital_modele.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True, key='cap_tpl_xlsx',
            )

        with _dl2:
            st.download_button(
                "⬇️ Modèle CSV",
                data=_tpl.to_csv(sep=';', decimal=',').encode('utf-8-sig'),
                file_name="deah_capital_modele.csv", mime="text/csv",
                use_container_width=True, key='cap_tpl_csv',
            )

        with _dl3:
            st.caption(
                "**Colonnes attendues** — `hotel_name` (colonne A, noms identiques), "
                "`surface_m2`, `capex_annuel (k€)`, `gop (k€)`, `classement (★)`."
            )

        _cap_up = st.file_uploader(
            "Réimporter le modèle rempli",
            type=['xlsx', 'xls', 'csv'], key='cap_upload',
        )

        if _cap_up is not None:
            try:
                if (_cap_up.name or '').lower().endswith(('.xlsx', '.xls')):
                    _up = pd.read_excel(_cap_up, index_col=0)
                else:
                    _rawc = _cap_up.read().decode('utf-8-sig', errors='replace')
                    _sepc = ';' if _rawc.count(';') > _rawc.count(',') else ','
                    _up = pd.read_csv(BytesIO(_rawc.encode('utf-8')), index_col=0,
                                      sep=_sepc, decimal=',' if _sepc == ';' else '.')

                _up.index   = _up.index.astype(str).str.strip()
                _up.columns = [str(c).strip() for c in _up.columns]

                _need = ['surface_m2', 'capex_annuel (k€)', 'gop (k€)', 'classement (★)']
                _miss = [c for c in _need if c not in _up.columns]
                if _miss:
                    st.error(f"❌ Colonnes manquantes : {', '.join(_miss)}")
                else:
                    _matched = [h for h in dea.hotels if h in _up.index]
                    _unknown = [str(h) for h in _up.index if h not in dea.hotels]

                    if not _matched:
                        st.error(
                            "❌ Aucun nom d'hôtel ne correspond au portefeuille chargé. "
                            "Les noms doivent être identiques à ceux du fichier principal."
                        )
                    else:
                        _new = st.session_state['capital_input'].copy()
                        for _c in _need:
                            _vals = pd.to_numeric(_up.loc[_matched, _c], errors='coerce').fillna(0)
                            _new.loc[_matched, _c] = _vals.values
                        _new['classement (★)'] = (_new['classement (★)']
                                                  .clip(1, 5).round().astype(int))
                        st.session_state['capital_input'] = _new.loc[dea.hotels]

                        st.success(
                            f"✅ {len(_matched)} hôtel(s) mis à jour sur {len(dea.hotels)}."
                        )
                        if _unknown:
                            st.warning(
                                f"⚠️ {len(_unknown)} ligne(s) ignorée(s) — nom inconnu : "
                                + ", ".join(_unknown[:4])
                                + (" …" if len(_unknown) > 4 else "")
                            )
                        _absents = [h for h in dea.hotels if h not in _up.index]
                        if _absents:
                            st.info(
                                f"ℹ️ {len(_absents)} hôtel(s) absent(s) du fichier — "
                                "valeurs actuelles conservées."
                            )
            except Exception as _e:
                st.error(f"❌ Lecture impossible : {_e}")

    cap_input = st.data_editor(
        st.session_state['capital_input'], use_container_width=True, num_rows='fixed',
        column_config={
            'surface_m2'        : st.column_config.NumberColumn('Surface (m²)',      min_value=0, format='%.0f'),
            'capex_annuel (k€)' : st.column_config.NumberColumn('CAPEX annuel (k€)', min_value=0, format='%.1f'),
            'gop (k€)'          : st.column_config.NumberColumn('GOP (k€)',           min_value=0, format='%.1f'),
            'classement (★)'    : st.column_config.SelectboxColumn('Classement', options=[1,2,3,4,5]),
        },
        key='capital_editor',
    )
    st.session_state['capital_input'] = cap_input

    # ── Export du tableau tel que saisi ─────────────────────────────────────
    _ex = cap_input.copy(); _ex.index.name = 'hotel_name'
    _e1, _e2, _e3 = st.columns([1, 1, 3])
    with _e1:
        _ex_xlsx = BytesIO()
        with pd.ExcelWriter(_ex_xlsx, engine='openpyxl') as _w:
            _ex.to_excel(_w, sheet_name='Capital')
            _ws = _w.sheets['Capital']
            _ws.column_dimensions['A'].width = 38
            for _c in 'BCDE':
                _ws.column_dimensions[_c].width = 20
            _ws.freeze_panes = 'B2'
        st.download_button(
            "⬇️ Excel", data=_ex_xlsx.getvalue(),
            file_name="deah_capital_saisie.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True, key='cap_exp_xlsx',
        )
    with _e2:
        st.download_button(
            "⬇️ CSV", data=_ex.to_csv(sep=';', decimal=',').encode('utf-8-sig'),
            file_name="deah_capital_saisie.csv", mime="text/csv",
            use_container_width=True, key='cap_exp_csv',
        )
    with _e3:
        st.caption("Sauvegarde de la saisie — réimportable à la session suivante.")

    has_any = (cap_input[['surface_m2','capex_annuel (k€)','gop (k€)']].sum().sum() > 0)
    if not has_any:
        st.info("👆 Renseignez surface_m2, CAPEX et GOP dans le tableau ci-dessus pour les analyses capital.")

    # Initialisation systématique — les sections avales (Expense Flex, Flow Through)
    # référencent cap_rows même quand le tableau capital n'est pas encore rempli
    cap_rows = []

    if has_any:

        st.markdown("---")
        for hotel in dea.hotels:
            row      = cap_input.loc[hotel]
            surf     = float(row['surface_m2']); capex_ke = float(row['capex_annuel (k€)'])
            gop_ke   = float(row['gop (k€)']);   stars    = int(row['classement (★)'])
            lits = float(dea.df.loc[hotel, 'nb_chambres']) if dea.has_chambres else None
            goppam   = round(gop_ke * 1000 / surf, 2)    if surf > 0    else None
            capex_ch = round(capex_ke * 1000 / lits, 0)  if lits else None
            # CA : exige un vrai nombre de chambres ou un total_revenue réel —
            # plus de repli automatique, cf. audit portefeuille Ibis Atream
            # (rapport chambres/lits de 1:1 à 7:1 selon l'hôtel).
            if dea.has_trevpar:
                rev_est = float(dea.df.loc[hotel, 'total_revenue'])
            elif lits:
                revpar = float(dea.df.loc[hotel, 'revpar'])
                rev_est  = revpar * jours_exploit * lits
            else:
                rev_est = None
            rendement= round(rev_est / (capex_ke * 1000), 2) if (rev_est and capex_ke > 0) else None
            ft       = ft_pct
            slack_r  = dea.slacks.get(hotel, {}).get('outputs', {}).get('revpar', 0)
            up_gop   = round(slack_r * lits * jours_exploit * ft / 1_000_000, 3) if (slack_r > 0 and lits) else 0
            gop_margin = round(gop_ke * 1000 / rev_est * 100, 1) if (gop_ke > 0 and rev_est) else None
            cap_rows.append({
                'Hôtel': hotel, 'BCC': f"{dea.bcc_scores.get(hotel, 0):.1%}", 'Classement': '★' * stars,
                'Surface (m²)': int(surf) if surf > 0 else '—',
                'm²/chambre': round(surf/lits,1) if surf>0 and lits else '—',
                'CAPEX/chambre (k€)': round(capex_ch/1000,1) if capex_ch else '—',
                'Rendement CAPEX (x)': rendement if rendement else '—',
                'GOP (k€)': gop_ke if gop_ke > 0 else '—',
                'Marge GOP %': f"{gop_margin:.1f}%" if gop_margin else '—',
                'GOPPAM (€/m²)': goppam if goppam else '—',
                'FT% benchmark': f"{ft:.0%}",
                'Upside GOP /FT (M€/an)': up_gop if up_gop > 0 else '—',
            })
        cap_df = pd.DataFrame(cap_rows)
    
        c1, c2, c3, c4 = st.columns(4)
        goppam_vals = [r['GOPPAM (€/m²)'] for r in cap_rows if r['GOPPAM (€/m²)'] != '—']
        capex_vals  = [r['CAPEX/chambre (k€)'] for r in cap_rows if r['CAPEX/chambre (k€)'] != '—']
        margin_vals = [float(r['Marge GOP %'].replace('%','')) for r in cap_rows if r['Marge GOP %'] != '—']
        upside_vals = [r['Upside GOP /FT (M€/an)'] for r in cap_rows if r['Upside GOP /FT (M€/an)'] != '—']
        with c1:
            if goppam_vals: st.metric("GOPPAM moyen", f"{sum(goppam_vals)/len(goppam_vals):.2f} €/m²")
        with c2:
            if capex_vals:  st.metric("CAPEX/chambre moyen", f"{sum(capex_vals)/len(capex_vals):.1f} k€")
        with c3:
            if margin_vals: st.metric("Marge GOP moyenne", f"{sum(margin_vals)/len(margin_vals):.1f}%")
        with c4:
            if upside_vals: st.metric("Upside GOP total /FT", f"{sum(upside_vals):.2f} M€/an")
    
        st.dataframe(cap_df, use_container_width=True, hide_index=True)
    
        col_l, col_r = st.columns(2)
        with col_l:
            _cat_colors = {'3★': '#3498db', '4★': '#2ecc71', '4★ Sup': '#f39c12', '5★': '#9b59b6'}
            plot_data = []
            for r in cap_rows:
                if r['GOPPAM (€/m²)'] != '—' and r['CAPEX/chambre (k€)'] != '—':
                    h = r['Hôtel']
                    _cat = dea.df.loc[h, 'categorie'] if 'categorie' in dea.df.columns else '4★'
                    plot_data.append((h, r['GOPPAM (€/m²)'], r['CAPEX/chambre (k€)'],
                                      dea.bcc_scores.get(h, 0), _cat))
            if plot_data:
                fig_cap = go.Figure()
                # Labels en mode hover uniquement — évite la superposition dans le cluster 4★
                _seen_cats = set()
                for h, gop_, capx, bcc, cat in plot_data:
                    color = _cat_colors.get(cat, '#95a5a6')
                    _show_leg = cat not in _seen_cats
                    _seen_cats.add(cat)
                    fig_cap.add_trace(go.Scatter(
                        x=[capx], y=[gop_], mode='markers', text=[h],
                        marker=dict(size=12 + bcc * 10, color=color, opacity=0.85,
                                    line=dict(width=1.5, color='white')),
                        name=cat, legendgroup=cat, showlegend=_show_leg,
                        hovertemplate=f"<b>{h}</b><br>Catégorie: {cat}<br>"
                                      f"CAPEX/chambre: %{{x:.1f}} k€<br>GOPPAM: %{{y:.2f}} €/m²<br>"
                                      f"BCC: {bcc:.1%}<extra></extra>",
                    ))
                # Annotations texte uniquement pour les outliers (top/bottom GOPPAM)
                _gop_sorted = sorted(plot_data, key=lambda x: x[1])
                _label_set = set([_gop_sorted[-1][0], _gop_sorted[-2][0],
                                   _gop_sorted[0][0], _gop_sorted[1][0]])
                # Toujours labelliser les 5★ (rares, bien séparés)
                _label_set |= {h for h, *_, cat in plot_data if cat == '5★'}
                _label_set |= {h for h, *_, cat in plot_data if cat == '4★ Sup'}
                for h, gop_, capx, bcc, cat in plot_data:
                    if h in _label_set:
                        fig_cap.add_annotation(
                            x=capx, y=gop_, text=f"<b>{h.split('(')[0].strip()}</b>",
                            showarrow=False, yshift=12, font=dict(size=8),
                            bgcolor='rgba(255,255,255,0.7)', borderpad=2,
                        )
                fig_cap.update_layout(
                    xaxis=dict(title="CAPEX annuel / chambre (k€) — benchmark par catégorie"),
                    yaxis=dict(title="GOPPAM (€/m²)"),
                    legend=dict(title="Catégorie", orientation="h", yanchor="bottom", y=1.02),
                    height=420, paper_bgcolor='rgba(0,0,0,0)',
                    annotations=[dict(
                        text="⚠️ CAPEX = taux benchmark secteur (données réelles non disponibles)",
                        xref="paper", yref="paper", x=0, y=-0.15,
                        showarrow=False, font=dict(size=10, color='grey'), align='left',
                    )],
                )
                st.plotly_chart(fig_cap, use_container_width=True)
        with col_r:
            margin_data = [(r['Hôtel'], float(r['Marge GOP %'].replace('%','')), dea.bcc_scores.get(r['Hôtel'],0))
                           for r in cap_rows if r['Marge GOP %'] != '—']
            if margin_data:
                fig_gop = go.Figure()
                for h, margin, bcc in margin_data:
                    color = '#27ae60' if bcc>=0.90 else '#f39c12' if bcc>=0.80 else '#e74c3c'
                    fig_gop.add_trace(go.Scatter(
                        x=[bcc], y=[margin], mode='markers', text=[h],
                        hovertemplate="<b>%{text}</b><br>BCC : %{x:.1%}<br>Marge GOP : %{y:.1f}%<extra></extra>",
                        marker=dict(size=11, color=color, opacity=0.8), showlegend=False,
                    ))
                fig_gop.update_layout(xaxis=dict(title="Score BCC", tickformat='.0%', range=[0.3,1.05]),
                                      yaxis=dict(title="Marge GOP %"),
                                      height=400, paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_gop, use_container_width=True)
    
        st.markdown("---")
        if not dea.has_chambres:
            st.info("Upside Revenu/GOP par hôtel non calculable — nombre de chambres réel non fourni.")
        else:
            upside_rows = []
            for r in cap_rows:
                hotel  = r['Hôtel']
                ft     = ft_pct
                slk_r  = dea.slacks.get(hotel,{}).get('outputs',{}).get('revpar', 0)
                lits   = float(dea.df.loc[hotel, 'nb_chambres'])
                up_rev = round(slk_r * lits * 365 / 1_000_000, 3)
                up_gop = round(up_rev * ft, 3)
                upside_rows.append({
                    'Hôtel': hotel, 'BCC': f"{dea.bcc_scores.get(hotel,0):.1%}",
                    'Classement': '★' * stars, 'FT% benchmark': f"{ft:.0%}",
                    'Slack RevPAR (€)': round(slk_r,1) if slk_r > 0 else '—',
                    'Upside Revenu brut (M€)': up_rev if up_rev > 0 else '—',
                    'Upside GOP /FT (M€)': up_gop if up_gop > 0 else '—',
                    'Priorité': ('🔴 Urgent' if up_gop>1.5 else '🟡 Moyen' if up_gop>0.5
                                 else '✅ RAS' if up_gop==0 else '🟢 Faible'),
                })
            upside_df = pd.DataFrame(upside_rows)
            upside_df['_sort'] = pd.to_numeric(upside_df['Upside GOP /FT (M€)'], errors='coerce')
            upside_df = upside_df.sort_values('_sort', ascending=False, na_position='last').drop(columns=['_sort'])
            st.dataframe(upside_df, use_container_width=True, hide_index=True)
    
    
        # ── DEA Capital vs DEA Opérationnel ─────────────────────────────────
        st.markdown('<p class="section-title">DEA Capital vs DEA Opérationnel</p>', unsafe_allow_html=True)
        st.caption(
            "Score opérationnel (BCC) = efficience de gestion. "
            "Score capital (DEA sur surface + CAPEX → GOP + CA) = efficience du capital immobilisé. "
            "Au-dessus de la diagonale : capital bien employé. "
            "En dessous : surcoût immobilier ou CAPEX mal alloué — signal de renégociation de bail ou révision plan CAPEX."
        )
    
        # Préparer les données capital depuis cap_input
        _has_cap_data = (cap_input[['surface_m2','capex_annuel (k€)']].sum().sum() > 0)
    
        if not _has_cap_data:
            st.info("Renseigner **surface_m2** et **capex_annuel** dans le tableau ci-dessus pour activer le DEA Capital.")
        else:
            # Construire un df temporaire pour compute_capital_dea
            _cap_df = dea.df.copy()
            for _hotel in dea.hotels:
                _row_cap = cap_input.loc[_hotel]
                if _row_cap['surface_m2'] > 0:
                    _cap_df.loc[_hotel, 'surface_m2'] = _row_cap['surface_m2']
                if _row_cap['capex_annuel (k€)'] > 0:
                    _cap_df.loc[_hotel, 'capex_annuel'] = _row_cap['capex_annuel (k€)'] * 1000
                if _row_cap['gop (k€)'] > 0:
                    _cap_df.loc[_hotel, 'gop'] = _row_cap['gop (k€)'] * 1000
    
            # Mettre à jour dea.df temporairement pour compute_capital_dea
            _dea_df_orig = dea.df.copy()
            dea.df = _cap_df
            dea.has_surface = 'surface_m2' in _cap_df.columns and _cap_df['surface_m2'].sum() > 0
            dea.has_capex   = 'capex_annuel' in _cap_df.columns and _cap_df['capex_annuel'].sum() > 0
            dea.has_gop     = 'gop' in _cap_df.columns and _cap_df['gop'].sum() > 0
    
            _cap_dea_df = dea.compute_capital_dea()
            dea.df = _dea_df_orig  # restaurer
    
            if _cap_dea_df is not None and not _cap_dea_df.empty:
                st.dataframe(_cap_dea_df, use_container_width=True, hide_index=True)
    
                # Scatter : DEA Opérationnel vs DEA Capital
                _op_vals  = [dea.bcc_scores.get(h, 0) for h in _cap_dea_df['Hôtel']]
                _cap_vals = [float(str(v).replace('%',''))/100 if isinstance(v, str) else v
                             for v in _cap_dea_df['DEA Capital']]
    
                fig_dea_cap = go.Figure()
                for i, h in enumerate(_cap_dea_df['Hôtel']):
                    _op  = _op_vals[i]
                    _cap = _cap_vals[i] if isinstance(_cap_vals[i], float) else 0.0
                    _lecture = str(_cap_dea_df.iloc[i]['Lecture'])
                    _color = ('#27ae60' if 'bien' in _lecture
                              else '#f39c12' if 'modéré' in _lecture
                              else '#e74c3c')
                    fig_dea_cap.add_trace(go.Scatter(
                        x=[_op], y=[_cap],
                        mode='markers', text=[h],
                        marker=dict(size=11, color=_color, opacity=0.85),
                        showlegend=False,
                        hovertemplate=f"<b>{h}</b><br>DEA Opérationnel : {_op:.1%}<br>DEA Capital : {_cap:.1%}<br>{_lecture}<extra></extra>",
                    ))
    
                # Diagonale
                fig_dea_cap.add_shape(type='line', x0=0.3, y0=0.3, x1=1.0, y1=1.0,
                                      line=dict(dash='dash', color='gray', width=1))
                fig_dea_cap.add_annotation(x=0.95, y=0.97, text="Au-dessus = capital bien employé",
                                           showarrow=False, font=dict(size=9, color='#27ae60'))
                fig_dea_cap.add_annotation(x=0.95, y=0.60, text="En dessous = surcoût capital",
                                           showarrow=False, font=dict(size=9, color='#e74c3c'))
    
                fig_dea_cap.update_layout(
                    title="DEA Capital vs DEA Opérationnel — Efficience immobilière",
                    xaxis=dict(title="DEA Opérationnel (BCC)", range=[0.3, 1.08], tickformat='.0%'),
                    yaxis=dict(title="DEA Capital", range=[0.3, 1.08], tickformat='.0%'),
                    height=480, paper_bgcolor='rgba(0,0,0,0)',
                )
                st.plotly_chart(fig_dea_cap, use_container_width=True)
    
                # Alertes
                _critiques = _cap_dea_df[_cap_dea_df['Lecture'].str.contains('sous-productif', na=False)]
                if not _critiques.empty:
                    st.error(
                        f"**Capital sous-productif détecté :** "
                        + ", ".join(_critiques['Hôtel'].tolist())
                        + " — Signal de renégociation de bail ou révision du plan CAPEX."
                    )
            else:
                st.info("Données capital insuffisantes pour le calcul DEA Capital (surface + CAPEX + revenus requis).")
    
    
    # ── MDEA Room / F&B Decomposition (Yu 2012) ─────────────────────────
    st.markdown("---")
    st.markdown('<p class="section-title">🏨 Décomposition Room / F&B — MDEA (Yu 2012)</p>', unsafe_allow_html=True)
    st.caption(
        "Yu, M.-M. (2012) Current Issues in Tourism 15(5), 461-476. "
        "Décompose l'efficience globale en efficience Hébergement et efficience F&B. "
        "Localise la source d'inefficience par département. "
        "Nécessite : rooms_cost + rooms_revenue + fb_cost + fb_revenue dans le CSV."
    )

    if st.button("Calculer décomposition Room / F&B", key="mdea_rf_btn"):
        with st.spinner("Calcul MDEA Room/F&B..."):
            try:
                _rf = compute_mdea_room_fb(dea)
                st.session_state["mdea_rf_results"] = _rf
            except Exception as _e:
                st.error(f"Erreur MDEA : {_e}")

    _rf = st.session_state.get("mdea_rf_results")
    if _rf is not None:
        if not _rf["feasible"]:
            st.warning(
                f"Colonnes manquantes pour MDEA Room/F&B : "
                f"{', '.join(_rf['missing_cols'])}. "
                "Ajoutez rooms_cost, rooms_revenue, fb_cost, fb_revenue au CSV."
            )
        else:
            if _rf["shared_used"]:
                st.success(f"Inputs partagés détectés : {', '.join(_rf['shared_used'])}")

            # Tableau scores
            st.dataframe(
                _rf["scores"][["Hôtel","BCC Global","BCC Room","BCC F&B","Δ (F&B - Room)","Source inefficience"]],
                use_container_width=True, hide_index=True
            )

            # Summary stats
            col_rf1, col_rf2 = st.columns(2)
            with col_rf1:
                st.markdown("**Statistiques par division**")
                st.dataframe(_rf["summary"], use_container_width=True, hide_index=True)
            with col_rf2:
                if _rf["mw_test"]:
                    st.markdown("**Mann-Whitney Room vs F&B**")
                    st.info(_rf["mw_test"]["conclusion"])
                    st.caption(_rf["mw_test"]["note"])

            # Scatter BCC Room vs BCC F&B
            _sc = _rf["scores"].copy()
            _sc["_r"] = pd.to_numeric(_sc["BCC Room"], errors="coerce")
            _sc["_f"] = pd.to_numeric(_sc["BCC F&B"], errors="coerce")
            _sc = _sc.dropna(subset=["_r","_f"])

            fig_rf = go.Figure()
            for _, row_rf in _sc.iterrows():
                _col_rf = ("#27ae60" if "Équilibré" in row_rf["Source inefficience"]
                           else "#e74c3c" if "Double" in row_rf["Source inefficience"]
                           else "#f39c12")
                fig_rf.add_trace(go.Scatter(
                    x=[row_rf["_r"]], y=[row_rf["_f"]],
                    mode="markers", text=[row_rf["Hôtel"]],
                    marker=dict(size=11, color=_col_rf), showlegend=False,
                    hovertemplate=f"<b>{row_rf['Hôtel']}</b><br>Room: {row_rf['_r']:.3f}<br>F&B: {row_rf['_f']:.3f}<extra></extra>",
                ))
            # Diagonale
            fig_rf.add_shape(type="line", x0=0.5, y0=0.5, x1=1.0, y1=1.0,
                             line=dict(dash="dot", color="gray"))
            fig_rf.add_annotation(x=0.97, y=0.97, text="Room = F&B",
                                   showarrow=False, font=dict(size=9, color="gray"))
            fig_rf.add_annotation(x=0.65, y=0.95, text="F&B > Room",
                                   showarrow=False, font=dict(size=9, color="#27ae60"))
            fig_rf.add_annotation(x=0.95, y=0.65, text="Room > F&B",
                                   showarrow=False, font=dict(size=9, color="#f39c12"))
            fig_rf.update_layout(
                title="BCC Room vs BCC F&B — Localisation de l'inefficience (Yu 2012)",
                xaxis=dict(title="BCC Hébergement (Room)", range=[0.5,1.05]),
                yaxis=dict(title="BCC Restauration (F&B)", range=[0.5,1.05]),
                height=420, paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_rf, use_container_width=True)

            st.info(
                "Au-dessus de la diagonale : F&B plus efficient que Hébergement. "
                "En dessous : Hébergement plus efficient. "
                "Sur la diagonale : performance équilibrée entre les deux divisions. "
                "Source : Yu (2012) — l'efficience globale = moyenne pondérée Room + F&B."
            )

            # Export CSV
            _csv_rf = _rf["scores"].to_csv(index=False, sep=";", decimal=",")
            st.download_button(
                "⬇️ Exporter décomposition Room/F&B (CSV)",
                data=_csv_rf.encode("utf-8-sig"),
                file_name="deah_mdea_room_fb.csv", mime="text/csv",
            )

    # Expense Flex (Russo & Legel p.33) — requiert le tableau capital renseigné
    if not cap_rows:
        st.markdown("---")
        st.info(
            "💡 **Expense Flex & Flow Through** — renseignez surface, CAPEX et GOP "
            "dans le tableau ci-dessus pour débloquer cette analyse."
        )
    else:
        st.markdown('<p class="section-title">Expense Flex & Flow Through</p>', unsafe_allow_html=True)
        st.caption('Flow Through = delta_GOP / delta_CA | Expense Flex = 1 - FT quand CA baisse | Cible standard 50%')

        # ── Source de la baseline N-1 ────────────────────────────────────────
        # Priorité 1 : colonnes revpar_n1 / gop_n1 par hôtel (précis)
        # Priorité 2 : saisie globale unique (approximatif, portefeuille homogène)
        _has_rev_n1 = 'revpar_n1' in dea.df.columns and dea.df['revpar_n1'].fillna(0).sum() > 0
        _has_gop_n1 = 'gop_n1'    in dea.df.columns and dea.df['gop_n1'].fillna(0).sum() > 0
        _n1_mode    = _has_rev_n1 and _has_gop_n1

        if _n1_mode:
            st.success(
                "✅ Baseline N-1 lue par hôtel (`revpar_n1`, `gop_n1`) — "
                "Flow Through calculé individuellement."
            )
            base_revpar_ft, base_gop_pct_ft = 0.0, 0.0
        else:
            _manque = []
            if not _has_rev_n1: _manque.append('`revpar_n1`')
            if not _has_gop_n1: _manque.append('`gop_n1`')
            st.warning(
                f"⚠️ Colonne(s) manquante(s) : {', '.join(_manque)}. "
                "Baseline unique appliquée à tous les hôtels — approximation "
                "grossière si les RevPAR sont hétérogènes."
            )
            col_fx1, col_fx2 = st.columns(2)
            with col_fx1:
                base_revpar_ft = st.number_input('RevPAR baseline N-1 ou Budget (€)', value=0.0, step=1.0, key='ft_revpar')
            with col_fx2:
                base_gop_pct_ft = st.number_input('Marge GOP% baseline (%)', value=35.0, step=0.5, key='ft_gop')

        flex_rows = []
        if not dea.has_chambres:
            st.info("Flow Through par hôtel non calculable — nombre de chambres réel non fourni.")
        else:
          for r in cap_rows:
            hotel_ft = r['Hôtel']
            stars_ft = int(cap_input.loc[hotel_ft, 'classement (★)']) if 'classement (★)' in cap_input.columns and hotel_ft in cap_input.index else 3
            ft_bench_ft = ft_pct
            lits_ft  = float(dea.df.loc[hotel_ft, 'nb_chambres'])
            revpar_ft = float(dea.df.loc[hotel_ft, 'revpar'])
            # CA = RevPAR × lits × jours (RevPAR intègre déjà l'occupation)
            ca_ft    = revpar_ft * lits_ft * jours_exploit

            gop_h_ft = (float(cap_input.loc[hotel_ft, 'gop (k€)']) * 1000
                        if 'gop (k€)' in cap_input.columns
                        and float(cap_input.loc[hotel_ft, 'gop (k€)']) > 0 else None)

            ca_base_ft = gop_base_ft = None
            if _n1_mode:
                _rv1 = float(dea.df.loc[hotel_ft, 'revpar_n1'] or 0)
                _gp1 = float(dea.df.loc[hotel_ft, 'gop_n1']    or 0)
                if _rv1 > 0 and _gp1 > 0:
                    ca_base_ft  = _rv1 * lits_ft * jours_exploit
                    gop_base_ft = _gp1 * 1000          # k€ → €
            elif base_revpar_ft > 0 and base_gop_pct_ft > 0:
                ca_base_ft  = base_revpar_ft * lits_ft * jours_exploit
                gop_base_ft = ca_base_ft * base_gop_pct_ft / 100

            if ca_base_ft is not None and gop_h_ft is not None:
                delta_ca_ft = ca_ft - ca_base_ft
                if abs(delta_ca_ft) > 1e-6:
                    ft_val   = round((gop_h_ft - gop_base_ft) / delta_ca_ft, 3)
                    flex_val = round(1 - ft_val, 3) if delta_ca_ft < 0 else None
                    src      = 'Calculé N-1' if _n1_mode else 'Calculé (global)'
                else:
                    ft_val = ft_bench_ft; flex_val = None; src = 'Benchmark'
            else:
                ft_val = ft_bench_ft; flex_val = None; src = 'Benchmark'
    
            slack_r_ft = dea.slacks.get(hotel_ft, {}).get('outputs', {}).get('revpar', 0)
            driver_ft = 'Rate-driven' if slack_r_ft > 0 else 'Volume-driven'
            if isinstance(ft_val, float):
                if ft_val >= 0.60: ftq = '✅ Excellent'
                elif ft_val >= 0.45: ftq = '🟡 Correct'
                elif ft_val >= 0.30: ftq = '🟠 Faible'
                else: ftq = '🔴 Tres faible'
            else: ftq = '--'
    
            flex_rows.append({
                'Hôtel'          : hotel_ft,
                'BCC'            : f"{dea.bcc_scores.get(hotel_ft,0):.1%}",
                'Classement'     : '★' * stars_ft,
                'Flow Through %' : f"{ft_val:.1%}" if isinstance(ft_val, float) else '--',
                'Source'         : src,
                'Qualite FT'     : ftq,
                'Expense Flex %' : f"{flex_val:.1%}" if flex_val is not None else '--',
                'Cible std'      : '50%',
                'Ecart cible'    : f"{(ft_val - 0.50):+.1%}" if isinstance(ft_val, float) else '--',
                'Driver revenu'  : driver_ft if src.startswith('Calculé') else '--',
            })
    
        st.dataframe(pd.DataFrame(flex_rows), use_container_width=True, hide_index=True)
        st.info('FT > 50% = bonne conversion revenus -> profit. FT < 50% = charges variables elevees. Expense Flex calcule quand CA baisse.')

# TAB 10 — MALMQUIST & TOBIT
# ══════════════════════════════════════════════
with tab_malm:
    st.markdown('<p class="section-title">📈 Malmquist Productivity Index — Évolution temporelle</p>', unsafe_allow_html=True)
    st.caption(
        "Décompose la variation de productivité entre N-1 et N en deux effets : "
        "Catch-up (mérite de l\'opérateur) × Frontier Shift (progrès du marché). "
        "Voir documentation méthodologique pour les références complètes."
    )

    # Vérifier colonnes N-1
    _n1_present = has_n1_cols(dea.df)
    _n1_required = ["revpar_n1", "nb_employes_n1", "couts_op_ex_n1",
                    "taux_occupation_n1"]
    _n1_missing  = [c for c in _n1_required if c not in dea.df.columns]

    if _n1_missing:
        _warn_msg = (
            f"Colonnes N-1 manquantes : {', '.join(_n1_missing)}. "
            "Ajoutez nb_employes_n1, couts_op_ex_n1, "
            "revpar_n1, taux_occupation_n1 (+ satisfaction_n1 optionnel) "
            "a votre CSV pour activer le Malmquist."
        )
        st.warning(_warn_msg)
    else:
        st.success(f"✅ Colonnes N-1 détectées : {', '.join(_n1_present)}")

        # ── Déflateur optionnel (biais d'inflation) ─────────────────────────
        with st.expander("⚙️ Correction inflation (optionnel — recommandé)"):
            st.caption(
                "L'inflation sur les charges op' et le RevPAR peut simuler un Frontier Shift "
                "artificiel. Renseignez le taux d'inflation entre N-1 et N pour corriger. "
                "Réf. : Färe, Grosskopf, Norris & Zhang (1994)."
            )
            _deflate_on = st.checkbox("Appliquer la correction inflation", value=False)
            _inflation_rate = st.number_input(
                "Taux d'inflation N-1→N (%)",
                min_value=0.0, max_value=20.0, value=3.0, step=0.1,
                format="%.1f",
                help="Ex : 3.0% pour corriger un Frontier Shift en données nominales. "
                     "Source : INSEE (France), INE (Espagne), ONS (UK)."
            ) / 100.0 if _deflate_on else 0.0

        if _deflate_on and _inflation_rate > 0:
            st.info(
                f"✅ Correction inflation activée : {_inflation_rate*100:.1f}% — "
                "les variables monétaires N-1 (revpar_n1, couts_op_ex_n1) seront "
                "déflatées avant le calcul Malmquist."
            )

        if st.button("🔄 Calculer Malmquist", key="mq_btn"):
            with st.spinner("Calcul des 4 problèmes DEA par DMU..."):
                try:
                    _mq = compute_malmquist(dea)
                    st.session_state["malmquist_results"] = _mq
                    if _mq is not None:
                        st.success(f"✅ Malmquist calculé — {len(_mq)} DMUs")
                except Exception as _e:
                    st.error(f"Erreur : {_e}")

        if st.session_state.get("malmquist_results") is not None:
            _mq = st.session_state["malmquist_results"]
            _mq_display_cols = ['Hôtel', 'BCC N-1', 'BCC N', 'Catch-up',
                                 'Frontier Shift', 'Malmquist TFP', 'Catégorie', 'Interprétation']
            _mq_display_cols = [c for c in _mq_display_cols if c in _mq.columns]
            st.dataframe(_mq[_mq_display_cols], use_container_width=True, hide_index=True)

            # ── Résumé Alpha — qui a créé de la valeur propre ? ──────────────
            st.markdown('<p class="section-title">Synthèse — Alpha opérateur vs Marée montante</p>', unsafe_allow_html=True)
            st.caption(
                "**Alpha opérateur** = Catch-up > 1 — progrès dû au management, indépendamment du marché. "
                "**Marée montante** = TFP > 1 uniquement via Frontier Shift — l\'opérateur a suivi la vague "
                "sans créer de valeur propre. Voir documentation méthodologique pour les références."
            )
            _mq_s2 = _mq[_mq['Malmquist TFP'] != '—'].copy()
            for _c in ['Catch-up','Frontier Shift','Malmquist TFP']:
                _mq_s2[_c] = pd.to_numeric(_mq_s2[_c], errors='coerce')
            _n_alpha_op   = (_mq_s2['Catch-up'] > 1.0).sum()
            _n_vague_only = ((_mq_s2['Malmquist TFP'] > 1.0) & (_mq_s2['Catch-up'] <= 1.0)).sum()
            _n_regress    = (_mq_s2['Malmquist TFP'] < 1.0).sum()
            _n_tot        = len(_mq_s2)
            _sa1, _sa2, _sa3 = st.columns(3)
            _sa1.metric("💼 Alpha opérateur",      f"{_n_alpha_op}/{_n_tot}",
                        help="Catch-up > 1 — progrès managérial réel, indépendant du marché")
            _sa2.metric("🌊 Marée montante seule", f"{_n_vague_only}/{_n_tot}",
                        help="TFP > 1 uniquement via Frontier Shift — pas de valeur propre créée par l\'opérateur")
            _sa3.metric("📉 En régression",         f"{_n_regress}/{_n_tot}",
                        help="TFP < 1 — recul de productivité globale")

            if 'Catégorie' in _mq_s2.columns and not _mq_s2.empty:
                _color_alpha = {
                    'Alpha total'    : '#1e8449',
                    'Alpha pur'      : '#2e6da4',
                    'Marée montante' : '#f39c12',
                    'Résistance'     : '#8e44ad',
                    'Régression'     : '#c0392b',
                }
                _mq_sorted = _mq_s2.sort_values('Malmquist TFP', ascending=True)
                _bcolors   = [_color_alpha.get(str(s), '#888') for s in _mq_sorted['Catégorie']]
                fig_alpha  = go.Figure(go.Bar(
                    x=_mq_sorted['Malmquist TFP'], y=_mq_sorted['Hôtel'],
                    orientation='h', marker_color=_bcolors, opacity=0.85,
                    text=[f"{v:.3f}" for v in _mq_sorted['Malmquist TFP']],
                    textposition='outside',
                ))
                fig_alpha.add_vline(x=1.0, line_dash='dash', line_color='gray',
                                    annotation_text="TFP = 1")
                fig_alpha.update_layout(
                    title="Malmquist TFP coloré par source de progrès",
                    xaxis_title="TFP (> 1 = progrès · < 1 = régression)",
                    height=max(350, _n_tot * 30),
                    paper_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=40, r=60, t=40, b=40),
                )
                st.plotly_chart(fig_alpha, use_container_width=True)

            # Graphiques
            _mq_num = _mq[_mq["Malmquist TFP"] != "—"].copy()
            for c in ["Catch-up", "Frontier Shift", "Malmquist TFP"]:
                _mq_num[c] = pd.to_numeric(_mq_num[c], errors="coerce")

            if not _mq_num.empty:
                _mc1, _mc2 = st.columns(2)

                with _mc1:
                    # Barres TFP
                    _mq_s = _mq_num.sort_values("Malmquist TFP")
                    fig_mq = go.Figure(go.Bar(
                        x=_mq_s["Malmquist TFP"], y=_mq_s["Hôtel"], orientation="h",
                        marker=dict(
                            color=_mq_s["Malmquist TFP"],
                            colorscale=[[0,"#e74c3c"],[0.5,"#f1c40f"],[1.0,"#2ecc71"]],
                            cmin=0.8, cmax=1.2, showscale=True,
                            colorbar=dict(title="TFP"),
                        ),
                        text=[f"{v:.3f}" for v in _mq_s["Malmquist TFP"]],
                        textposition="outside",
                    ))
                    fig_mq.add_vline(x=1.0, line_dash="dash", line_color="#27ae60",
                                     annotation_text="TFP = 1 (stable)")
                    fig_mq.update_layout(
                        title="Malmquist TFP (> 1 = progrès, < 1 = régression)",
                        height=max(350, len(_mq_num)*28),
                        paper_bgcolor="rgba(0,0,0,0)",
                    )
                    st.plotly_chart(fig_mq, use_container_width=True)

                with _mc2:
                    # Scatter Catch-up vs Frontier Shift
                    fig_decomp = go.Figure()
                    for _, r in _mq_num.iterrows():
                        _cu = r["Catch-up"]; _fs = r["Frontier Shift"]
                        _col = ("#27ae60" if _cu >= 1 and _fs >= 1
                                else "#e74c3c" if _cu < 1 and _fs < 1
                                else "#f39c12")
                        fig_decomp.add_trace(go.Scatter(
                            x=[_cu], y=[_fs], mode="markers",
                            text=[r["Hôtel"]],
                            marker=dict(size=11, color=_col), showlegend=False,
                            hovertemplate=(f"<b>{r['Hôtel']}</b><br>"
                                          f"Catch-up: {_cu:.3f}<br>"
                                          f"Frontier Shift: {_fs:.3f}<br>"
                                          f"TFP: {r['Malmquist TFP']:.3f}<extra></extra>"),
                        ))
                    # Quadrants
                    fig_decomp.add_hline(y=1, line_dash="dot", line_color="gray")
                    fig_decomp.add_vline(x=1, line_dash="dot", line_color="gray")
                    fig_decomp.add_annotation(x=1.08, y=1.08, text="Progrès total ✅",
                        showarrow=False, font=dict(size=9, color="#27ae60"))
                    fig_decomp.add_annotation(x=0.93, y=0.93, text="Régression totale 🔴",
                        showarrow=False, font=dict(size=9, color="#e74c3c"))
                    fig_decomp.add_annotation(x=0.93, y=1.08,
                        text="Progrès sectoriel (frontière avance)", showarrow=False,
                        font=dict(size=8, color="#3498db"))
                    fig_decomp.add_annotation(x=1.08, y=0.93,
                        text="Rattrapage (convergence)", showarrow=False,
                        font=dict(size=8, color="#f39c12"))
                    fig_decomp.update_layout(
                        title="Décomposition : Catch-up × Frontier Shift",
                        xaxis=dict(title="Catch-up (efficience)", zeroline=False),
                        yaxis=dict(title="Frontier Shift (technologie)", zeroline=False),
                        height=400, paper_bgcolor="rgba(0,0,0,0)",
                    )
                    st.plotly_chart(fig_decomp, use_container_width=True)

                # Résumé portfolio
                st.markdown("---")
                _mc3, _mc4, _mc5 = st.columns(3)
                _tfp_vals = _mq_num["Malmquist TFP"]
                _mc3.metric("TFP moyen portfolio", f"{_tfp_vals.mean():.3f}",
                            help="< 1 = régression globale | > 1 = progrès global")
                _mc4.metric("DMUs en progrès (TFP>1)",
                            f"{(_tfp_vals > 1).sum()}/{len(_tfp_vals)}")
                _mc5.metric("Frontier Shift moyen",
                            f"{_mq_num['Frontier Shift'].mean():.3f}",
                            help="Progrès sectoriel indépendant de la gestion")

                st.info(
                    "Catch-up > 1 : hotel rattrape la frontiere = merite de gestion. "
                    "Frontier Shift > 1 : frontiere avance = progres sectoriel. "
                    "TFP = Catch-up x Frontier Shift. "
                    "Cas optimal : TFP > 1 ET Catch-up > 1 = progres propre independant du marche."
                )

                # Export CSV
                _csv_mq = _mq.to_csv(index=False, sep=";", decimal=",")
                st.download_button(
                    "⬇️ Exporter Malmquist (CSV)", data=_csv_mq.encode("utf-8-sig"),
                    file_name="deah_malmquist.csv", mime="text/csv",
                )



    st.markdown('---')
    st.markdown('---')
    st.markdown('''<p class="section-title">🧪 Stage 2 — Déterminants de l'Efficience</p>''', unsafe_allow_html=True)
    st.caption(
        "Deux estimateurs complémentaires : Tobit censuré (Tobin 1958) pour la lisibilité, "
        "Simar-Wilson (2007) pour la robustesse statistique. "
        "Variables financières et de saisonnalité issues de Pulina & Santoni (2018)."
    )

    # ── Calcul variables dérivées Stage 2 ────────────────────────────────────
    _s2_added = build_stage2_vars(dea)

    # Sélection des régresseurs disponibles
    _tobit_candidates = {
        # Structurelles
        'classement_etoiles': 'Classement (★)',
        'nb_chambres'           : 'Nombre de chambres',
        'log_nb_chambres'       : 'Taille — log(chambres)',
        'surface_m2'        : 'Surface totale (m²)',
        # Financières — Pulina & Santoni (2018)
        'ltv_proxy'         : 'Intensité capital / CA (proxy LTV %)',
        'asset_yield'       : 'Rendement actifs CA/BV (proxy ROA)',
        'capex_per_room'    : 'CAPEX par chambre (k€)',
        'gop_margin_pct'    : 'Marge GOP (%)',
        'capex_annuel'      : 'CAPEX annuel total (k€)',
        # Saisonnalité
        'saison_dummy'      : 'Saisonnalité (resort=1 / urbain=0)',
        # ESG / opérationnel
        'energy_kwh'        : 'Énergie consommée (kWh)',
        'payroll_total'     : 'Masse salariale (k€)',
    }
    _tobit_avail = {c: l for c, l in _tobit_candidates.items() if c in dea.df.columns}

    # Variables du profil compset (session state Tab 10)
    _cs_profile = st.session_state.get('compset_profile', pd.DataFrame())
    _tobit_cs = {}
    if not _cs_profile.empty:
        if 'Localisation' in _cs_profile.columns:
            _loc_dummies = pd.get_dummies(_cs_profile['Localisation'], prefix='loc')
            for col in _loc_dummies.columns:
                dea.df[col] = [_loc_dummies.loc[h, col] if h in _loc_dummies.index else 0
                               for h in dea.hotels]
                _tobit_cs[col] = col.replace('loc_', 'Loc. ')
        if 'Affiliation' in _cs_profile.columns:
            _aff_dummies = pd.get_dummies(_cs_profile['Affiliation'], prefix='aff')
            for col in _aff_dummies.columns:
                dea.df[col] = [_aff_dummies.loc[h, col] if h in _aff_dummies.index else 0
                               for h in dea.hotels]
                _tobit_cs[col] = col.replace('aff_', 'Affiliation: ')

    _all_tobit = {**_tobit_avail, **_tobit_cs}

    if not _all_tobit:
        st.info("Aucune variable environnementale disponible. Enrichissez le CSV avec `classement_etoiles`, `surface_m2`, etc. ou remplissez le Profil Compset (Tab 10).")
    else:
        _selected_env = st.multiselect(
            "Variables environnementales (régresseurs)",
            options=list(_all_tobit.keys()),
            default=list(_all_tobit.keys())[:min(3, len(_all_tobit))],
            format_func=lambda x: _all_tobit.get(x, x),
            key='tobit_vars',
        )

        if _selected_env and st.button("Estimer modèle Tobit", key="tobit_btn"):
            if len(_selected_env) >= len(dea.hotels) - 2:
                st.error("Trop de régresseurs pour le nombre de DMUs. Réduisez la sélection.")
            else:
                with st.spinner("Estimation MLE Tobit..."):
                    try:
                        _tb = compute_tobit(
                            dea,
                            env_vars=_selected_env,
                            env_labels=_all_tobit,
                        )
                        st.session_state["tobit_results"] = _tb
                    except Exception as _e:
                        st.error(f"Erreur Tobit : {_e}")

        if st.session_state.get("tobit_results") and "error" not in st.session_state["tobit_results"]:
            _tb = st.session_state["tobit_results"]
            _tb_col1, _tb_col2 = st.columns([3, 1])
            with _tb_col1:
                st.markdown("**Résultats de la régression Tobit**")
                st.dataframe(_tb["coef_df"], use_container_width=True, hide_index=True)
                st.caption(
                    "Sig. : *** p<0.01  ** p<0.05  * p<0.10  — "
                    "Coefficients interprétés en variation du score BCC (0 à 1) "
                    "toutes choses égales par ailleurs."
                )
            with _tb_col2:
                st.metric("N observations", _tb["n"])
                st.metric("N censurées (BCC=1)", _tb["n_censored"])
                st.metric("Log-vraisemblance", _tb["log_lik"])
                st.metric("σ (bruit)", _tb["sigma"])
                st.metric("Convergence", "✅ Oui" if _tb["converged"] else "⚠️ Non")

            # Bar chart des coefficients significatifs
            _sig_coef = _tb["coef_df"][_tb["coef_df"]["Sig."] != ""].copy()
            if not _sig_coef.empty and len(_sig_coef) > 1:
                _sig_coef = _sig_coef[_sig_coef["Variable"] != "Constante"]
                if not _sig_coef.empty:
                    _coef_vals = pd.to_numeric(_sig_coef["Coeff."], errors="coerce")
                    fig_tobit = go.Figure(go.Bar(
                        x=_sig_coef["Variable"],
                        y=_coef_vals,
                        marker=dict(color=["#27ae60" if v > 0 else "#e74c3c"
                                           for v in _coef_vals]),
                        text=[f"{v:+.3f}{s}" for v, s in zip(_coef_vals, _sig_coef["Sig."])],
                        textposition="outside",
                    ))
                    fig_tobit.add_hline(y=0, line_color="gray", line_width=1)
                    fig_tobit.update_layout(
                        title="Coefficients Tobit significatifs — impact sur BCC",
                        yaxis=dict(title="Effet marginal sur score BCC", zeroline=True),
                        height=350, paper_bgcolor="rgba(0,0,0,0)",
                    )
                    st.plotly_chart(fig_tobit, use_container_width=True)

            st.info(
                "**Lecture :** Un coefficient de +0.08 sur Classement(★) signifie que "
                "chaque étoile supplémentaire améliore le score BCC de 0.08 points "
                "toutes choses égales par ailleurs (ETP, charges, localisation inchangés). "
                "Le Tobit corrige le biais d'estimation lié au plafond BCC=1.0 "
                "(Simar & Wilson 2007)."
            )
        elif st.session_state.get("tobit_results") and "error" in st.session_state["tobit_results"]:
            st.error(st.session_state["tobit_results"]["error"])

    # ── SIMAR-WILSON (2007) ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('''<p class="section-title">📐 Simar-Wilson (2007) — Régression Tronquée Bootstrappée</p>''', unsafe_allow_html=True)

    with st.expander("ℹ️ Différence Tobit vs Simar-Wilson", expanded=False):
        st.markdown("""
| | **Tobit censuré** | **Simar-Wilson tronqué** |
|---|---|---|
| DMUs efficients (θ=1) | Inclus — censurés via P(Y≥1) | **Exclus** de l'estimation |
| Biais correlation scores DEA | Non corrigé | **Corrigé** par bootstrap |
| Standard errors | Hessien numérique | **Bootstrap paramétrique** B=200 |
| IC 95% | Non disponible | **Percentiles 2.5/97.5** |
| Usage recommandé | Lecture rapide, interprétation | **Publication, validation** |

*Réf. : Simar & Wilson (2007) Journal of Econometrics 136(1), 31-64*
        """)

    _sw_selected = st.multiselect(
        "Variables environnementales — Simar-Wilson",
        options=list(_all_tobit.keys()),
        default=list(_all_tobit.keys())[:min(4, len(_all_tobit))],
        format_func=lambda x: _all_tobit.get(x, x),
        key='sw_vars',
    )
    _sw_boot = st.slider("Itérations bootstrap (B)", 100, 500, 200, 50, key='sw_boot')

    if _sw_selected and st.button("⚙️ Estimer Simar-Wilson", key="sw_btn"):
        if len(_sw_selected) >= len(dea.hotels) - 2:
            st.error("Trop de régresseurs. Réduisez la sélection.")
        else:
            with st.spinner(f"Bootstrap tronqué — {_sw_boot} itérations…"):
                try:
                    _sw = compute_simar_wilson(
                        dea,
                        env_vars=_sw_selected,
                        env_labels=_all_tobit,
                        n_bootstrap=_sw_boot,
                    )
                    st.session_state["sw_results"] = _sw
                except Exception as _e_sw:
                    st.error(f"Erreur Simar-Wilson : {_e_sw}")

    if st.session_state.get("sw_results") and "error" not in st.session_state["sw_results"]:
        _sw = st.session_state["sw_results"]
        _sw_c1, _sw_c2 = st.columns([3, 1])
        with _sw_c1:
            st.markdown("**Résultats — Régression tronquée bootstrappée**")
            st.dataframe(_sw["coef_df"], use_container_width=True, hide_index=True)
            st.caption(
                "SE Bootstrap : écart-type sur B itérations bootstrap. "
                "IC95% : percentiles 2.5/97.5 du bootstrap. "
                "Sig. : *** p<0.01  ** p<0.05  * p<0.10"
            )
        with _sw_c2:
            st.metric("N total", _sw["n"])
            st.metric("N inefficients utilisés", _sw["n_inefficients"])
            st.metric("Bootstrap convergés", _sw["n_bootstrap"])
            st.metric("Log-vraisemblance", _sw["log_lik"])
            st.metric("σ (bruit)", _sw["sigma"])
            st.metric("Convergence MLE", "✅ Oui" if _sw["converged"] else "⚠️ Non")

        # Bar chart coefficients significatifs S-W
        _sw_sig = _sw["coef_df"][_sw["coef_df"]["Sig."] != ""].copy()
        _sw_sig = _sw_sig[_sw_sig["Variable"] != "Constante"]
        if not _sw_sig.empty:
            _sw_vals = pd.to_numeric(_sw_sig["Coeff."], errors="coerce")
            _sw_lo   = pd.to_numeric(_sw_sig["IC95% Lo"], errors="coerce")
            _sw_hi   = pd.to_numeric(_sw_sig["IC95% Hi"], errors="coerce")
            fig_sw = go.Figure()
            fig_sw.add_trace(go.Bar(
                x=_sw_sig["Variable"], y=_sw_vals,
                marker=dict(color=["#27ae60" if v > 0 else "#e74c3c" for v in _sw_vals]),
                error_y=dict(
                    type='data', symmetric=False,
                    array=(_sw_hi - _sw_vals).tolist(),
                    arrayminus=(_sw_vals - _sw_lo).tolist(),
                    color='rgba(0,0,0,0.4)', thickness=2,
                ),
                text=[f"{v:+.3f}{s}" for v, s in zip(_sw_vals, _sw_sig["Sig."])],
                textposition="outside",
            ))
            fig_sw.add_hline(y=0, line_color="gray", line_width=1)
            fig_sw.update_layout(
                title="Coefficients Simar-Wilson significatifs + IC95% bootstrap",
                yaxis=dict(title="Effet marginal sur BCC", zeroline=True),
                height=380, paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_sw, use_container_width=True)

        st.info(
            "**Lecture investisseur :** Un coefficient négatif sur *Saisonnalité* confirme que "
            "les actifs resort sont structurellement moins efficients (sous-utilisation hors saison). "
            "Un coefficient positif sur *Rendement actifs* valide que les hôtels générant plus "
            "de CA par € immobilisé atteignent une meilleure efficience opérationnelle. "
            "Ces signaux guident la décision de repositionnement ou de cession."
        )

    elif st.session_state.get("sw_results") and "error" in st.session_state["sw_results"]:
        st.error(st.session_state["sw_results"]["error"])

    if "sw_results" not in st.session_state:
        st.session_state["sw_results"] = None


# TAB 11 — SYNTHÈSE MULTI-MODULE (v3.2)
# ══════════════════════════════════════════════
with tab_synth:
    _module_results = st.session_state.get("module_results", {})

    if not _module_results:
        st.info("""
**🔀 Synthèse Multi-Module DEA**

Aucun résultat disponible pour l'instant.

**Pour activer cette analyse :**
1. Dans la sidebar, sélectionner les **modules DEA** souhaités (section *🔬 Modules DEA*)
2. Cliquer sur **🚀 LANCER L'ANALYSE DEA COMPLÈTE**
3. Revenir sur cet onglet

**Données disponibles avec le CSV actuel :**
""")
        # Afficher la couverture des modules avec le dataset courant
        _available_now = [c for c in _df_mm.columns if c != "hotel_name"]
        cov_cols = st.columns(min(len(MODULES), 4))
        for i, (mod_id, mod_cfg) in enumerate(MODULES.items()):
            f = check_module_feasibility(mod_id, _available_now, proxy_cols=_proxy_cols)
            with cov_cols[i % len(cov_cols)]:
                if f["feasible"]:
                    st.success(f"**{mod_cfg['label_fr']}**\n\n✅ {f['coverage_pct']:.0f}% couverture")
                else:
                    st.error(f"**{mod_cfg['label_fr']}**\n\n❌ Manque : {', '.join(f['missing_required'][:2])}")

        st.markdown("---")
        st.markdown("""
**💡 Pour activer tous les modules**, enrichissez votre CSV avec les colonnes standard :

| Colonne | Description |
|---------|-------------|
| `nb_rooms` | Nombre de chambres (lits DEA-H utilisés par défaut si absent) |
| `fte_total` | ETP totaux |
| `total_revenue` | CA total (€) |
| `gop` | GOP (€) |
| `rooms_revenue` | CA Hébergement (€) |
| `rooms_cost` | Coûts dept. Hébergement (€) |
| `payroll_total` | Masse salariale (€) |
| `book_value_assets` | Valeur comptable actifs (€) |
| `marketing_cost` | Dépenses marketing (€) |
| `trevpar` | TRevPAR (€/chambre) |
| `satisfaction_score` | Score NPS/Google/TA |
| `maintenance_cost` | Coûts maintenance (€) |
| `energy_kwh` | Consommation énergie |
| `water_m3` | Consommation eau |

*Note : les colonnes DEA-H standard (`nb_chambres`, `nb_employes`, `couts_op_ex`, `revpar`, `taux_occupation`, `satisfaction`)
sont automatiquement mappées vers les noms standard des modules.*
""")
    else:
        if illustrative_data:
            st.error(
                "⚠️ **Données illustratives** — les résultats affichés ci-dessous reposent "
                "en tout ou partie sur des estimations, pas sur des données vérifiées par "
                "le client. Les scores ne doivent pas être présentés comme des résultats "
                "définitifs."
            )
        render_synthesis_tab(_module_results, dmu_col="hotel_name")

        # ── Diagnostic croisé multi-dimensions ──────────────────────────────────
        # Le BCC de base (opérationnel : chambres/ETP/charges → RevPAR/satisfaction/
        # occupation) compare les hôtels à leurs pairs mais ne dit jamais POURQUOI
        # un hôtel sous-performe. Chaque module actif (Financier, Commercial/Revenue
        # Mgmt, RH) mesure une dimension différente de la même question — donc dès
        # qu'un module est actif, on le croise avec les hôtels déjà signalés
        # critiques en BCC, plutôt que de le laisser comme un onglet isolé à
        # consulter à part. Un hôtel signalé sur plusieurs dimensions à la fois
        # pose un problème structurel, pas un simple ajustement managérial.
        _dim_modules = {
            'financial_usali'   : '💰 Financier',
            'revenue_management': '📈 Commercial / Revenue Mgmt',
            'workforce'         : '👥 Ressources Humaines',
        }
        _dim_scores = {}
        for _mid, _label in _dim_modules.items():
            _res = _module_results.get(_mid)
            if _res is not None and not _res.error and not _res.scores.empty:
                _dim_scores[_label] = dict(zip(_res.scores['dmu_name'], _res.scores['efficiency']))

        st.markdown("---")
        st.markdown('<p class="section-title">🎯 Diagnostic croisé — Opérationnel × Financier × Commercial × RH</p>', unsafe_allow_html=True)

        if not _dim_scores:
            st.caption(
                "Aucun module Financier / Commercial / RH actif avec de vraies données — "
                "seul le diagnostic opérationnel (BCC) est disponible. Uploadez "
                "l'enrichissement pour débloquer les autres dimensions."
            )
        else:
            _diag_rows = []
            for h in dea.hotels:
                _bcc_h = dea.bcc_scores.get(h, 1.0)
                if _bcc_h >= 0.85:
                    continue
                _weak_dims = [lbl for lbl, sc in _dim_scores.items() if sc.get(h) is not None and sc[h] < 0.85]
                if not _weak_dims:
                    continue
                _row = {'Hôtel': h, 'BCC (opérationnel)': f"{_bcc_h:.1%}"}
                for lbl, sc in _dim_scores.items():
                    _v = sc.get(h)
                    _row[lbl] = f"{_v:.1%}" if _v is not None else "n/d"
                _row['Dimensions en cause'] = ", ".join(d.split(" ", 1)[1] if " " in d else d for d in _weak_dims)
                _row['Lecture'] = (
                    "Problème structurel multi-dimensions — pas qu'un levier managérial isolé."
                    if len(_weak_dims) >= 2 else
                    f"Sous-performance concentrée sur une dimension identifiable ({_weak_dims[0].split(' ',1)[-1]})."
                )
                _diag_rows.append(_row)

            if _diag_rows:
                _n_multi = sum(1 for r in _diag_rows if len(r['Dimensions en cause'].split(', ')) >= 2)
                st.warning(
                    f"{len(_diag_rows)} hôtel(s) critique(s) en BCC (<85%) montrent aussi une "
                    f"faiblesse identifiable sur au moins une autre dimension"
                    + (f", dont {_n_multi} sur plusieurs dimensions à la fois — signal de "
                       f"problème structurel, pas d'un simple ajustement opérationnel."
                       if _n_multi else ".")
                )
                st.dataframe(pd.DataFrame(_diag_rows), use_container_width=True, hide_index=True)
            else:
                st.caption(
                    "Les hôtels critiques en BCC ne montrent pas de faiblesse identifiable sur "
                    "les autres dimensions actives — la sous-performance semble d'origine "
                    "purement opérationnelle."
                )

    # ══════════════════════════════════════════════════════════════════════
    #  RAPPORT PORTFOLIO — finalité du parcours d'analyse
    #  Placé en dernier onglet : toutes les sections sont alimentées une fois
    #  les modules amont parcourus (Capital, Stage 2, Metafrontière...).
    # ══════════════════════════════════════════════════════════════════════

    # ── Export Portfolio PDF ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<p class="section-title">📥 Rapport Portfolio PDF</p>', unsafe_allow_html=True)
    st.caption(
        "Document de comité d'investissement — 13 sections. Chaque section comprend "
        "la méthode, un tableau et une lecture investisseur interprétant vos chiffres. "
        "Le rapport s'adapte automatiquement au portefeuille chargé : variables, "
        "nombre d'actifs et données disponibles."
    )

    # ── Diagnostic de complétude ────────────────────────────────────────────
    # Le rapport calcule lui-même Simar-Wilson et récupère les données capital
    # dans le fichier source : la complétude ne dépend QUE des données fournies,
    # jamais du fait d'avoir visité tel ou tel onglet.
    _cap_st = st.session_state.get('capital_input')
    _cap_saisi = (_cap_st is not None
                  and 'gop (k€)' in getattr(_cap_st, 'columns', [])
                  and float(_cap_st[['surface_m2', 'capex_annuel (k€)',
                                     'gop (k€)']].sum().sum()) > 0)
    _cap_source   = all(c in dea.df.columns for c in ('surface_m2', 'capex_annuel', 'gop'))
    _has_cap_data = _cap_saisi or _cap_source

    _has_n1  = len(has_n1_cols(dea.df)) >= 4
    _has_ft  = ('revpar_n1' in dea.df.columns and 'gop_n1' in dea.df.columns
                and dea.df.get('revpar_n1', pd.Series(dtype=float)).fillna(0).sum() > 0
                and dea.df.get('gop_n1',    pd.Series(dtype=float)).fillna(0).sum() > 0)
    # Simar-Wilson : calculé à la volée si au moins 2 variables structurelles
    # exploitables et assez d'unités inefficientes pour les degrés de liberté
    _n_ineff_ui = sum(1 for s in dea.bcc_scores.values() if s < 1 - 1e-8)
    _dea_vars_ui = set(getattr(dea, 'input_cols', [])) | set(getattr(dea, 'output_cols', []))
    _sw_cands   = [c for c in ('classement_etoiles', 'saison_dummy', 'surface_m2',
                               'capex_annuel', 'gop', 'energy_kwh', 'payroll_total',
                               'book_value_assets', 'total_revenue')
                   if c in dea.df.columns and c not in _dea_vars_ui
                   and pd.to_numeric(dea.df[c], errors='coerce').nunique() > 1]
    _has_sw = (bool(st.session_state.get('sw_results'))
               or (len(_sw_cands) >= 2 and _n_ineff_ui >= 4))

    _SECTIONS = [
        ("1. Dashboard portefeuille",        True,           ""),
        ("2. Quadrants gestion x échelle",   True,           ""),
        ("3. Classement TOPSIS",             True,           ""),
        ("4. Super-efficience",              True,           ""),
        ("5. SBM (Tone 2001)",               True,           ""),
        ("6. Slacks & gaspillages",          True,           ""),
        ("7. Métafrontière GTE/MTE/TGR",     True,           ""),
        ("8. Déterminants (Simar-Wilson)",   _has_sw,
         "Au moins 2 variables structurelles et 4 actifs inefficients requis"),
        ("9. Efficience du capital",         _has_cap_data,
         "Colonnes surface_m2, capex_annuel et gop — fichier source ou onglet Capital"),
        ("10. DEA Capital vs Opérationnel",  _has_cap_data,  "Idem section 9"),
        ("11. Expense Flex & Flow Through",  _has_cap_data and _has_ft,
         "Colonnes revpar_n1 et gop_n1 requises en plus des données capital"),
        ("12. Malmquist (productivité N-1)", _has_n1,
         "Colonnes _n1 requises dans le fichier source"),
        ("13. Diagnostic croisé multi-dimensions", any(
            (_module_results.get(m) is not None and not _module_results[m].error
             and not _module_results[m].scores.empty)
            for m in ('financial_usali', 'revenue_management', 'workforce')),
         "Au moins un module Financier/Commercial/RH actif avec de vraies données"),
    ]
    _ready = sum(1 for _, ok, _ in _SECTIONS if ok)

    with st.expander(f"📋 Contenu du rapport — {_ready}/13 sections alimentées",
                     expanded=(_ready < 12)):
        _sc1, _sc2 = st.columns(2)
        for _i, (_nm, _ok, _fix) in enumerate(_SECTIONS):
            with (_sc1 if _i < 7 else _sc2):
                if _ok:
                    st.markdown(f"✅ {_nm}")
                else:
                    st.markdown(f"⚪ {_nm}  \n&nbsp;&nbsp;&nbsp;<small>*{_fix}*</small>",
                                unsafe_allow_html=True)
        st.caption(
            "Le rapport calcule lui-même Simar-Wilson, le Malmquist, le SBM et la "
            "métafrontière : aucun passage préalable par les autres onglets n'est "
            "nécessaire. La complétude ne dépend que des colonnes présentes dans "
            "votre fichier."
        )
        if _ready < 13:
            st.caption(
                "Les sections non alimentées restent dans le PDF avec leur explication "
                "méthodologique et l'indication des données manquantes."
            )
            st.markdown("---")
            st.markdown(
                f"**Pour atteindre 12/12** — votre fichier couvre {_ready} sections. "
                "Les colonnes suivantes débloquent les autres :"
            )
            _MANQUE = []
            if not _has_cap_data:
                _MANQUE += [("surface_m2",   "Surface totale de l'actif (m²)",        "9, 10"),
                            ("capex_annuel", "CAPEX annuel (€)",                      "9, 10"),
                            ("gop",          "Gross Operating Profit (€)",            "9, 10")]
            if not _has_ft:
                _MANQUE += [("revpar_n1", "RevPAR de l'exercice précédent (€)",       "11"),
                            ("gop_n1",    "GOP de l'exercice précédent (k€)",         "11")]
            if not _has_n1:
                _MANQUE += [("nb_employes_n1 / couts_op_ex_n1 / revpar_n1 / taux_occupation_n1",
                             "Jeu complet de l'exercice précédent", "12")]
            if not _has_sw:
                _MANQUE += [("classement_etoiles, saison_dummy, energy_kwh…",
                             "Variables structurelles hors modèle DEA", "8")]
            if _MANQUE:
                st.dataframe(
                    pd.DataFrame(_MANQUE, columns=["Colonne", "Contenu", "Débloque"]),
                    use_container_width=True, hide_index=True,
                )
                # Modèle Excel pré-rempli aux noms des actifs du portefeuille
                _tpl_cols = [c for c, _, _ in _MANQUE if ' ' not in c and '/' not in c]
                if _tpl_cols:
                    _tpl_full = pd.DataFrame(
                        {c: [0.0] * dea.n for c in _tpl_cols},
                        index=pd.Index(dea.hotels, name='hotel_name'))
                    _bio = BytesIO()
                    with pd.ExcelWriter(_bio, engine='openpyxl') as _w:
                        _tpl_full.to_excel(_w, sheet_name='Compléments')
                        _w.sheets['Compléments'].column_dimensions['A'].width = 38
                    st.download_button(
                        "⬇️ Modèle Excel des colonnes manquantes",
                        data=_bio.getvalue(),
                        file_name="deah_colonnes_manquantes.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="tpl_missing_cols",
                    )
                    st.caption(
                        "Complétez ce fichier, fusionnez-le avec votre fichier source, "
                        "puis rechargez — les sections correspondantes s'activeront."
                    )

    _rpc1, _rpc2 = st.columns([3, 1])
    with _rpc1:
        # Garde-fou : petit portefeuille (n < 4) → slider impossible (min > max)
        _tn_max = min(20, dea.n)
        if _tn_max <= 3:
            top_n_portfolio = dea.n
            st.caption(f"Portefeuille de {dea.n} hôtels — Top complet inclus.")
        else:
            top_n_portfolio = st.slider(
                "Hôtels affichés dans le classement TOPSIS", 3, _tn_max,
                min(10, _tn_max), key="top_n_slider")
    with _rpc2:
        st.markdown("<div style='height:1.7rem'></div>", unsafe_allow_html=True)
        if st.button("📊 Générer le rapport", key="portfolio_pdf_btn",
                     type="primary", use_container_width=True):
            with st.spinner("Génération du rapport — calcul Malmquist, SBM, "
                            "super-efficience et métafrontière..."):
                try:
                    _rp = generate_portfolio_report_pdf(
                        dea=dea, quadrant_labels=QUADRANT_LABELS, top_n=top_n_portfolio,
                        cap_input=_cap_st,
                        sw_results=st.session_state.get('sw_results'),
                        jours_exploit=jours_exploit, ft_pct=ft_pct,
                        avg_salary=avg_salary,
                        thresholds=REPORT_THRESHOLDS,
                        module_results=st.session_state.get('module_results', {}),
                    )
                    if _rp:
                        st.session_state["portfolio_pdf"]      = _rp
                        st.session_state["portfolio_pdf_meta"] = {
                            'ko': len(_rp) // 1024, 'sections': _ready,
                            'ts': datetime.now().strftime('%d/%m/%Y %H:%M'),
                        }
                    else:
                        st.error("ReportLab non disponible sur ce serveur.")
                except Exception as _e:
                    st.error(f"Erreur de génération : {_e}")

    if st.session_state.get("portfolio_pdf"):
        _meta = st.session_state.get("portfolio_pdf_meta", {})
        _d1, _d2 = st.columns([1, 2])
        with _d1:
            st.download_button(
                "⬇️ Télécharger le rapport",
                data=st.session_state["portfolio_pdf"],
                file_name=f"rapport_portfolio_DEA-H_{datetime.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf", key="dl_portfolio_pdf",
                use_container_width=True,
            )
        with _d2:
            if _meta:
                st.caption(
                    f"Généré le {_meta.get('ts','—')} · {_meta.get('ko','—')} Ko · "
                    f"{_meta.get('sections','—')}/13 sections alimentées"
                )




# ══════════════════════════════════════════════
# TAB 13 -- VARIANCE BUDGET
# ══════════════════════════════════════════════
with tab_budget:
    st.markdown('<p class="section-title">Analyse de Variance Budget -- Format USALI</p>', unsafe_allow_html=True)
    st.caption('Russo & Legel Exhibit 6 : N-1 / Budget / Realise en PAR (Per Available Room), POR (Per Occupied Room), % CA')

    sel_var = st.selectbox('Actif a analyser', options=dea.hotels, key='var_hotel_sel')
    if dea.has_chambres:
        lits_v  = float(dea.df.loc[sel_var, 'nb_chambres'])
        occ_v   = float(dea.df.loc[sel_var, 'taux_occupation']) / 100
        revpar_v= float(dea.df.loc[sel_var, 'revpar'])
        nights_v= lits_v * 365 * occ_v
        par_v   = lits_v * 365
        ca_v    = revpar_v * lits_v * 365
    else:
        lits_v = nights_v = par_v = ca_v = None

    USALI_ITEMS = [
        ('Rooms Revenue', 'POR'),
        ('F&B Revenue', 'POR'),
        ('Other Revenue', 'POR'),
        ('Rooms Expense', 'POR'),
        ('F&B Expense', '% Rev'),
        ('Other Dept Expense', '% Rev'),
        ('Admin & General', 'PAR'),
        ('Sales & Marketing', 'PAR'),
        ('Property Ops & Maintenance', 'PAR'),
        ('Utilities', 'PAR'),
        ('GOP', 'PAR'),
        ('Management Fees', '% CA'),
        ('Property Taxes', 'PAR'),
        ('Insurance', 'PAR'),
        ('EBITDA', 'PAR'),
    ]

    _vkey = f'var_{sel_var}'
    if _vkey not in st.session_state:
        st.session_state[_vkey] = {ln: {'n1':0.0,'bud':0.0,'rea':0.0} for ln,_ in USALI_ITEMS}

    _var_rows_in = [{'Poste': ln, 'Metrique': mt,
        'N-1 (e)': st.session_state[_vkey].get(ln,{}).get('n1',0.0),
        'Budget (e)': st.session_state[_vkey].get(ln,{}).get('bud',0.0),
        'Realise (e)': st.session_state[_vkey].get(ln,{}).get('rea',0.0),
    } for ln, mt in USALI_ITEMS]

    _edited = st.data_editor(
        pd.DataFrame(_var_rows_in), use_container_width=True, hide_index=True,
        column_config={
            'Poste'      : st.column_config.TextColumn('Poste USALI', disabled=True),
            'Metrique'   : st.column_config.TextColumn('Metrique', disabled=True),
            'N-1 (e)'    : st.column_config.NumberColumn('N-1 Realise (e)', format='%.0f', step=1000.0),
            'Budget (e)' : st.column_config.NumberColumn('Budget N (e)', format='%.0f', step=1000.0),
            'Realise (e)': st.column_config.NumberColumn('N Realise (e)', format='%.0f', step=1000.0),
        }, key=f'veditor_{sel_var}',
    )

    # Variances
    st.markdown('---')
    st.markdown('<p class="section-title">Tableau de Variance PAR / POR / % CA</p>', unsafe_allow_html=True)
    st.caption(
        f'PAR base = {par_v:,.0f} room-nights | POR base = {nights_v:,.0f} nuitees | CA ref = {ca_v:,.0f} euros'
        if dea.has_chambres else
        'PAR/POR non calculables — nombre de chambres réel non fourni. Colonnes € brutes disponibles ci-dessous.'
    )

    _vres = []
    for _, row_v in _edited.iterrows():
        ln_v = row_v['Poste']; mt_v = row_v['Metrique']
        n1_v  = float(row_v['N-1 (e)'])    if row_v['N-1 (e)']    else 0.0
        bud_v = float(row_v['Budget (e)'])  if row_v['Budget (e)'] else 0.0
        rea_v = float(row_v['Realise (e)']) if row_v['Realise (e)'] else 0.0

        var_b = rea_v - bud_v; var_n1 = rea_v - n1_v
        pct_b = var_b / abs(bud_v) * 100 if bud_v != 0 else 0
        pct_n1= var_n1/ abs(n1_v)  * 100 if n1_v  != 0 else 0

        base_v = (par_v if mt_v=='PAR' else nights_v if mt_v=='POR' else ca_v) if dea.has_chambres else None
        def _fmt(v):
            if base_v is None: return '--'
            return f'{v/base_v*100:.1f}%' if mt_v=='% CA' and base_v>0 else (f'{v/base_v:.2f}' if base_v>0 else '--')

        _vres.append({
            'Poste'          : ln_v,
            'N-1'            : f"{n1_v:,.0f}" if n1_v else '--',
            'Budget'         : f"{bud_v:,.0f}" if bud_v else '--',
            'Realise'        : f"{rea_v:,.0f}" if rea_v else '--',
            'Ecart/Budget'   : f"{var_b:+,.0f} ({pct_b:+.1f}%)" if bud_v else '--',
            'Ecart/N-1'      : f"{var_n1:+,.0f} ({pct_n1:+.1f}%)" if n1_v else '--',
            f'N {mt_v}'      : _fmt(rea_v),
            f'Bud {mt_v}'    : _fmt(bud_v),
            f'N-1 {mt_v}'    : _fmt(n1_v),
        })

    if _vres:
        st.dataframe(pd.DataFrame(_vres), use_container_width=True, hide_index=True)
        _csv_v = pd.DataFrame(_vres).to_csv(index=False, sep=';', decimal=',')
        st.download_button('Exporter Variance (CSV)', data=_csv_v.encode('utf-8-sig'),
                           file_name=f'variance_{sel_var}.csv', mime='text/csv')

        # Bar chart realise vs budget vs N-1
        def _peur(s): return float(s.replace(',','').replace(' ','')) if s != '--' else 0
        _chart_r = [r for r in _vres if any(v != '--' for v in [r['Realise'],r['Budget'],r['N-1']])]
        if _chart_r:
            _postes_v = [r['Poste'] for r in _chart_r]
            fig_var = go.Figure()
            for _s, _col in [('N-1','#95a5a6'),('Budget','#f39c12'),('Realise','#2ecc71')]:
                _vals_v = [_peur(r[_s]) for r in _chart_r]
                fig_var.add_trace(go.Bar(name=_s, x=_postes_v, y=_vals_v, marker_color=_col, opacity=0.85))
            fig_var.update_layout(barmode='group', title='N-1 vs Budget vs Realise',
                                  xaxis_tickangle=-30, height=400, paper_bgcolor='rgba(0,0,0,0)',
                                  yaxis=dict(title='Montant (e)', tickformat=',.0f'),
                                  legend=dict(orientation='h', yanchor='bottom', y=1.02))
            st.plotly_chart(fig_var, use_container_width=True)

    c_ft1, c_ft2 = st.columns(2)
    c_ft1.info('**Flow Through** (CA augmente)\nFT = delta GOP / delta CA\nCible : 50% | Rate-driven : FT eleve | Volume-driven : FT faible')
    c_ft2.info('**Expense Flex** (CA baisse)\nFlex = 1 - FT\nMesure la capacite a reduire les couts quand le CA recule.')


# ════════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────
#  Footer
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("---")
st.markdown(
    """<div style='text-align:center;color:#94a3b8;font-size:0.75rem;line-height:1.8;padding:0.5rem 0;'>
    <b>DEA-H v3.9 · REIV Hospitality · Mehdi Sayyou</b><br>
    Modèles : BCC/CCR Input/Output-Oriented · TOPSIS Shannon entropy · K-means · Metafrontière GTE/MTE/TGR ·
    Malmquist TFP (Catch-up × Frontier Shift) · Tobit Second Stage · Multi-Module DEA (7 dimensions)<br>
    <b>Références :</b>
    Charnes, Cooper &amp; Rhodes (1978) CCR ·
    Banker, Charnes &amp; Cooper (1984) BCC ·
    Barros (2005) format Barros ·
    Min, Min &amp; Joo (2009) USALI ·
    Assaf, Barros &amp; Josiassen (2010) Metafrontière ·
    Shang, Wang &amp; Hung (2010) Stochastic DEA ·
    Yu (2012) MDEA Room/F&amp;B · Mann-Whitney ·
    Simar &amp; Wilson (2007) Tobit bootstrap ·
    Caves, Christensen &amp; Diewert (1982) Malmquist ·
    Färe, Grosskopf, Norris &amp; Zhang (1994) Décomposition TFP ·
    Poldrugovac, Tekavcic &amp; Jankovic (2016) Mahalanobis ·
    Vlad, Toma &amp; Fîntîneru (2026) TOPSIS entropy ·
    Shirouyehzad et al. (2012) SERVQUAL ·
    Via@ (2013) Benchmark 15–25% · Alpha opérateur
    </div>""",
    unsafe_allow_html=True,
)
