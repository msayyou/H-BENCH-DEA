"""
app.py — DEA-H v3.9
Application Streamlit : Analyse DEA BCC/CCR pour Asset Management Hôtelier
11 onglets : Board | KPIs | TOPSIS | K-means | Quadrants | Slacks | Fiche Actif
           | Metafrontière | Capital & Flow Through | Benchmark Marché
           | 🔀 Synthèse Multi-Module  ← NOUVEAU v3.2
"""

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
from io import BytesIO
from datetime import datetime

import numpy as np


def generate_fiche_actif_pdf(
    hotel: str,
    dea,
    quadrant_labels: dict,
    avg_salary: float = 35_000,
    revpar_value: float = 1_000,
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
    lits  = float(raw["nb_lits"])
    emp   = float(raw["nb_employes"])
    costs = float(raw["couts_op_ex"])
    rvp   = float(raw["revpar"])
    sat   = float(raw["satisfaction"])
    occ   = float(raw["taux_occupation"])
    nights = lits * jours_exploit * occ / 100
    ca_est = rvp * lits * jours_exploit

    slk_emp = dea.slacks.get(hotel, {}).get("inputs", {}).get("nb_employes", 0)
    slk_rvp = dea.slacks.get(hotel, {}).get("outputs", {}).get("revpar", 0)
    up_fte  = round(slk_emp * avg_salary / 1000)
    up_rev  = round(slk_rvp * nights * revpar_value / 1_000_000, 2)

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
        ["Chambres",         f"{int(lits)}",         "RevPAR",              f"{rvp:.0f} €"],
        ["ETP",              f"{emp:.0f}",           "Taux occupation",     f"{occ:.1f}%"],
        ["Charges op.",      f"{costs:.2f} M€",      "Satisfaction",        f"{sat:.1f}/10"],
        ["CA estimé",        f"{ca_est/1e6:.2f} M€", "Nuitées estimées",    f"{nights:,.0f}"],
    ]
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
    for col in ["nb_lits", "nb_employes", "couts_op_ex"]:
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
    upside_txt = f"Upside ETP : {up_fte} k€/an  |  Upside RevPAR : {up_rev:.2f} M€/an"
    story.append(Spacer(1, 4))
    if up_fte > 0 or up_rev > 0:
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

# --- malmquist_tobit inline ---
"""
malmquist_tobit.py — Modules Malmquist & Tobit pour DEA-H v3.7
REIV Hospitality

Malmquist : Caves, Christensen & Diewert (1982) ; Färe et al. (1994)
Tobit     : Tobin (1958) ; Simar & Wilson (2007) second stage DEA
"""
import numpy as np
import pandas as pd
import pulp
from scipy import stats, optimize

# ══════════════════════════════════════════════════════════════════════════════
# MALMQUIST PRODUCTIVITY INDEX
# ══════════════════════════════════════════════════════════════════════════════

_N1_COLS = {
    'nb_lits'         : 'nb_lits_n1',
    'nb_employes'     : 'nb_employes_n1',
    'couts_op_ex'     : 'couts_op_ex_n1',
    'revpar'          : 'revpar_n1',
    'satisfaction'    : 'satisfaction_n1',
    'taux_occupation' : 'taux_occupation_n1',
}

def _dea_cross(j: int,
               X_eval: np.ndarray, Y_eval: np.ndarray,
               X_ref:  np.ndarray, Y_ref:  np.ndarray) -> float | None:
    """
    Input-orienté BCC.
    Évalue DMU j (données X_eval[j], Y_eval[j])
    sur la frontière construite depuis (X_ref, Y_ref).
    """
    n_ref = X_ref.shape[0]
    m_in  = X_ref.shape[1]
    m_out = Y_ref.shape[1]

    mdl = pulp.LpProblem(f"MQ_{j}", pulp.LpMinimize)
    th  = pulp.LpVariable("theta", lowBound=0)
    lam = pulp.LpVariable.dicts("lam", range(n_ref), lowBound=0)

    mdl += th
    for i in range(m_in):
        mdl += pulp.lpSum(lam[k] * X_ref[k, i] for k in range(n_ref)) <= th * X_eval[j, i]
    for r in range(m_out):
        mdl += pulp.lpSum(lam[k] * Y_ref[k, r] for k in range(n_ref)) >= Y_eval[j, r]
    mdl += pulp.lpSum(lam.values()) == 1  # BCC

    mdl.solve(pulp.PULP_CBC_CMD(msg=False))
    return round(pulp.value(th), 6) if mdl.status == 1 else None


def compute_malmquist(dea) -> pd.DataFrame | None:
    """
    Calcule l'indice de productivité de Malmquist pour chaque DMU.

    Nécessite dans dea.df les colonnes _n1 :
      nb_lits_n1, nb_employes_n1, couts_op_ex_n1,
      revpar_n1, taux_occupation_n1 (+ satisfaction_n1 optionnel)

    Décomposition (Färe et al. 1994) :
      M = Catch-up × Frontier Shift
      Catch-up      = D¹(x¹,y¹) / D⁰(x⁰,y⁰)   — convergence vers frontière
      Frontier Shift = √[ D⁰(x¹,y¹)/D¹(x¹,y¹) × D⁰(x⁰,y⁰)/D¹(x⁰,y⁰) ]
                                                   — déplacement de la frontière

    Réf. : Caves, Christensen & Diewert (1982) Econometrica
           Färe, Grosskopf, Norris & Zhang (1994) American Economic Review
    """
    df = dea.df
    # Vérifier colonnes _n1
    missing = [v for v in _N1_COLS.values()
               if v not in df.columns and v != 'satisfaction_n1']
    if missing:
        return None

    input_cols  = ['nb_lits', 'nb_employes', 'couts_op_ex']
    output_cols = ['revpar', 'taux_occupation']
    if 'satisfaction_n1' in df.columns:
        output_cols.append('satisfaction')

    X1 = df[input_cols].values.astype(float)
    Y1 = df[output_cols].values.astype(float)
    X0 = df[[_N1_COLS[c] for c in input_cols]].values.astype(float)
    Y0 = df[[_N1_COLS[c] for c in output_cols]].values.astype(float)

    hotels = dea.hotels
    rows   = []

    for j, hotel in enumerate(hotels):
        d00 = _dea_cross(j, X0, Y0, X0, Y0)  # frontier t=0, data t=0 (BCC N-1)
        d11 = _dea_cross(j, X1, Y1, X1, Y1)  # frontier t=1, data t=1 (BCC N)
        d01 = _dea_cross(j, X1, Y1, X0, Y0)  # frontier t=0, data t=1 (cross)
        d10 = _dea_cross(j, X0, Y0, X1, Y1)  # frontier t=1, data t=0 (cross)

        if not all(v is not None and v > 0 for v in [d00, d11, d01, d10]):
            rows.append({'Hôtel': hotel, 'BCC N-1': '—', 'BCC N': '—',
                         'Catch-up': '—', 'Frontier Shift': '—',
                         'Malmquist TFP': '—', 'Interprétation': 'Non calculable'})
            continue

        catchup        = round(d11 / d00, 4)
        frontier_shift = round(np.sqrt((d01 / d11) * (d00 / d10)), 4)
        tfp            = round(catchup * frontier_shift, 4)

        # Interprétation
        if tfp > 1.0 and catchup > 1.0 and frontier_shift > 1.0:
            interp = '🟢 Progrès total — gestion ET technologie'
        elif tfp > 1.0 and catchup >= 1.0:
            interp = '🟢 Progrès — rattrapage de la frontière'
        elif tfp > 1.0 and frontier_shift > 1.0:
            interp = '🔵 Progrès sectoriel — hôtel suit la marée montante'
        elif tfp < 1.0 and catchup < 1.0 and frontier_shift < 1.0:
            interp = '🔴 Régression totale — gestion ET technologie en recul'
        elif tfp < 1.0 and catchup < 1.0:
            interp = '🟠 Régression gestion — s\'éloigne de la frontière'
        else:
            interp = '🟡 Stable ou mixte'

        rows.append({
            'Hôtel'          : hotel,
            'BCC N-1'        : f'{d00:.3f}',
            'BCC N'          : f'{d11:.3f}',
            'Catch-up'       : f'{catchup:.3f}',
            'Frontier Shift' : f'{frontier_shift:.3f}',
            'Malmquist TFP'  : f'{tfp:.3f}',
            'Interprétation' : interp,
        })

    return pd.DataFrame(rows).sort_values('Malmquist TFP', ascending=False,
                                          key=lambda x: pd.to_numeric(x, errors='coerce')).reset_index(drop=True)


def has_n1_cols(df: pd.DataFrame) -> list[str]:
    """Retourne la liste des colonnes _n1 présentes dans df."""
    return [c for c in _N1_COLS.values() if c in df.columns]


# ══════════════════════════════════════════════════════════════════════════════
# TOBIT SECOND STAGE
# ══════════════════════════════════════════════════════════════════════════════

def _tobit_loglik(params: np.ndarray, y: np.ndarray,
                  X: np.ndarray, limit: float = 1.0) -> float:
    """
    Log-vraisemblance Tobit right-censored à `limit`.
    Signe négatif pour minimisation.
    Réf. : Tobin (1958) ; Simar & Wilson (2007)
    """
    beta  = params[:-1]
    sigma = max(abs(params[-1]), 1e-6)
    Xb    = X @ beta
    ll    = 0.0
    for i in range(len(y)):
        if y[i] < limit - 1e-8:
            ll += stats.norm.logpdf(y[i], Xb[i], sigma)
        else:
            ll += np.log(max(1.0 - stats.norm.cdf(limit, Xb[i], sigma), 1e-10))
    return -ll


def compute_tobit(
    dea,
    env_vars: list[str],
    env_labels: dict[str, str] | None = None,
    n_bootstrap: int = 0,
) -> dict:
    """
    Tobit second stage sur les scores BCC.
    
    Args:
        dea        : HotelDEAAnalyzer (scores BCC calculés)
        env_vars   : liste de colonnes de dea.df à utiliser comme régresseurs
        env_labels : dict {colonne: label lisible}
        n_bootstrap: 0 = pas de bootstrap ; >0 = bootstrap SE (Simar & Wilson 2007)
    
    Returns: dict avec clés
        'coef_df'   : pd.DataFrame coefficients + SE + t + p-value
        'n'         : nombre d'observations
        'n_censored': DMUs à score = 1.0
        'log_lik'   : log-vraisemblance optimale
        'converged' : bool
        'sigma'     : bruit estimé
    
    Réf. : Tobin (1958) Econometrica
           Simar & Wilson (2007) Journal of Econometrics
    """
    if env_labels is None:
        env_labels = {}

    # Préparer les données
    valid = [h for h in dea.hotels if all(
        c in dea.df.columns and not pd.isna(dea.df.loc[h, c])
        for c in env_vars
    )]
    if len(valid) < len(env_vars) + 3:
        return {'error': f'Trop peu d\'observations valides ({len(valid)}) '
                         f'pour {len(env_vars)} régresseurs.'}

    y = np.array([dea.bcc_scores[h] for h in valid])
    X_raw = np.column_stack([dea.df.loc[valid, c].values.astype(float)
                             for c in env_vars])

    # Normalisation des régresseurs (meilleure convergence)
    X_means = X_raw.mean(axis=0)
    X_stds  = X_raw.std(axis=0)
    X_stds[X_stds == 0] = 1.0
    X_norm  = (X_raw - X_means) / X_stds
    X_fit   = np.column_stack([np.ones(len(valid)), X_norm])

    # Optimisation MLE
    k     = X_fit.shape[1]
    p0    = np.zeros(k + 1)
    p0[0] = float(np.mean(y))
    p0[-1] = max(float(np.std(y)), 0.05)

    res = optimize.minimize(
        _tobit_loglik, p0, args=(y, X_fit, 1.0),
        method='Nelder-Mead',
        options={'maxiter': 10000, 'xatol': 1e-7, 'fatol': 1e-7},
    )

    beta_hat  = res.x[:-1]
    sigma_hat = max(abs(res.x[-1]), 1e-6)
    log_lik   = -res.fun
    n_cens    = int((y >= 1.0 - 1e-8).sum())

    # Standard errors via Hessian numérique
    try:
        from scipy.optimize import approx_fprime
        eps_h  = 1e-5
        hess   = np.zeros((len(res.x), len(res.x)))
        for i in range(len(res.x)):
            def grad_i(p, i=i):
                return approx_fprime(p, _tobit_loglik, eps_h,
                                     y, X_fit, 1.0)[i]
            hess[i] = approx_fprime(res.x, grad_i, eps_h)
        cov = np.linalg.pinv(hess)
        se  = np.sqrt(np.abs(np.diag(cov)[:k]))
    except Exception:
        se = np.full(k, np.nan)

    # Dé-normalisation des coefficients (hors constante)
    labels = ['Constante'] + [env_labels.get(c, c) for c in env_vars]
    beta_denorm = beta_hat.copy()
    for i in range(1, k):
        beta_denorm[i] = beta_hat[i] / X_stds[i-1]
    se_denorm = se.copy()
    for i in range(1, k):
        se_denorm[i] = se[i] / X_stds[i-1]

    t_stats = np.where(se_denorm > 0, beta_denorm / se_denorm, np.nan)
    p_vals  = np.where(np.isnan(t_stats), np.nan,
                       2 * (1 - stats.norm.cdf(np.abs(t_stats))))

    # Interprétation des coefficients
    def stars(p):
        if np.isnan(p): return ''
        return '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.10 else ''

    coef_df = pd.DataFrame({
        'Variable'  : labels,
        'Coeff.'    : [round(b, 4) for b in beta_denorm],
        'Std. Error': [round(s, 4) if not np.isnan(s) else '—' for s in se_denorm],
        't-stat'    : [round(t, 2) if not np.isnan(t) else '—' for t in t_stats],
        'p-value'   : [round(p, 4) if not np.isnan(p) else '—' for p in p_vals],
        'Sig.'      : [stars(p) for p in p_vals],
        'Effet'     : [('↑ améliore efficience' if b > 0 else '↓ réduit efficience')
                       if l != 'Constante' else '—'
                       for b, l in zip(beta_denorm, labels)],
    })

    return {
        'coef_df'    : coef_df,
        'n'          : len(valid),
        'n_censored' : n_cens,
        'log_lik'    : round(log_lik, 4),
        'sigma'      : round(sigma_hat, 4),
        'converged'  : bool(res.success),
        'hotels'     : valid,
        'y'          : y,
        'X'          : X_raw,
        'env_vars'   : env_vars,
    }


# ══════════════════════════════════════════════════════════════════════════════
# MDEA ROOM / F&B DECOMPOSITION — Yu (2012)
# ══════════════════════════════════════════════════════════════════════════════

MDEA_COL_MAP = {
    # Division Rooms
    'room': {
        'inputs' : ['rooms_cost', 'nb_lits'],
        'outputs': ['rooms_revenue'],
        'label'  : 'Hébergement (Rooms)',
    },
    # Division F&B
    'fb': {
        'inputs' : ['fb_cost'],
        'outputs': ['fb_revenue'],
        'label'  : 'Restauration (F&B)',
    },
    # Shared inputs (utilisés dans les deux)
    'shared_inputs' : ['payroll_total', 'undistributed_expenses'],
    # Common output
    'common_outputs': ['other_dept_revenue'],
}

def _bcc_input(j: int, X: np.ndarray, Y: np.ndarray) -> float | None:
    """BCC input-orienté. Retourne theta ∈ ]0,1]."""
    n, m_in = X.shape; m_out = Y.shape[1]
    mp  = pulp.LpProblem(f"BCC_{j}", pulp.LpMinimize)
    th  = pulp.LpVariable("theta", lowBound=0)
    lam = pulp.LpVariable.dicts("lam", range(n), lowBound=0)
    mp += th
    for i in range(m_in):
        mp += pulp.lpSum(lam[k] * X[k, i] for k in range(n)) <= th * X[j, i]
    for r in range(m_out):
        mp += pulp.lpSum(lam[k] * Y[k, r] for k in range(n)) >= Y[j, r]
    mp += pulp.lpSum(lam.values()) == 1
    mp.solve(pulp.PULP_CBC_CMD(msg=False))
    return round(float(pulp.value(th)), 4) if mp.status == 1 else None


def compute_mdea_room_fb(dea) -> dict:
    """
    Décomposition MDEA Room / F&B selon Yu (2012).

    Calcule trois scores BCC indépendants :
      - BCC Global     : modèle standard DEA-H
      - BCC Room       : efficience du département Hébergement
      - BCC F&B        : efficience du département Restauration
    
    L'écart entre BCC Room et BCC F&B localise la source d'inefficience
    — réplication simplifiée de la décomposition MDEA/GAR (Yu 2012, Eq. 3-6).
    
    Inputs Room  : rooms_cost + nb_lits
    Inputs F&B   : fb_cost (+ fb_area si disponible)
    Outputs Room : rooms_revenue
    Outputs F&B  : fb_revenue
    Shared       : payroll_total, undistributed_expenses (informatif)
    
    Réf. : Yu, M.-M. (2012) Current Issues in Tourism 15(5), 461-476
           Jahanshahloo, Amirteimoori & Kordrostami (2004)
           Mann-Whitney test : Yu Table 4-5 (non-parametric)
    
    Returns dict avec :
        'scores'      : pd.DataFrame scores par hôtel
        'feasible'    : bool
        'missing_cols': list colonnes manquantes
        'summary'     : pd.DataFrame résumé Room vs F&B
        'mw_test'     : dict Mann-Whitney entre divisions
    """
    from scipy import stats as _stats

    df = dea.df
    hotels = dea.hotels

    # Détection colonnes disponibles
    room_in_avail  = [c for c in MDEA_COL_MAP['room']['inputs']   if c in df.columns]
    room_out_avail = [c for c in MDEA_COL_MAP['room']['outputs']  if c in df.columns]
    fb_in_avail    = [c for c in MDEA_COL_MAP['fb']['inputs']     if c in df.columns]
    fb_out_avail   = [c for c in MDEA_COL_MAP['fb']['outputs']    if c in df.columns]

    missing = []
    if not room_in_avail  : missing += MDEA_COL_MAP['room']['inputs']
    if not room_out_avail : missing += MDEA_COL_MAP['room']['outputs']
    if not fb_in_avail    : missing += MDEA_COL_MAP['fb']['inputs']
    if not fb_out_avail   : missing += MDEA_COL_MAP['fb']['outputs']

    if missing:
        return {'feasible': False, 'missing_cols': list(set(missing)),
                'scores': pd.DataFrame(), 'summary': pd.DataFrame(), 'mw_test': {}}

    X_room = df[room_in_avail].values.astype(float)
    Y_room = df[room_out_avail].values.astype(float)
    X_fb   = df[fb_in_avail].values.astype(float)
    Y_fb   = df[fb_out_avail].values.astype(float)

    # Colonnes partagées disponibles (pour info seulement)
    shared_avail = [c for c in MDEA_COL_MAP['shared_inputs'] if c in df.columns]

    rows = []
    scores_room = []; scores_fb = []

    for j, hotel in enumerate(hotels):
        bcc_g = dea.bcc_scores.get(hotel, None)
        er    = _bcc_input(j, X_room, Y_room)
        ef    = _bcc_input(j, X_fb, Y_fb)

        if er is not None: scores_room.append(er)
        if ef is not None: scores_fb.append(ef)

        # Diagnostic source d'inefficience
        if er is not None and ef is not None:
            delta = round(ef - er, 4)
            if er < 0.85 and ef >= 0.90:
                source = '🏨 Room sous-performant'
            elif ef < 0.85 and er >= 0.90:
                source = '🍽️ F&B sous-performant'
            elif er < 0.85 and ef < 0.85:
                source = '🔴 Double inefficience'
            else:
                source = '✅ Équilibré'
        else:
            delta = None; source = '—'

        rows.append({
            'Hôtel'          : hotel,
            'BCC Global'     : f"{bcc_g:.3f}" if bcc_g else '—',
            'BCC Room'       : f"{er:.3f}" if er else '—',
            'BCC F&B'        : f"{ef:.3f}" if ef else '—',
            'Δ (F&B - Room)' : f"{delta:+.3f}" if delta is not None else '—',
            'Source inefficience': source,
            'Inputs Room'    : ', '.join(room_in_avail),
            'Inputs F&B'     : ', '.join(fb_in_avail),
        })

    scores_df = pd.DataFrame(rows).sort_values(
        'BCC Global', ascending=True,
        key=lambda x: pd.to_numeric(x, errors='coerce')
    ).reset_index(drop=True)

    # Résumé statistique Room vs F&B
    summary_rows = []
    for label, vals in [('BCC Room', scores_room), ('BCC F&B', scores_fb)]:
        if vals:
            summary_rows.append({
                'Division'  : label,
                'Moyenne'   : round(np.mean(vals), 4),
                'Médiane'   : round(np.median(vals), 4),
                'Std'       : round(np.std(vals), 4),
                'Min'       : round(np.min(vals), 4),
                'Max'       : round(np.max(vals), 4),
                'N < 0.85'  : sum(1 for v in vals if v < 0.85),
                'N = 1.0'   : sum(1 for v in vals if v >= 0.999),
            })

    summary_df = pd.DataFrame(summary_rows)

    # Mann-Whitney test Room vs F&B (Yu 2012, Table 4-5)
    mw_test = {}
    if len(scores_room) >= 4 and len(scores_fb) >= 4:
        stat, pval = _stats.mannwhitneyu(scores_room, scores_fb, alternative='two-sided')
        mw_test = {
            'stat'         : round(stat, 4),
            'p_value'      : round(pval, 4),
            'significatif' : pval < 0.05,
            'conclusion'   : (
                f"Différence significative Room vs F&B (p={pval:.4f}) "
                f"— {'Room' if np.mean(scores_room) < np.mean(scores_fb) else 'F&B'} "
                f"est le département le moins efficient."
                if pval < 0.05
                else f"Pas de différence significative Room vs F&B (p={pval:.4f})"
            ),
            'test'         : 'Mann-Whitney U (Yu 2012, Table 4-5)',
            'note'         : 'Test non-paramétrique recommandé pour scores DEA (censurés en 1.0)',
        }

    return {
        'feasible'    : True,
        'missing_cols': [],
        'scores'      : scores_df,
        'summary'     : summary_df,
        'mw_test'     : mw_test,
        'shared_used' : shared_avail,
    }


def mann_whitney_groups(dea, groups: pd.Series) -> dict:
    """
    Mann-Whitney U test entre groupes sur scores BCC.
    Alternative non-paramétrique à l'ANOVA.
    Recommandé pour scores DEA (Simar & Wilson 2007 ; Yu 2012).

    Args:
        dea    : HotelDEAAnalyzer
        groups : pd.Series index=hotel_name, values=groupe

    Returns: dict avec tableau comparaisons par paires
    """
    from scipy import stats as _stats
    from itertools import combinations

    unique_g = [g for g in groups.loc[dea.hotels].unique() if str(g) != 'nan']
    grp_scores = {}
    for g in unique_g:
        h_list = [h for h in dea.hotels if groups.get(h) == g]
        if len(h_list) >= 2:
            grp_scores[g] = [dea.bcc_scores[h] for h in h_list]

    if len(grp_scores) < 2:
        return {'error': 'Moins de 2 groupes avec ≥ 2 DMUs'}

    results = []
    alpha_bonf = 0.05 / max(1, len(grp_scores) * (len(grp_scores) - 1) / 2)

    for (g1, s1), (g2, s2) in combinations(grp_scores.items(), 2):
        stat, pval = _stats.mannwhitneyu(s1, s2, alternative='two-sided')
        results.append({
            'Groupe A'     : g1,
            'Groupe B'     : g2,
            'N A'          : len(s1),
            'N B'          : len(s2),
            'Moy. BCC A'   : round(np.mean(s1), 4),
            'Moy. BCC B'   : round(np.mean(s2), 4),
            'U-stat'       : round(stat, 3),
            'p-value'      : round(pval, 4),
            'Bonf. α'      : round(alpha_bonf, 4),
            'Sig.*'        : '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.10 else 'ns',
            'Verdict'      : (f"{g1} > {g2}" if np.mean(s1) > np.mean(s2) else f"{g2} > {g1}")
                             if pval < 0.05 else 'Pas de différence significative',
        })

    return {
        'pairs'       : pd.DataFrame(results),
        'test'        : 'Mann-Whitney U (bilatéral, correction Bonferroni)',
        'ref'         : 'Yu (2012) Table 4-5 ; Simar & Wilson (2007)',
        'alpha_bonf'  : round(alpha_bonf, 4),
        'n_groupes'   : len(grp_scores),
        'summary'     : pd.DataFrame([
            {'Groupe': g, 'N': len(s), 'Moy. BCC': round(np.mean(s),4),
             'Med. BCC': round(np.median(s),4), 'Std': round(np.std(s),4)}
            for g, s in grp_scores.items()
        ]).sort_values('Moy. BCC', ascending=False),
    }

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
.main-header {
    font-size: 2.2rem; font-weight: 800;
    color: #1a3a5c; text-align: center; margin-bottom: 0.5rem;
}
.sub-header {
    text-align: center; color: #555; font-size: 0.95rem; margin-bottom: 1.5rem;
}
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
.section-title {
    font-size: 1.3rem; font-weight: 700;
    color: #1a3a5c; border-left: 4px solid #2e6da4;
    padding-left: 0.7rem; margin: 1.2rem 0 0.8rem 0;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  Titre
# ─────────────────────────────────────────────
st.markdown('<h1 class="main-header">📊 DEA-H — Asset Manager Benchmarking</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Analyse d\'efficacité BCC/CCR · TOPSIS · K-means · Portefeuille hôtelier</p>', unsafe_allow_html=True)

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
    avg_salary   = st.number_input("Coût FTE (€/an)",           value=35_000, step=5_000, format="%i")
    revpar_value = st.number_input("Valeur 1 pt RevPAR (€/an)", value=1_000,  step=100,   format="%i")
    jours_exploit = st.number_input(
        "Jours d'exploitation / an",
        value=365, min_value=30, max_value=365, step=1, format="%i",
        help="Resort saisonnier : 180-240 | Urban : 340-365. Corrige PAR, GOPPAM, TREVPAR."
    )

    st.markdown("---")
    st.subheader("📐 Seuils Quadrants")
    bcc_threshold   = st.slider("Seuil BCC efficience",   0.70, 0.99, 0.90, 0.01)
    scale_threshold = st.slider("Seuil Scale Efficiency", 0.70, 0.99, 0.90, 0.01)

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
        "CSV : hotel_name (index), nb_lits, nb_employes, couts_op_ex, revpar, satisfaction, taux_occupation",
        type=['csv'],
    )
    use_sample = st.checkbox("📋 Données d'exemple (15 hôtels FR)", value=(uploaded_file is None))

# ─────────────────────────────────────────────
#  Chargement des données
# ─────────────────────────────────────────────
NUMERIC_COLS = ['nb_lits', 'nb_employes', 'couts_op_ex', 'revpar', 'satisfaction', 'taux_occupation']

def load_sample() -> pd.DataFrame:
    d = {
        'hotel_name':       ['Paris_Opéra','Lyon_PartDieu','Marseille_VP','Nice_Prom','Bordeaux_Ctr',
                             'Cannes_Carlton','Toulouse_Cap','Nantes_Atl','Strasbourg_Cat',
                             'Lille_GP','Rennes_Rep','Montpellier_Ant','Reims_Champ','Dijon_Palace','Annecy_Lac'],
        'nb_lits':          [120,180,85,150,95,200,140,110,130,160,100,125,145,135,90],
        'nb_employes':      [105,112,30,144,63,163,77,40,77,96,41,57,82,80,70],
        'couts_op_ex':      [8.5,8.5,3.1,10.9,4.6,15.9,4.8,3.8,7.4,8.6,3.3,5.2,6.7,6.3,8.4],
        'revpar':           [280,110,160,105,75,115,130,95,140,120,110,105,125,115,135],
        'satisfaction':     [9.2,8.6,9.2,9.1,8.6,7.9,7.9,8.3,9.0,8.2,8.0,7.9,8.9,7.8,8.8],
        'taux_occupation':  [85,78,82,75,70,76,79,74,81,77,72,75,80,73,78],
    }
    df = pd.DataFrame(d).set_index('hotel_name')
    df.loc['Toulouse_Cap',   'nb_employes'] = int(df.loc['Toulouse_Cap', 'nb_employes'] * 1.4)
    df.loc['Lille_GP',       'couts_op_ex'] = df.loc['Lille_GP', 'couts_op_ex'] * 1.3
    return df

if uploaded_file is not None:
    import io
    raw = uploaded_file.read().decode('utf-8', errors='replace')
    sep = ';' if raw.count(';') > raw.count(',') else ','
    df = pd.read_csv(io.StringIO(raw), index_col=0, sep=sep)
    missing_cols = [c for c in NUMERIC_COLS if c not in df.columns]
    if missing_cols:
        st.sidebar.error(f"❌ Colonnes manquantes : {', '.join(missing_cols)}")
        st.sidebar.caption("Colonnes requises : " + ", ".join(NUMERIC_COLS))
        st.stop()
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=NUMERIC_COLS)
    st.sidebar.success(f"✅ {len(df)} hôtels chargés (séparateur : '{sep}')")
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

# Mapping automatique colonnes DEA-H v3 → noms standard modules_config
_COL_ALIAS = {
    "nb_lits"         : "nb_rooms",
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
# ADR = RevPAR / (TO/100)  — toujours calculable depuis le dataset DEA-H base
if "revpar" in _df_mm.columns and "occupancy_rate" in _df_mm.columns:
    if "adr" not in _df_mm.columns:
        _df_mm["adr"] = (
            _df_mm["revpar"] / (_df_mm["occupancy_rate"] / 100).replace(0, np.nan)
        ).round(2)

# TRevPAR = CA total / nb_rooms (si total_revenue disponible)
if "total_revenue" in _df_mm.columns and "nb_rooms" in _df_mm.columns:
    if "trevpar" not in _df_mm.columns:
        _df_mm["trevpar"] = (
            _df_mm["total_revenue"] / (_df_mm["nb_rooms"] * 365)
        ).round(2)

# revenue_per_fte = CA / ETP
if "total_revenue" in _df_mm.columns and "fte_total" in _df_mm.columns:
    if "revenue_per_fte" not in _df_mm.columns:
        _df_mm["revenue_per_fte"] = (
            _df_mm["total_revenue"] / _df_mm["fte_total"].replace(0, np.nan)
        ).round(0)
elif "revpar" in _df_mm.columns and "nb_rooms" in _df_mm.columns and "fte_total" in _df_mm.columns:
    if "revenue_per_fte" not in _df_mm.columns:
        _df_mm["revenue_per_fte"] = (
            _df_mm["revpar"] * 365 * _df_mm["nb_rooms"] / _df_mm["fte_total"].replace(0, np.nan)
        ).round(0)

# rooms_revenue estimé = revpar × nb_rooms × 365 (proxy si absent)
if "revpar" in _df_mm.columns and "nb_rooms" in _df_mm.columns:
    if "rooms_revenue" not in _df_mm.columns:
        _df_mm["rooms_revenue"] = (
            _df_mm["revpar"] * (_df_mm["occupancy_rate"] / 100 if "occupancy_rate" in _df_mm.columns else 1)
            * _df_mm["nb_rooms"] * 365
        ).round(0)

# total_revenue estimé = rooms_revenue (proxy minimal si absent)
if "total_revenue" not in _df_mm.columns and "rooms_revenue" in _df_mm.columns:
    _df_mm["total_revenue"] = _df_mm["rooms_revenue"]

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
_active_modules, _variable_overrides = render_module_selector(_available_mm)
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
        dea.has_trevpar  = 'total_revenue'    in df.columns
        dea.has_surface  = 'surface_m2'       in df.columns
        dea.has_capex    = 'capex_annuel'     in df.columns
        dea.has_gop      = 'gop'              in df.columns
        dea.has_goppam   = 'surface_m2'       in df.columns and 'gop' in df.columns
        dea.has_stars    = 'classement_etoiles' in df.columns
        dea.has_flow     = 'gop_n1'           in df.columns and 'revenu_n1' in df.columns
        dea.inputs          = df[['nb_lits','nb_employes','couts_op_ex']].values.astype(float)
        dea.outputs         = df[['revpar','satisfaction','taux_occupation']].values.astype(float)
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
            )
            st.session_state["module_results"] = _module_results
        _ok = sum(1 for r in _module_results.values() if not r.error and not r.scores.empty)
        st.success(f"✅ Analyse terminée ! ({_ok}/{len(_active)} modules multi-dim OK)")
    else:
        st.success("✅ Analyse terminée !")

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

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric("📊 Efficacité BCC moy.", f"{avg_bcc:.1%}")
with c2:
    st.metric("🏆 Hôtels efficaces",    f"{n_efficient}/{dea.n}")
with c3:
    st.metric("⚙️ Eff. Échelle moy.",   f"{avg_scale:.1%}")
with c4:
    st.metric("🔴 Critiques (<85%)",    n_critical)
with c5:
    best_topsis = min(dea.topsis_ranks, key=dea.topsis_ranks.get)
    st.metric("🥇 Leader TOPSIS",       best_topsis)

st.markdown("---")

# ─────────────────────────────────────────────
#  11 ONGLETS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12 = st.tabs([
    "📋 Rapport Board",
    "📈 Dashboard KPIs",
    "🏆 Classement TOPSIS",
    "🗂️ Segmentation K-means",
    "📐 Quadrants & Échelle",
    "🔥 Slacks & Gaspillages",
    "🔍 Fiche Actif",
    "🌐 Metafrontière",
    "💰 Capital & Flow Through",
    "📊 Benchmark Marché & Dynamique",
    "🔀 Synthese Multi-Module",
    "📊 Variance Budget",
])

# ══════════════════════════════════════════════
# TAB 1 — RAPPORT BOARD
# ══════════════════════════════════════════════
with tab1:
    # Alerte ratio DMUs/variables
    if getattr(dea, '_dmu_ratio_warning', False):
        st.warning(
            f"⚠️ **Ratio DMUs/variables insuffisant** : {dea._dmu_ratio_info} — "
            "Scores DEA potentiellement sur-efficients. Augmentez le compset ou réduisez les variables. "
            "Réf. : Poldrugovac et al. (2016), Färe et al. (1994), Yu (2012), Tobin (1958) ; Raab & Lichty (2002)."
        )
    st.markdown('<p class="section-title">📋 Rapport Stratégique — Comité d\'Investissement</p>', unsafe_allow_html=True)
    st.dataframe(board, use_container_width=True, hide_index=True)
    st.markdown('<p class="section-title">Répartition par Quadrant</p>', unsafe_allow_html=True)
    q_summary = dea.get_quadrant_summary()
    st.dataframe(q_summary.drop(columns=['Hôtels'], errors='ignore'), use_container_width=True, hide_index=True)

    # ── Détection outliers Mahalanobis (Poldrugovac et al. 2016) ─────────────────────
    st.markdown('<p class="section-title">Détection Outliers — Distance de Mahalanobis</p>', unsafe_allow_html=True)
    st.caption("Poldrugovac et al. (2016), Färe et al. (1994), Yu (2012), Tobin (1958) ; Kerstens (1996) — D² suit une loi χ² à k degrés de liberté. Outlier si p < 0.01. Exclure les outliers avant interprétation des scores DEA.")

    _mah_df = dea.detect_outliers_mahalanobis(threshold_p=0.01)
    _n_outliers = _mah_df['Outlier'].sum()

    if _n_outliers > 0:
        _outlier_names = _mah_df[_mah_df['Outlier']]['Hôtel'].tolist()
        st.error(f"**{_n_outliers} outlier(s) détecté(s) :** {', '.join(_outlier_names)} — vérifier la cohérence du compset avant interprétation.")
    else:
        st.success("✅ Aucun outlier détecté (p > 0.01 pour tous les DMUs) — compset homogène.")

    st.dataframe(_mah_df, use_container_width=True, hide_index=True)
    st.caption("D² = distance de Mahalanobis au centre du nuage de points | p-value = probabilité sous H0 : 'ce DMU appartient à la distribution' | Seuil : p < 0.01")


# ══════════════════════════════════════════════
# TAB 2 — DASHBOARD KPIs
# ══════════════════════════════════════════════
with tab2:
    st.markdown('<p class="section-title">Distribution des scores d\'efficacité</p>', unsafe_allow_html=True)
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
        colors = ['#27ae60' if dea.bcc_scores[h] >= 0.95
                  else '#f39c12' if dea.bcc_scores[h] >= 0.85 else '#e74c3c'
                  for h in sorted_hotels]
        fig_bars = go.Figure(go.Bar(
            x=[dea.bcc_scores[h] for h in sorted_hotels], y=sorted_hotels,
            orientation='h', marker_color=colors, opacity=0.85,
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
            x=[dea.ccr_scores[h]], y=[dea.bcc_scores[h]], mode='markers+text',
            text=[h], textposition='top center', textfont=dict(size=9),
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

# ══════════════════════════════════════════════
# TAB 3 — CLASSEMENT TOPSIS
# ══════════════════════════════════════════════
with tab3:
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
            _se_num["_se_val"] = _se_num["Super-Efficience"].str.replace("%","").astype(float)
            fig_se = go.Figure(go.Bar(
                x=_se_num["_se_val"], y=_se_num["Hôtel"], orientation="h",
                marker=dict(
                    color=_se_num["_se_val"],
                    colorscale=[[0,"#e74c3c"],[0.7,"#f39c12"],[0.999,"#f1c40f"],[1.0,"#27ae60"],[1.5,"#1a8a4a"]],
                    cmin=0.3, cmax=1.5, showscale=True,
                    colorbar=dict(title="Score SE"),
                ),
                text=_se_num["Super-Efficience"], textposition="outside",
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
                    mode="markers+text", text=[r["Hôtel"]],
                    textposition="top center", textfont=dict(size=8),
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
                    z=_ce_matrix.values,
                    x=_ce_matrix.columns.tolist(),
                    y=_ce_matrix.index.tolist(),
                    colorscale="RdYlGn", zmin=0, zmax=1,
                    text=[[f"{v:.2f}" for v in row] for row in _ce_matrix.values],
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
        st.caption("Doyle & Green (1994) ; Andersen & Petersen (1993) — Un actif robuste est performant sur les 4 dimensions.")

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


# ══════════════════════════════════════════════
# TAB 4 — SEGMENTATION K-MEANS
# ══════════════════════════════════════════════
with tab4:
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
            _fig_elbow = _go.Figure(_go.Scatter(x=list(_inertias.keys()), y=list(_inertias.values()),
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
                mode='markers+text', text=hotels_c, textposition='top center',
                textfont=dict(size=9), marker=dict(size=12, color=color), name=lbl,
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
with tab5:
    st.markdown('<p class="section-title">📐 4 Quadrants — Gestion Pure × Efficacité d\'Échelle</p>', unsafe_allow_html=True)
    quadrant_colors = {'Q1': '#1e8449', 'Q2': '#2e6da4', 'Q3': '#f39c12', 'Q4': '#c0392b'}
    fig_q = go.Figure()
    for q, label in QUADRANT_LABELS.items():
        hotels_q = [h for h in dea.hotels if dea.quadrants.get(h) == q]
        if hotels_q:
            fig_q.add_trace(go.Scatter(
                x=[dea.scale_efficiency[h] for h in hotels_q],
                y=[dea.bcc_scores[h] for h in hotels_q],
                mode='markers+text', text=hotels_q, textposition='top center',
                textfont=dict(size=9), marker=dict(size=13, color=quadrant_colors[q], symbol='circle'),
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
- **Q2 ⚙️ Échelle** : BCC ≥ seuil MAIS Scale < seuil → Problème de taille
- **Q3 🧠 Gestion** : BCC < seuil MAIS Scale ≥ seuil → Problème de gestion pure
- **Q4 🔴 Double** : BCC < seuil ET Scale < seuil → Intervention prioritaire
        """)
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
with tab6:
    st.markdown('<p class="section-title">🔥 Slacks — Gaspillages et Potentiels d\'Amélioration</p>', unsafe_allow_html=True)
    input_names  = ['nb_lits', 'nb_employes', 'couts_op_ex']
    output_names = ['revpar', 'satisfaction', 'taux_occupation']
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
                z=slack_matrix_in, x=['Lits', 'Employés', 'Coûts Op.'], y=hotels_inefficient,
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
        upside_data = []
        for h in hotels_inefficient:
            slack_emp    = dea.slacks[h]['inputs'].get('nb_employes', 0)
            slack_revpar = dea.slacks[h]['outputs'].get('revpar', 0)
            lits         = float(dea.df.loc[h, 'nb_lits'])
            upside_fte   = round(slack_emp * avg_salary / 1000)
            upside_rev   = round(slack_revpar * lits * 365 * revpar_value / 1_000_000, 1)
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
        st.caption('PAR = Per Available Room-Night | POR = Per Occupied Room | % CA = % Chiffre Affaires total')

        par_por_rows = []
        for h in hotels_inefficient:
            lits_h   = float(dea.df.loc[h, 'nb_lits'])
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
with tab7:
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
        raw_df = pd.DataFrame({
            'Indicateur': ['Nombre de lits', 'Employés', 'Coûts Op. (M€)', 'RevPAR (€)', 'Satisfaction', 'Taux Occup. (%)'],
            'Valeur': [raw['nb_lits'], raw['nb_employes'], raw['couts_op_ex'],
                       raw['revpar'], raw['satisfaction'], raw['taux_occupation']],
        })
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
        color = '#e74c3c' if h == selected else '#aec6e8'
        size  = 16        if h == selected else 9
        fig_pos.add_trace(go.Scatter(
            x=[dea.scale_efficiency[h]], y=[dea.bcc_scores[h]], mode='markers+text',
            text=[h], textposition='top center',
            textfont=dict(size=8 if h != selected else 11, color='red' if h == selected else 'gray'),
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
    _wi_lits = float(_wi_raw['nb_lits'])
    _wi_emp  = float(_wi_raw['nb_employes'])
    _wi_opex = float(_wi_raw['couts_op_ex'])
    _wi_rvp  = float(_wi_raw['revpar'])
    _wi_sat  = float(_wi_raw['satisfaction'])
    _wi_occ  = float(_wi_raw['taux_occupation'])

    # Sliders — inputs
    _wi_col1, _wi_col2 = st.columns(2)
    with _wi_col1:
        st.markdown("**📥 Inputs (ressources)**")
        _wi_new_lits = st.slider(
            "Nombre de lits (chambres)", 
            min_value=max(10,  int(_wi_lits * 0.5)),
            max_value=int(_wi_lits * 1.5),
            value=int(_wi_lits),
            step=5, key="wi_lits"
        )
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
        "Lits"   : (_wi_lits, _wi_new_lits,  "↓ moins = mieux"),
        "ETP"    : (_wi_emp,  _wi_new_emp,   "↓ moins = mieux"),
        "OpEx"   : (_wi_opex, _wi_new_opex,  "↓ moins = mieux"),
        "RevPAR" : (_wi_rvp,  _wi_new_rvp,   "↑ plus = mieux"),
        "Sat."   : (_wi_sat,  _wi_new_sat,   "↑ plus = mieux"),
        "TO"     : (_wi_occ,  _wi_new_occ,   "↑ plus = mieux"),
    }
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
                    f"<span style='color:{"#1e8449" if d["Actuel"]>d["Simulé"] else "#c0392b"}'>{d['Δ']}</span>"
                    f"</div>", unsafe_allow_html=True)
        with _dc2:
            st.markdown("**Variation des Outputs**")
            for d in _outputs_d:
                _ico = "🟢" if d["Simulé"] > d["Actuel"] else "🔴" if d["Simulé"] < d["Actuel"] else "⚪"
                st.markdown(
                    f"<div style='padding:6px 10px;background:#f8f9fa;border-radius:4px;margin-bottom:4px;'>"
                    f"{_ico} <b>{d['Paramètre']}</b> : {d['Actuel']} → {d['Simulé']} "
                    f"<span style='color:{"#1e8449" if d["Simulé"]>d["Actuel"] else "#c0392b"}'>{d['Δ']}</span>"
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
                _df_mod.loc[selected, 'nb_lits']       = _wi_new_lits
                _df_mod.loc[selected, 'nb_employes']   = _wi_new_emp
                _df_mod.loc[selected, 'couts_op_ex']   = _wi_new_opex
                _df_mod.loc[selected, 'revpar']        = _wi_new_rvp
                _df_mod.loc[selected, 'satisfaction']  = _wi_new_sat
                _df_mod.loc[selected, 'taux_occupation'] = _wi_new_occ

                _wi_inputs_mat  = _df_mod[['nb_lits','nb_employes','couts_op_ex']].values.astype(float)
                _wi_outputs_mat = _df_mod[['revpar','satisfaction','taux_occupation']].values.astype(float)
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
with tab8:
    st.markdown('<p class="section-title">🌐 Metafrontière — Analyse GTE / MTE / TGR</p>', unsafe_allow_html=True)
    st.info("""
**Principe (Assaf et al., 2010) :**
- **GTE** — efficience relative à la frontière de son groupe
- **MTE** — efficience relative à la meilleure technologie toutes catégories
- **TGR** (MTE/GTE) — écart entre technologie du groupe et meilleure technologie possible
    """)
    seg_options = {'🏠 Taille (auto depuis nb_lits)': 'taille'}
    if 'type_gestion' in dea.df.columns:
        seg_options['⚙️ Type de gestion'] = 'type_gestion'
    if 'classement_etoiles' in dea.df.columns:
        seg_options['⭐ Classement étoiles'] = 'classement_etoiles'
    seg_label = st.selectbox("Dimension de segmentation", options=list(seg_options.keys()))
    seg_key = seg_options[seg_label]
    groups = dea.get_auto_size_groups() if seg_key == 'taille' else dea.df[seg_key].rename('groupe')
    with st.spinner("Calcul GTE / MTE / TGR…"):
        meta_df  = dea.compute_metafrontier(groups)
        meta_sum = dea.get_metafrontier_summary(meta_df)
    warns = [w for w in meta_df['_warn'].dropna().unique() if w]
    for w in warns:
        st.warning(w)
    st.dataframe(meta_sum, use_container_width=True, hide_index=True)
    display_cols = ['Hôtel', 'Groupe', 'GTE', 'MTE', 'TGR', 'Interprétation']
    st.dataframe(meta_df[display_cols].sort_values('TGR'), use_container_width=True, hide_index=True)

    grp_colors = px.colors.qualitative.Set2
    unique_grps = meta_df['Groupe'].unique()
    color_map   = {g: grp_colors[i % len(grp_colors)] for i, g in enumerate(unique_grps)}
    fig_meta = go.Figure()
    for grp in unique_grps:
        sub = meta_df[meta_df['Groupe'] == grp]
        fig_meta.add_trace(go.Scatter(
            x=sub['GTE'], y=sub['TGR'], mode='markers+text',
            text=sub['Hôtel'], textposition='top center', textfont=dict(size=9),
            marker=dict(size=12, color=color_map[grp]), name=grp,
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
    st.dataframe(_enr_df, use_container_width=True, hide_index=True)

    # Scatter market share vs BCC
    _enr_bcc = [dea.bcc_scores.get(h, 0) for h in _enr_df['Hôtel']]
    fig_ms = go.Figure()
    for i, h in enumerate(_enr_df['Hôtel']):
        _ms = _enr_df.iloc[i]['Market share (%)']
        _bcc = _enr_bcc[i]
        _color = '#27ae60' if _bcc >= 0.90 else '#f39c12' if _bcc >= 0.80 else '#e74c3c'
        fig_ms.add_trace(go.Scatter(
            x=[_ms], y=[_bcc], mode='markers+text', text=[h],
            textposition='top center', textfont=dict(size=8),
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
with tab9:
    st.markdown('<p class="section-title">💰 Efficience Capital & Flow Through</p>', unsafe_allow_html=True)
    FT_BENCH = {1: 0.30, 2: 0.35, 3: 0.45, 4: 0.55, 5: 0.65}
    FT_DEFAULT = 0.45

    init_data = {}
    for hotel in dea.hotels:
        row = {
            'surface_m2'        : float(dea.df.loc[hotel, 'surface_m2'])        if getattr(dea,'has_surface',False) else 0.0,
            'capex_annuel (k€)' : float(dea.df.loc[hotel, 'capex_annuel'])      if getattr(dea,'has_capex',  False) else 0.0,
            'gop (k€)'          : float(dea.df.loc[hotel, 'gop'])               if getattr(dea,'has_gop',    False) else 0.0,
            'classement (★)'    : int(dea.df.loc[hotel, 'classement_etoiles'])  if getattr(dea,'has_stars',  False) else 3,
        }
        init_data[hotel] = row
    df_init = pd.DataFrame(init_data).T

    if ('capital_input' not in st.session_state or
            set(st.session_state['capital_input'].index) != set(dea.hotels)):
        st.session_state['capital_input'] = df_init
    else:
        for col in ['surface_m2','capex_annuel (k€)','gop (k€)','classement (★)']:
            if df_init[col].sum() > 0:
                st.session_state['capital_input'][col] = df_init[col]

    csv_cols = []
    if getattr(dea,'has_surface',False): csv_cols.append('surface_m2')
    if getattr(dea,'has_capex',  False): csv_cols.append('capex_annuel')
    if getattr(dea,'has_gop',    False): csv_cols.append('gop')
    if getattr(dea,'has_stars',  False): csv_cols.append('classement_etoiles')
    if csv_cols:
        st.success(f"✅ Données lues depuis le CSV : {', '.join(csv_cols)}")
    else:
        st.info("Aucune colonne capital dans le CSV — saisissez les valeurs ci-dessous.")

    cap_input = st.data_editor(
        st.session_state['capital_input'], use_container_width=True, num_rows='fixed',
        column_config={
            'surface_m2'        : st.column_config.NumberColumn('Surface (m²)', min_value=0, format='%d m²'),
            'capex_annuel (k€)' : st.column_config.NumberColumn('CAPEX annuel (k€)', min_value=0, format='%.0f k€'),
            'gop (k€)'          : st.column_config.NumberColumn('GOP (k€)', min_value=0, format='%.0f k€'),
            'classement (★)'    : st.column_config.SelectboxColumn('Classement', options=[1,2,3,4,5]),
        },
        key='capital_editor',
    )
    st.session_state['capital_input'] = cap_input

    has_any = (cap_input[['surface_m2','capex_annuel (k€)','gop (k€)']].sum().sum() > 0)
    if not has_any:
        st.info("👆 Renseignez surface_m2, CAPEX et GOP dans le tableau ci-dessus pour les analyses capital.")

    if has_any:
    
        st.markdown("---")
        cap_rows = []
        for hotel in dea.hotels:
            row      = cap_input.loc[hotel]
            surf     = float(row['surface_m2']); capex_ke = float(row['capex_annuel (k€)'])
            gop_ke   = float(row['gop (k€)']);   stars    = int(row['classement (★)'])
            lits     = float(dea.df.loc[hotel, 'nb_lits']); revpar = float(dea.df.loc[hotel, 'revpar'])
            to       = float(dea.df.loc[hotel, 'taux_occupation']) / 100
            goppam   = round(gop_ke * 1000 / surf, 2)    if surf > 0    else None
            capex_ch = round(capex_ke * 1000 / lits, 0)  if lits > 0    else None
            rev_est  = revpar * to * jours_exploit * lits
            rendement= round(rev_est / (capex_ke * 1000), 2) if capex_ke > 0 else None
            ft       = FT_BENCH.get(stars, FT_DEFAULT)
            slack_r  = dea.slacks.get(hotel, {}).get('outputs', {}).get('revpar', 0)
            up_gop   = round(slack_r * lits * jours_exploit * ft / 1_000_000, 3) if slack_r > 0 else 0
            gop_margin = round(gop_ke * 1000 / rev_est * 100, 1) if gop_ke > 0 and rev_est > 0 else None
            cap_rows.append({
                'Hôtel': hotel, 'BCC': f"{dea.bcc_scores.get(hotel, 0):.1%}", 'Classement': '★' * stars,
                'Surface (m²)': int(surf) if surf > 0 else '—',
                'm²/chambre': round(surf/lits,1) if surf>0 and lits>0 else '—',
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
            if capex_vals:  st.metric("CAPEX/ch moyen", f"{sum(capex_vals)/len(capex_vals):.1f} k€")
        with c3:
            if margin_vals: st.metric("Marge GOP moyenne", f"{sum(margin_vals)/len(margin_vals):.1f}%")
        with c4:
            if upside_vals: st.metric("Upside GOP total /FT", f"{sum(upside_vals):.2f} M€/an")
    
        st.dataframe(cap_df, use_container_width=True, hide_index=True)
    
        col_l, col_r = st.columns(2)
        with col_l:
            plot_data = [(r['Hôtel'], r['GOPPAM (€/m²)'], r['CAPEX/chambre (k€)'], dea.bcc_scores.get(r['Hôtel'],0))
                         for r in cap_rows if r['GOPPAM (€/m²)'] != '—' and r['CAPEX/chambre (k€)'] != '—']
            if plot_data:
                fig_cap = go.Figure()
                for h, gop_, capx, bcc in plot_data:
                    color = '#27ae60' if bcc>=0.90 else '#f39c12' if bcc>=0.80 else '#e74c3c'
                    fig_cap.add_trace(go.Scatter(
                        x=[capx], y=[gop_], mode='markers+text', text=[h], textposition='top center',
                        textfont=dict(size=8), marker=dict(size=10+bcc*8, color=color, opacity=0.8),
                        showlegend=False,
                    ))
                fig_cap.update_layout(xaxis=dict(title="CAPEX annuel / chambre (k€)"),
                                      yaxis=dict(title="GOPPAM (€/m²)"),
                                      height=400, paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_cap, use_container_width=True)
        with col_r:
            margin_data = [(r['Hôtel'], float(r['Marge GOP %'].replace('%','')), dea.bcc_scores.get(r['Hôtel'],0))
                           for r in cap_rows if r['Marge GOP %'] != '—']
            if margin_data:
                fig_gop = go.Figure()
                for h, margin, bcc in margin_data:
                    color = '#27ae60' if bcc>=0.90 else '#f39c12' if bcc>=0.80 else '#e74c3c'
                    fig_gop.add_trace(go.Scatter(
                        x=[bcc], y=[margin], mode='markers+text', text=[h], textposition='top center',
                        textfont=dict(size=8), marker=dict(size=11, color=color, opacity=0.8), showlegend=False,
                    ))
                fig_gop.update_layout(xaxis=dict(title="Score BCC", tickformat='.0%', range=[0.3,1.05]),
                                      yaxis=dict(title="Marge GOP %"),
                                      height=400, paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_gop, use_container_width=True)
    
        st.markdown("---")
        upside_rows = []
        for r in cap_rows:
            hotel  = r['Hôtel']; stars  = int(cap_input.loc[hotel,'classement (★)'])
            ft     = FT_BENCH.get(stars, FT_DEFAULT)
            slk_r  = dea.slacks.get(hotel,{}).get('outputs',{}).get('revpar', 0)
            lits   = float(dea.df.loc[hotel, 'nb_lits'])
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
                        mode='markers+text', text=[h],
                        textposition='top center', textfont=dict(size=8),
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
                    mode="markers+text", text=[row_rf["Hôtel"]],
                    textposition="top center", textfont=dict(size=8),
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

    # Expense Flex (Russo & Legel p.33)
        st.markdown('<p class="section-title">Expense Flex & Flow Through</p>', unsafe_allow_html=True)
        st.caption('Flow Through = delta_GOP / delta_CA | Expense Flex = 1 - FT quand CA baisse | Cible standard 50%')
    
        col_fx1, col_fx2 = st.columns(2)
        with col_fx1:
            base_revpar_ft = st.number_input('RevPAR baseline N-1 ou Budget (e)', value=0.0, step=1.0, key='ft_revpar')
        with col_fx2:
            base_gop_pct_ft = st.number_input('Marge GOP% baseline (%)', value=35.0, step=0.5, key='ft_gop')
    
        flex_rows = []
        for r in cap_rows:
            hotel_ft = r['Hôtel']
            stars_ft = int(cap_input.loc[hotel_ft, 'classement (★)'])
            ft_bench_ft = FT_BENCH.get(stars_ft, FT_DEFAULT)
            lits_ft  = float(dea.df.loc[hotel_ft, 'nb_lits'])
            revpar_ft = float(dea.df.loc[hotel_ft, 'revpar'])
            ca_ft    = revpar_ft * lits_ft * 365
    
            if base_revpar_ft > 0 and base_gop_pct_ft > 0:
                ca_base_ft  = base_revpar_ft * lits_ft * 365
                gop_base_ft = ca_base_ft * base_gop_pct_ft / 100
                gop_h_ft = (float(cap_input.loc[hotel_ft, 'gop (k€)']) * 1000 if 'gop (k€)' in cap_input.columns and float(cap_input.loc[hotel_ft, 'gop (k€)']) > 0 else None)
                delta_ca_ft  = ca_ft - ca_base_ft
                if gop_h_ft is not None and abs(delta_ca_ft) > 0:
                    ft_val = round((gop_h_ft - gop_base_ft) / delta_ca_ft, 3)
                    flex_val = round(1 - ft_val, 3) if delta_ca_ft < 0 else None
                    src = 'Calcule'
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
                'Driver revenu'  : driver_ft if src == 'Calcule' else '--',
            })
    
        st.dataframe(pd.DataFrame(flex_rows), use_container_width=True, hide_index=True)
        st.info('FT > 50% = bonne conversion revenus -> profit. FT < 50% = charges variables elevees. Expense Flex calcule quand CA baisse.')

# ══════════════════════════════════════════════
# TAB 10 — BENCHMARK MARCHÉ
# ══════════════════════════════════════════════
with tab10:
    st.markdown('<p class="section-title">📊 Benchmark Marché — STR Indices & Quartiles</p>', unsafe_allow_html=True)
    col_a, col_b, col_c = st.columns(3)
    with col_a: mkt_revpar = st.number_input("RevPAR marché (€)", value=100.0, step=1.0, format="%.1f")
    with col_b: mkt_adr    = st.number_input("ADR marché (€) — optionnel", value=0.0, step=1.0, format="%.1f")
    with col_c: mkt_occ    = st.number_input("OCC marché (%) — optionnel", value=0.0, step=0.5, format="%.1f")
    st.markdown("---")

    str_rows = []
    for hotel in dea.hotels:
        rvp = float(dea.df.loc[hotel, 'revpar'])
        rgi = round(rvp / mkt_revpar * 100, 1) if mkt_revpar > 0 else None
        row = {
            'Hôtel': hotel, 'RevPAR (€)': rvp, 'RGI': rgi,
            'Signal RGI': ('🟢 Leader' if rgi and rgi >= 110 else
                           '🟡 Dans le marché' if rgi and rgi >= 90 else '🔴 Sous le marché'),
        }
        if 'taux_occupation' in dea.df.columns and mkt_occ > 0:
            occ = float(dea.df.loc[hotel, 'taux_occupation'])
            row['OCC (%)'] = occ; row['MPI'] = round(occ / mkt_occ * 100, 1)
            row['Signal MPI'] = ('🟢 Leader' if row['MPI'] >= 110 else '🟡 Marché' if row['MPI'] >= 90 else '🔴 Sous')
        # ADR : RevPAR/TO par defaut, colonne explicite si dispo
        _occ_s = float(dea.df.loc[hotel, 'taux_occupation']) / 100
        _rvp_s = float(dea.df.loc[hotel, 'revpar'])
        _adr_s = float(dea.df.loc[hotel, 'adr']) if 'adr' in dea.df.columns else (_rvp_s / _occ_s if _occ_s > 0 else 0.0)
        row['ADR (€)'] = round(_adr_s, 2)
        if mkt_adr > 0 and _adr_s > 0:
            row['ARI'] = round(_adr_s / mkt_adr * 100, 1)
            row['Signal ARI'] = ('🟢 Leader' if row['ARI'] >= 110
                                 else '🟡 Marche' if row['ARI'] >= 90 else '🔴 Sous')
        str_rows.append(row)

    str_df = pd.DataFrame(str_rows).sort_values('RGI', ascending=False)
    st.dataframe(str_df, use_container_width=True, hide_index=True)

    fig_rgi = go.Figure(go.Bar(
        x=str_df['RGI'], y=str_df['Hôtel'], orientation='h',
        marker=dict(color=str_df['RGI'], colorscale='RdYlGn', cmin=70, cmax=140,
                    showscale=True, colorbar=dict(title="RGI")),
        text=[f"{v:.0f}" for v in str_df['RGI']], textposition='outside',
    ))
    fig_rgi.add_vline(x=100, line_dash='dash', line_color='gray', annotation_text="Base marché = 100")
    fig_rgi.update_layout(title="RevPAR Generation Index (RGI)",
                          xaxis=dict(title="RGI", range=[50, max(str_df['RGI'])*1.15]),
                          height=max(350, dea.n * 30), paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_rgi, use_container_width=True)

    # Barres MPI et ARI si disponibles
    str_idx_charts = [(col, ttl, clr) for col, ttl, clr in [
        ('MPI', 'MPI Market Penetration Index (Occupation)', '#3498db'),
        ('ARI', 'ARI Average Rate Index (Tarif)', '#9b59b6'),
    ] if col in str_df.columns]

    if str_idx_charts:
        idx_c = st.columns(len(str_idx_charts))
        for ci, (icol, ititle, icolor) in enumerate(str_idx_charts):
            sub_i = str_df.dropna(subset=[icol]).sort_values(icol)
            if sub_i.empty: continue
            fig_i = go.Figure(go.Bar(
                x=sub_i[icol], y=sub_i['Hôtel'], orientation='h',
                marker=dict(color=sub_i[icol], colorscale='RdYlGn', cmin=70, cmax=140,
                            showscale=True, colorbar=dict(title=icol)),
                text=[f"{v:.0f}" for v in sub_i[icol]], textposition='outside',
            ))
            fig_i.add_vline(x=100, line_dash='dash', line_color='gray')
            fig_i.update_layout(title=ititle, xaxis=dict(range=[50, max(sub_i[icol])*1.15]),
                                height=max(280, len(sub_i)*28), paper_bgcolor='rgba(0,0,0,0)')
            idx_c[ci].plotly_chart(fig_i, use_container_width=True)

        st.info('RGI = RevPAR hôtel / RevPAR marche x100 | MPI = OCC hôtel / OCC marche x100 | ARI = ADR hôtel / ADR marche x100 | >100 = au-dessus fair share | <80 ou >130 = revalider compset (Russo & Legel Exhibit 9)')

    # Profil Compset (Exhibit 8)
    st.markdown('---')
    st.markdown('<p class="section-title">Profil Compset - Grille de coherence (Exhibit 8)</p>', unsafe_allow_html=True)
    st.caption('Valider que les DMUs sont comparables avant interpretation DEA. RGI doit rester entre 80 et 130% pour valider le compset.')

    _compset_init = pd.DataFrame({
        'Hôtel'           : dea.hotels,
        'Nb chambres'     : [int(dea.df.loc[h, 'nb_lits']) for h in dea.hotels],
        'Annee ouv.'      : [0]*dea.n,
        'Dern. renov.'    : [0]*dea.n,
        'Classement (e)'  : [3]*dea.n,
        'Affiliation'     : ['Independant']*dea.n,
        'Meeting (m2)'    : [0]*dea.n,
        'Localisation'    : ['Centre-ville']*dea.n,
        'Gestion'         : ['3rd party']*dea.n,
    }).set_index('Hôtel')
    if 'compset_profile' not in st.session_state or set(st.session_state.get('compset_profile', pd.DataFrame()).index) != set(dea.hotels):
        st.session_state['compset_profile'] = _compset_init

    _cs = st.data_editor(
        st.session_state['compset_profile'], use_container_width=True,
        column_config={
            'Nb chambres'   : st.column_config.NumberColumn('Nb ch.', min_value=0, format='%d'),
            'Annee ouv.'    : st.column_config.NumberColumn('Annee ouv.', min_value=1800, max_value=2030, format='%d'),
            'Dern. renov.'  : st.column_config.NumberColumn('Dern. renov.', min_value=1800, max_value=2030, format='%d'),
            'Classement (e)': st.column_config.SelectboxColumn('Classement', options=[1,2,3,4,5]),
            'Affiliation'   : st.column_config.SelectboxColumn('Affiliation', options=['Independant','Franchise','Mgmt contract','Owner-operated']),
            'Localisation'  : st.column_config.SelectboxColumn('Localisation', options=['Centre-ville','Suburban','Airport','Resort','Route']),
            'Gestion'       : st.column_config.SelectboxColumn('Gestion', options=['3rd party','Brand-managed','Owner-operated']),
            'Meeting (m2)'  : st.column_config.NumberColumn('Meeting m2', min_value=0, format='%d'),
        }, key='compset_editor',
    )
    st.session_state['compset_profile'] = _cs
    _ch_vals = _cs['Nb chambres'].values
    if _ch_vals.max() > 0 and _ch_vals.min() > 0:
        _ratio = _ch_vals.max() / _ch_vals.min()
        if _ratio > 3: st.warning(f'Ratio taille max/min = {_ratio:.1f}x - compset heterogene. Segmenter ou affiner.')
        else: st.success(f'Homogeneite capacite OK : ratio max/min = {_ratio:.1f}x')

    st.markdown("---")
    kpis_def = {
        'RevPAR (€)': (dea.df['revpar'], 'benefit'), 'Satisfaction': (dea.df['satisfaction'], 'benefit'),
        'TO (%)': (dea.df['taux_occupation'], 'benefit'),
        'ETP / chambre': (dea.df['nb_employes'] / dea.df['nb_lits'], 'cost'),
        'CPOR (k€/ch)': (dea.df['couts_op_ex'] / dea.df['nb_lits'], 'cost'),
        'Score BCC': (pd.Series(dea.bcc_scores), 'benefit'),
    }
    q_stats = []
    for kpi_name, (series, direction) in kpis_def.items():
        s = series.dropna()
        q_stats.append({'KPI': kpi_name, 'Direction': '↑ max' if direction=='benefit' else '↓ min',
                        'Min': round(s.min(),2), 'P25': round(s.quantile(0.25),2),
                        'Médiane': round(s.median(),2), 'P75': round(s.quantile(0.75),2),
                        'Max': round(s.max(),2), 'Moy.': round(s.mean(),2)})
    st.dataframe(pd.DataFrame(q_stats), use_container_width=True, hide_index=True)

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
    st.dataframe(pd.DataFrame(pos_rows), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown('<p class="section-title">TOPSIS Composite DEA + KPIs financiers</p>', unsafe_allow_html=True)
    st.caption("Pondération hybride Shannon entropy (α=0.6) — Vlad, Toma & Fîntîneru (2026)")

    bcc_s = pd.Series(dea.bcc_scores); etp_r = dea.df['nb_employes'] / dea.df['nb_lits']
    cpor  = dea.df['couts_op_ex'] / dea.df['nb_lits']
    dm = pd.DataFrame({
        'DEA BCC': bcc_s, 'RevPAR': dea.df['revpar'], 'Satisfaction': dea.df['satisfaction'],
        'TO%': dea.df['taux_occupation'],
        'ETP_inv': (1 / etp_r).replace([np.inf], 0), 'CPOR_inv': (1 / cpor).replace([np.inf], 0),
    }, index=dea.hotels).astype(float)

    cap_inp_tab10 = st.session_state.get('capital_input', None)
    if cap_inp_tab10 is not None and 'gop (k€)' in cap_inp_tab10.columns:
        rev_est = dea.df['revpar'] * dea.df['taux_occupation']/100 * 365 * dea.df['nb_lits']
        gop_vals = cap_inp_tab10['gop (k€)'].astype(float) * 1000
        gop_margin = (gop_vals / rev_est.replace(0, np.nan)).fillna(0)
        dm['GOP Margin'] = gop_margin

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
    w_ent = dj / dj.sum() if dj.sum() > 0 else np.ones(n_crit)/n_crit
    w = (1-alpha)*(dj/dj.sum() if dj.sum()>0 else np.ones(n_crit)/n_crit) + alpha*(np.ones(n_crit)/n_crit)
    V = Xn * w; V_plus = V.max(axis=0); V_minus = V.min(axis=0)
    S_plus = np.sqrt(((V-V_plus)**2).sum(axis=1)); S_minus = np.sqrt(((V-V_minus)**2).sum(axis=1))
    Pi = S_minus / (S_plus + S_minus + 1e-10)
    q25t, q50t, q75t = np.percentile(Pi, [25,50,75])
    topsis_bm_rows = [{'Rang': 0, 'Hôtel': hotel, 'Score Pi': round(Pi[i], 4),
                       'DEA BCC': f"{dea.bcc_scores.get(hotel,0):.1%}",
                       'RGI': round(float(dea.df.loc[hotel,'revpar'])/mkt_revpar*100,1) if mkt_revpar>0 else '—',
                       'Quartile': ('Q4 — Top 25%' if Pi[i]>=q75t else 'Q3' if Pi[i]>=q50t
                                    else 'Q2' if Pi[i]>=q25t else 'Q1 — Bottom 25%')}
                      for i, hotel in enumerate(dea.hotels)]
    topsis_bm_df = (pd.DataFrame(topsis_bm_rows).sort_values('Score Pi', ascending=False).reset_index(drop=True))
    topsis_bm_df['Rang'] = range(1, len(topsis_bm_df)+1)
    col_x, col_y = st.columns([2,1])
    with col_x: st.dataframe(topsis_bm_df, use_container_width=True, hide_index=True)
    with col_y:
        st.markdown("**Poids des critères**")
        st.dataframe(pd.DataFrame({'Critère': crit_labels, 'Poids': [f"{v:.1%}" for v in w]}),
                     use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════
    st.markdown('---')
    st.markdown('<p class="section-title">🚀 Malmquist Productivity Index — Évolution temporelle</p>', unsafe_allow_html=True)
    st.caption('Caves, Christensen & Diewert (1982) ; Fare et al. (1994) -- Catch-up x Frontier Shift = TFP total.')
    st.markdown('<p class="section-title">🚀 Malmquist Productivity Index — Évolution temporelle</p>', unsafe_allow_html=True)
    st.caption(
        "Caves, Christensen & Diewert (1982) Econometrica ; "
        "Färe, Grosskopf, Norris & Zhang (1994) American Economic Review. "
        "Décompose la variation de productivité entre N-1 et N en deux effets : "
        "Catch-up (gestion) × Frontier Shift (progrès sectoriel)."
    )

    # Vérifier colonnes N-1
    _n1_present = has_n1_cols(dea.df)
    _n1_required = ["revpar_n1", "nb_employes_n1", "couts_op_ex_n1",
                    "taux_occupation_n1", "nb_lits_n1"]
    _n1_missing  = [c for c in _n1_required if c not in dea.df.columns]

    if _n1_missing:
        _warn_msg = (
            f"Colonnes N-1 manquantes : {', '.join(_n1_missing)}. "
            "Ajoutez nb_lits_n1, nb_employes_n1, couts_op_ex_n1, "
            "revpar_n1, taux_occupation_n1 (+ satisfaction_n1 optionnel) "
            "a votre CSV pour activer le Malmquist."
        )
        st.warning(_warn_msg)
    else:
        st.success(f"✅ Colonnes N-1 détectées : {', '.join(_n1_present)}")

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
            st.dataframe(_mq, use_container_width=True, hide_index=True)

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
                            x=[_cu], y=[_fs], mode="markers+text",
                            text=[r["Hôtel"]], textposition="top center",
                            textfont=dict(size=8),
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
    st.markdown('''<p class="section-title">🧪 Tobit Second Stage -- Déterminants Efficience</p>''', unsafe_allow_html=True)
    st.caption('Tobin (1958) ; Simar & Wilson (2007) -- Regression censuree en 1.0. Quantifie l\'effet marginal de chaque variable environnementale sur le score BCC.')
    # ── Tobit Second Stage (Simar & Wilson 2007) ──────────────────────────────
    st.markdown("---")
    st.markdown('''<p class="section-title">🧪 Tobit Second Stage -- Déterminants Efficience</p>''', unsafe_allow_html=True)
    st.caption(
        "Régression Tobit censurée à droite en 1.0 sur les scores BCC. "
        "Quantifie l'effet marginal de chaque variable environnementale sur l'efficience. "
        "Réf. : Tobin (1958) Econometrica ; Simar & Wilson (2007) Journal of Econometrics."
    )

    # Sélection des régresseurs disponibles
    _tobit_candidates = {
        'classement_etoiles': 'Classement (★)',
        'surface_m2'        : 'Surface totale (m²)',
        'capex_annuel'      : 'CAPEX annuel (€)',
        'nb_lits'           : 'Nombre de chambres',
        'energy_kwh'        : 'Énergie consommée (kWh)',
        'payroll_total'     : 'Masse salariale (€)',
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
        if 'Gestion' in _cs_profile.columns:
            _gest_dummies = pd.get_dummies(_cs_profile['Gestion'], prefix='gest')
            for col in _gest_dummies.columns:
                dea.df[col] = [_gest_dummies.loc[h, col] if h in _gest_dummies.index else 0
                               for h in dea.hotels]
                _tobit_cs[col] = col.replace('gest_', 'Gestion: ')

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


# TAB 11 — SYNTHÈSE MULTI-MODULE (v3.2)
# ══════════════════════════════════════════════
with tab11:
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
            f = check_module_feasibility(mod_id, _available_now)
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
| `nb_rooms` | Nombre de chambres |
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

*Note : les colonnes DEA-H standard (`nb_lits`, `nb_employes`, `couts_op_ex`, `revpar`, `taux_occupation`, `satisfaction`)
sont automatiquement mappées vers les noms standard des modules.*
""")
    else:
        render_synthesis_tab(_module_results, dmu_col="hotel_name")


# ══════════════════════════════════════════════
# TAB 12 -- VARIANCE BUDGET
# ══════════════════════════════════════════════
with tab12:
    st.markdown('<p class="section-title">Analyse de Variance Budget -- Format USALI</p>', unsafe_allow_html=True)
    st.caption('Russo & Legel Exhibit 6 : N-1 / Budget / Realise en PAR (Per Available Room), POR (Per Occupied Room), % CA')

    sel_var = st.selectbox('Actif a analyser', options=dea.hotels, key='var_hotel_sel')
    lits_v  = float(dea.df.loc[sel_var, 'nb_lits'])
    occ_v   = float(dea.df.loc[sel_var, 'taux_occupation']) / 100
    revpar_v= float(dea.df.loc[sel_var, 'revpar'])
    nights_v= lits_v * 365 * occ_v
    par_v   = lits_v * 365
    ca_v    = revpar_v * lits_v * 365

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
    st.caption(f'PAR base = {par_v:,.0f} room-nights | POR base = {nights_v:,.0f} nuitees | CA ref = {ca_v:,.0f} euros')

    _vres = []
    for _, row_v in _edited.iterrows():
        ln_v = row_v['Poste']; mt_v = row_v['Metrique']
        n1_v  = float(row_v['N-1 (e)'])    if row_v['N-1 (e)']    else 0.0
        bud_v = float(row_v['Budget (e)'])  if row_v['Budget (e)'] else 0.0
        rea_v = float(row_v['Realise (e)']) if row_v['Realise (e)'] else 0.0

        var_b = rea_v - bud_v; var_n1 = rea_v - n1_v
        pct_b = var_b / abs(bud_v) * 100 if bud_v != 0 else 0
        pct_n1= var_n1/ abs(n1_v)  * 100 if n1_v  != 0 else 0

        base_v = par_v if mt_v=='PAR' else nights_v if mt_v=='POR' else ca_v
        def _fmt(v): return f'{v/base_v*100:.1f}%' if mt_v=='% CA' and base_v>0 else (f'{v/base_v:.2f}' if base_v>0 else '--')

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
st.caption(
    f"DEA-H v3.9 · REIV Hospitality · {datetime.now().strftime('%d/%m/%Y')} · "
    "Modèles : BCC/CCR · TOPSIS · K-means · Metafrontière · Multi-Module DEA (7 dimensions) · "
    "Méthodologie : Charnes et al. (1978), Banker et al. (1984), Min et al. (2009), Assaf et al. (2009), Poldrugovac et al. (2016), Färe et al. (1994), Yu (2012), Tobin (1958)"
)
