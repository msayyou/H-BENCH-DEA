"""
app.py — DEA-H v3.3
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
if "module_results" not in st.session_state:
    st.session_state["module_results"] = {}
if "active_modules_mm" not in st.session_state:
    st.session_state["active_modules_mm"] = []
if "variable_overrides_mm" not in st.session_state:
    st.session_state["variable_overrides_mm"] = {}

# ─────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    st.subheader("💼 Paramètres économiques")
    avg_salary   = st.number_input("Coût FTE (€/an)",           value=35_000, step=5_000, format="%i")
    revpar_value = st.number_input("Valeur 1 pt RevPAR (€/an)", value=1_000,  step=100,   format="%i")

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
    "📊 Benchmark Marché",
    "🔀 Synthese Multi-Module",
    "📊 Variance Budget",
])

# ══════════════════════════════════════════════
# TAB 1 — RAPPORT BOARD
# ══════════════════════════════════════════════
with tab1:
    st.markdown('<p class="section-title">📋 Rapport Stratégique — Comité d\'Investissement</p>', unsafe_allow_html=True)
    st.dataframe(board, use_container_width=True, hide_index=True)
    st.markdown('<p class="section-title">Répartition par Quadrant</p>', unsafe_allow_html=True)
    q_summary = dea.get_quadrant_summary()
    st.dataframe(q_summary.drop(columns=['Hôtels'], errors='ignore'), use_container_width=True, hide_index=True)

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

# ══════════════════════════════════════════════
# TAB 4 — SEGMENTATION K-MEANS
# ══════════════════════════════════════════════
with tab4:
    st.markdown('<p class="section-title">🗂️ Segmentation K-means — 4 Clusters</p>', unsafe_allow_html=True)
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
            nights_h = lits_h * 365 * occ_h
            ca_h     = revpar_h * lits_h * 365
            par_h    = lits_h * 365

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
        st.caption("👆 Aucune donnée capital disponible.")
        st.stop()

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
        rev_est  = revpar * to * 365 * lits
        rendement= round(rev_est / (capex_ke * 1000), 2) if capex_ke > 0 else None
        ft       = FT_BENCH.get(stars, FT_DEFAULT)
        slack_r  = dea.slacks.get(hotel, {}).get('outputs', {}).get('revpar', 0)
        up_gop   = round(slack_r * lits * 365 * ft / 1_000_000, 3) if slack_r > 0 else 0
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

# ─────────────────────────────────────────────
#  Footer
# ─────────────────────────────────────────────
st.markdown("---")
st.caption(
    f"DEA-H v3.3 · REIV Hospitality · {datetime.now().strftime('%d/%m/%Y')} · "
    "Modèles : BCC/CCR · TOPSIS · K-means · Metafrontière · Multi-Module DEA (7 dimensions) · "
    "Méthodologie : Charnes et al. (1978), Banker et al. (1984), Min et al. (2009), Assaf (2010)"
)
