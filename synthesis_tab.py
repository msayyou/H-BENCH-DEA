"""
synthesis_tab.py — Onglet Synthèse Croisée Multi-Module DEA-H
REIV Hospitality / VLTION PRO
-------------------------------------------------------
Affiche la vue consolidée des scores d'efficience par DMU
sur l'ensemble des modules actifs.

À appeler depuis app.py :
    from synthesis_tab import render_synthesis_tab
    render_synthesis_tab(module_results, dmu_col)
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from modules_config import MODULES, get_col_label
from dea_multi_module import (
    build_cross_synthesis,
    get_module_ranking,
    get_dmu_weaknesses,
    ModuleResult,
)

# Palette couleurs REIV
PALETTE = {
    "efficient": "#2ecc71",
    "near": "#f39c12",
    "weak": "#e74c3c",
    "neutral": "#3498db",
    "bg": "#0f1117",
    "card": "#1a1d27",
    "border": "#2d3147",
    "text": "#e8eaf6",
    "muted": "#7986cb",
}

MODULE_COLORS = {
    "operational":        "#3498db",
    "financial_usali":    "#2ecc71",
    "capital_assets":     "#9b59b6",
    "workforce":          "#f39c12",
    "revenue_management": "#1abc9c",
    "esg":                "#27ae60",
    "quality":            "#e67e22",
}


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS UI
# ─────────────────────────────────────────────────────────────────────────────

def _score_color(score: float) -> str:
    if pd.isna(score):
        return PALETTE["muted"]
    if score >= 0.999:
        return PALETTE["efficient"]
    elif score >= 0.80:
        return PALETTE["near"]
    else:
        return PALETTE["weak"]


def _score_badge(score: float) -> str:
    if pd.isna(score):
        return "—"
    color = _score_color(score)
    label = f"{score:.2%}"
    return f'<span style="color:{color};font-weight:700">{label}</span>'


def _format_score(val):
    """Formatage conditionnel pour st.dataframe."""
    if pd.isna(val):
        return ""
    return f"{val:.2%}"


# ─────────────────────────────────────────────────────────────────────────────
# COMPOSANTS GRAPHIQUES
# ─────────────────────────────────────────────────────────────────────────────

def _radar_chart(synthesis: pd.DataFrame, score_cols: list[str], dmu_names: list[str]) -> go.Figure:
    """Radar chart multi-DMU — scores par module."""
    labels = [MODULES[c]["label_fr"].split(" ", 1)[-1] for c in score_cols]

    fig = go.Figure()
    color_list = px.colors.qualitative.Set2

    for i, dmu in enumerate(dmu_names[:8]):  # max 8 DMUs
        row = synthesis[synthesis["Hôtel"] == dmu]
        if row.empty:
            continue
        vals = [row[c].values[0] if c in row.columns else np.nan for c in score_cols]
        vals_pct = [v * 100 if not pd.isna(v) else 0 for v in vals]

        fig.add_trace(go.Scatterpolar(
            r=vals_pct + [vals_pct[0]],
            theta=labels + [labels[0]],
            name=dmu,
            fill="toself",
            opacity=0.3,
            line=dict(color=color_list[i % len(color_list)], width=2),
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                ticksuffix="%",
                tickfont=dict(size=10, color=PALETTE["muted"]),
                gridcolor=PALETTE["border"],
            ),
            angularaxis=dict(
                tickfont=dict(size=11, color=PALETTE["text"]),
                gridcolor=PALETTE["border"],
            ),
            bgcolor=PALETTE["card"],
        ),
        paper_bgcolor=PALETTE["bg"],
        plot_bgcolor=PALETTE["bg"],
        font=dict(color=PALETTE["text"]),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.25,
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(t=40, b=80, l=60, r=60),
        height=480,
    )
    return fig


def _heatmap_chart(synthesis: pd.DataFrame, score_cols: list[str]) -> go.Figure:
    """Heatmap scores d'efficience DMU × Module."""
    dmus = synthesis["Hôtel"].tolist()
    z = []
    for _, row in synthesis.iterrows():
        z.append([row[c] if c in row.index and not pd.isna(row[c]) else np.nan
                  for c in score_cols])

    labels_x = [MODULES[c]["label_fr"].split(" ", 1)[-1] for c in score_cols]

    fig = go.Figure(go.Heatmap(
        z=z,
        x=labels_x,
        y=dmus,
        colorscale=[
            [0.0, "#e74c3c"],
            [0.65, "#f39c12"],
            [0.80, "#f1c40f"],
            [0.999, "#2ecc71"],
            [1.0, "#27ae60"],
        ],
        zmin=0,
        zmax=1,
        text=[[f"{v:.1%}" if not pd.isna(v) else "—" for v in row] for row in z],
        texttemplate="%{text}",
        textfont=dict(size=11, color="white"),
        hovertemplate="Hôtel : %{y}<br>Module : %{x}<br>Score : %{text}<extra></extra>",
        showscale=True,
        colorbar=dict(
            title="Efficience",
            tickformat=".0%",
            tickfont=dict(color=PALETTE["text"]),
            bgcolor=PALETTE["card"],
        ),
    ))

    fig.update_layout(
        paper_bgcolor=PALETTE["bg"],
        plot_bgcolor=PALETTE["bg"],
        font=dict(color=PALETTE["text"]),
        xaxis=dict(tickfont=dict(size=11), side="bottom"),
        yaxis=dict(tickfont=dict(size=11), autorange="reversed"),
        margin=dict(t=30, b=100, l=150, r=60),
        height=max(300, len(dmus) * 38 + 80),
    )
    return fig


def _bar_module_ranking(ranking_df: pd.DataFrame) -> go.Figure:
    """Bar chart — score moyen par module (difficulté relative)."""
    labels = [MODULES[r["module_id"]]["label_fr"].split(" ", 1)[-1]
              for _, r in ranking_df.iterrows()]
    scores = ranking_df["score_moyen"].tolist()
    colors = [MODULE_COLORS.get(r["module_id"], PALETTE["neutral"])
              for _, r in ranking_df.iterrows()]

    fig = go.Figure(go.Bar(
        x=labels,
        y=scores,
        marker_color=colors,
        text=[f"{s:.1%}" for s in scores],
        textposition="outside",
        textfont=dict(size=12, color=PALETTE["text"]),
        hovertemplate="%{x}<br>Score moyen : %{y:.2%}<extra></extra>",
    ))

    fig.add_hline(y=1.0, line_dash="dot", line_color=PALETTE["efficient"],
                  annotation_text="Frontière efficiente",
                  annotation_font_color=PALETTE["efficient"])

    fig.update_layout(
        yaxis=dict(range=[0, 1.1], tickformat=".0%", gridcolor=PALETTE["border"]),
        xaxis=dict(tickfont=dict(size=11)),
        paper_bgcolor=PALETTE["bg"],
        plot_bgcolor=PALETTE["bg"],
        font=dict(color=PALETTE["text"]),
        margin=dict(t=40, b=80, l=60, r=30),
        height=360,
        showlegend=False,
    )
    return fig


def _profil_donut(synthesis: pd.DataFrame) -> go.Figure:
    """Donut chart — répartition des profils."""
    counts = synthesis["profil"].value_counts()
    fig = go.Figure(go.Pie(
        labels=counts.index.tolist(),
        values=counts.values.tolist(),
        hole=0.55,
        marker=dict(colors=[
            PALETTE["efficient"], PALETTE["neutral"],
            PALETTE["near"], "#e67e22", PALETTE["weak"], "#95a5a6"
        ]),
        textfont=dict(size=11, color="white"),
        hovertemplate="%{label}<br>%{value} hôtels (%{percent})<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor=PALETTE["bg"],
        font=dict(color=PALETTE["text"]),
        legend=dict(font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=10, b=10, l=10, r=10),
        height=260,
        showlegend=True,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# ONGLET SYNTHÈSE PRINCIPALE
# ─────────────────────────────────────────────────────────────────────────────

def render_synthesis_tab(
    module_results: dict[str, ModuleResult],
    dmu_col: str = "hotel_name",
) -> None:
    """
    Render complet de l'onglet Synthèse Croisée.
    À appeler dans la section tabs de app.py.
    """

    st.markdown("## 🔀 Synthèse Croisée Multi-Module")

    # ── Statut des modules ──────────────────────────────────────────────────
    st.markdown("### Statut des modules")
    status_cols = st.columns(min(len(module_results), 4))
    for i, (mod_id, result) in enumerate(module_results.items()):
        col = status_cols[i % len(status_cols)]
        with col:
            if result.error:
                st.error(f"**{result.label_fr}**\n\n❌ {result.error}")
            else:
                n_eff = (result.scores["efficiency"] >= 0.999).sum()
                avg = result.scores["efficiency"].mean()
                st.success(
                    f"**{result.label_fr}**\n\n"
                    f"✅ {result.n_dmus} DMUs | Eff. moy : {avg:.1%} | "
                    f"Frontière : {n_eff} hôtel{'s' if n_eff > 1 else ''}"
                )

    st.markdown("---")

    # ── Construction synthèse ───────────────────────────────────────────────
    ok_results = {k: v for k, v in module_results.items() if not v.error and not v.scores.empty}
    if not ok_results:
        st.warning("Aucun module n'a produit de résultats. Vérifiez vos données.")
        return

    synthesis = build_cross_synthesis(ok_results)
    score_cols = [c for c in ok_results.keys() if c in synthesis.columns]

    if synthesis.empty:
        st.warning("Impossible de construire la synthèse — aucune DMU commune entre modules.")
        return

    # ── KPIs synthèse ───────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    n_dmus = len(synthesis)
    n_all_efficient = (synthesis["nb_modules_efficient"] == len(score_cols)).sum()
    avg_global = synthesis["score_moyen"].mean()
    n_critical = (synthesis["score_moyen"] < 0.65).sum()

    k1.metric("Hôtels analysés", f"{n_dmus}")
    k2.metric("Excellence globale", f"{n_all_efficient}", help="Efficient sur tous les modules actifs")
    k3.metric("Score moyen global", f"{avg_global:.1%}")
    k4.metric("Situations critiques", f"{n_critical}", help="Score moyen < 65%",
              delta=f"-{n_critical}" if n_critical > 0 else "0",
              delta_color="inverse")

    st.markdown("---")

    # ── Vue tabulaire ───────────────────────────────────────────────────────
    st.markdown("### 📊 Tableau de synthèse — Scores par module")

    display_df = synthesis.copy()
    # Renommer colonnes modules pour affichage
    rename_map = {
        c: MODULES[c]["label_fr"].split(" ", 1)[-1]
        for c in score_cols if c in display_df.columns
    }
    display_df = display_df.rename(columns=rename_map)
    display_df["Score moyen"] = display_df["score_moyen"].apply(_format_score)
    display_df["Modules efficients"] = display_df["nb_modules_efficient"].apply(
        lambda x: f"{x} / {len(score_cols)}"
    )

    show_cols = ["Hôtel"] + list(rename_map.values()) + ["Score moyen", "Modules efficients", "profil"]
    show_cols = [c for c in show_cols if c in display_df.columns]

    # Formatage scores en %
    styled = display_df[show_cols].style.format(
        {v: lambda x: f"{x:.1%}" if pd.notna(x) and isinstance(x, float) else x
         for v in rename_map.values()},
        na_rep="—"
    ).background_gradient(
        subset=list(rename_map.values()),
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
    )

    st.dataframe(styled, use_container_width=True, height=min(600, n_dmus * 38 + 60))

    # ── Export CSV ──────────────────────────────────────────────────────────
    csv = synthesis.to_csv(index=False, sep=";", decimal=",")
    st.download_button(
        "⬇️ Exporter synthèse (CSV)",
        data=csv.encode("utf-8-sig"),
        file_name="deah_synthese_multimodule.csv",
        mime="text/csv",
    )

    st.markdown("---")

    # ── Heatmap ─────────────────────────────────────────────────────────────
    st.markdown("### 🌡️ Heatmap d'efficience — DMU × Module")
    st.plotly_chart(_heatmap_chart(synthesis, score_cols), use_container_width=True)

    # ── Radar + Donut ────────────────────────────────────────────────────────
    col_r, col_d = st.columns([2, 1])

    with col_r:
        st.markdown("### 🕸️ Radar multi-DMU")
        if len(score_cols) >= 3:
            all_dmus = synthesis["Hôtel"].tolist()
            selected_dmus = st.multiselect(
                "Sélectionner les hôtels à comparer",
                options=all_dmus,
                default=all_dmus[:min(5, len(all_dmus))],
                key="synthesis_radar_dmus",
            )
            if selected_dmus and len(score_cols) >= 3:
                st.plotly_chart(
                    _radar_chart(synthesis, score_cols, selected_dmus),
                    use_container_width=True,
                )
        else:
            st.info("Radar disponible avec ≥ 3 modules actifs.")

    with col_d:
        st.markdown("### 🍩 Profils")
        st.plotly_chart(_profil_donut(synthesis), use_container_width=True)
        st.markdown("""
**Légende profils :**
- 🟢 Excellence globale — efficient sur ≥ 80% modules
- 🔵 Efficience équilibrée — performant, homogène
- 🟡 Spécialiste — forces et faiblesses marquées
- 🟠 Proche frontière — potentiel sous-exploité
- 🔴 Inefficience significative — action prioritaire
""")

    st.markdown("---")

    # ── Ranking modules par difficulté ──────────────────────────────────────
    st.markdown("### 📉 Difficulté relative par module")
    st.caption(
        "Un score moyen bas indique un module plus discriminant (frontière plus difficile à atteindre). "
        "Utile pour calibrer les priorités opérationnelles."
    )
    ranking_df = get_module_ranking(synthesis, score_cols)
    if not ranking_df.empty:
        ranking_df["label"] = ranking_df["module_id"].apply(
            lambda x: MODULES[x]["label_fr"].split(" ", 1)[-1]
        )
        st.plotly_chart(_bar_module_ranking(ranking_df), use_container_width=True)

        ranking_display = ranking_df[["label", "score_moyen", "score_median", "nb_efficients", "std"]].copy()
        ranking_display.columns = ["Module", "Score moyen", "Médiane", "Nb efficients", "Écart-type"]
        for col in ["Score moyen", "Médiane", "Écart-type"]:
            ranking_display[col] = ranking_display[col].apply(lambda x: f"{x:.2%}")
        st.dataframe(ranking_display, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Drill-down DMU ──────────────────────────────────────────────────────
    st.markdown("### 🔍 Drill-down Hôtel")

    selected_dmu = st.selectbox(
        "Sélectionner un hôtel",
        options=synthesis["Hôtel"].tolist(),
        key="synthesis_dmu_drilldown",
    )

    if selected_dmu:
        dmu_row = synthesis[synthesis["Hôtel"] == selected_dmu].iloc[0]

        # Fiche synthèse DMU
        c1, c2, c3 = st.columns(3)
        c1.metric("Score moyen", f"{dmu_row['score_moyen']:.1%}")
        c2.metric(
            "Modules efficients",
            f"{int(dmu_row['nb_modules_efficient'])} / {len(score_cols)}"
        )
        c3.markdown(f"**Profil :** {dmu_row['profil']}")

        # Scores par module
        st.markdown("**Scores par module :**")
        mod_cols = st.columns(min(len(score_cols), 4))
        for i, mod_id in enumerate(score_cols):
            val = dmu_row[mod_id] if mod_id in dmu_row.index else np.nan
            label = MODULES[mod_id]["label_fr"]
            color = _score_color(val)
            mod_cols[i % len(mod_cols)].markdown(
                f"""
                <div style="
                    background:{PALETTE['card']};
                    border:1px solid {color};
                    border-radius:8px;
                    padding:10px;
                    text-align:center;
                    margin-bottom:8px;
                ">
                    <div style="font-size:12px;color:{PALETTE['muted']}">{label}</div>
                    <div style="font-size:22px;font-weight:700;color:{color}">
                        {'—' if pd.isna(val) else f'{val:.1%}'}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Faiblesses et priorités
        weaknesses = get_dmu_weaknesses(selected_dmu, synthesis, score_cols, threshold=0.85)
        if weaknesses:
            st.markdown("**⚠️ Axes d'amélioration prioritaires :**")
            for w in weaknesses:
                mod_label = MODULES[w["module_id"]]["label_fr"]
                prio_color = PALETTE["weak"] if w["priority"] == "haute" else PALETTE["near"]
                st.markdown(
                    f"""
                    <div style="
                        background:{PALETTE['card']};
                        border-left:4px solid {prio_color};
                        padding:10px 14px;
                        border-radius:4px;
                        margin-bottom:6px;
                    ">
                        <b style="color:{prio_color}">[Priorité {w['priority']}]</b>
                        &nbsp; {mod_label} — Score : {w['score']:.1%}
                        &nbsp;|&nbsp; Écart frontière : -{w['gap_to_frontier']:.1%}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.success(f"✅ {selected_dmu} est efficient ou proche de la frontière sur tous les modules actifs.")

        # Détail slacks par module
        with st.expander("📋 Détail slacks & pairs par module"):
            for mod_id in score_cols:
                result = ok_results.get(mod_id)
                if result is None or result.scores.empty:
                    continue
                dmu_scores = result.scores[result.scores["dmu_name"] == selected_dmu]
                if dmu_scores.empty:
                    continue
                row = dmu_scores.iloc[0]
                st.markdown(f"**{result.label_fr}** — Efficience : {row['efficiency']:.2%}")
                if isinstance(row.get("slacks_in"), dict):
                    slack_data = {
                        **{f"Slack IN — {get_col_label(mod_id, k)}": v
                           for k, v in row["slacks_in"].items()},
                        **{f"Slack OUT — {get_col_label(mod_id, k)}": v
                           for k, v in row["slacks_out"].items()},
                    }
                    slack_df = pd.DataFrame(
                        slack_data.items(), columns=["Variable", "Slack (unité)"]
                    )
                    st.dataframe(slack_df, hide_index=True, use_container_width=True)
                st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR MODULE SELECTOR (à appeler depuis app.py)
# ─────────────────────────────────────────────────────────────────────────────

def render_module_selector(available_cols: list[str]) -> tuple[list[str], dict]:
    """
    Affiche le sélecteur de modules dans la sidebar.
    Retourne :
        - active_modules : liste des modules cochés
        - variable_overrides : {module_id: {"inputs": [...], "outputs": [...]}}
    """
    from modules_config import (
        MODULES, check_module_feasibility,
        get_all_input_cols, get_all_output_cols
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔬 Modules DEA")
    st.sidebar.caption("Sélectionner les dimensions d'analyse")

    active_modules = []
    variable_overrides = {}

    for mod_id, mod_cfg in MODULES.items():
        feasibility = check_module_feasibility(mod_id, available_cols)
        label = mod_cfg["label_fr"]
        coverage = feasibility["coverage_pct"]

        # Badge couverture
        if feasibility["feasible"]:
            badge = f"✅ {coverage:.0f}% données"
        else:
            badge = f"❌ Données manquantes"

        enabled = st.sidebar.checkbox(
            f"{label}",
            value=feasibility["feasible"],
            disabled=not feasibility["feasible"],
            key=f"module_check_{mod_id}",
            help=f"{mod_cfg['description']}\n\n{badge}\n\nRéférence : {mod_cfg['references']}",
        )

        st.sidebar.caption(badge)

        if enabled and feasibility["feasible"]:
            active_modules.append(mod_id)

            # Sélecteur de variables (expander dans sidebar)
            with st.sidebar.expander(f"⚙️ Variables — {label.split(' ', 1)[-1]}", expanded=False):
                all_in = get_all_input_cols(mod_id)
                all_out = get_all_output_cols(mod_id)

                avail_in = [c for c in all_in if c in available_cols]
                avail_out = [c for c in all_out if c in available_cols]

                if avail_in:
                    sel_in = st.multiselect(
                        "Inputs",
                        options=avail_in,
                        default=avail_in,
                        format_func=lambda c: mod_cfg["inputs"].get(c, {}).get("label", c),
                        key=f"var_in_{mod_id}",
                    )
                else:
                    sel_in = []

                if avail_out:
                    sel_out = st.multiselect(
                        "Outputs",
                        options=avail_out,
                        default=avail_out,
                        format_func=lambda c: mod_cfg["outputs"].get(c, {}).get("label", c),
                        key=f"var_out_{mod_id}",
                    )
                else:
                    sel_out = []

                if sel_in != avail_in or sel_out != avail_out:
                    variable_overrides[mod_id] = {"inputs": sel_in, "outputs": sel_out}

    st.sidebar.markdown("---")
    return active_modules, variable_overrides
