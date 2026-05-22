"""
dea_model.py — DEA-H v3.2
═══════════════════════════════════════════════════════════════════════════════
Moteur DEA unifié — REIV Hospitality

PARTIE 1 — HotelDEAAnalyzer  (solveur PuLP/CBC)
  BCC/CCR · Slacks & Targets · TOPSIS · K-means · 4 Quadrants
  Metafrontière GTE/MTE/TGR · Capital & Flow Through

PARTIE 2 — Multi-Module DEA  (moteur scipy/HiGHS)
  7 dimensions d'analyse · runner multi-module · synthèse croisée DMU × module

Réf. : Charnes et al. (1978), Banker et al. (1984), Barros (2005),
       Min et al. (2009), Assaf et al. (2010)
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pulp
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════════════════════════════════
#  CONSTANTES
# ═══════════════════════════════════════════════════════════════════════════════

INPUT_COLS  = ['nb_lits', 'nb_employes', 'couts_op_ex']
OUTPUT_COLS = ['revpar', 'satisfaction', 'taux_occupation']

QUADRANT_LABELS = {
    'Q1': '🏆 Efficient',
    'Q2': '⚙️ Problème Échelle',
    'Q3': '🧠 Problème Gestion',
    'Q4': '🔴 Double Peine',
}


# ═══════════════════════════════════════════════════════════════════════════════
#  PARTIE 1 — HotelDEAAnalyzer  (PuLP/CBC)
# ═══════════════════════════════════════════════════════════════════════════════

class HotelDEAAnalyzer:
    """
    Analyse DEA complète (BCC + CCR) pour portefeuille hôtelier.
    orientation='input'  → minimisation des ressources  (Charnes et al. 1978)
    orientation='output' → maximisation des revenus     (Barros 2005)
    Enrichi : TOPSIS · K-means · 4 Quadrants · Metafrontière · Capital & Flow Through
    """

    def __init__(self, df: pd.DataFrame, orientation: str = 'input'):
        self.df          = df.copy()
        self.hotels      = df.index.tolist()
        self.n           = len(self.hotels)
        self.orientation = orientation.lower()

        self.has_trevpar = 'total_revenue'      in df.columns
        self.has_goppam  = ('surface_m2' in df.columns and 'gop' in df.columns)
        self.has_surface = 'surface_m2'         in df.columns
        self.has_capex   = 'capex_annuel'       in df.columns
        self.has_gop     = 'gop'                in df.columns
        self.has_stars   = 'classement_etoiles' in df.columns
        self.has_flow    = ('gop_n1' in df.columns and 'revenu_n1' in df.columns)

        self.inputs  = df[INPUT_COLS].values.astype(float)
        self.outputs = df[OUTPUT_COLS].values.astype(float)

        self.bcc_scores      : Dict[str, float]            = {}
        self.ccr_scores      : Dict[str, float]            = {}
        self.scale_efficiency: Dict[str, float]            = {}
        self.slacks          : Dict[str, Dict]             = {}
        self.peers           : Dict[str, Dict[str, float]] = {}
        self.targets         : Dict[str, Dict]             = {}
        self.topsis_scores   : Dict[str, float]            = {}
        self.topsis_ranks    : Dict[str, int]              = {}
        self.kmeans_clusters : Dict[str, int]              = {}
        self.kmeans_labels   : Dict[str, str]              = {}
        self.quadrants       : Dict[str, str]              = {}
        self.trevpar         : Dict[str, Optional[float]]  = {}
        self.goppam          : Dict[str, Optional[float]]  = {}
        self.capital_metrics : Dict[str, Dict]             = {}

        self._run_analysis()

    # ─────────────────────────────────────────────
    #  Orchestration
    # ─────────────────────────────────────────────
    def _run_analysis(self):
        for i, hotel in enumerate(self.hotels):
            bcc_theta, bcc_lambdas = self._solve_dea(i, rts='vrs')
            ccr_theta, _           = self._solve_dea(i, rts='crs')
            self.bcc_scores[hotel] = round(min(bcc_theta, 1.0), 6)
            self.ccr_scores[hotel] = round(min(ccr_theta, 1.0), 6)
            self.peers[hotel] = {
                self.hotels[k]: v for k, v in bcc_lambdas.items()
                if v > 1e-5 and k != i
            }
            self._compute_slacks_and_targets(i, hotel, bcc_theta, bcc_lambdas)

        for hotel in self.hotels:
            bcc = self.bcc_scores[hotel]; ccr = self.ccr_scores[hotel]
            self.scale_efficiency[hotel] = round(ccr / bcc, 6) if bcc > 0 else 0.0

        self._compute_optional_metrics()
        self._compute_quadrants()
        self._compute_topsis()
        self._compute_kmeans()

    # ─────────────────────────────────────────────
    #  Solveur PuLP
    # ─────────────────────────────────────────────
    def _solve_dea(self, idx: int, rts: str = 'vrs') -> Tuple[float, Dict[int, float]]:
        return (self._solve_input(idx, rts) if self.orientation == 'input'
                else self._solve_output(idx, rts))

    def _solve_input(self, idx: int, rts: str) -> Tuple[float, Dict[int, float]]:
        model   = pulp.LpProblem(f"DEA_in_{idx}_{rts}", pulp.LpMinimize)
        theta   = pulp.LpVariable("theta", lowBound=0)
        lambdas = pulp.LpVariable.dicts("lambda", range(self.n), lowBound=0)
        model  += theta
        for j in range(self.inputs.shape[1]):
            model += (pulp.lpSum(lambdas[k] * self.inputs[k, j] for k in range(self.n))
                      <= theta * self.inputs[idx, j])
        for j in range(self.outputs.shape[1]):
            model += (pulp.lpSum(lambdas[k] * self.outputs[k, j] for k in range(self.n))
                      >= self.outputs[idx, j])
        if rts == 'vrs':
            model += pulp.lpSum(lambdas.values()) == 1
        model.solve(pulp.PULP_CBC_CMD(msg=False))
        return (pulp.value(theta) or 1.0), {k: (pulp.value(lambdas[k]) or 0.0) for k in range(self.n)}

    def _solve_output(self, idx: int, rts: str) -> Tuple[float, Dict[int, float]]:
        model   = pulp.LpProblem(f"DEA_out_{idx}_{rts}", pulp.LpMaximize)
        phi     = pulp.LpVariable("phi", lowBound=1)
        lambdas = pulp.LpVariable.dicts("lambda", range(self.n), lowBound=0)
        model  += phi
        for j in range(self.inputs.shape[1]):
            model += (pulp.lpSum(lambdas[k] * self.inputs[k, j] for k in range(self.n))
                      <= self.inputs[idx, j])
        for j in range(self.outputs.shape[1]):
            model += (pulp.lpSum(lambdas[k] * self.outputs[k, j] for k in range(self.n))
                      >= phi * self.outputs[idx, j])
        if rts == 'vrs':
            model += pulp.lpSum(lambdas.values()) == 1
        model.solve(pulp.PULP_CBC_CMD(msg=False))
        phi_star = pulp.value(phi) or 1.0
        lam_star = {k: (pulp.value(lambdas[k]) or 0.0) for k in range(self.n)}
        return (1.0 / phi_star if phi_star > 0 else 0.0), lam_star

    # ─────────────────────────────────────────────
    #  Slacks & Targets
    # ─────────────────────────────────────────────
    def _compute_slacks_and_targets(self, idx, hotel, theta, lambdas):
        in_slacks  = [max(0.0, self.inputs[idx, j] * theta
                          - sum(lambdas[k] * self.inputs[k, j] for k in range(self.n)))
                      for j in range(self.inputs.shape[1])]
        out_slacks = [max(0.0, sum(lambdas[k] * self.outputs[k, j] for k in range(self.n))
                          - self.outputs[idx, j])
                      for j in range(self.outputs.shape[1])]
        self.slacks[hotel] = {
            'inputs' : {INPUT_COLS[j]: in_slacks[j]  for j in range(len(INPUT_COLS))},
            'outputs': {OUTPUT_COLS[j]: out_slacks[j] for j in range(len(OUTPUT_COLS))},
        }
        self.targets[hotel] = {
            'inputs' : {INPUT_COLS[j]: max(0.0, self.inputs[idx, j] * theta - in_slacks[j])
                        for j in range(len(INPUT_COLS))},
            'outputs': {OUTPUT_COLS[j]: self.outputs[idx, j] + out_slacks[j]
                        for j in range(len(OUTPUT_COLS))},
        }

    # ─────────────────────────────────────────────
    #  Colonnes optionnelles / Capital
    # ─────────────────────────────────────────────
    def _compute_optional_metrics(self):
        FT_BENCHMARK = {1: 0.30, 2: 0.35, 3: 0.45, 4: 0.55, 5: 0.65}
        FT_DEFAULT   = 0.45

        for hotel in self.hotels:
            h    = self.df.loc[hotel]
            lits = float(h['nb_lits'])

            self.trevpar[hotel] = (
                round(float(h['total_revenue']) / (lits * 365), 2)
                if self.has_trevpar and lits > 0 else None
            )
            self.goppam[hotel] = (
                round(float(h['gop']) / float(h['surface_m2']), 2)
                if self.has_goppam and float(h['surface_m2']) > 0 else None
            )

            cap = {
                'surface_m2'     : float(h['surface_m2']) if self.has_surface else None,
                'm2_par_chambre' : (round(float(h['surface_m2']) / lits, 1)
                                    if self.has_surface and lits > 0 else None),
                'goppam'         : self.goppam[hotel],
            }

            if self.has_capex:
                capex = float(h['capex_annuel'])
                total_rev = (float(h['total_revenue']) if self.has_trevpar
                             else float(h['revpar']) * float(h['taux_occupation']) / 100 * 365 * lits)
                cap['capex_annuel']      = capex
                cap['capex_par_chambre'] = round(capex / lits, 0) if lits > 0 else None
                cap['rendement_capex']   = round(total_rev / capex, 2) if capex > 0 else None
            else:
                cap['capex_annuel'] = cap['capex_par_chambre'] = cap['rendement_capex'] = None

            if self.has_flow:
                delta_gop = float(h['gop']) - float(h['gop_n1'])
                delta_rev = ((float(h['total_revenue']) if self.has_trevpar else 0)
                             - float(h['revenu_n1']))
                ft = round(delta_gop / delta_rev, 3) if abs(delta_rev) > 0 else None
                cap['flow_through'] = ft
                cap['ft_source']    = 'historique' if ft is not None else 'n/a (Δ revenu = 0)'
            else:
                stars = int(h['classement_etoiles']) if self.has_stars else 3
                ft    = FT_BENCHMARK.get(stars, FT_DEFAULT)
                cap['flow_through'] = ft
                cap['ft_source']    = 'benchmark marché'

            cap['ft_qualite'] = ('✅ Excellent' if ft and ft >= 0.55 else
                                 '🟡 Correct'  if ft and ft >= 0.40 else
                                 '🔴 Faible'   if ft else '—')

            slack_r = self.slacks.get(hotel, {}).get('outputs', {}).get('revpar', 0)
            cap['upside_gop_ft'] = (round(slack_r * lits * 365 * ft / 1_000_000, 3)
                                    if ft else None)
            self.capital_metrics[hotel] = cap

    # ─────────────────────────────────────────────
    #  4 Quadrants
    # ─────────────────────────────────────────────
    def _compute_quadrants(self, bcc_threshold: float = 0.90, scale_threshold: float = 0.90):
        for hotel in self.hotels:
            bcc   = self.bcc_scores[hotel]
            scale = self.scale_efficiency[hotel]
            if   bcc >= bcc_threshold   and scale >= scale_threshold: q = 'Q1'
            elif bcc >= bcc_threshold   and scale < scale_threshold:  q = 'Q2'
            elif bcc < bcc_threshold    and scale >= scale_threshold:  q = 'Q3'
            else:                                                       q = 'Q4'
            self.quadrants[hotel] = q

    # ─────────────────────────────────────────────
    #  TOPSIS
    # ─────────────────────────────────────────────
    def _compute_topsis(self):
        matrix = np.array([
            [self.bcc_scores[h], self.scale_efficiency[h],
             float(self.df.loc[h, 'revpar']), float(self.df.loc[h, 'taux_occupation'])]
            for h in self.hotels
        ], dtype=float)
        weights = np.array([0.35, 0.25, 0.25, 0.15])
        norms   = np.sqrt((matrix ** 2).sum(axis=0)); norms[norms == 0] = 1e-10
        weighted   = (matrix / norms) * weights
        ideal_pos  = weighted.max(axis=0); ideal_neg = weighted.min(axis=0)
        d_pos = np.sqrt(((weighted - ideal_pos) ** 2).sum(axis=1))
        d_neg = np.sqrt(((weighted - ideal_neg) ** 2).sum(axis=1))
        denom = d_pos + d_neg; denom[denom == 0] = 1e-10
        scores = d_neg / denom
        ranks  = (-scores).argsort().argsort() + 1
        for i, h in enumerate(self.hotels):
            self.topsis_scores[h] = round(float(scores[i]), 4)
            self.topsis_ranks[h]  = int(ranks[i])

    # ─────────────────────────────────────────────
    #  K-means
    # ─────────────────────────────────────────────
    def _compute_kmeans(self, n_clusters: int = 4, random_state: int = 42):
        if self.n < n_clusters:
            for h in self.hotels:
                self.kmeans_clusters[h] = 0; self.kmeans_labels[h] = 'Groupe Unique'
            return
        features = np.array([
            [self.bcc_scores[h], self.scale_efficiency[h],
             self.topsis_scores[h], float(self.df.loc[h, 'revpar'])]
            for h in self.hotels
        ], dtype=float)
        features_scaled = MinMaxScaler().fit_transform(features)
        km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        labels = km.fit_predict(features_scaled)
        cluster_bcc = {c: features[labels == c, 0].mean() for c in range(n_clusters) if (labels == c).any()}
        sorted_c = sorted(cluster_bcc, key=lambda c: cluster_bcc[c], reverse=True)
        name_map = {sorted_c[0]: '🏆 Leaders', sorted_c[1]: '📈 Intermédiaires',
                    sorted_c[2]: '⚠️ Sous-performants', sorted_c[3]: '🔴 Critiques'}
        for i, h in enumerate(self.hotels):
            c = int(labels[i])
            self.kmeans_clusters[h] = c
            self.kmeans_labels[h]   = name_map.get(c, f'Cluster {c}')

    # ─────────────────────────────────────────────
    #  Metafrontière (Assaf et al. 2010)
    # ─────────────────────────────────────────────
    def _solve_dea_subgroup(self, idx_in_group, grp_inputs, grp_outputs, rts='vrs'):
        n_grp = grp_inputs.shape[0]
        if self.orientation == 'input':
            model   = pulp.LpProblem(f"DEA_grp_in_{idx_in_group}", pulp.LpMinimize)
            theta   = pulp.LpVariable("theta", lowBound=0)
            lambdas = pulp.LpVariable.dicts("lam", range(n_grp), lowBound=0)
            model  += theta
            for j in range(grp_inputs.shape[1]):
                model += (pulp.lpSum(lambdas[k]*grp_inputs[k,j] for k in range(n_grp))
                          <= theta * grp_inputs[idx_in_group, j])
            for j in range(grp_outputs.shape[1]):
                model += (pulp.lpSum(lambdas[k]*grp_outputs[k,j] for k in range(n_grp))
                          >= grp_outputs[idx_in_group, j])
            if rts == 'vrs': model += pulp.lpSum(lambdas.values()) == 1
            model.solve(pulp.PULP_CBC_CMD(msg=False))
            return min(pulp.value(theta) or 1.0, 1.0)
        else:
            model   = pulp.LpProblem(f"DEA_grp_out_{idx_in_group}", pulp.LpMaximize)
            phi     = pulp.LpVariable("phi", lowBound=1)
            lambdas = pulp.LpVariable.dicts("lam", range(n_grp), lowBound=0)
            model  += phi
            for j in range(grp_inputs.shape[1]):
                model += (pulp.lpSum(lambdas[k]*grp_inputs[k,j] for k in range(n_grp))
                          <= grp_inputs[idx_in_group, j])
            for j in range(grp_outputs.shape[1]):
                model += (pulp.lpSum(lambdas[k]*grp_outputs[k,j] for k in range(n_grp))
                          >= phi * grp_outputs[idx_in_group, j])
            if rts == 'vrs': model += pulp.lpSum(lambdas.values()) == 1
            model.solve(pulp.PULP_CBC_CMD(msg=False))
            phi_val = pulp.value(phi) or 1.0
            return min(1.0 / phi_val, 1.0) if phi_val > 0 else 0.0

    def get_auto_size_groups(self) -> pd.Series:
        def size_label(n):
            if n < 100: return '🏠 Petit (<100 ch.)'
            if n < 200: return '🏨 Moyen (100-199 ch.)'
            return '🏢 Grand (>=200 ch.)'
        return self.df['nb_lits'].apply(size_label).rename('groupe')

    def compute_metafrontier(self, groups: pd.Series, rts: str = 'vrs') -> pd.DataFrame:
        mte_scores  = {h: min(self.bcc_scores[h], 1.0) for h in self.hotels}
        gte_scores  = {}; group_warns = {}
        for grp in groups.loc[self.hotels].unique():
            grp_hotels = [h for h in self.hotels if groups.get(h) == grp]
            group_warns[grp] = None
            if len(grp_hotels) < 3:
                group_warns[grp] = f"Groupe '{grp}' : {len(grp_hotels)} DMU(s) — GTE non significatif"
                for h in grp_hotels: gte_scores[h] = 1.0
                continue
            grp_idx     = [self.hotels.index(h) for h in grp_hotels]
            grp_inp     = self.inputs[grp_idx]; grp_out = self.outputs[grp_idx]
            for i, h in enumerate(grp_hotels):
                gte_scores[h] = self._solve_dea_subgroup(i, grp_inp, grp_out, rts=rts)

        rows = []
        for h in self.hotels:
            grp = groups.get(h, '—'); gte = gte_scores.get(h, 1.0); mte = mte_scores.get(h, 1.0)
            tgr = round(min(mte / gte, 1.0), 4) if gte > 0 else 0.0
            rows.append({'Hôtel': h, 'Groupe': grp, 'GTE': round(gte, 4), 'MTE': round(mte, 4),
                         'TGR': tgr,
                         'Interprétation': ('✅ Leader absolu'               if gte >= 0.90 and tgr >= 0.90 else
                                            '⚙️ Bon gestionnaire, segment faible' if gte >= 0.90 else
                                            '🧠 Segment fort, gestion à améliorer' if tgr >= 0.90 else
                                            '🔴 Double gap'),
                         '_warn': group_warns.get(grp)})
        df_out = pd.DataFrame(rows); df_out['_warn'] = df_out['_warn'].fillna('')
        return df_out

    def get_metafrontier_summary(self, meta_df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for grp, sub in meta_df.groupby('Groupe'):
            rows.append({'Groupe': grp, 'N hôtels': len(sub),
                         'GTE moyen': f"{sub['GTE'].mean():.1%}", 'MTE moyen': f"{sub['MTE'].mean():.1%}",
                         'TGR moyen': f"{sub['TGR'].mean():.1%}", 'Meilleur TGR': f"{sub['TGR'].max():.1%}",
                         'Pire TGR': f"{sub['TGR'].min():.1%}"})
        return pd.DataFrame(rows).sort_values('TGR moyen', ascending=False)

    # ─────────────────────────────────────────────
    #  Format Barros (2005)
    # ─────────────────────────────────────────────
    def get_barros_table(self, hotel: str) -> pd.DataFrame:
        rows = []; bcc = self.bcc_scores[hotel]
        for col_name in INPUT_COLS:
            current  = float(self.df.loc[hotel, col_name])
            target   = self.targets[hotel]['inputs'].get(col_name, current)
            slack_v  = self.slacks[hotel]['inputs'].get(col_name, 0)
            radial   = current * bcc - current if self.orientation == 'input' else 0.0
            if self.orientation == 'output': slack_v = current - target
            rows.append({'Variable': f"↓ {col_name.replace('_',' ').title()}",
                         'Valeur Actuelle': round(current, 2), 'Mvt. Radial': round(radial, 2),
                         'Slack': round(slack_v, 2), 'Valeur Projetée': round(target, 2),
                         'Amélioration %': f"{(target - current) / current * 100:+.1f}%" if current > 0 else '—'})
        for col_name in OUTPUT_COLS:
            current  = float(self.df.loc[hotel, col_name])
            target   = self.targets[hotel]['outputs'].get(col_name, current)
            slack_v  = self.slacks[hotel]['outputs'].get(col_name, 0)
            radial   = current * (1 / bcc) - current if self.orientation == 'output' else 0.0
            rows.append({'Variable': f"↑ {col_name.replace('_',' ').title()}",
                         'Valeur Actuelle': round(current, 2), 'Mvt. Radial': round(radial, 2),
                         'Slack': round(slack_v, 2), 'Valeur Projetée': round(target, 2),
                         'Amélioration %': f"{(target - current) / current * 100:+.1f}%" if current > 0 else '—'})
        return pd.DataFrame(rows)

    @property
    def orientation_label(self) -> str:
        return ("📥 Input-Oriented (réduction des ressources)" if self.orientation == 'input'
                else "📤 Output-Oriented (maximisation des revenus)")

    # ─────────────────────────────────────────────
    #  Rapport Board
    # ─────────────────────────────────────────────
    def generate_board_report(self, avg_salary: float = 35_000, revpar_value: float = 1_000) -> pd.DataFrame:
        rows = []
        for hotel in self.hotels:
            bcc = self.bcc_scores[hotel]; ccr = self.ccr_scores[hotel]; scale = self.scale_efficiency[hotel]
            slack_emp  = self.slacks[hotel]['inputs'].get('nb_employes', 0)
            slack_rev  = self.slacks[hotel]['outputs'].get('revpar', 0)
            lits       = float(self.df.loc[hotel, 'nb_lits'])
            upside_fte = round(slack_emp * avg_salary)
            upside_rev = round(slack_rev * lits * 365 * revpar_value / 1_000)
            row = {'Hôtel': hotel, 'Score BCC': f"{bcc:.1%}", 'Score CCR': f"{ccr:.1%}",
                   'Eff. Échelle': f"{scale:.1%}",
                   'Quadrant': QUADRANT_LABELS.get(self.quadrants.get(hotel, ''), '—'),
                   'Rang TOPSIS': self.topsis_ranks.get(hotel, '—'),
                   'Score TOPSIS': f"{self.topsis_scores.get(hotel, 0):.3f}",
                   'Segment K-means': self.kmeans_labels.get(hotel, '—'),
                   'Upside ETP (€/an)': f"{upside_fte:,}" if upside_fte > 0 else '—',
                   'Upside RevPAR (k€)': f"{upside_rev:,}" if upside_rev > 0 else '—',
                   'Upside Total (k€)': f"{upside_fte + upside_rev:,}" if upside_fte + upside_rev > 0 else '—',
                   'Priorité': ('✅ RAS' if bcc >= 0.95 else '🟡 Surveiller' if bcc >= 0.85 else '🔴 Action urgente')}
            if self.has_trevpar: row['TRevPAR (€/lit/j)'] = self.trevpar.get(hotel, '—') or '—'
            if self.has_goppam:  row['GOPPAM (€/m²)']     = self.goppam.get(hotel, '—')  or '—'
            rows.append(row)
        return pd.DataFrame(rows).sort_values('Rang TOPSIS').reset_index(drop=True)

    # ─────────────────────────────────────────────
    #  Helpers
    # ─────────────────────────────────────────────
    def get_quadrant_summary(self) -> pd.DataFrame:
        data = []
        for q, label in QUADRANT_LABELS.items():
            hotels_q = [h for h in self.hotels if self.quadrants.get(h) == q]
            if hotels_q:
                data.append({'Quadrant': label, 'N hôtels': len(hotels_q),
                             'BCC moyen': f"{np.mean([self.bcc_scores[h] for h in hotels_q]):.1%}",
                             'Eff. Éch.': f"{np.mean([self.scale_efficiency[h] for h in hotels_q]):.1%}",
                             'Hôtels': ', '.join(hotels_q)})
        return pd.DataFrame(data)

    def get_cluster_summary(self) -> pd.DataFrame:
        seen = {}
        for h in self.hotels:
            lbl = self.kmeans_labels[h]; seen.setdefault(lbl, []).append(h)
        data = [{'Segment': lbl, 'N hôtels': len(hotels_c),
                 'BCC moyen': f"{np.mean([self.bcc_scores[h] for h in hotels_c]):.1%}",
                 'TOPSIS moyen': f"{np.mean([self.topsis_scores[h] for h in hotels_c]):.3f}",
                 'Hôtels': ', '.join(hotels_c)} for lbl, hotels_c in seen.items()]
        return pd.DataFrame(data).sort_values('TOPSIS moyen', ascending=False)

    def get_topsis_ranking(self) -> pd.DataFrame:
        rows = [{'Rang': self.topsis_ranks[h], 'Hôtel': h, 'Score TOPSIS': self.topsis_scores[h],
                 'BCC': f"{self.bcc_scores[h]:.1%}", 'Eff. Échelle': f"{self.scale_efficiency[h]:.1%}",
                 'RevPAR': self.df.loc[h, 'revpar'], 'TO': f"{self.df.loc[h, 'taux_occupation']:.0f}%",
                 'Quadrant': QUADRANT_LABELS.get(self.quadrants.get(h, ''), '—'),
                 'Segment': self.kmeans_labels.get(h, '—')} for h in self.hotels]
        return pd.DataFrame(rows).sort_values('Rang').reset_index(drop=True)

    def get_capital_report(self) -> pd.DataFrame:
        rows = []
        for hotel in self.hotels:
            cap = self.capital_metrics.get(hotel, {}); bcc = self.bcc_scores.get(hotel, 0)
            rows.append({'Hôtel': hotel, 'BCC': f"{bcc:.1%}",
                         'Rang TOPSIS': self.topsis_ranks.get(hotel, '—'),
                         'Surface m²': cap.get('surface_m2'), 'm²/chambre': cap.get('m2_par_chambre'),
                         'CAPEX annuel (€)': cap.get('capex_annuel'), 'CAPEX/chambre (€)': cap.get('capex_par_chambre'),
                         'Rendement CAPEX (x)': cap.get('rendement_capex'), 'GOPPAM (€/m²)': cap.get('goppam'),
                         'Flow Through %': f"{cap.get('flow_through', 0):.1%}" if cap.get('flow_through') else '—',
                         'Source FT': cap.get('ft_source', '—'), 'Qualité FT': cap.get('ft_qualite', '—'),
                         'Upside GOP/FT (M€/an)': cap.get('upside_gop_ft')})
        return pd.DataFrame(rows).sort_values('Rang TOPSIS')

    def compute_capital_dea(self) -> Optional[pd.DataFrame]:
        needed = ['surface_m2', 'capex_annuel']
        if not all(c in self.df.columns for c in needed): return None
        df = self.df.copy()
        cap_in  = ['surface_m2', 'capex_annuel', 'nb_lits']
        df['_rev'] = (df['total_revenue'] if 'total_revenue' in df.columns
                      else df['revpar'] * df['taux_occupation'] / 100 * 365 * df['nb_lits'])
        cap_out = ['_rev'] + (['gop'] if 'gop' in df.columns else [])
        X_in = df[cap_in].values.astype(float); X_out = df[cap_out].values.astype(float); n = len(self.hotels)
        cap_scores = {}
        for i, hotel in enumerate(self.hotels):
            model   = pulp.LpProblem(f"CAP_DEA_{i}", pulp.LpMinimize)
            theta   = pulp.LpVariable("theta", lowBound=0)
            lambdas = pulp.LpVariable.dicts("lam", range(n), lowBound=0)
            model  += theta
            for j in range(X_in.shape[1]):
                model += pulp.lpSum(lambdas[k]*X_in[k,j] for k in range(n)) <= theta * X_in[i,j]
            for j in range(X_out.shape[1]):
                model += pulp.lpSum(lambdas[k]*X_out[k,j] for k in range(n)) >= X_out[i,j]
            model += pulp.lpSum(lambdas.values()) == 1
            model.solve(pulp.PULP_CBC_CMD(msg=False))
            cap_scores[hotel] = round(min(pulp.value(theta) or 1.0, 1.0), 4)
        rows = []
        for hotel in self.hotels:
            op = self.bcc_scores.get(hotel, 0); cp = cap_scores[hotel]; d = round(cp - op, 4)
            cap = self.capital_metrics.get(hotel, {})
            rows.append({'Hôtel': hotel, 'DEA Opérationnel': f"{op:.1%}", 'DEA Capital': f"{cp:.1%}",
                         'Δ (Capital-Opérat.)': f"{d:+.1%}",
                         'Lecture': ('✅ Capital bien employé' if d >= 0 else
                                     '⚠️ Surcoût capital modéré' if d >= -0.10 else '🔴 Capital sous-productif'),
                         'CAPEX/chambre (€)': cap.get('capex_par_chambre'),
                         'Rendement CAPEX': cap.get('rendement_capex'), 'GOPPAM (€/m²)': cap.get('goppam')})
        return pd.DataFrame(rows).sort_values('DEA Capital', ascending=False)


# ═══════════════════════════════════════════════════════════════════════════════
#  PARTIE 2 — Multi-Module DEA  (scipy/HiGHS)
# ═══════════════════════════════════════════════════════════════════════════════
#
#  Importer depuis modules_config.py (même répertoire) :
#    MODULES, check_module_feasibility, get_all_input_cols,
#    get_all_output_cols, get_col_label
#
#  Point d'entrée principal : run_multi_module()
#  Synthèse croisée         : build_cross_synthesis()
# ═══════════════════════════════════════════════════════════════════════════════

# Import local (même répertoire que dea_model.py)
from modules_config import (
    MODULES,
    check_module_feasibility,
    get_all_input_cols,
    get_all_output_cols,
    get_col_label,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Moteur DEA scipy/HiGHS  (BCC/CCR, Input/Output orienté)
# ─────────────────────────────────────────────────────────────────────────────

def _run_dea_internal(
    df: pd.DataFrame,
    inputs: list,
    outputs: list,
    orientation: str = 'output',
    model: str = 'BCC',
) -> pd.DataFrame:
    """
    Solveur DEA BCC/CCR via scipy.optimize.linprog (HiGHS).
    Utilisé exclusivement par run_multi_module().
    Retourne DataFrame : dmu_index | efficiency | raw_score | peers | lambdas | slacks_in | slacks_out
    """
    from scipy.optimize import linprog

    df      = df.copy().reset_index(drop=True)
    n       = len(df)
    m_in    = len(inputs)
    m_out   = len(outputs)
    X       = df[inputs].values.astype(float)
    Y       = df[outputs].values.astype(float)
    x_scale = X.mean(axis=0); x_scale[x_scale == 0] = 1
    y_scale = Y.mean(axis=0); y_scale[y_scale == 0] = 1
    X_n = X / x_scale; Y_n = Y / y_scale

    results = []
    for j in range(n):
        x0 = X_n[j]; y0 = Y_n[j]

        if orientation == 'output':
            n_vars = 1 + n; c = np.zeros(n_vars); c[0] = -1
            A_ub_list = []
            for i in range(m_in):
                row = np.zeros(n_vars); row[1:] = X_n[:, i]; A_ub_list.append(row)
            b_ub = list(x0)
            for k in range(m_out):
                row = np.zeros(n_vars); row[0] = y0[k]; row[1:] = -Y_n[:, k]; A_ub_list.append(row)
            b_ub += [0.0] * m_out
            A_eq = b_eq = None
            if model == 'BCC':
                A_eq = np.zeros((1, n_vars)); A_eq[0, 1:] = 1; b_eq = np.array([1.0])
            bounds = [(1, None)] + [(0, None)] * n
        else:
            n_vars = 1 + n; c = np.zeros(n_vars); c[0] = 1
            A_ub_list = []
            for i in range(m_in):
                row = np.zeros(n_vars); row[0] = -x0[i]; row[1:] = X_n[:, i]; A_ub_list.append(row)
            b_ub = [0.0] * m_in
            for k in range(m_out):
                row = np.zeros(n_vars); row[1:] = -Y_n[:, k]; A_ub_list.append(row)
            b_ub += list(-y0)
            A_eq = b_eq = None
            if model == 'BCC':
                A_eq = np.zeros((1, n_vars)); A_eq[0, 1:] = 1; b_eq = np.array([1.0])
            bounds = [(0, 1)] + [(0, None)] * n

        res = linprog(c, A_ub=np.array(A_ub_list), b_ub=np.array(b_ub),
                      A_eq=A_eq, b_eq=b_eq, bounds=bounds,
                      method='highs', options={'disp': False})

        if res.success:
            phi_theta = res.x[0]; lmbds = res.x[1:]
            eff = (1 / phi_theta) if orientation == 'output' else phi_theta
            proj_x = X_n.T @ lmbds; proj_y = Y_n.T @ lmbds
            if orientation == 'output':
                slacks_in  = np.maximum(0, x0 - proj_x)
                slacks_out = np.maximum(0, proj_y - y0 * phi_theta)
            else:
                slacks_in  = np.maximum(0, x0 * phi_theta - proj_x)
                slacks_out = np.maximum(0, proj_y - y0)
            peer_idx = np.where(lmbds > 0.01)[0]
        else:
            eff = np.nan; slacks_in = [np.nan]*m_in; slacks_out = [np.nan]*m_out
            peer_idx = np.array([]); lmbds = np.zeros(n)

        results.append({
            'dmu_index': j,
            'efficiency': round(min(eff, 1.0), 4) if not np.isnan(eff) else np.nan,
            'raw_score':  round(eff, 4) if not np.isnan(eff) else np.nan,
            'peers':      list(peer_idx),
            'lambdas':    lmbds[peer_idx].tolist(),
            'slacks_in':  {inputs[i]:  round(float(slacks_in[i])  * x_scale[i], 2) for i in range(m_in)},
            'slacks_out': {outputs[k]: round(float(slacks_out[k]) * y_scale[k], 2) for k in range(m_out)},
        })

    result_df = pd.DataFrame(results); result_df.index = df.index
    return result_df


# ─────────────────────────────────────────────────────────────────────────────
#  ModuleResult
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModuleResult:
    module_id   : str
    label_fr    : str
    orientation : str
    model_type  : str
    inputs_used : List[str]
    outputs_used: List[str]
    n_dmus      : int
    scores      : pd.DataFrame
    feasibility : dict
    error       : Optional[str]        = None
    warnings    : List[str]            = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
#  Runner multi-module
# ─────────────────────────────────────────────────────────────────────────────

def run_multi_module(
    df: pd.DataFrame,
    dmu_col: str,
    active_modules: list,
    variable_overrides: Optional[dict] = None,
) -> Dict[str, ModuleResult]:
    """
    Exécute le moteur DEA scipy pour chaque module actif sur le même dataset.

    Args:
        df              : DataFrame avec colonnes = variables hôtelières
        dmu_col         : colonne identifiant les DMUs (nom hôtel)
        active_modules  : liste des module_ids à exécuter
        variable_overrides : {module_id: {"inputs": [...], "outputs": [...]}}

    Returns:
        {module_id: ModuleResult}
    """
    available_cols = [c for c in df.columns if c != dmu_col]
    results: Dict[str, ModuleResult] = {}

    for mod_id in active_modules:
        if mod_id not in MODULES:
            continue

        mod_cfg    = MODULES[mod_id]
        feasibility = check_module_feasibility(mod_id, available_cols)

        if not feasibility['feasible']:
            results[mod_id] = ModuleResult(
                module_id=mod_id, label_fr=mod_cfg['label_fr'],
                orientation=mod_cfg['orientation'], model_type=mod_cfg['model'],
                inputs_used=[], outputs_used=[], n_dmus=len(df),
                scores=pd.DataFrame(), feasibility=feasibility,
                error=f"Colonnes obligatoires manquantes : {', '.join(feasibility['missing_required'])}",
            )
            continue

        # Variables effectives
        if variable_overrides and mod_id in variable_overrides:
            inputs_used  = variable_overrides[mod_id].get('inputs', [])
            outputs_used = variable_overrides[mod_id].get('outputs', [])
        else:
            inputs_used  = feasibility['available_inputs']
            outputs_used = feasibility['available_outputs']

        # Vérifications
        warn_list = []
        for inp in inputs_used:
            for out in outputs_used:
                if inp in df.columns and out in df.columns:
                    corr = df[inp].corr(df[out])
                    if corr < 0:
                        warn_list.append(f"Corrélation négative : {inp} ↔ {out} (r={corr:.2f})")
        n_vars = len(inputs_used) + len(outputs_used)
        if len(df) < 3 * n_vars:
            warn_list.append(f"Ratio DMUs/variables faible : {len(df)} DMUs / {n_vars} variables")

        # Nettoyage NaN
        df_clean = df[[dmu_col] + inputs_used + outputs_used].dropna()
        if len(df_clean) < 2:
            results[mod_id] = ModuleResult(
                module_id=mod_id, label_fr=mod_cfg['label_fr'],
                orientation=mod_cfg['orientation'], model_type=mod_cfg['model'],
                inputs_used=inputs_used, outputs_used=outputs_used, n_dmus=len(df),
                scores=pd.DataFrame(), feasibility=feasibility,
                error=f"Données insuffisantes après nettoyage NaN ({len(df_clean)} DMUs).",
                warnings=warn_list,
            )
            continue

        # Calcul DEA
        try:
            raw = _run_dea_internal(df_clean, inputs_used, outputs_used,
                                    mod_cfg['orientation'], mod_cfg['model'])
            scores = raw.copy()
            scores.insert(0, 'dmu_name', df_clean[dmu_col].values)
            scores['rank']      = scores['efficiency'].rank(ascending=False, method='min').astype(int)
            scores['efficient'] = scores['efficiency'] >= 0.999
            scores['module']    = mod_id

            results[mod_id] = ModuleResult(
                module_id=mod_id, label_fr=mod_cfg['label_fr'],
                orientation=mod_cfg['orientation'], model_type=mod_cfg['model'],
                inputs_used=inputs_used, outputs_used=outputs_used,
                n_dmus=len(df_clean), scores=scores, feasibility=feasibility,
                warnings=warn_list,
            )
        except Exception as e:
            results[mod_id] = ModuleResult(
                module_id=mod_id, label_fr=mod_cfg['label_fr'],
                orientation=mod_cfg['orientation'], model_type=mod_cfg['model'],
                inputs_used=inputs_used, outputs_used=outputs_used, n_dmus=len(df),
                scores=pd.DataFrame(), feasibility=feasibility,
                error=str(e), warnings=warn_list,
            )

    return results


# ─────────────────────────────────────────────────────────────────────────────
#  Synthèse croisée
# ─────────────────────────────────────────────────────────────────────────────

def build_cross_synthesis(module_results: Dict[str, ModuleResult]) -> pd.DataFrame:
    """
    DataFrame de synthèse : ligne = DMU, colonne = score par module.
    Colonnes calculées : score_moyen, nb_modules_efficient, profil.
    """
    dfs = []
    for mod_id, result in module_results.items():
        if result.error or result.scores.empty: continue
        tmp = result.scores[['dmu_name', 'efficiency']].copy()
        tmp = tmp.rename(columns={'efficiency': mod_id})
        dfs.append(tmp.set_index('dmu_name'))
    if not dfs: return pd.DataFrame()
    synthesis = pd.concat(dfs, axis=1)
    score_cols = list(synthesis.columns)
    synthesis['score_moyen']           = synthesis.mean(axis=1).round(4)
    synthesis['nb_modules_efficient']  = (synthesis[score_cols] >= 0.999).sum(axis=1)
    synthesis['profil'] = synthesis.apply(lambda row: _classify_profile(row, score_cols, len(score_cols)), axis=1)
    return synthesis.reset_index().rename(columns={'dmu_name': 'Hôtel'})


def _classify_profile(row: pd.Series, score_cols: list, n_active: int) -> str:
    if n_active == 0: return '—'
    avg   = row['score_moyen']; n_eff = row['nb_modules_efficient']
    pct   = n_eff / n_active
    avail = [row[c] for c in score_cols if pd.notna(row[c])]
    spread = max(avail) - min(avail) if len(avail) >= 2 else 0
    if   pct >= 0.80:                        return '🟢 Excellence globale'
    elif pct >= 0.50 and spread < 0.15:      return '🔵 Efficience équilibrée'
    elif pct >= 0.50 and spread >= 0.15:     return '🟡 Spécialiste (forces & faiblesses)'
    elif avg >= 0.75 and pct < 0.30:         return '🟠 Sous-efficient mais proche frontière'
    elif avg < 0.60:                          return '🔴 Inefficience significative'
    else:                                     return '🟡 Profil mixte'


def get_module_ranking(synthesis: pd.DataFrame, score_cols: list) -> pd.DataFrame:
    """Classement des modules par difficulté (score moyen croissant)."""
    if synthesis.empty: return pd.DataFrame()
    rows = []
    for col in score_cols:
        if col in synthesis.columns:
            vals = synthesis[col].dropna()
            rows.append({'module_id': col, 'score_moyen': vals.mean().round(4),
                         'score_median': vals.median().round(4),
                         'nb_efficients': (vals >= 0.999).sum(), 'std': vals.std().round(4)})
    return pd.DataFrame(rows).sort_values('score_moyen')


def get_dmu_weaknesses(
    dmu_name: str, synthesis: pd.DataFrame, score_cols: list, threshold: float = 0.80
) -> list:
    """Modules sous le seuil pour un DMU — drill-down Fiche Actif."""
    row = synthesis[synthesis['Hôtel'] == dmu_name]
    if row.empty: return []
    weaknesses = []
    for col in score_cols:
        if col in row.columns:
            val = row[col].values[0]
            if pd.notna(val) and val < threshold:
                weaknesses.append({'module_id': col, 'score': val,
                                   'gap_to_frontier': round(1 - val, 4),
                                   'priority': 'haute' if val < 0.65 else 'moyenne'})
    return sorted(weaknesses, key=lambda x: x['score'])
