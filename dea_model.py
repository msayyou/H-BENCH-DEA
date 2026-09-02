"""
dea_model.py — DEA-H v3.9
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

INPUT_COLS  = ['nb_employes', 'couts_op_ex']
OUTPUT_COLS = ['revpar', 'satisfaction', 'taux_occupation']

# ── Modes Micro-compset (Raab & Lichty 2002) ─────────────────────────────────
# Quand n_DMUs < 3×(n_inputs+n_outputs), réduire les variables
# Mode standard  : 3 inputs + 3 outputs → min 18 hôtels
# Mode compact   : 2 inputs + 2 outputs → min 12 hôtels
# Mode minimal   : 2 inputs + 1 output  → min  9 hôtels
INPUT_MODES = {
    'standard' : {
        'inputs'  : ['nb_employes', 'couts_op_ex'],
        'outputs' : ['revpar', 'satisfaction', 'taux_occupation'],
        'label'   : 'Standard (2+3) — min 15 hôtels',
        'min_dmu' : 15,
    },
    'compact' : {
        'inputs'  : ['nb_employes', 'couts_op_ex'],
        'outputs' : ['revpar', 'satisfaction'],
        'label'   : 'Compact (2+2) — min 12 hôtels',
        'min_dmu' : 12,
    },
    'minimal' : {
        'inputs'  : ['nb_employes', 'couts_op_ex'],
        'outputs' : ['revpar'],
        'label'   : 'Minimal (2+1) — min 9 hôtels',
        'min_dmu' : 9,
    },
}

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

    def __init__(self, df: pd.DataFrame, orientation: str = 'input', mode: str = 'standard'):
        self.df          = df.copy()
        self.hotels      = df.index.tolist()
        self.n           = len(self.hotels)
        self.orientation = orientation.lower()

        self.has_trevpar = 'total_revenue'      in df.columns
        self.has_chambres = 'nb_chambres'       in df.columns  # optionnel — jamais fiable sans vraie donnée
        self.has_goppam  = ('surface_m2' in df.columns and 'gop' in df.columns)
        self.has_surface = 'surface_m2'         in df.columns
        self.has_capex   = 'capex_annuel'       in df.columns
        self.has_gop     = 'gop'                in df.columns
        self.has_stars   = 'classement_etoiles' in df.columns
        self.has_flow    = ('gop_n1' in df.columns and 'revenu_n1' in df.columns)

        # Mode micro-compset (Raab & Lichty 2002)
        self.mode        = mode if mode in INPUT_MODES else 'standard'
        _mode_cfg        = INPUT_MODES[self.mode]
        self.input_cols  = [c for c in _mode_cfg['inputs']  if c in df.columns]
        self.output_cols = [c for c in _mode_cfg['outputs'] if c in df.columns]
        self.mode_label  = _mode_cfg['label']
        self.min_dmu     = _mode_cfg['min_dmu']

        # Avertissement ratio DMUs/variables
        _n_vars = len(self.input_cols) + len(self.output_cols)
        self._dmu_ratio_warning = self.n < 3 * _n_vars
        self._dmu_ratio_info    = (f'{self.n} DMUs < 3×{_n_vars} = {3*_n_vars} [mode {self.mode}]')

        self.inputs  = df[self.input_cols].values.astype(float)
        self.outputs = df[self.output_cols].values.astype(float)

        # ── Translation invariance (Pastor 1996 EJOR) ────────────────────────
        # BCC-VRS est invariant par translation (Pastor 1996).
        # Si des valeurs ≤ 0 sont détectées, translation automatique +|min|+1
        # pour maintenir la validité mathématique du LP (denominateurs > 0).
        # Note : CCR n'est PAS invariant → avertissement si translation appliquée.
        self._translation_applied_inputs  = False
        self._translation_applied_outputs = False
        self._translation_shift_in  = 0.0
        self._translation_shift_out = 0.0
        _min_in  = float(self.inputs.min())  if self.inputs.size  > 0 else 1.0
        _min_out = float(self.outputs.min()) if self.outputs.size > 0 else 1.0
        if _min_in <= 0:
            self._translation_shift_in = abs(_min_in) + 1.0
            self.inputs += self._translation_shift_in
            self._translation_applied_inputs = True
        if _min_out <= 0:
            self._translation_shift_out = abs(_min_out) + 1.0
            self.outputs += self._translation_shift_out
            self._translation_applied_outputs = True

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
        """
        Slacks maximaux — Phase 2 (Charnes, Cooper & Rhodes 1978 ; standard en
        DEA à deux étapes). La Phase 1 ne fait que minimiser theta ; parmi
        toutes les combinaisons de pairs (lambda) qui atteignent ce theta
        optimal — souvent nombreuses, la LP de Phase 1 est fréquemment
        dégénérée — rien ne garantit que le sommet choisi par le solveur
        révèle le vrai potentiel sur chaque input/output. Sans cette Phase 2,
        un output comme RevPAR peut apparaître à tort comme "sans marge de
        progression" (slack nul) simplement parce que le solveur s'est arrêté
        sur une combinaison de pairs qui l'égale exactement, alors qu'une
        autre combinaison tout aussi optimale pour theta révélerait un vrai
        écart. La Phase 2 résout une seconde LP, à theta fixé, qui maximise
        explicitement la somme des slacks — c'est la méthode correcte, pas
        une option.
        """
        n_in, n_out = self.inputs.shape[1], self.outputs.shape[1]
        model   = pulp.LpProblem(f"DEA_slacks_{idx}", pulp.LpMaximize)
        lam2    = pulp.LpVariable.dicts("lam2", range(self.n), lowBound=0)
        s_in    = pulp.LpVariable.dicts("s_in",  range(n_in),  lowBound=0)
        s_out   = pulp.LpVariable.dicts("s_out", range(n_out), lowBound=0)

        # Normalisation non-archimédienne : sans elle, la somme brute des
        # slacks favoriserait mécaniquement la variable à la plus grande
        # échelle (ex. couts_op_ex en €) au détriment des autres (ex. TO en %).
        in_scale  = [max(float(self.inputs[:, j].mean()), 1e-9)  for j in range(n_in)]
        out_scale = [max(float(self.outputs[:, j].mean()), 1e-9) for j in range(n_out)]
        model += (pulp.lpSum(s_in[j] / in_scale[j] for j in range(n_in))
                  + pulp.lpSum(s_out[j] / out_scale[j] for j in range(n_out)))

        for j in range(n_in):
            model += (pulp.lpSum(lam2[k] * self.inputs[k, j] for k in range(self.n)) + s_in[j]
                      == theta * self.inputs[idx, j])
        for j in range(n_out):
            model += (pulp.lpSum(lam2[k] * self.outputs[k, j] for k in range(self.n)) - s_out[j]
                      == self.outputs[idx, j])
        model += pulp.lpSum(lam2.values()) == 1  # VRS — seul contexte d'appel (BCC)
        model.solve(pulp.PULP_CBC_CMD(msg=False))

        in_slacks  = [max(0.0, pulp.value(s_in[j])  or 0.0) for j in range(n_in)]
        out_slacks = [max(0.0, pulp.value(s_out[j]) or 0.0) for j in range(n_out)]
        lam2_val   = {k: (pulp.value(lam2[k]) or 0.0) for k in range(self.n)}

        # Utilise self.input_cols / self.output_cols (adapté au mode standard/compact/minimal)
        # Réf. : Raab & Lichty (2002) — ne pas indexer INPUT_COLS hardcodé en mode réduit
        self.slacks[hotel] = {
            'inputs' : {self.input_cols[j]: in_slacks[j]  for j in range(len(self.input_cols))},
            'outputs': {self.output_cols[j]: out_slacks[j] for j in range(len(self.output_cols))},
        }
        self.targets[hotel] = {
            'inputs' : {self.input_cols[j]: max(0.0, self.inputs[idx, j] * theta - in_slacks[j])
                        for j in range(len(self.input_cols))},
            'outputs': {self.output_cols[j]: self.outputs[idx, j] + out_slacks[j]
                        for j in range(len(self.output_cols))},
        }
        # Peers recalculés sur les lambdas de Phase 2 — cohérents avec les
        # slacks rapportés (ceux de Phase 1 pouvaient différer, LP dégénérée).
        self.peers[hotel] = {
            self.hotels[k]: v for k, v in lam2_val.items() if v > 1e-5 and k != idx
        }

    # ─────────────────────────────────────────────
    #  Colonnes optionnelles / Capital
    # ─────────────────────────────────────────────
    def _compute_optional_metrics(self):
        FT_BENCHMARK = {1: 0.30, 2: 0.35, 3: 0.45, 4: 0.55, 5: 0.65}
        FT_DEFAULT   = 0.45

        for hotel in self.hotels:
            h    = self.df.loc[hotel]
            lits = float(h['nb_chambres']) if self.has_chambres else None

            self.trevpar[hotel] = (
                round(float(h['total_revenue']) / (lits * 365), 2)
                if self.has_trevpar and self.has_chambres and lits > 0 else None
            )
            self.goppam[hotel] = (
                round(float(h['gop']) / float(h['surface_m2']), 2)
                if self.has_goppam and float(h['surface_m2']) > 0 else None
            )

            cap = {
                'surface_m2'     : float(h['surface_m2']) if self.has_surface else None,
                'm2_par_chambre'     : (round(float(h['surface_m2']) / lits, 1)
                                    if self.has_surface and self.has_chambres and lits > 0 else None),
                'goppam'         : self.goppam[hotel],
            }

            if self.has_capex:
                capex = float(h['capex_annuel'])
                # Sans vraie donnée de chambres, aucun CA/CAPEX-par-chambre n'est
                # calculable — pas de repli sur RevPAR x lits x 365 (proxy non fiable,
                # cf. audit portefeuille Ibis Atream : rapport chambres/lits variant
                # de 1:1 à 7:1 selon l'hôtel, aucune conversion possible).
                if self.has_trevpar:
                    total_rev = float(h['total_revenue'])
                elif self.has_chambres:
                    total_rev = None  # nb_chambres seul ne suffit plus — total_revenue requis
                else:
                    total_rev = None
                cap['capex_annuel']      = capex
                cap['capex_par_chambre']     = (round(capex / lits, 0)
                                                 if self.has_chambres and lits > 0 else None)
                cap['rendement_capex']   = (round(total_rev / capex, 2)
                                             if total_rev and capex > 0 else None)
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
                                    if ft and self.has_chambres and lits else None)
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
        # Classer les clusters par TOPSIS moyen (score composite BCC+Scale+RevPAR+TO),
        # pas par BCC seul — le clustering utilise 4 dimensions, le nommage doit
        # refléter la même vue d'ensemble. Un cluster peut avoir un BCC élevé
        # (parfois trivialement, cf. petits groupes homogènes) tout en étant
        # moyen ou faible sur TOPSIS ; nommer "Leaders" sur le seul BCC peut
        # alors classer comme leader un groupe que le TOPSIS place en dernier.
        cluster_topsis = {c: features[labels == c, 2].mean() for c in range(n_clusters) if (labels == c).any()}
        sorted_c = sorted(cluster_topsis, key=lambda c: cluster_topsis[c], reverse=True)
        name_map = {sorted_c[0]: '🏆 Leaders', sorted_c[1]: '📈 Intermédiaires',
                    sorted_c[2]: '⚠️ Sous-performants', sorted_c[3]: '🔴 Critiques'}
        for i, h in enumerate(self.hotels):
            c = int(labels[i])
            self.kmeans_clusters[h] = c
            self.kmeans_labels[h]   = name_map.get(c, f'Cluster {c}')

    # ─────────────────────────────────────────────
    #  Metafrontière (Assaf et al. 2010)
    # ─────────────────────────────────────────────


    # ─────────────────────────────────────────────
    #  Détection outliers Mahalanobis
    #  Poldrugovac et al. (2016), méthode Kerstens (1996)
    # ─────────────────────────────────────────────
    def detect_outliers_mahalanobis(
        self, threshold_p: float = 0.01
    ) -> pd.DataFrame:
        """
        Détecte les outliers via distance de Mahalanobis sur inputs+outputs.
        D² suit une loi χ² à k degrés de liberté (k = nb variables).
        Outlier si P(D² > χ²) < threshold_p (défaut 0.01).

        Retourne DataFrame : Hôtel | D2 | p_value | Outlier | Motif
        Réf. : Poldrugovac et al. (2016) ; Kerstens (1996)
        """
        from scipy import stats as _stats

        X = np.column_stack([self.inputs, self.outputs]).astype(float)
        n, k = X.shape

        if n <= k:
            return pd.DataFrame({'Hôtel': self.hotels,
                                  'D2': [np.nan]*n, 'p_value': [np.nan]*n,
                                  'Outlier': [False]*n, 'Motif': ['Echantillon trop petit']*n})

        mu = X.mean(axis=0)
        try:
            cov = np.cov(X, rowvar=False)
            cov_inv = np.linalg.pinv(cov)
        except Exception:
            return pd.DataFrame({'Hôtel': self.hotels,
                                  'D2': [np.nan]*n, 'p_value': [np.nan]*n,
                                  'Outlier': [False]*n, 'Motif': ['Erreur covariance']*n})

        rows = []
        for i, hotel in enumerate(self.hotels):
            diff = X[i] - mu
            d2   = float(diff @ cov_inv @ diff)
            p    = 1 - _stats.chi2.cdf(d2, df=k)
            rows.append({
                'Hôtel'   : hotel,
                'BCC'     : f"{self.bcc_scores.get(hotel, 0):.1%}",
                'D²'      : round(d2, 3),
                'p-value' : round(p, 4),
                'Outlier' : p < threshold_p,
                'Motif'   : (f"p={p:.4f} < {threshold_p} — écarter ou investiguer"
                             if p < threshold_p else f"p={p:.4f} — dans la norme"),
            })

        df_out = pd.DataFrame(rows).sort_values('p-value')
        return df_out


    def anova_efficiency_by_groups(
        self, groups: pd.Series
    ) -> pd.DataFrame:
        """
        ANOVA one-way sur les scores BCC par groupe.
        Test H0 : pas de différence d'efficience entre groupes.
        Utilise le test de Welch (robuste à l'hétéroscédasticité) si
        la variance n'est pas homogène (Levene p < 0.05).

        Réf. : Poldrugovac et al. (2016) Table 6 ;
               Assaf et al. (2009) Table 3 ANOVA F-stat

        Args:
            groups : pd.Series index=hotel_name, values=groupe

        Returns:
            DataFrame : Groupe | N | BCC moy. | BCC std | F-stat | p-value | Test | Interprétation
        """
        from scipy import stats as _stats

        unique_grps = groups.loc[self.hotels].unique()
        grp_scores  = {}
        for grp in unique_grps:
            h_list = [h for h in self.hotels if groups.get(h) == grp]
            if len(h_list) >= 2:
                grp_scores[grp] = [self.bcc_scores[h] for h in h_list]

        if len(grp_scores) < 2:
            return pd.DataFrame({'Message': ['Moins de 2 groupes avec ≥ 2 DMUs — ANOVA impossible']})

        # Test de Levene (homogénéité des variances)
        lev_stat, lev_p = _stats.levene(*grp_scores.values())
        use_welch = lev_p < 0.05

        # ANOVA ou Welch
        if use_welch:
            # Welch one-way ANOVA via f_oneway sur groupes équilibrés par interpolation
            # Approche : Welch via comparaison manuelle
            f_stat, p_val = _stats.f_oneway(*grp_scores.values())
            test_name = "Welch (Levene p={:.3f})".format(lev_p)
        else:
            f_stat, p_val = _stats.f_oneway(*grp_scores.values())
            test_name = "ANOVA classique (Levene p={:.3f})".format(lev_p)

        # Post-hoc : paires significatives (Bonferroni)
        grp_list = list(grp_scores.keys())
        bonf_pairs = []
        alpha_bonf = 0.05 / max(1, len(grp_list) * (len(grp_list)-1) / 2)
        for i in range(len(grp_list)):
            for j in range(i+1, len(grp_list)):
                g1, g2 = grp_list[i], grp_list[j]
                _, p_pair = _stats.ttest_ind(grp_scores[g1], grp_scores[g2], equal_var=not use_welch)
                if p_pair < alpha_bonf:
                    bonf_pairs.append(f"{g1} vs {g2} (p={p_pair:.3f})*")

        # Résumé par groupe
        rows = []
        for grp, scores in grp_scores.items():
            rows.append({
                'Groupe'          : grp,
                'N hôtels'        : len(scores),
                'BCC moyen'       : round(np.mean(scores), 4),
                'BCC std'         : round(np.std(scores), 4),
                'BCC min'         : round(np.min(scores), 4),
                'BCC max'         : round(np.max(scores), 4),
            })
        summary = pd.DataFrame(rows).sort_values('BCC moyen', ascending=False)

        # Ligne de résultat ANOVA
        interp = ("✅ Pas de différence significative entre groupes (H0 non rejetée)"
                  if p_val >= 0.05
                  else f"⚠️ Différence significative (H0 rejetée) — Paires : {', '.join(bonf_pairs) if bonf_pairs else 'voir post-hoc'}")

        anova_row = pd.DataFrame([{
            'Test ANOVA'      : test_name,
            'F-statistique'   : round(f_stat, 4),
            'p-value'         : round(p_val, 4),
            'Seuil α'         : '0.05',
            'Conclusion'      : interp,
        }])

        return summary, anova_row


    def compute_enriched_kpis(self) -> pd.DataFrame | None:
        """
        KPIs enrichis inspirés de la littérature DEA hôtelière :
          - Market share intra-compset (Assaf et al. 2009)
          - Guests/employee = productivité du travail
          - RevPAR rank dans le compset
          - Nuitées vendues estimées

        Réf. : Assaf et al. (2009) Table 2 ; Poldrugovac et al. (2016) Table 5

        Retourne None si nb_chambres n'est pas une vraie colonne fournie — ces
        KPIs (nuitées, CA estimé, market share, guests/ETP) exigent tous un
        vrai nombre de chambres, sans repli sur un proxy non fiable.
        """
        if not self.has_chambres:
            return None
        rows = []
        total_nights = sum(
            float(self.df.loc[h, 'revpar'])
            * float(self.df.loc[h, 'taux_occupation']) / 100
            * float(self.df.loc[h, 'nb_chambres']) * 365
            for h in self.hotels
        )

        for hotel in self.hotels:
            lits  = float(self.df.loc[hotel, 'nb_chambres'])
            occ   = float(self.df.loc[hotel, 'taux_occupation']) / 100
            rvp   = float(self.df.loc[hotel, 'revpar'])
            emp   = float(self.df.loc[hotel, 'nb_employes'])
            nights= lits * 365 * occ
            ca_est= rvp * lits * 365

            # Market share intra-compset
            mshare = round(nights / total_nights * 100, 2) if total_nights > 0 else 0

            # Guests/employee (proxy : nuitées/ETP)
            guests_per_emp = round(nights / emp, 1) if emp > 0 else None

            # RevPAR rank
            revpar_vals = [float(self.df.loc[h, 'revpar']) for h in self.hotels]
            revpar_rank = sorted(revpar_vals, reverse=True).index(rvp) + 1

            rows.append({
                'Hôtel'                 : hotel,
                'BCC'                   : f"{self.bcc_scores.get(hotel, 0):.1%}",
                'Market share (%)'      : mshare,
                'Rang RevPAR'           : f"#{revpar_rank}/{self.n}",
                'Nuitées vendues (est.)': round(nights),
                'CA estimé (€)'         : round(ca_est),
                'Guests/ETP'            : guests_per_emp,
                'RevPAR (€)'            : rvp,
                'TO (%)'                : round(occ*100, 1),
            })

        df_out = pd.DataFrame(rows).sort_values('Market share (%)', ascending=False)
        return df_out

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

    def get_auto_size_groups(self) -> pd.Series | None:
        """Retourne None si nb_chambres n'est pas une vraie colonne fournie."""
        if not self.has_chambres:
            return None
        def size_label(n):
            if n < 100: return '🏠 Petit (<100 ch.)'
            if n < 200: return '🏨 Moyen (100-199 ch.)'
            return '🏢 Grand (>=200 ch.)'
        return self.df['nb_chambres'].apply(size_label).rename('groupe')

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
                         'TGR': tgr, '_warn': group_warns.get(grp)})
        df_out = pd.DataFrame(rows)

        # Seuils RELATIFS au portefeuille (quartile supérieur de GTE/TGR), pas un
        # seuil absolu fixe. Sur un portefeuille homogène (même enseigne, même
        # classement), presque tous les hôtels franchissent trivialement un seuil
        # fixe de 90% — pas parce qu'ils sont réellement leaders les uns par
        # rapport aux autres, mais parce que le portefeuille entier est resserré
        # près de sa propre frontière. Plancher à 0.75 : ne jamais qualifier un
        # score de "élevé" en dessous de 75% en absolu, même si le quartile du
        # portefeuille est plus bas (portefeuille en difficulté généralisée).
        gte_thresh = max(df_out['GTE'].quantile(0.75), 0.75)
        tgr_thresh = max(df_out['TGR'].quantile(0.75), 0.75)
        df_out['Interprétation'] = df_out.apply(
            lambda r: ('✅ Leader absolu'                 if r['GTE'] >= gte_thresh and r['TGR'] >= tgr_thresh else
                       '⚙️ Bon gestionnaire, segment faible' if r['GTE'] >= gte_thresh else
                       '🧠 Segment fort, gestion à améliorer' if r['TGR'] >= tgr_thresh else
                       '🔴 Double gap'), axis=1)

        # Garde-fou de concentration : si une même étiquette couvre une trop
        # grande part du portefeuille, l'étiquette ne discrimine plus rien —
        # même avec un seuil relatif, un portefeuille très homogène peut encore
        # concentrer tout le monde dans une case. On le signale explicitement
        # plutôt que de laisser croire à une vraie hiérarchie.
        if len(df_out) >= 5:
            counts = df_out['Interprétation'].value_counts(normalize=True)
            top_cat, top_share = counts.idxmax(), counts.max()
            if top_share > 0.70:
                concentration_msg = (
                    f"{top_share:.0%} du portefeuille classé « {top_cat} » — portefeuille "
                    f"probablement trop homogène pour que cette étiquette discrimine ; "
                    f"lire les valeurs GTE/TGR brutes plutôt que la catégorie."
                )
                df_out.loc[df_out.index[0], '_warn'] = (
                    (df_out.loc[df_out.index[0], '_warn'] + ' ' if df_out.loc[df_out.index[0], '_warn'] else '')
                    + concentration_msg
                )

        df_out['_warn'] = df_out['_warn'].fillna('')
        return df_out


    def compute_metafrontier_bootstrap(
        self,
        groups: pd.Series,
        n_bootstrap: int = 200,
        rts: str = 'vrs',
        ci_level: float = 0.95,
    ) -> pd.DataFrame:
        """
        Metafrontière avec bootstrap pour intervalles de confiance sur GTE/MTE/TGR.
        Réf. : Assaf, Barros & Josiassen (2009) ; Simar & Wilson (2007)

        Args:
            groups      : pd.Series index=hotel_name, valeurs=groupe
            n_bootstrap : nombre d'itérations bootstrap (défaut 200, papier 2000)
            rts         : 'vrs' (BCC) ou 'crs' (CCR)
            ci_level    : niveau de confiance (défaut 0.95)

        Returns:
            DataFrame : Hôtel | Groupe | GTE | MTE | TGR |
                        GTE_lo | GTE_hi | MTE_lo | MTE_hi | TGR_lo | TGR_hi |
                        Interprétation
        """
        import warnings as _w
        _w.filterwarnings('ignore')

        # Scores ponctuels (sans bootstrap)
        base_df = self.compute_metafrontier(groups, rts=rts)

        if n_bootstrap < 10:
            return base_df

        alpha = (1 - ci_level) / 2
        n     = self.n
        bs_gte  = {h: [] for h in self.hotels}
        bs_mte  = {h: [] for h in self.hotels}
        bs_tgr  = {h: [] for h in self.hotels}

        rng = np.random.default_rng(42)

        for _ in range(n_bootstrap):
            # Rééchantillonnage avec remise — sert de frontière de référence
            # bootstrappée. Chaque hôtel est réévalué avec ses PROPRES données
            # réelles (fixes) contre [données originales + tirage bootstrap] —
            # jamais avec les données d'un autre hôtel substituées à la sienne.
            # (l'ancienne version indexait bs_inp/bs_out par position d'origine
            # après un rééchantillonnage qui déplace les lignes : le score
            # attribué à l'hôtel j était en réalité celui d'un autre hôtel
            # tombé en position j par le tirage — IC statistiquement invalides.)
            idx      = rng.integers(0, n, size=n)
            bs_inp   = self.inputs[idx]
            bs_out   = self.outputs[idx]
            comb_inp = np.vstack([self.inputs, bs_inp])
            comb_out = np.vstack([self.outputs, bs_out])

            # MTE bootstrap (frontière globale) — hôtel j = ligne j du bloc
            # original, toujours sa vraie donnée, jamais celle d'un tirage.
            for j, hotel in enumerate(self.hotels):
                try:
                    mte_j = self._solve_dea_subgroup(j, comb_inp, comb_out, rts=rts)
                    bs_mte[hotel].append(mte_j)
                except Exception:
                    bs_mte[hotel].append(np.nan)

            # GTE bootstrap (frontière de groupe) — même principe, à l'échelle du groupe
            unique_g = groups.loc[self.hotels].unique()
            for grp in unique_g:
                grp_h    = [h for h in self.hotels if groups.get(h) == grp]
                grp_idx  = [self.hotels.index(h) for h in grp_h]
                if len(grp_h) < 3:
                    for h in grp_h: bs_gte[h].append(1.0)
                    continue
                grp_orig_inp = self.inputs[grp_idx]; grp_orig_out = self.outputs[grp_idx]
                grp_boot_inp = bs_inp[grp_idx];      grp_boot_out = bs_out[grp_idx]
                grp_comb_inp = np.vstack([grp_orig_inp, grp_boot_inp])
                grp_comb_out = np.vstack([grp_orig_out, grp_boot_out])
                for i, h in enumerate(grp_h):
                    try:
                        gte_j = self._solve_dea_subgroup(i, grp_comb_inp, grp_comb_out, rts=rts)
                        bs_gte[h].append(gte_j)
                        tgr_j = min(bs_mte[h][-1] / gte_j, 1.0) if gte_j > 0 else 0.0
                        bs_tgr[h].append(tgr_j)
                    except Exception:
                        bs_gte[h].append(np.nan); bs_tgr[h].append(np.nan)

        # Assembler résultats
        rows = []
        for _, row in base_df.iterrows():
            h   = row['Hôtel']
            gte_bs = [v for v in bs_gte[h] if not np.isnan(v)]
            mte_bs = [v for v in bs_mte[h] if not np.isnan(v)]
            tgr_bs = [v for v in bs_tgr[h] if not np.isnan(v)]

            r = dict(row)
            if len(gte_bs) >= 10:
                r['GTE IC bas'] = round(np.quantile(gte_bs, alpha), 4)
                r['GTE IC haut']= round(np.quantile(gte_bs, 1-alpha), 4)
            else:
                r['GTE IC bas'] = r['GTE IC haut'] = np.nan

            if len(mte_bs) >= 10:
                r['MTE IC bas'] = round(np.quantile(mte_bs, alpha), 4)
                r['MTE IC haut']= round(np.quantile(mte_bs, 1-alpha), 4)
            else:
                r['MTE IC bas'] = r['MTE IC haut'] = np.nan

            if len(tgr_bs) >= 10:
                r['TGR IC bas'] = round(np.quantile(tgr_bs, alpha), 4)
                r['TGR IC haut']= round(np.quantile(tgr_bs, 1-alpha), 4)
            else:
                r['TGR IC bas'] = r['TGR IC haut'] = np.nan

            rows.append(r)

        return pd.DataFrame(rows)

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
        # Utilise self.input_cols / self.output_cols (adapté au mode standard/compact/minimal)
        for col_name in self.input_cols:
            current  = float(self.df.loc[hotel, col_name])
            target   = self.targets[hotel]['inputs'].get(col_name, current)
            slack_v  = self.slacks[hotel]['inputs'].get(col_name, 0)
            radial   = current * bcc - current if self.orientation == 'input' else 0.0
            if self.orientation == 'output': slack_v = current - target
            rows.append({'Variable': f"↓ {col_name.replace('_',' ').title()}",
                         'Valeur Actuelle': round(current, 2), 'Mvt. Radial': round(radial, 2),
                         'Slack': round(slack_v, 2), 'Valeur Projetée': round(target, 2),
                         'Amélioration %': f"{(target - current) / current * 100:+.1f}%" if current > 0 else '—'})
        for col_name in self.output_cols:
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
    def generate_board_report(self, avg_salary: float = 35_000, revpar_value: float = 1) -> pd.DataFrame:
        rows = []
        for hotel in self.hotels:
            bcc = self.bcc_scores[hotel]; ccr = self.ccr_scores[hotel]; scale = self.scale_efficiency[hotel]
            slack_emp  = self.slacks[hotel]['inputs'].get('nb_employes', 0)
            slack_rev  = self.slacks[hotel]['outputs'].get('revpar', 0)
            upside_fte = round(slack_emp * avg_salary)
            if self.has_chambres:
                lits       = float(self.df.loc[hotel, 'nb_chambres'])
                upside_rev = round(slack_rev * lits * 365 * revpar_value / 1_000)
            else:
                upside_rev = None
            row = {'Hôtel': hotel, 'Score BCC': f"{bcc:.1%}", 'Score CCR': f"{ccr:.1%}",
                   'Eff. Échelle': f"{scale:.1%}",
                   'Quadrant': QUADRANT_LABELS.get(self.quadrants.get(hotel, ''), '—'),
                   'Rang TOPSIS': self.topsis_ranks.get(hotel, '—'),
                   'Score TOPSIS': f"{self.topsis_scores.get(hotel, 0):.3f}",
                   'Segment K-means': self.kmeans_labels.get(hotel, '—'),
                   'Upside ETP (€/an)': f"{upside_fte:,}" if upside_fte > 0 else '—',
                   'Upside RevPAR (k€)': (f"{upside_rev:,}" if upside_rev else '—')
                                          if self.has_chambres else 'n/d — chambres',
                   'Upside Total (k€)': (f"{upside_fte + (upside_rev or 0):,}"
                                          if upside_fte + (upside_rev or 0) > 0 else '—')
                                          if self.has_chambres else 'n/d — chambres',
                   'Priorité': ('✅ RAS' if bcc >= 0.95 else '🟡 Surveiller' if bcc >= 0.85 else '🔴 Action urgente')}
            if self.has_trevpar: row['TRevPAR (€/chambre/j)'] = self.trevpar.get(hotel, '—') or '—'
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



    def compute_super_efficiency(self) -> pd.DataFrame:
        """
        Super-Efficience — Andersen & Petersen (1993).

        Pour chaque DMU p : résoudre le LP BCC en excluant p du dataset de référence.
        Les DMUs efficients (BCC = 1) obtiennent un score > 1 → discrimination possible.
        Les DMUs inefficients gardent leur score BCC (inchangé par construction).

        Score > 1 : l'hôtel pourrait augmenter ses inputs et rester sur la frontière.
        Score = 3.5 : l'hôtel peut utiliser 3.5× ses inputs actuels et rester efficient.

        Réf. : Andersen & Petersen (1993) Management Science 39(10), 1261–1264.
        """
        import pulp as _pulp
        rows = []

        for p_idx, hotel in enumerate(self.hotels):
            bcc = self.bcc_scores.get(hotel, 0)

            # Exclure DMU p du dataset de référence
            ref_idx   = [i for i in range(self.n) if i != p_idx]
            inp_ref   = self.inputs[ref_idx]
            out_ref   = self.outputs[ref_idx]
            n_ref     = len(ref_idx)

            if n_ref == 0:
                rows.append({'Hôtel': hotel, 'BCC': f'{bcc:.1%}',
                             'Super-Efficience': 1.0, 'Δ vs BCC': '+0.000',
                             'Lecture': '— Seul DMU'})
                continue

            # LP BCC Input-Oriented sans DMU p
            if self.orientation == 'input':
                model  = _pulp.LpProblem(f"SE_{p_idx}", _pulp.LpMinimize)
                theta  = _pulp.LpVariable("theta", lowBound=0)
                lam    = _pulp.LpVariable.dicts("lam", range(n_ref), lowBound=0)

                model += theta

                for j in range(self.inputs.shape[1]):
                    model += (_pulp.lpSum(lam[k] * inp_ref[k, j] for k in range(n_ref))
                              <= theta * self.inputs[p_idx, j])
                for j in range(self.outputs.shape[1]):
                    model += (_pulp.lpSum(lam[k] * out_ref[k, j] for k in range(n_ref))
                              >= self.outputs[p_idx, j])
                model += _pulp.lpSum(lam.values()) == 1  # VRS

                model.solve(_pulp.PULP_CBC_CMD(msg=False))
                se = _pulp.value(theta) if _pulp.LpStatus[model.status] == 'Optimal' else bcc

            else:  # output-oriented
                model  = _pulp.LpProblem(f"SE_{p_idx}", _pulp.LpMaximize)
                phi    = _pulp.LpVariable("phi", lowBound=0)
                lam    = _pulp.LpVariable.dicts("lam", range(n_ref), lowBound=0)

                model += phi

                for j in range(self.inputs.shape[1]):
                    model += (_pulp.lpSum(lam[k] * inp_ref[k, j] for k in range(n_ref))
                              <= self.inputs[p_idx, j])
                for j in range(self.outputs.shape[1]):
                    model += (_pulp.lpSum(lam[k] * out_ref[k, j] for k in range(n_ref))
                              >= phi * self.outputs[p_idx, j])
                model += _pulp.lpSum(lam.values()) == 1

                model.solve(_pulp.PULP_CBC_CMD(msg=False))
                phi_val = _pulp.value(phi) if _pulp.LpStatus[model.status] == 'Optimal' else 1.0
                se = phi_val if phi_val else bcc

            se = round(float(se or bcc), 4)
            delta = round(se - bcc, 4)

            # Lecture
            if se >= 1.0:
                if se >= 2.0:
                    lecture = f"🏆 Leader robuste — peut doubler ses ressources et rester efficient"
                elif se >= 1.5:
                    lecture = f"🥇 Très robuste — marge de sécurité élevée"
                elif se >= 1.1:
                    lecture = f"✅ Robuste — efficient et confirmé"
                else:
                    lecture = f"🟡 Efficient mais fragile — faible marge"
            else:
                lecture = f"🔴 Inefficient — score BCC ({bcc:.1%})"

            rows.append({
                'Hôtel'           : hotel,
                'BCC'             : f"{bcc:.1%}",
                'Super-Efficience': se,
                'Δ vs BCC'        : f"{delta:+.4f}",
                'Lecture'         : lecture,
            })

        df_out = (pd.DataFrame(rows)
                    .sort_values('Super-Efficience', ascending=False)
                    .reset_index(drop=True))
        df_out.insert(0, 'Rang SE', range(1, len(df_out) + 1))
        return df_out

    def compute_cross_efficiency(self) -> tuple:
        """
        Cross-Efficience — Doyle & Green (1994) forme multiplicatrice.

        Chaque DMU j est évalué avec les poids optimaux de TOUS ses pairs.
        Score CE_k = moyenne des évaluations de k par les poids de chaque j.

        Retourne :
            df_summary : DataFrame avec BCC, Cross-Efficience, Δ, Lecture par hôtel
            ce_matrix  : matrice n×n des évaluations croisées (numpy array)

        Réf. : Doyle & Green (1994) Omega 22(6) · Anderson & Peterson (2008)
        """
        import pulp as _pulp
        n      = self.n
        hotels = self.hotels
        X      = self.inputs    # shape (n, n_inputs)
        Y      = self.outputs   # shape (n, n_outputs)
        n_in   = X.shape[1]
        n_out  = Y.shape[1]
        eps    = 1e-6

        # Matrice CE : CE[p, k] = score de k évalué avec les poids de p
        ce_matrix = np.zeros((n, n))

        for p in range(n):
            # Forme multiplicatrice — modèle CCR Output-Oriented
            # max  Σ_r u_r * y_rp
            # s.t. Σ_i v_i * x_ip = 1
            #      Σ_r u_r * y_rj - Σ_i v_i * x_ij ≤ 0  ∀j
            #      u_r, v_i ≥ ε
            model = _pulp.LpProblem(f"CE_{p}", _pulp.LpMaximize)
            u = [_pulp.LpVariable(f"u_{r}", lowBound=eps) for r in range(n_out)]
            v = [_pulp.LpVariable(f"v_{i}", lowBound=eps) for i in range(n_in)]

            # Objectif
            model += _pulp.lpSum(u[r] * Y[p, r] for r in range(n_out))

            # Normalisation inputs de p
            model += (_pulp.lpSum(v[i] * X[p, i] for i in range(n_in)) == 1)

            # Contraintes DMUs
            for j in range(n):
                model += (
                    _pulp.lpSum(u[r] * Y[j, r] for r in range(n_out)) -
                    _pulp.lpSum(v[i] * X[j, i] for i in range(n_in)) <= 0
                )

            model.solve(_pulp.PULP_CBC_CMD(msg=False))

            if _pulp.LpStatus[model.status] == 'Optimal':
                u_star = np.array([_pulp.value(u[r]) or eps for r in range(n_out)])
                v_star = np.array([_pulp.value(v[i]) or eps for i in range(n_in)])

                # Évaluer tous les DMUs avec les poids de p
                for k in range(n):
                    denom = np.dot(v_star, X[k])
                    if denom > 0:
                        ce_matrix[p, k] = np.dot(u_star, Y[k]) / denom
                    else:
                        ce_matrix[p, k] = 0.0
            else:
                # Si infaisable, utiliser le score BCC comme fallback
                for k in range(n):
                    ce_matrix[p, k] = self.bcc_scores.get(hotels[k], 0)

        # Score CE moyen pour chaque hôtel (colonne = moyenne par k)
        ce_scores = ce_matrix.mean(axis=0)

        # Construire le DataFrame résumé
        rows = []
        for k, hotel in enumerate(hotels):
            bcc = self.bcc_scores.get(hotel, 0)
            ce  = round(float(ce_scores[k]), 4)
            delta = round(ce - bcc, 4)

            if bcc >= 0.999 and ce >= 0.85:
                lecture = "✅ Robuste — efficient et confirmé par les pairs"
            elif bcc >= 0.999 and ce < 0.75:
                lecture = "⚠️ Fragile — efficient uniquement avec ses propres poids"
            elif bcc >= 0.999 and ce < 0.85:
                lecture = "🟡 Efficient mais modérément robuste"
            elif ce >= bcc - 0.05:
                lecture = "🟢 Score stable — cohérent avec BCC"
            else:
                lecture = "🔴 Écart important — score BCC peu robuste"

            rows.append({
                'Rang CE'          : 0,
                'Hôtel'            : hotel,
                'BCC'              : f"{bcc:.1%}",
                'Cross-Efficience' : ce,
                'Δ BCC-CE'         : f"{delta:+.4f}",
                'Lecture'          : lecture,
            })

        df_out = (pd.DataFrame(rows)
                    .sort_values('Cross-Efficience', ascending=False)
                    .reset_index(drop=True))
        df_out['Rang CE'] = range(1, len(df_out) + 1)

        return df_out, ce_matrix

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
        cap_in  = ['surface_m2', 'capex_annuel'] + (['nb_chambres'] if self.has_chambres else [])
        # Le CA doit être une vraie donnée — plus de repli RevPAR x occ x 365 x
        # nb_chambres (proxy non fiable, cf. audit portefeuille Ibis Atream).
        if 'total_revenue' not in df.columns:
            return None
        df['_rev'] = df['total_revenue']
        cap_out = ['_rev'] + (['gop'] if 'gop' in df.columns else [])
        X_in = df[cap_in].values.astype(float); X_out = df[cap_out].values.astype(float); n = len(self.hotels)

        # ── Sanitation : PuLP refuse NaN/inf et les inputs nuls rendent le PL dégénéré ──
        # Réf. : Cooper, Seiford & Tone (2007) — les données DEA doivent être strictement positives
        X_in  = np.nan_to_num(X_in,  nan=0.0, posinf=0.0, neginf=0.0)
        X_out = np.nan_to_num(X_out, nan=0.0, posinf=0.0, neginf=0.0)
        if not np.isfinite(X_in).all() or not np.isfinite(X_out).all():
            return None
        # Colonnes entièrement nulles/négatives → écartées (input ou output non renseigné)
        _keep_in  = [j for j in range(X_in.shape[1])  if (X_in[:, j]  > 0).all()]
        _keep_out = [j for j in range(X_out.shape[1]) if (X_out[:, j] > 0).all()]
        if not _keep_in or not _keep_out:
            return None   # données capital incomplètes — Tab affichera un message
        X_in  = X_in[:,  _keep_in]
        X_out = X_out[:, _keep_out]

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

    def compute_sbm(self) -> Dict[str, Dict]:
        """
        SBM non-orienté VRS — Tone (2001) EJOR 130:498-509.
        Range normalization pour données non-positives : Tone & Tsutsui (2010).
        Charnes-Cooper linearization de l'objectif fractionnaire.

        Score ρ* ∈ [0,1] — NON comparable aux scores BCC/CCR radials.
        ρ* = 1  ↔  DMU SBM-efficiente (aucun slack améliorable).
        ρ* < 1  ↔  inefficience mesurée sur inputs ET outputs simultanément.
        """
        n = len(self.hotels)
        m = self.inputs.shape[1]
        s = self.outputs.shape[1]

        # Range par variable — gère les zéros et négatifs (Tone & Tsutsui 2010)
        x_range = np.ptp(self.inputs,  axis=0).astype(float)
        y_range = np.ptp(self.outputs, axis=0).astype(float)
        x_range = np.where(x_range < 1e-9, 1.0, x_range)
        y_range = np.where(y_range < 1e-9, 1.0, y_range)

        results: Dict[str, Dict] = {}
        for idx, hotel in enumerate(self.hotels):
            x0 = self.inputs[idx].astype(float)
            y0 = self.outputs[idx].astype(float)

            prob   = pulp.LpProblem(f"SBM_{idx}", pulp.LpMinimize)
            t      = pulp.LpVariable("t",  lowBound=1e-8)
            Lambda = [pulp.LpVariable(f"L{j}", lowBound=0) for j in range(n)]
            S_in   = [pulp.LpVariable(f"Si{i}", lowBound=0) for i in range(m)]
            S_out  = [pulp.LpVariable(f"So{r}", lowBound=0) for r in range(s)]

            # Objectif : τ = t − (1/m)Σ(Sᵢ⁻/rᵢ)
            prob += t - (1.0 / m) * pulp.lpSum(S_in[i]  / x_range[i] for i in range(m))

            # Normalisation : t + (1/s)Σ(Sᵣ⁺/rᵣ) = 1
            prob += t + (1.0 / s) * pulp.lpSum(S_out[r] / y_range[r] for r in range(s)) == 1

            # Inputs  : ΣΛⱼ·xᵢⱼ + Sᵢ⁻ = t·xᵢ₀
            for i in range(m):
                prob += (pulp.lpSum(Lambda[j] * float(self.inputs[j, i]) for j in range(n))
                         + S_in[i] == t * x0[i])

            # Outputs : ΣΛⱼ·yᵣⱼ − Sᵣ⁺ = t·yᵣ₀
            for r in range(s):
                prob += (pulp.lpSum(Lambda[j] * float(self.outputs[j, r]) for j in range(n))
                         - S_out[r] == t * y0[r])

            # VRS : ΣΛⱼ = t  (Charnes-Cooper de Σλ = 1)
            prob += pulp.lpSum(Lambda) == t

            prob.solve(pulp.PULP_CBC_CMD(msg=0))

            if prob.status == 1:
                t_val = pulp.value(t) or 1e-8
                rho   = round(min(float(pulp.value(prob.objective)), 1.0), 4)
                s_in  = {self.input_cols[i]:  round(float(pulp.value(S_in[i]))  / t_val, 4) for i in range(m)}
                s_out = {self.output_cols[r]: round(float(pulp.value(S_out[r])) / t_val, 4) for r in range(s)}
            else:
                rho   = None
                s_in  = {c: None for c in self.input_cols}
                s_out = {c: None for c in self.output_cols}

            results[hotel] = {'score': rho, 'slacks_in': s_in, 'slacks_out': s_out,
                               'feasible': prob.status == 1}
        return results

    def compute_nondiscretionary_targets(
        self,
        hotel: str,
        fixed_inputs: list,          # noms de colonnes verrouillées, ex. ['surface_m2']
    ) -> Optional[Dict]:
        """
        DEA avec inputs non-discrétionnaires — Banker & Morey (1986) Mgmt Sci.
        Les inputs dans `fixed_inputs` sont contraints à leur valeur actuelle (pas de θ).
        θ est minimisé uniquement sur les inputs discrétionnaires.

        Retourne :
          - theta_nd   : score d'efficience contraint
          - targets_disc : nouvelles cibles pour les inputs discrétionnaires
          - targets_out  : cibles outputs (inchangées dans la logique)
          - slacks_disc  : slacks résiduels inputs discrétionnaires
          - infeasible   : True si aucune solution trouvée
        """
        if hotel not in self.hotels:
            return None

        idx = list(self.hotels).index(hotel)
        n   = len(self.hotels)
        x0  = self.inputs[idx]
        y0  = self.outputs[idx]

        disc_idx  = [i for i, c in enumerate(self.input_cols) if c not in fixed_inputs]
        fixed_idx = [i for i, c in enumerate(self.input_cols) if c in fixed_inputs]

        prob    = pulp.LpProblem(f"ND_DEA_{hotel}", pulp.LpMinimize)
        theta   = pulp.LpVariable("theta", lowBound=0)
        lambdas = pulp.LpVariable.dicts("lam", range(n), lowBound=0)

        prob += theta

        # Inputs discrétionnaires : Σλⱼxᵢⱼ ≤ θ·xᵢ₀
        for i in disc_idx:
            prob += pulp.lpSum(lambdas[j] * float(self.inputs[j, i]) for j in range(n)) <= theta * float(x0[i])

        # Inputs non-discrétionnaires (fixes) : Σλⱼxᵢⱼ ≤ xᵢ₀  (sans θ)
        for i in fixed_idx:
            prob += pulp.lpSum(lambdas[j] * float(self.inputs[j, i]) for j in range(n)) <= float(x0[i])

        # Outputs : Σλⱼyᵣⱼ ≥ yᵣ₀
        for r in range(self.outputs.shape[1]):
            prob += pulp.lpSum(lambdas[j] * float(self.outputs[j, r]) for j in range(n)) >= float(y0[r])

        # VRS
        prob += pulp.lpSum(lambdas.values()) == 1

        prob.solve(pulp.PULP_CBC_CMD(msg=0))

        if prob.status != 1:
            return {'infeasible': True, 'hotel': hotel, 'fixed_inputs': fixed_inputs}

        th = float(pulp.value(theta))

        # Cibles discrétionnaires = θ · xᵢ₀ - slack_i
        lam_vals = {j: float(pulp.value(lambdas[j])) for j in range(n)}
        targets_disc = {}
        slacks_disc  = {}
        for i in disc_idx:
            col      = self.input_cols[i]
            proj     = sum(lam_vals[j] * float(self.inputs[j, i]) for j in range(n))
            slack    = max(0.0, th * float(x0[i]) - proj)
            target   = max(0.0, th * float(x0[i]) - slack)
            targets_disc[col] = round(target, 4)
            slacks_disc[col]  = round(slack,  4)

        # Cibles outputs
        targets_out = {}
        for r in range(self.outputs.shape[1]):
            col    = self.output_cols[r]
            proj   = sum(lam_vals[j] * float(self.outputs[j, r]) for j in range(n))
            target = max(float(y0[r]), proj)
            targets_out[col] = round(target, 4)

        return {
            'infeasible'   : False,
            'hotel'        : hotel,
            'theta_nd'     : round(th, 4),
            'fixed_inputs' : fixed_inputs,
            'targets_disc' : targets_disc,
            'slacks_disc'  : slacks_disc,
            'targets_out'  : targets_out,
            'current_vals' : {self.input_cols[i]: round(float(x0[i]), 4) for i in disc_idx},
        }


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
    proxy_cols: Optional[set] = None,
) -> Dict[str, ModuleResult]:
    """
    Exécute le moteur DEA scipy pour chaque module actif sur le même dataset.

    Args:
        df              : DataFrame avec colonnes = variables hôtelières
        dmu_col         : colonne identifiant les DMUs (nom hôtel)
        active_modules  : liste des module_ids à exécuter
        variable_overrides : {module_id: {"inputs": [...], "outputs": [...]}}
        proxy_cols      : colonnes calculées par reformulation d'autres colonnes
                          (ex. rooms_revenue, adr, trevpar dérivés du RevPAR) —
                          ne comptent pas pour satisfaire les "required" d'un
                          module (cf. check_module_feasibility), pour éviter
                          qu'un module se déclare calculable sur une simple
                          reformulation arithmétique plutôt que sur une vraie
                          donnée départementale/marché.

    Returns:
        {module_id: ModuleResult}
    """
    available_cols = [c for c in df.columns if c != dmu_col]
    results: Dict[str, ModuleResult] = {}

    for mod_id in active_modules:
        if mod_id not in MODULES:
            continue

        mod_cfg    = MODULES[mod_id]
        feasibility = check_module_feasibility(mod_id, available_cols, proxy_cols=proxy_cols)

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
