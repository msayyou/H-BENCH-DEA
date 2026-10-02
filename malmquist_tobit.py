"""
malmquist_tobit.py — Modules Malmquist & Tobit pour DEA-H v3.7
REIV Hospitality

Malmquist : Caves, Christensen & Diewert (1982) ; Färe et al. (1994)
Tobit     : Tobin (1958) ; Simar & Wilson (2007) second stage DEA
"""
from __future__ import annotations
import numpy as np
import pandas as pd
try:
    import pulp
except Exception:  # solveur optionnel : les calculs DEA deviennent non calculables
    pulp = None
from scipy import stats, optimize



def _get_df(dea):
    """Retourne un DataFrame exploitable, sinon DataFrame vide."""
    df = getattr(dea, 'df', None)
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()


def _get_hotels(dea, df=None):
    """Accès sûr aux hôtels, limité aux lignes réellement présentes."""
    hotels = getattr(dea, 'hotels', []) or []
    if df is None:
        df = _get_df(dea)
    return [h for h in hotels if h in df.index] if isinstance(df, pd.DataFrame) else []


def _find_col(df, name):
    """Résout un nom de colonne exact ou insensible à la casse."""
    if name in df.columns:
        return name
    wanted = str(name).casefold()
    return next((c for c in df.columns if str(c).casefold() == wanted), None)


def _numeric_frame(df, cols, index=None):
    """Extrait des colonnes en numérique, valeurs invalides -> NaN."""
    out = df.reindex(index=index, columns=cols) if index is not None else df.reindex(columns=cols)
    return out.apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)


# ══════════════════════════════════════════════════════════════════════════════
# MALMQUIST PRODUCTIVITY INDEX
# ══════════════════════════════════════════════════════════════════════════════

_N1_COLS = {
    'nb_employes'     : 'nb_employes_n1',
    'couts_op_ex'     : 'couts_op_ex_n1',
    'revpar'          : 'revpar_n1',
    'satisfaction'    : 'satisfaction_n1',
    'taux_occupation' : 'taux_occupation_n1',
}

def _dea_cross(j: int, X_eval: np.ndarray, Y_eval: np.ndarray, X_ref: np.ndarray, Y_ref: np.ndarray) -> float | None:
    """Input-orienté BCC, sécurisé contre solveur indisponible/infeasible."""
    if pulp is None:
        return None
    try:
        X_eval, Y_eval = np.asarray(X_eval, float), np.asarray(Y_eval, float)
        X_ref, Y_ref = np.asarray(X_ref, float), np.asarray(Y_ref, float)
        if j < 0 or j >= len(X_eval) or not (np.all(np.isfinite(X_eval)) and np.all(np.isfinite(Y_eval)) and np.all(np.isfinite(X_ref)) and np.all(np.isfinite(Y_ref))):
            return None
        n_ref, m_in, m_out = X_ref.shape[0], X_ref.shape[1], Y_ref.shape[1]
        mdl = pulp.LpProblem(f"MQ_{j}", pulp.LpMinimize)
        th = pulp.LpVariable("theta", lowBound=0)
        lam = pulp.LpVariable.dicts("lam", range(n_ref), lowBound=0)
        mdl += th
        for i in range(m_in):
            mdl += pulp.lpSum(lam[k] * X_ref[k, i] for k in range(n_ref)) <= th * X_eval[j, i]
        for r in range(m_out):
            mdl += pulp.lpSum(lam[k] * Y_ref[k, r] for k in range(n_ref)) >= Y_eval[j, r]
        mdl += pulp.lpSum(lam.values()) == 1
        status = mdl.solve(pulp.PULP_CBC_CMD(msg=False))
        value = pulp.value(th)
        return round(float(value), 6) if status == pulp.LpStatusOptimal and value is not None and np.isfinite(value) else None
    except Exception:
        return None


def compute_malmquist(dea) -> pd.DataFrame | None:
    """
    Calcule l'indice de productivité de Malmquist pour chaque DMU.

    Nécessite dans dea.df les colonnes _n1 :
      nb_employes_n1, couts_op_ex_n1,
      revpar_n1, taux_occupation_n1 (+ satisfaction_n1 optionnel)

    Décomposition (Färe et al. 1994) :
      M = Catch-up × Frontier Shift
      Catch-up      = D¹(x¹,y¹) / D⁰(x⁰,y⁰)   — convergence vers frontière
      Frontier Shift = √[ D⁰(x¹,y¹)/D¹(x¹,y¹) × D⁰(x⁰,y⁰)/D¹(x⁰,y⁰) ]
                                                   — déplacement de la frontière

    Réf. : Caves, Christensen & Diewert (1982) Econometrica
           Färe, Grosskopf, Norris & Zhang (1994) American Economic Review
    """
    df = _get_df(dea)
    # Vérifier colonnes _n1 (exactes ou casse insensible)
    resolved = {k: _find_col(df, v) for k, v in _N1_COLS.items()}
    missing = [v for k, v in _N1_COLS.items() if resolved[k] is None and v != 'satisfaction_n1']
    if missing:
        return None

    input_cols  = ['nb_employes', 'couts_op_ex']
    output_cols = ['revpar', 'taux_occupation']
    if resolved.get('satisfaction') is not None:
        output_cols.append('satisfaction')

    cols_now = [_find_col(df, c) for c in input_cols + output_cols]
    cols_old = [resolved.get(c) for c in input_cols + output_cols]
    hotels = _get_hotels(dea, df)
    if not hotels or any(c is None for c in cols_now + cols_old):
        return None
    X1 = _numeric_frame(df, cols_now[:len(input_cols)], hotels)
    Y1 = _numeric_frame(df, cols_now[len(input_cols):], hotels)
    X0 = _numeric_frame(df, cols_old[:len(input_cols)], hotels)
    Y0 = _numeric_frame(df, cols_old[len(input_cols):], hotels)
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

        # ── Interprétation — 5 cas Alpha vs Marée montante ──────────────────
        # Réf. : DEA-H v3.9 Documentation — Section "Alpha vs Marée montante"
        #        Caves et al. (1982) ; Färe et al. (1994)
        #
        # Alpha total   : CU > 1 ET FS > 1 → progrès gestion + secteur
        # Alpha pur     : CU > 1 ET FS ≤ 1 → l'hôtel progresse contre la tendance
        # Marée montante: CU ≤ 1 ET FS > 1 ET TFP > 1 → porté par le secteur
        # Résistance    : CU ≤ 1 ET FS > 1 ET TFP ≤ 1 → secteur monte mais hôtel recule
        # Régression    : CU ≤ 1 ET FS ≤ 1 → double recul
        if catchup > 1.0 and frontier_shift > 1.0:
            interp    = '🟢 Alpha total — progrès gestion + progrès sectoriel (TFP porté par les deux)'
            alpha_cat = 'Alpha total'
        elif catchup > 1.0 and frontier_shift <= 1.0:
            interp    = '🏆 Alpha pur — hôtel progresse malgré frontière stable/régressante (mérite pur)'
            alpha_cat = 'Alpha pur'
        elif catchup <= 1.0 and frontier_shift > 1.0 and tfp > 1.0:
            interp    = '🔵 Marée montante — TFP > 1 grâce au secteur, pas à la gestion propre'
            alpha_cat = 'Marée montante'
        elif catchup <= 1.0 and frontier_shift > 1.0 and tfp <= 1.0:
            interp    = '🟠 Résistance — secteur progresse mais efficience de l\'hôtel en recul'
            alpha_cat = 'Résistance'
        else:
            interp    = '🔴 Régression — gestion ET technologie sectorielle en recul'
            alpha_cat = 'Régression'

        rows.append({
            'Hôtel'          : hotel,
            'BCC N-1'        : f'{d00:.3f}',
            'BCC N'          : f'{d11:.3f}',
            'Catch-up'       : f'{catchup:.3f}',
            'Frontier Shift' : f'{frontier_shift:.3f}',
            'Malmquist TFP'  : f'{tfp:.3f}',
            'Catégorie'      : alpha_cat,
            'Interprétation' : interp,
        })

    return pd.DataFrame(rows).sort_values('Malmquist TFP', ascending=False,
                                          key=lambda x: pd.to_numeric(x, errors='coerce')).reset_index(drop=True)


def has_n1_cols(df: pd.DataFrame) -> list[str]:
    """Retourne la liste des colonnes _n1 présentes dans df."""
    return [c for c in _N1_COLS.values() if _find_col(df, c) is not None]


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
    df = _get_df(dea)
    hotels = _get_hotels(dea, df)
    resolved_env = {c: _find_col(df, c) for c in env_vars}
    valid = [h for h in hotels if all(resolved_env[c] is not None for c in env_vars)]
    valid = [h for h in valid if all(pd.to_numeric(df.loc[h, resolved_env[c]], errors='coerce') == pd.to_numeric(df.loc[h, resolved_env[c]], errors='coerce') for c in env_vars)]
    if len(valid) < len(env_vars) + 3:
        return {'error': f'Trop peu d\'observations valides ({len(valid)}) '
                         f'pour {len(env_vars)} régresseurs.'}

    scores = getattr(dea, 'bcc_scores', {}) or {}
    valid = [h for h in valid if pd.to_numeric(scores.get(h), errors='coerce') == pd.to_numeric(scores.get(h), errors='coerce')]
    y = pd.to_numeric(pd.Series([scores.get(h) for h in valid]), errors='coerce').to_numpy(dtype=float)
    X_raw = _numeric_frame(df, [resolved_env[c] for c in env_vars], valid)

    # Rejeter les régresseurs constants — même raisonnement que Simar-Wilson
    # ci-dessous : un écart-type nul rend le coefficient non identifié.
    X_means = X_raw.mean(axis=0)
    X_stds  = X_raw.std(axis=0)
    _const_vars = [env_vars[i] for i in range(len(env_vars)) if X_stds[i] == 0]
    if _const_vars:
        _const_labels = [ (env_labels or {}).get(v, v) for v in _const_vars]
        return {'error': (
            f"Variable(s) constante(s) sur l'échantillon (aucune variance à expliquer) : "
            f"{', '.join(_const_labels)}. Retirez-la(les) de la sélection."
        )}

    # Normalisation des régresseurs (meilleure convergence)
    X_norm  = (X_raw - X_means) / X_stds
    X_fit   = np.column_stack([np.ones(len(valid)), X_norm])

    # Optimisation MLE
    k     = X_fit.shape[1]
    p0    = np.zeros(k + 1)
    p0[0] = float(np.mean(y))
    p0[-1] = max(float(np.std(y)), 0.05)

    try:
        res = optimize.minimize(_tobit_loglik, p0, args=(y, X_fit, 1.0), method='Nelder-Mead', options={'maxiter': 10000, 'xatol': 1e-7, 'fatol': 1e-7})
    except Exception as exc:
        return {'error': f'Optimisation Tobit échouée : {exc}'}
    if not getattr(res, 'success', False) or not np.all(np.isfinite(res.x)) or not np.isfinite(res.fun):
        return {'error': f'Optimisation Tobit non convergente : {getattr(res, "message", "statut inconnu")}'}

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
        'inputs' : ['rooms_cost'],
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
    """BCC input-orienté sécurisé."""
    if pulp is None:
        return None
    try:
        X, Y = np.asarray(X, float), np.asarray(Y, float)
        if j < 0 or j >= len(X) or not (np.all(np.isfinite(X)) and np.all(np.isfinite(Y))):
            return None
        n, m_in, m_out = X.shape[0], X.shape[1], Y.shape[1]
        mp = pulp.LpProblem(f"BCC_{j}", pulp.LpMinimize)
        th = pulp.LpVariable("theta", lowBound=0)
        lam = pulp.LpVariable.dicts("lam", range(n), lowBound=0)
        mp += th
        for i in range(m_in):
            mp += pulp.lpSum(lam[k] * X[k, i] for k in range(n)) <= th * X[j, i]
        for r in range(m_out):
            mp += pulp.lpSum(lam[k] * Y[k, r] for k in range(n)) >= Y[j, r]
        mp += pulp.lpSum(lam.values()) == 1
        status = mp.solve(pulp.PULP_CBC_CMD(msg=False)); value = pulp.value(th)
        return round(float(value), 4) if status == pulp.LpStatusOptimal and value is not None and np.isfinite(value) else None
    except Exception:
        return None


def compute_mdea_room_fb(dea) -> dict:
    """
    Décomposition MDEA Room / F&B selon Yu (2012).

    Calcule trois scores BCC indépendants :
      - BCC Global     : modèle standard DEA-H
      - BCC Room       : efficience du département Hébergement
      - BCC F&B        : efficience du département Restauration
    
    L'écart entre BCC Room et BCC F&B localise la source d'inefficience
    — réplication simplifiée de la décomposition MDEA/GAR (Yu 2012, Eq. 3-6).
    
    Inputs Room  : rooms_cost
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

    df = _get_df(dea)
    hotels = _get_hotels(dea, df)

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

    X_room = _numeric_frame(df, room_in_avail, hotels)
    Y_room = _numeric_frame(df, room_out_avail, hotels)
    X_fb   = _numeric_frame(df, fb_in_avail, hotels)
    Y_fb   = _numeric_frame(df, fb_out_avail, hotels)

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

    hotels = _get_hotels(dea, _get_df(dea))
    groups = groups if isinstance(groups, pd.Series) else pd.Series(dtype=object)
    unique_g = [g for g in groups.reindex(hotels).dropna().unique()]
    grp_scores = {}
    for g in unique_g:
        h_list = [h for h in hotels if groups.get(h) == g and h in (getattr(dea, 'bcc_scores', {}) or {}) and pd.to_numeric((getattr(dea, 'bcc_scores', {}) or {}).get(h), errors='coerce') == pd.to_numeric((getattr(dea, 'bcc_scores', {}) or {}).get(h), errors='coerce')]
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


# ══════════════════════════════════════════════════════════════════════════════
# BUILD STAGE 2 VARIABLES — FINANCIER + SAISONNALITÉ
# ══════════════════════════════════════════════════════════════════════════════

def build_stage2_vars(dea) -> list:
    """
    Calcule les variables environnementales dérivées pour le Stage 2 (Tobit / Simar-Wilson).
    Injecte directement dans dea.df. Retourne liste des colonnes créées.

    Variables ajoutées :
      ltv_proxy      : CAPEX annuel / CA total × 100  — intensité capital (proxy LTV)
      asset_yield    : CA total / Book value assets    — rendement actifs (proxy ROA)
      capex_per_room : CAPEX annuel / nb_chambres          — CAPEX par chambre (k€)
      gop_margin_pct : GOP / CA total × 100            — marge opérationnelle
      log_nb_chambres    : log(nb_chambres)                   — effet taille (log-linéaire)

    Réf. : Pulina & Santoni (2018) Tourism Economics
           Simar & Wilson (2007) Journal of Econometrics
    """
    df  = dea.df
    added = []

    # 1. LTV proxy — intensité capital / dette implicite
    if 'capex_annuel' in df.columns and 'total_revenue' in df.columns:
        _tr = df['total_revenue'].replace(0, np.nan)
        df['ltv_proxy'] = (df['capex_annuel'] / _tr * 100).round(2)
        added.append('ltv_proxy')

    # 2. Asset yield — retour sur actifs (ROA proxy)
    if 'total_revenue' in df.columns and 'book_value_assets' in df.columns:
        _bv = df['book_value_assets'].replace(0, np.nan)
        df['asset_yield'] = (df['total_revenue'] / _bv).round(4)
        added.append('asset_yield')

    # 3. CAPEX par chambre
    if 'capex_annuel' in df.columns and 'nb_chambres' in df.columns:
        _lits = df['nb_chambres'].replace(0, np.nan)
        df['capex_per_room'] = (df['capex_annuel'] / _lits).round(2)
        added.append('capex_per_room')

    # 4. Marge GOP %
    if 'gop' in df.columns and 'total_revenue' in df.columns:
        _tr = df['total_revenue'].replace(0, np.nan)
        df['gop_margin_pct'] = (df['gop'] / _tr * 100).round(2)
        added.append('gop_margin_pct')

    # 5. Log taille
    if 'nb_chambres' in df.columns:
        df['log_nb_chambres'] = np.log(pd.to_numeric(df['nb_chambres'], errors='coerce').replace(0, np.nan)).round(4)
        added.append('log_nb_chambres')

    return added


# ══════════════════════════════════════════════════════════════════════════════
# SIMAR & WILSON (2007) — RÉGRESSION TRONQUÉE BOOTSTRAPPÉE
# ══════════════════════════════════════════════════════════════════════════════

def _sw_truncated_loglik(params: np.ndarray, y: np.ndarray,
                         X: np.ndarray, upper: float = 1.0) -> float:
    """
    Log-vraisemblance régression tronquée normale (tronquée à droite en `upper`).
    Signe négatif pour minimisation.
    f(y|X) = φ((y-Xβ)/σ) / [σ·Φ((upper-Xβ)/σ)]
    Réf. : Simar & Wilson (2007) eq. (4)
    """
    beta  = params[:-1]
    sigma = max(abs(params[-1]), 1e-6)
    mu    = X @ beta
    resid = (y - mu) / sigma
    trunc = stats.norm.cdf((upper - mu) / sigma)
    trunc = np.maximum(trunc, 1e-10)
    ll    = np.sum(-np.log(sigma) + stats.norm.logpdf(resid) - np.log(trunc))
    return -ll


def _fit_sw(y: np.ndarray, X: np.ndarray,
            upper: float = 1.0, p0: np.ndarray | None = None):
    """MLE régression tronquée. Retourne (beta, sigma, success, log_lik)."""
    k = X.shape[1]
    if p0 is None:
        p0       = np.zeros(k + 1)
        p0[0]    = float(np.mean(y))
        p0[-1]   = max(float(np.std(y)), 0.05)
    try:
        res = optimize.minimize(_sw_truncated_loglik, p0, args=(y, X, upper), method='Nelder-Mead', options={'maxiter': 15000, 'xatol': 1e-8, 'fatol': 1e-8})
    except Exception:
        return np.full(k, np.nan), np.nan, False, np.nan
    if not getattr(res, 'success', False) or not np.all(np.isfinite(res.x)) or not np.isfinite(res.fun):
        return np.full(k, np.nan), np.nan, False, np.nan
    beta_hat  = res.x[:-1]
    sigma_hat = max(abs(res.x[-1]), 1e-6)
    return beta_hat, sigma_hat, res.success, -res.fun


def compute_simar_wilson(
    dea,
    env_vars   : list,
    env_labels : dict | None = None,
    n_bootstrap: int  = 200,
) -> dict:
    """
    Simar & Wilson (2007) Algorithm 1 — Régression tronquée bootstrappée.

    Différences vs Tobit censuré :
    ┌─────────────────────────────────────────────────────┐
    │ Tobit        : CENSURE les DMUs efficients (θ=1)   │
    │               → les inclut via P(Y≥1) dans L       │
    │ Simar-Wilson : TRONQUE — exclut θ=1 de l'estimation│
    │               → corrige la corrélation des scores  │
    │               via bootstrap paramétrique           │
    └─────────────────────────────────────────────────────┘

    Bootstrap (B itérations) :
      1. Génère θ*_i ~ N(Zβ̂, σ̂²) tronquée à droite en 1
         pour TOUS les DMUs
      2. Ré-estime régression tronquée sur {θ*_i < 1}
      3. SE = σ(β*_b) ; IC95% = percentiles 2.5 / 97.5

    n_bootstrap=200 recommandé (Simar & Wilson 2007, p.48)

    Réf. : Simar L. & Wilson P.W. (2007)
           Journal of Econometrics 136(1), 31-64.
    """
    from scipy.stats import truncnorm as _tnorm

    if env_labels is None:
        env_labels = {}

    # ── Données ──────────────────────────────────────────────────────────────
    df = _get_df(dea)
    hotels = _get_hotels(dea, df)
    resolved_env = {c: _find_col(df, c) for c in env_vars}
    valid = [h for h in hotels if all(resolved_env[c] is not None for c in env_vars)]
    valid = [h for h in valid if all(pd.to_numeric(df.loc[h, resolved_env[c]], errors='coerce') == pd.to_numeric(df.loc[h, resolved_env[c]], errors='coerce') for c in env_vars)]
    if len(valid) < len(env_vars) + 3:
        return {'error': f'Trop peu d\'observations valides ({len(valid)}) '
                         f'pour {len(env_vars)} régresseurs.'}

    scores = getattr(dea, 'bcc_scores', {}) or {}
    valid = [h for h in valid if pd.to_numeric(scores.get(h), errors='coerce') == pd.to_numeric(scores.get(h), errors='coerce')]
    y_all = pd.to_numeric(pd.Series([scores.get(h) for h in valid]), errors='coerce').to_numpy(dtype=float)
    Z_raw = _numeric_frame(df, [resolved_env[c] for c in env_vars], valid)

    # Rejeter les régresseurs constants — un écart-type nul rend le
    # coefficient non identifié par les données (la colonne normalisée
    # devient une suite de zéros, sans contenu informatif). Sans ce
    # contrôle, l'ancien code masquait le problème en remplaçant l'écart-type
    # nul par 1.0, produisant un coefficient/IC/p-value d'apparence légitime
    # mais purement artefactuel — ex. une variable de localisation identique
    # pour tous les hôtels du portefeuille (grille jamais modifiée).
    Z_means = Z_raw.mean(axis=0);  Z_stds = Z_raw.std(axis=0)
    _const_vars = [env_vars[i] for i in range(len(env_vars)) if Z_stds[i] == 0]
    if _const_vars:
        _const_labels = [env_labels.get(v, v) for v in _const_vars]
        return {'error': (
            f"Variable(s) constante(s) sur l'échantillon (aucune variance à expliquer) : "
            f"{', '.join(_const_labels)}. Retirez-la(les) de la sélection — un coefficient "
            f"estimé dessus ne serait pas identifié par les données, quel que soit le chiffre "
            f"que produirait le solveur."
        )}

    # Normalisation
    Z_norm = (Z_raw - Z_means) / Z_stds
    Z_fit  = np.column_stack([np.ones(len(valid)), Z_norm])

    # ── Step 1 : Régression tronquée initiale (θ < 1 seulement) ─────────────
    mask_ineff = y_all < 1.0 - 1e-8
    n_ineff    = int(mask_ineff.sum())
    if n_ineff < len(env_vars) + 2:
        return {'error': f'Trop peu de DMUs inefficients ({n_ineff}) '
                         f'pour l\'estimation tronquée (besoin ≥ {len(env_vars)+2}).'}

    y_trunc = y_all[mask_ineff]
    X_trunc = Z_fit[mask_ineff]
    beta_hat, sigma_hat, converged, log_lik = _fit_sw(y_trunc, X_trunc)
    mu_all  = Z_fit @ beta_hat

    # ── Step 2 : Bootstrap Algorithm 1 ───────────────────────────────────────
    rng        = np.random.default_rng(42)
    boot_betas = []
    p0_b       = np.append(beta_hat, sigma_hat)

    for _ in range(n_bootstrap):
        # Générer θ* pour TOUS les DMUs — N(μ_i, σ²) tronquée droite à 1
        theta_star = np.empty(len(valid))
        for i in range(len(valid)):
            mu_i    = float(mu_all[i])
            b_upper = (1.0 - mu_i) / sigma_hat
            a_lower = max(-10.0, (-mu_i) / sigma_hat)  # borne inférieure ≈ 0
            try:
                theta_star[i] = _tnorm.rvs(
                    a_lower, b_upper, loc=mu_i, scale=sigma_hat,
                    random_state=rng,
                )
            except Exception:
                theta_star[i] = min(max(mu_i, 1e-4), 0.9999)

        # Ré-estimer sur {θ* < 1}
        mask_b = theta_star < 1.0 - 1e-8
        if mask_b.sum() < len(env_vars) + 2:
            continue
        try:
            beta_b, _, _, _ = _fit_sw(theta_star[mask_b], Z_fit[mask_b], p0=p0_b)
            boot_betas.append(beta_b)
        except Exception:
            continue

    if not boot_betas:
        return {'error': 'Bootstrap échoué — aucune itération n\'a convergé.'}

    boot_arr  = np.array(boot_betas)           # (B, k)
    boot_se   = boot_arr.std(axis=0)
    ci_lo     = np.percentile(boot_arr, 2.5,  axis=0)
    ci_hi     = np.percentile(boot_arr, 97.5, axis=0)

    # ── Dé-normalisation ──────────────────────────────────────────────────────
    k      = Z_fit.shape[1]
    labels = ['Constante'] + [env_labels.get(c, c) for c in env_vars]

    if not np.all(np.isfinite(beta_hat)) or not np.isfinite(sigma_hat):
        return {'error': 'Régression Simar-Wilson non convergente ou résultat non fini.'}

    def _denorm(arr):
        out = arr.copy()
        for i in range(1, k):
            out[i] /= Z_stds[i - 1]
        return out

    beta_dn  = _denorm(beta_hat)
    se_dn    = _denorm(boot_se)
    ci_lo_dn = _denorm(ci_lo)
    ci_hi_dn = _denorm(ci_hi)

    t_stats = np.where(se_dn > 1e-10, beta_dn / se_dn, np.nan)
    p_vals  = np.where(np.isnan(t_stats), np.nan,
                       2 * (1 - stats.norm.cdf(np.abs(t_stats))))

    def stars(p):
        if np.isnan(p): return ''
        return '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.10 else ''

    coef_df = pd.DataFrame({
        'Variable'     : labels,
        'Coeff.'       : [round(b, 4) for b in beta_dn],
        'SE Bootstrap' : [round(s, 4) for s in se_dn],
        'IC95% Lo'     : [round(lo, 4) for lo in ci_lo_dn],
        'IC95% Hi'     : [round(hi, 4) for hi in ci_hi_dn],
        't-stat'       : [round(t, 2) if not np.isnan(t) else '—' for t in t_stats],
        'p-value'      : [round(p, 4) if not np.isnan(p) else '—' for p in p_vals],
        'Sig.'         : [stars(p) for p in p_vals],
        'Effet'        : [('↑ améliore efficience' if b > 0 else '↓ réduit efficience')
                          if l != 'Constante' else '—'
                          for b, l in zip(beta_dn, labels)],
    })

    return {
        'coef_df'        : coef_df,
        'n'              : len(valid),
        'n_inefficients' : n_ineff,
        'n_bootstrap'    : len(boot_betas),
        'log_lik'        : round(log_lik, 4),
        'sigma'          : round(sigma_hat, 4),
        'converged'      : bool(converged),
    }
