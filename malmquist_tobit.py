"""
malmquist_tobit.py — Modules Malmquist & Tobit pour DEA-H v3.7
REIV Hospitality

Malmquist : Caves, Christensen & Diewert (1982) ; Färe et al. (1994)
Tobit     : Tobin (1958) ; Simar & Wilson (2007) second stage DEA
"""
from __future__ import annotations
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
