"""
modules_config.py — DEA-H Multi-Module Configuration
REIV Hospitality / VLTION PRO
-------------------------------------------------------
Définit les 7 modules d'analyse DEA pour l'industrie hôtelière.
Chaque module contient :
  - inputs / outputs : colonnes attendues dans le dataset
  - orientation : 'input' ou 'output'
  - model : 'BCC' ou 'CCR'
  - label_fr : nom affiché dans l'interface
  - description : objet d'analyse
  - references : littérature académique
  - required_cols : colonnes minimum obligatoires
  - optional_cols : colonnes enrichissant l'analyse si disponibles
"""

MODULES: dict[str, dict] = {

    # ─────────────────────────────────────────────────────────────────────────
    # MODULE 1 — OPÉRATIONNEL GLOBAL
    # ─────────────────────────────────────────────────────────────────────────
    "operational": {
        "label_fr": "🏨 Opérationnel Global",
        "description": (
            "Mesure l'efficience globale de l'exploitation hôtelière : "
            "capacité à transformer les ressources physiques et humaines "
            "en revenus et résultat opérationnel."
        ),
        "references": "Anderson et al. (1999) ; Barros (2005) ; Hwang & Chang (2003)",
        "orientation": "output",
        "model": "BCC",
        "inputs": {
            "nb_rooms": {
                "label": "Nombre de chambres (lits DEA-H si non fournies)",
                "unit": "unité",
                "required": True,
            },
            "fte_total": {
                "label": "Effectifs totaux (ETP)",
                "unit": "ETP",
                "required": True,
            },
            "opex_total": {
                "label": "Charges opérationnelles totales",
                "unit": "€",
                "required": False,
            },
            "surface_m2": {
                "label": "Surface totale exploitée (m²)",
                "unit": "m²",
                "required": False,
            },
        },
        "outputs": {
            "revpar": {
                "label": "RevPAR (€/chambre dispo.)",
                "unit": "€",
                "required": True,
            },
            "occupancy_rate": {
                "label": "Taux d'occupation (%)",
                "unit": "%",
                "required": True,
            },
            "total_revenue": {
                "label": "Chiffre d'affaires total",
                "unit": "€",
                "required": False,
            },
            "gop": {
                "label": "GOP (Gross Operating Profit)",
                "unit": "€",
                "required": False,
            },
            "room_nights_sold": {
                "label": "Nuitées vendues",
                "unit": "nuits",
                "required": False,
            },
        },
    },

    # ─────────────────────────────────────────────────────────────────────────
    # MODULE 2 — FINANCIER USALI
    # ─────────────────────────────────────────────────────────────────────────
    "financial_usali": {
        "label_fr": "💰 Financier USALI",
        "description": (
            "Efficience par centre de profit selon la nomenclature USALI. "
            "Analyse la transformation des coûts départementaux en revenus "
            "par département (Rooms, F&B, Autres)."
        ),
        "references": "Min et al. (2009) ; USALI 11th Edition",
        "orientation": "input",
        "model": "BCC",
        "inputs": {
            "opex_total": {
                "label": "Charges opérationnelles totales (proxy USALI)",
                "unit": "€",
                "required": False,
            },
            "rooms_cost": {
                "label": "Coûts dept. Hébergement",
                "unit": "€",
                "required": False,
            },
            "fb_cost": {
                "label": "Coûts dept. F&B",
                "unit": "€",
                "required": False,
            },
            "other_dept_cost": {
                "label": "Coûts autres départements",
                "unit": "€",
                "required": False,
            },
            "undistributed_expenses": {
                "label": "Charges non distribuées",
                "unit": "€",
                "required": False,
            },
            "fixed_charges": {
                "label": "Charges fixes (loyer, amort.)",
                "unit": "€",
                "required": False,
            },
        },
        "outputs": {
            "rooms_revenue": {
                "label": "Revenus Hébergement",
                "unit": "€",
                "required": True,
            },
            "fb_revenue": {
                "label": "Revenus F&B",
                "unit": "€",
                "required": False,
            },
            "other_dept_revenue": {
                "label": "Revenus autres départements",
                "unit": "€",
                "required": False,
            },
            "ebitda": {
                "label": "EBITDA / NOP",
                "unit": "€",
                "required": False,
            },
        },
    },

    # ─────────────────────────────────────────────────────────────────────────
    # MODULE 3 — CAPITAL / ACTIFS
    # ─────────────────────────────────────────────────────────────────────────
    "capital_assets": {
        "label_fr": "🏗️ Capital & Actifs",
        "description": (
            "Efficience du capital immobilisé : mesure la capacité de l'actif "
            "hôtelier (valeur comptable, surface, CAPEX) à générer du chiffre "
            "d'affaires et de la rentabilité. Adapté aux portefeuilles "
            "institutionnels et aux comparaisons intra-chaîne."
        ),
        "references": "Barros & Alves (2004) ; Assaf (2010) ; Barros (2005)",
        "orientation": "output",
        "model": "CCR",
        "inputs": {
            "nb_rooms": {
                "label": "Capacité (chambres, ou lits DEA-H par défaut)",
                "unit": "unité",
                "required": False,
            },
            "book_value_assets": {
                "label": "Valeur comptable des actifs",
                "unit": "€",
                "required": False,
            },
            "surface_m2": {
                "label": "Surface exploitée (m²)",
                "unit": "m²",
                "required": False,
            },
            "capex_annuel": {
                "label": "CAPEX annuel / période",
                "unit": "€",
                "required": False,
            },
        },
        "outputs": {
            "total_revenue": {
                "label": "Chiffre d'affaires total",
                "unit": "€",
                "required": True,
            },
            "gop": {
                "label": "GOP",
                "unit": "€",
                "required": False,
            },
            "room_nights_sold": {
                "label": "Nuitées vendues",
                "unit": "nuits",
                "required": False,
            },
            "occupancy_rate": {
                "label": "Taux d'occupation (%)",
                "unit": "%",
                "required": False,
            },
        },
    },

    # ─────────────────────────────────────────────────────────────────────────
    # MODULE 4 — MAIN-D'ŒUVRE
    # ─────────────────────────────────────────────────────────────────────────
    "workforce": {
        "label_fr": "👥 Main-d'œuvre",
        "description": (
            "Efficience des ressources humaines : évalue la productivité "
            "du travail (masse salariale, ETP, heures) par rapport aux "
            "revenus générés et au taux d'occupation atteint."
        ),
        "references": "Hwang & Chang (2003) ; Barros (2005)",
        "orientation": "input",
        "model": "BCC",
        "inputs": {
            "fte_total": {
                "label": "Effectifs totaux (ETP)",
                "unit": "ETP",
                "required": True,
            },
            "payroll_total": {
                "label": "Masse salariale totale",
                "unit": "€",
                "required": False,
            },
            "hours_worked": {
                "label": "Heures travaillées",
                "unit": "heures",
                "required": False,
            },
            "training_cost": {
                "label": "Coût formation",
                "unit": "€",
                "required": False,
            },
        },
        "outputs": {
            "revenue_per_fte": {
                "label": "Revenu par ETP",
                "unit": "€/ETP",
                "required": True,
            },
            "occupancy_rate": {
                "label": "Taux d'occupation (%)",
                "unit": "%",
                "required": True,
            },
            "revpar": {
                "label": "RevPAR",
                "unit": "€",
                "required": False,
            },
            "gop": {
                "label": "GOP",
                "unit": "€",
                "required": False,
            },
        },
    },

    # ─────────────────────────────────────────────────────────────────────────
    # MODULE 5 — REVENUE MANAGEMENT / COMMERCIAL
    # ─────────────────────────────────────────────────────────────────────────
    "revenue_management": {
        "label_fr": "📈 Revenue Management",
        "description": (
            "Efficience commerciale et de distribution : mesure la capacité "
            "des investissements marketing et distribution à maximiser "
            "les revenus totaux (TRevPAR) et le positionnement tarifaire (ADR)."
        ),
        "references": "Sigala (2004) ; Anderson & Xie (2010)",
        "orientation": "output",
        "model": "BCC",
        "inputs": {
            "opex_total": {
                "label": "Charges opérationnelles (proxy coût distribution)",
                "unit": "€",
                "required": False,
            },
            "nb_rooms": {
                "label": "Capacité (chambres, ou lits DEA-H par défaut) — proxy exposition marché",
                "unit": "unité",
                "required": False,
            },
            "marketing_cost": {
                "label": "Dépenses marketing & distribution",
                "unit": "€",
                "required": False,
            },
            "ota_gds_cost": {
                "label": "Coûts canaux (OTA, GDS, commissions)",
                "unit": "€",
                "required": False,
            },
            "sales_fte": {
                "label": "ETP vente & réservations",
                "unit": "ETP",
                "required": False,
            },
            "promo_budget": {
                "label": "Budget promotionnel",
                "unit": "€",
                "required": False,
            },
        },
        "outputs": {
            "trevpar": {
                "label": "TRevPAR",
                "unit": "€/chambre",
                "required": True,
            },
            "adr": {
                "label": "ADR (Average Daily Rate)",
                "unit": "€",
                "required": True,
            },
            "occupancy_rate": {
                "label": "Taux d'occupation (%)",
                "unit": "%",
                "required": False,
            },
            "fb_revenue": {
                "label": "Revenus F&B & ancillaires",
                "unit": "€",
                "required": False,
            },
        },
    },

    # ─────────────────────────────────────────────────────────────────────────
    # MODULE 6 — ESG / ENVIRONNEMENTAL
    # ─────────────────────────────────────────────────────────────────────────
    "esg": {
        "label_fr": "🌿 ESG / Environnemental",
        "description": (
            "Efficience environnementale avec outputs désirables et non-désirables. "
            "Utilise un modèle SBM (Slack-Based Measure) pour traiter conjointement "
            "les outputs économiques (revenus, GOP) et les externalités négatives "
            "(CO₂, eau, déchets). Conforme au cadre CSRD/ESRS E1-E5."
        ),
        "references": "Zha et al. (2020) ; Camanho & Dyson (2005) ; modèle SBM Tone (2001)",
        "orientation": "output",
        "model": "BCC",  # remplacé par SBM si undesirable outputs actifs
        "inputs": {
            "energy_kwh": {
                "label": "Consommation énergie (kWh)",
                "unit": "kWh",
                "required": True,
            },
            "water_m3": {
                "label": "Consommation eau (m³)",
                "unit": "m³",
                "required": True,
            },
            "surface_m2": {
                "label": "Surface totale (m²)",
                "unit": "m²",
                "required": False,
            },
            "fte_total": {
                "label": "ETP (proxy empreinte sociale)",
                "unit": "ETP",
                "required": False,
            },
        },
        "outputs": {
            "total_revenue": {
                "label": "Chiffre d'affaires total [désirable]",
                "unit": "€",
                "required": False,
            },
            "revpar": {
                "label": "RevPAR [proxy intensité revenus]",
                "unit": "€",
                "required": False,
            },
            "gop": {
                "label": "GOP [désirable]",
                "unit": "€",
                "required": False,
            },
            "room_nights_sold": {
                "label": "Nuitées vendues [désirable]",
                "unit": "nuits",
                "required": False,
            },
        },
        "undesirable_outputs": {
            "co2_tonnes": {
                "label": "Émissions CO₂ (tonnes) [indésirable]",
                "unit": "t CO₂",
                "required": False,
            },
            "waste_tonnes": {
                "label": "Déchets produits (tonnes) [indésirable]",
                "unit": "tonnes",
                "required": False,
            },
            "water_waste_m3": {
                "label": "Eau non recyclée (m³) [indésirable]",
                "unit": "m³",
                "required": False,
            },
        },
    },

    # ─────────────────────────────────────────────────────────────────────────
    # MODULE 7 — QUALITÉ / SATISFACTION CLIENT
    # ─────────────────────────────────────────────────────────────────────────
    "quality": {
        "label_fr": "⭐ Qualité & Satisfaction",
        "description": (
            "Efficience de la qualité perçue : mesure la capacité des "
            "investissements en qualité (maintenance, formation, F&B) à "
            "générer de la satisfaction client et un positionnement tarifaire "
            "premium. Proxy de la valeur de marque hôtelière."
        ),
        "references": "Phua & Rasiah (2017) ; Assaf & Magnini (2012)",
        "orientation": "output",
        "model": "BCC",
        "inputs": {
            "opex_total": {
                "label": "Charges opérationnelles (proxy investissement qualité)",
                "unit": "€",
                "required": False,
            },
            "maintenance_cost": {
                "label": "Coûts maintenance & rénovation",
                "unit": "€",
                "required": False,
            },
            "fb_cost": {
                "label": "Coûts F&B",
                "unit": "€",
                "required": False,
            },
            "training_cost": {
                "label": "Investissement formation",
                "unit": "€",
                "required": False,
            },
            "guest_facing_fte": {
                "label": "ETP contact client",
                "unit": "ETP",
                "required": False,
            },
        },
        "outputs": {
            "satisfaction_score": {
                "label": "Score satisfaction (NPS / TripAdvisor / Google)",
                "unit": "score",
                "required": True,
            },
            "revpar": {
                "label": "RevPAR (proxy qualité perçue & pricing power)",
                "unit": "€",
                "required": True,
            },
            "adr": {
                "label": "ADR",
                "unit": "€",
                "required": False,
            },
            "repeat_guest_rate": {
                "label": "Taux de retour clientèle (%)",
                "unit": "%",
                "required": False,
            },
            "star_achievement": {
                "label": "Classement réalisé vs potentiel",
                "unit": "ratio",
                "required": False,
            },
        },
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def get_module_ids() -> list[str]:
    """Retourne la liste ordonnée des IDs de modules."""
    return list(MODULES.keys())


def get_module_labels() -> dict[str, str]:
    """Retourne {module_id: label_fr}."""
    return {k: v["label_fr"] for k, v in MODULES.items()}


def get_all_input_cols(module_id: str) -> list[str]:
    """Retourne toutes les colonnes inputs d'un module."""
    return list(MODULES[module_id]["inputs"].keys())


def get_all_output_cols(module_id: str) -> list[str]:
    """Retourne toutes les colonnes outputs désirables d'un module."""
    return list(MODULES[module_id]["outputs"].keys())


def get_undesirable_cols(module_id: str) -> list[str]:
    """Retourne les colonnes outputs non-désirables (ESG uniquement)."""
    return list(MODULES[module_id].get("undesirable_outputs", {}).keys())


def get_required_inputs(module_id: str) -> list[str]:
    return [
        col for col, meta in MODULES[module_id]["inputs"].items()
        if meta.get("required", False)
    ]


def get_required_outputs(module_id: str) -> list[str]:
    return [
        col for col, meta in MODULES[module_id]["outputs"].items()
        if meta.get("required", False)
    ]


def normalize_column_aliases(columns) -> list[str]:
    """Normalise les anciens noms de colonnes vers les clés canoniques.

    ``capex_annuel`` est le nom officiel. L'ancien ``capex`` reste accepté
    en entrée afin que les datasets/configurations historiques restent
    utilisables, sans jamais remplacer la colonne canonique si elle existe.
    Accepte une liste de noms ou tout itérable de noms (p. ex. ``df.columns``).
    """
    cols = list(columns)
    if "capex_annuel" not in cols and "capex" in cols:
        cols = ["capex_annuel" if col == "capex" else col for col in cols]
    return cols


def normalize_dataframe_columns(df):
    """Retourne une copie avec l'ancien alias ``capex`` mappé vers ``capex_annuel``.

    Le renommage n'est effectué que si la colonne canonique est absente,
    ce qui évite toute collision et tout accès à une colonne inexistante.
    """
    if "capex" in df.columns and "capex_annuel" not in df.columns:
        return df.rename(columns={"capex": "capex_annuel"})
    return df


def check_module_feasibility(module_id: str, available_cols: list[str],
                              proxy_cols: "set[str] | None" = None) -> dict:
    """
    Vérifie si un module est activable avec les colonnes disponibles.

    proxy_cols : colonnes calculées par reformulation arithmétique d'autres
    colonnes déjà utilisées ailleurs (ex. rooms_revenue = RevPAR × chambres × 365,
    adr = RevPAR / Occupation). Elles enrichissent l'affichage mais ne peuvent pas
    à elles seules satisfaire une colonne "required" : sinon un module comme
    Financier USALI ou Capital & Actifs se déclare calculable sur une simple
    reformulation du RevPAR, sans jamais voir de vraie donnée départementale
    (rooms_cost, fb_cost, total_revenue réel...). Un required satisfait
    uniquement par une colonne proxy reste listé dans missing_required.

    Retourne :
        {
          "feasible": bool,
          "missing_required": list[str],
          "available_inputs": list[str],
          "available_outputs": list[str],
          "coverage_pct": float,   # % colonnes disponibles / total attendues
        }
    """
    m = MODULES[module_id]
    available_cols = normalize_column_aliases(available_cols)
    proxy_cols = set(normalize_column_aliases(proxy_cols or set()))
    real_cols = [c for c in available_cols if c not in proxy_cols]
    all_expected = (
        list(m["inputs"].keys())
        + list(m["outputs"].keys())
        + list(m.get("undesirable_outputs", {}).keys())
    )
    available = [c for c in all_expected if c in available_cols]
    missing_req = [
        c for c in get_required_inputs(module_id) + get_required_outputs(module_id)
        if c not in real_cols
    ]
    # Un modèle DEA n'a de sens qu'avec au moins un vrai input ET un vrai
    # output — même si aucun des deux n'est marqué "required" (cas de modules
    # dont tous les inputs sont optionnels, ex. Capital & Actifs). Sans ce
    # plancher, "missing_required" pourrait être vide alors qu'il n'y a
    # littéralement aucune donnée réelle à mettre dans le modèle.
    _has_real_input  = any(c in real_cols for c in m["inputs"])
    _has_real_output = any(c in real_cols for c in m["outputs"])
    if not _has_real_input and not get_required_inputs(module_id):
        missing_req = missing_req + ["(aucun input réel disponible)"]
    if not _has_real_output and not get_required_outputs(module_id):
        missing_req = missing_req + ["(aucun output réel disponible)"]
    coverage = len(available) / len(all_expected) if all_expected else 0.0
    return {
        "feasible": (len(missing_req) == 0),
        "missing_required": missing_req,
        "available_inputs": [c for c in m["inputs"] if c in available_cols],
        "available_outputs": [c for c in m["outputs"] if c in available_cols],
        "available_undesirable": [
            c for c in m.get("undesirable_outputs", {}) if c in available_cols
        ],
        "coverage_pct": round(coverage * 100, 1),
    }


def get_col_label(module_id: str, col: str) -> str:
    """Retourne le label FR d'une colonne dans un module donné."""
    m = MODULES[module_id]
    for section in ("inputs", "outputs", "undesirable_outputs"):
        if col in m.get(section, {}):
            return m[section][col]["label"]
    return col
