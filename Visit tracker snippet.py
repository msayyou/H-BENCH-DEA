"""
=====================================================================
SNIPPET DE TRACKING DE VISITES — À AJOUTER EN HAUT DE app_v2.py
=====================================================================

Ce code fait 2 choses :
1. Envoie une notification push instantanée sur ton téléphone
   à chaque nouvelle visite (via ntfy.sh, gratuit, sans compte).
2. Enregistre un log local des visites (timestamp) visible
   via un petit tableau de bord caché dans la sidebar.

INSTALLATION (3 étapes) :
---------------------------------------------------------------------
1. Sur ton téléphone : installe l'app "ntfy" (iOS/Android, gratuite)
   ou ouvre https://ntfy.sh/TON_TOPIC_UNIQUE dans un navigateur.

2. Choisis un nom de "topic" UNIQUE et SECRET (personne ne doit le
   deviner, sinon n'importe qui peut spammer tes notifs).
   Exemple : "reiv-pl-usali-mehdi-9f3k2x"
   Remplace NTFY_TOPIC ci-dessous par ce nom.

3. Dans l'app ntfy (mobile), abonne-toi à ce même topic pour recevoir
   les notifications. Colle ce code en haut de app_v2.py (juste après
   les imports existants), puis redéploie sur Render.
---------------------------------------------------------------------
"""

import streamlit as st
import requests
from datetime import datetime
import json
import os

# ============ CONFIGURATION — À PERSONNALISER ============
NTFY_TOPIC = "KDS2124413"  # <-- CHANGE CECI (nom unique et secret)
LOG_FILE = "visit_log.json"
# ===========================================================

def _send_notification(session_id: str):
    """Envoie une notif push instantanée via ntfy.sh (silencieux si échec)."""
    try:
        requests.post(
            f"https://ntfy.sh/{NTFY_TOPIC}",
            data=f"Nouvelle visite sur P&L USALI — {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}".encode("utf-8"),
            headers={
                "Title": "REIV — P&L USALI",
                "Priority": "default",
                "Tags": "eyes"
            },
            timeout=3,
        )
    except Exception:
        pass  # ne jamais bloquer l'app si la notif échoue

def _log_visit():
    """Enregistre la visite dans un fichier JSON local (persiste tant que l'instance tourne)."""
    entry = {"timestamp": datetime.now().isoformat()}
    log = []
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "r") as f:
                log = json.load(f)
        except Exception:
            log = []
    log.append(entry)
    try:
        with open(LOG_FILE, "w") as f:
            json.dump(log, f)
    except Exception:
        pass

def track_visit():
    """À appeler UNE FOIS en haut du script principal, avant tout autre st.* """
    if "visit_tracked" not in st.session_state:
        st.session_state.visit_tracked = True
        _log_visit()
        _send_notification(st.session_state.get("session_id", "anonyme"))

def show_visit_dashboard():
    """
    Tableau de bord caché — affiche le nombre de visites et les dates.
    Accessible uniquement via l'URL avec ?admin=1 à la fin
    (ex: https://p-l-usali-1.onrender.com/?admin=1)
    Place cet appel n'importe où dans ton script (ex: dans la sidebar).
    """
    query_params = st.query_params
    if query_params.get("admin") == "1":
        with st.sidebar.expander("📊 Suivi des visites (admin)", expanded=True):
            if os.path.exists(LOG_FILE):
                with open(LOG_FILE, "r") as f:
                    log = json.load(f)
                st.metric("Total visites (depuis dernier redémarrage)", len(log))
                for entry in reversed(log[-15:]):
                    ts = datetime.fromisoformat(entry["timestamp"])
                    st.text(ts.strftime("%d/%m/%Y à %H:%M:%S"))
            else:
                st.text("Aucune visite enregistrée pour l'instant.")

# ============ À AJOUTER DANS TON SCRIPT PRINCIPAL ============
# Juste après tes imports, avant le reste du code de l'app :
#
#   track_visit()
#   show_visit_dashboard()
#
# ===============================================================
