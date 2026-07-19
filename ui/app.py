"""Point d'entrée Chainlit — UI conversationnelle minimale pour ClaimShield
Santé (pipeline V2 uniquement). Trois fonctionnalités : soumission d'un
dossier, consultation de statut, correction post-décision (override), plus
le chat conversationnel. Client HTTP pur de l'API (``ui/api_client_v2.py``)
— n'importe jamais ``graph.*``/``agents.*``, aucun accès direct au pipeline.

Lancer (API déjà démarrée séparément, voir ``api/main.py``) ::

    chainlit run ui/app.py -w
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any

# `DATABASE_URL` (chargée par Chainlit depuis notre .env — non lié à cette
# UI) est aussi le nom conventionnel que Chainlit utilise lui-même pour
# activer sa propre couche de persistance de conversations (postgres via
# asyncpg, voir chainlit/data/__init__.py::get_data_layer). Notre valeur
# (`sqlite+aiosqlite://...`, destinée à `database/audit_models.py`, sans
# rapport avec l'UI) fait donc planter Chainlit dès la première interaction
# (`ClientConfigurationError: invalid DSN`). Retrait avant tout import
# `chainlit` — `get_data_layer()` est appelé paresseusement (première
# interaction), jamais à l'import, donc ce retrait précoce suffit.
os.environ.pop("DATABASE_URL", None)

import chainlit as cl  # noqa: E402

# Chainlit charge ce fichier directement (importlib, hors package) — sans
# cet ajout, `from ui import ...` échoue (`ModuleNotFoundError: No module
# named 'ui'`), même patron que `scripts/run_agent_manual.py`.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.settings import get_settings  # noqa: E402
from ui import api_client_v2  # noqa: E402
from ui.uploads import stage_uploaded_files  # noqa: E402

# `.chainlit/config.toml` auto-généré au premier lancement fixe
# `[features.spontaneous_file_upload].accept = ["*/*"]` par défaut — un
# type MIME invalide (le propre commentaire du template Chainlit prévient :
# "Using '*/*' is not recommended as it may cause browser warnings").
# Cette fonctionnalité n'est pas lue par notre flux (entièrement guidé via
# `cl.AskFileMessage`), mais elle ne peut PAS être désactivée : le serveur
# Chainlit (`chainlit/server.py::validate_file_upload`) rejette aussi les
# uploads légitimes de `cl.AskFileMessage` quand `ask_parent_id` n'est pas
# reconnu côté serveur (observé en pratique dans cette version — l'upload
# retombe alors sur ce chemin "spontané" et un `enabled=False` le bloque
# avec un 400 "File upload is not enabled", même depuis le bon widget).
# On la laisse donc active, avec notre vraie liste de types MIME plutôt
# que `*/*` — supprime l'avertissement navigateur sans jamais bloquer un
# dépôt légitime.
if cl.config.config.features.spontaneous_file_upload is not None:
    cl.config.config.features.spontaneous_file_upload.enabled = True
    cl.config.config.features.spontaneous_file_upload.accept = get_settings().allowed_mime_types

_CASE_ID_PATTERN = re.compile(r"^CLM-\d{4,}$")


def _allowed_mime_types() -> list[str]:
    """`cl.AskFileMessage(accept=...)` attend des types MIME (transmis tels
    quels au composant de dépôt du frontend), jamais des extensions de
    fichier — réutilise la même liste que `Settings.claimshield_allowed_mime_types`
    (déjà utilisée par `tools/file_inspection.py`), pas une liste dupliquée."""
    return get_settings().allowed_mime_types


async def _ask_text(prompt: str, *, timeout: int = 180) -> str | None:
    res = await cl.AskUserMessage(content=prompt, timeout=timeout).send()
    if res is None:
        return None
    output = str(res.get("output", "")).strip()
    return output or None


async def _ask_case_id(prompt: str) -> str | None:
    for _ in range(3):
        case_id = await _ask_text(prompt)
        if case_id is None:
            return None
        if _CASE_ID_PATTERN.match(case_id):
            return case_id
        await cl.Message(content=f"Identifiant invalide ({case_id!r}) — format attendu CLM-XXXX.").send()
    return None


@cl.on_chat_start
async def on_chat_start() -> None:
    up = await api_client_v2.healthz()
    status = "en ligne" if up else "**injoignable** — vérifiez que l'API tourne"
    await cl.Message(
        content=f"ClaimShield Santé — API {status}.",
        actions=[
            cl.Action(name="menu_submit_v2", payload={}, label="Soumettre un dossier"),
            cl.Action(name="menu_status_v2", payload={}, label="Consulter un dossier"),
            cl.Action(name="menu_override_v2", payload={}, label="Corriger une décision"),
        ],
    ).send()


# ── Pipeline V2 (autonome, sans revue humaine bloquante) ─────────────────────
# Client HTTP dédié : `ui/api_client_v2.py`.


def _format_status_v2(body: dict[str, Any]) -> str:
    lines = [
        f"**Dossier {body['case_id']} (V2)**",
        f"- Étape courante : `{body.get('current_step')}`",
        f"- Étapes complétées : {', '.join(body.get('completed_steps') or []) or '—'}",
        f"- Décision finale : {body.get('final_decision') or '—'}",
    ]
    if body.get("decision_summary"):
        lines.append("- Justification : " + "; ".join(body["decision_summary"]))
    if body.get("bounded_by"):
        lines.append("- Garde-fous appliqués : " + "; ".join(body["bounded_by"]))
    if body.get("errors"):
        lines.append(f"- Erreurs : {'; '.join(body['errors'])}")
    if body.get("alerts"):
        lines.append(f"- Alertes : {'; '.join(body['alerts'])}")
    return "\n".join(lines)


async def _submit_flow_v2() -> None:
    case_id = await _ask_case_id("Identifiant du dossier à soumettre (V2, ex. CLM-0001) :")
    if case_id is None:
        await cl.Message(content="Soumission annulée.").send()
        return
    cl.user_session.set("last_case_id", case_id)

    files = await cl.AskFileMessage(
        content="Déposez les documents du dossier (PDF/image/JSON) :",
        accept=_allowed_mime_types(),
        max_files=10,
        max_size_mb=20,
        timeout=300,
    ).send()
    if not files:
        await cl.Message(content="Aucun fichier reçu — soumission annulée.").send()
        return

    source_path = stage_uploaded_files(case_id, files)
    response = await api_client_v2.submit_claim_v2(case_id, source_path)
    if response.status_code >= 400:
        await cl.Message(
            content=f"Échec de la soumission V2 ({response.status_code}) : {response.text}"
        ).send()
        return

    await cl.Message(content=_format_status_v2(response.json())).send()


async def _status_flow_v2() -> None:
    case_id = await _ask_case_id("Identifiant du dossier à consulter (V2, ex. CLM-0001) :")
    if case_id is None:
        await cl.Message(content="Consultation annulée.").send()
        return
    cl.user_session.set("last_case_id", case_id)

    response = await api_client_v2.get_status_v2(case_id)
    if response.status_code == 404:
        await cl.Message(content=f"Dossier {case_id!r} introuvable (V2).").send()
        return
    if response.status_code >= 400:
        await cl.Message(content=f"Erreur ({response.status_code}) : {response.text}").send()
        return

    await cl.Message(content=_format_status_v2(response.json())).send()


async def _override_flow_v2() -> None:
    case_id = await _ask_case_id("Identifiant du dossier à corriger (V2, ex. CLM-0001) :")
    if case_id is None:
        await cl.Message(content="Correction annulée.").send()
        return
    cl.user_session.set("last_case_id", case_id)

    actor = cl.user_session.get("actor")
    if not actor:
        actor = await _ask_text("Votre identifiant (acteur, pour l'audit) :")
        if not actor:
            await cl.Message(content="Correction annulée (acteur requis).").send()
            return
        cl.user_session.set("actor", actor)

    action = await _ask_text(
        "Action (CONFIRM / OVERRIDE_APPROVE / OVERRIDE_REJECT / REOPEN) :"
    )
    if not action:
        await cl.Message(content="Correction annulée.").send()
        return

    justification = await _ask_text("Justification (obligatoire) :")
    if not justification:
        await cl.Message(content="Correction annulée (justification requise).").send()
        return

    response = await api_client_v2.submit_override_v2(
        case_id, actor=actor, action=action.strip().upper(), justification=justification
    )
    if response.status_code >= 400:
        await cl.Message(content=f"Correction refusée ({response.status_code}) : {response.text}").send()
        return

    body = response.json()
    await cl.Message(content=f"Correction enregistrée pour {case_id} : {body['action']}.").send()


@cl.action_callback("menu_submit_v2")
async def on_menu_submit_v2(action: cl.Action) -> None:
    await _submit_flow_v2()


@cl.action_callback("menu_status_v2")
async def on_menu_status_v2(action: cl.Action) -> None:
    await _status_flow_v2()


@cl.action_callback("menu_override_v2")
async def on_menu_override_v2(action: cl.Action) -> None:
    await _override_flow_v2()


# ── Chat conversationnel (Chat Reasoning Agent, `chat/` + `/v2/chat`) ────────
#
# Tout message libre tapé hors d'une invite structurée (`cl.AskUserMessage`/
# `cl.AskFileMessage`, qui interceptent déjà la réponse suivante avant
# qu'elle n'atteigne ce gestionnaire) est transmis tel quel à
# `POST /v2/chat` — permet de demander « pourquoi ce dossier est refusé ? »,
# « simule sans l'ordonnance », etc. `last_case_id` (posé par les flux
# soumission/consultation/décision ci-dessus) sert de contexte par défaut :
# il prime sur un identifiant que le NLU croirait détecter dans le texte
# (voir `chat.agent.handle_message`), mais un identifiant explicite dans le
# message reste toujours possible si aucun dossier n'a encore été consulté.
@cl.on_message
async def on_message(message: cl.Message) -> None:
    text = (message.content or "").strip()
    if not text:
        return

    case_id = cl.user_session.get("last_case_id")
    # Mémoire conversationnelle (Phase 8) — entièrement opt-in côté API :
    # `thread_id` (session Chainlit, stable pour toute la conversation) et
    # `actor` (déjà demandé/caché en session par les flux de décision
    # humaine/override ci-dessus, jamais redemandé ici) activent la mémoire
    # uniquement si `actor` est déjà connu — sinon comportement inchangé.
    thread_id = cl.context.session.id
    actor = cl.user_session.get("actor")
    response = await api_client_v2.send_chat_message_v2(
        text, case_id=case_id, thread_id=thread_id, actor=actor
    )
    if response.status_code >= 400:
        await cl.Message(content=f"Erreur chat ({response.status_code}) : {response.text}").send()
        return

    await cl.Message(content=response.json()["reply"]).send()
