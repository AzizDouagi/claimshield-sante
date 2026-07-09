"""Point d'entrée Chainlit — UI conversationnelle minimale pour ClaimShield
Santé. Trois fonctionnalités : soumission d'un dossier, consultation de
statut, décision humaine (HITL). Client HTTP pur de l'API (``ui/api_client.py``)
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

# `.chainlit/config.toml` auto-généré au premier lancement fixe
# `[features.spontaneous_file_upload].accept = ["*/*"]` par défaut — un
# type MIME invalide (le propre commentaire du template Chainlit prévient :
# "Using '*/*' is not recommended as it may cause browser warnings") qui
# spamme la console à chaque interaction. Cette fonctionnalité (dépôt de
# fichier libre dans le champ de message général) n'est de toute façon pas
# utilisée par notre flux, entièrement guidé via `cl.AskFileMessage` — on la
# désactive plutôt que de la reconfigurer, pour ne pas laisser un second
# canal de dépôt de fichier sans effet réel (jamais lu par `ui/app.py`).
if cl.config.config.features.spontaneous_file_upload is not None:
    cl.config.config.features.spontaneous_file_upload.enabled = False

# Chainlit charge ce fichier directement (importlib, hors package) — sans
# cet ajout, `from ui import ...` échoue (`ModuleNotFoundError: No module
# named 'ui'`), même patron que `scripts/run_agent_manual.py`.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.settings import get_settings  # noqa: E402
from ui import api_client  # noqa: E402
from ui.forms import form_from_pending_review, render_for_chainlit_actions  # noqa: E402
from ui.uploads import stage_uploaded_files  # noqa: E402

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


def _format_status(body: dict[str, Any]) -> str:
    lines = [
        f"**Dossier {body['case_id']}**",
        f"- Étape courante : `{body.get('current_step')}`",
        f"- Étapes complétées : {', '.join(body.get('completed_steps') or []) or '—'}",
        f"- Recommandation finale : {body.get('final_recommendation') or '—'}",
        f"- Interrompu (revue humaine) : {'oui' if body.get('interrupted') else 'non'}",
    ]
    if body.get("errors"):
        lines.append(f"- Erreurs : {'; '.join(body['errors'])}")
    if body.get("alerts"):
        lines.append(f"- Alertes : {'; '.join(body['alerts'])}")
    return "\n".join(lines)


async def _show_status_and_maybe_decision(body: dict[str, Any]) -> None:
    await cl.Message(content=_format_status(body)).send()
    if body.get("interrupted") and body.get("pending_review"):
        await _show_decision_form(body["case_id"], body["pending_review"])


async def _show_decision_form(case_id: str, pending_review: dict[str, Any]) -> None:
    form = form_from_pending_review(pending_review)
    lines = [f"**Revue humaine requise — {case_id}**"]
    if form.summary:
        lines.append("Résumé : " + "; ".join(form.summary))
    if form.evidence:
        preuves = ", ".join(f"{k}={v}" for k, v in form.evidence.items())
        lines.append("Preuves : " + preuves)

    actions = [
        cl.Action(name=item["name"], payload={"case_id": case_id}, label=item["label"])
        for item in render_for_chainlit_actions(form)
    ]
    await cl.Message(content="\n".join(lines), actions=actions).send()


async def _submit_flow() -> None:
    case_id = await _ask_case_id("Identifiant du dossier à soumettre (ex. CLM-0001) :")
    if case_id is None:
        await cl.Message(content="Soumission annulée.").send()
        return

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
    response = await api_client.submit_claim(case_id, source_path)
    if response.status_code >= 400:
        await cl.Message(content=f"Échec de la soumission ({response.status_code}) : {response.text}").send()
        return

    await _show_status_and_maybe_decision(response.json())


async def _status_flow() -> None:
    case_id = await _ask_case_id("Identifiant du dossier à consulter (ex. CLM-0001) :")
    if case_id is None:
        await cl.Message(content="Consultation annulée.").send()
        return

    response = await api_client.get_status(case_id)
    if response.status_code == 404:
        await cl.Message(content=f"Dossier {case_id!r} introuvable.").send()
        return
    if response.status_code >= 400:
        await cl.Message(content=f"Erreur ({response.status_code}) : {response.text}").send()
        return

    await _show_status_and_maybe_decision(response.json())


async def _decision_flow(case_id: str, action: str) -> None:
    actor = cl.user_session.get("actor")
    if not actor:
        actor = await _ask_text("Votre identifiant (acteur, pour l'audit) :")
        if not actor:
            await cl.Message(content="Décision annulée (acteur requis).").send()
            return
        cl.user_session.set("actor", actor)

    justification = await _ask_text("Justification (obligatoire) :")
    if not justification:
        await cl.Message(content="Décision annulée (justification requise).").send()
        return

    target_node = None
    if action == "RETRY":
        target_node = await _ask_text("Étape à relancer (target_node, obligatoire pour RETRY) :")
        if not target_node:
            await cl.Message(content="Décision annulée (target_node requis pour RETRY).").send()
            return

    response = await api_client.submit_human_decision(
        case_id, actor=actor, action=action, justification=justification, target_node=target_node
    )
    if response.status_code >= 400:
        await cl.Message(content=f"Décision refusée ({response.status_code}) : {response.text}").send()
        return

    await _show_status_and_maybe_decision(response.json())


@cl.on_chat_start
async def on_chat_start() -> None:
    up = await api_client.healthz()
    status = "en ligne" if up else "**injoignable** — vérifiez que l'API tourne"
    await cl.Message(
        content=f"ClaimShield Santé — API {status}.",
        actions=[
            cl.Action(name="menu_submit", payload={}, label="Soumettre un dossier"),
            cl.Action(name="menu_status", payload={}, label="Consulter un dossier"),
        ],
    ).send()


@cl.action_callback("menu_submit")
async def on_menu_submit(action: cl.Action) -> None:
    await _submit_flow()


@cl.action_callback("menu_status")
async def on_menu_status(action: cl.Action) -> None:
    await _status_flow()


@cl.action_callback("APPROVE")
async def on_approve(action: cl.Action) -> None:
    await _decision_flow(action.payload["case_id"], "APPROVE")


@cl.action_callback("MODIFY")
async def on_modify(action: cl.Action) -> None:
    await _decision_flow(action.payload["case_id"], "MODIFY")


@cl.action_callback("REJECT")
async def on_reject(action: cl.Action) -> None:
    await _decision_flow(action.payload["case_id"], "REJECT")


@cl.action_callback("RETRY")
async def on_retry(action: cl.Action) -> None:
    await _decision_flow(action.payload["case_id"], "RETRY")
