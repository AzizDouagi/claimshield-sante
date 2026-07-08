"""Rédaction déterministe des payloads d'audit — tools/audit_redaction.py.

Fonction pure, sans E/S ni appel LLM, aucun état mutable. Appliquée en
défense en profondeur avant qu'un événement structuré soit soumis au LLM
normalizer de l'Audit Agent (``agents.audit_agent.agent._invoke_llm_audit``)
— point de passage unique pour tous les producteurs d'événements
(``security_gate_agent``, ``orchestrator/executor.py``,
``human_review/service.py``, ``audit_agent`` lui-même). Ne dépend jamais de
la bonne volonté du LLM : le contenu dangereux est retiré avant même
d'atteindre le prompt, pas seulement décrit comme « à ne pas répéter ».

Supprime (jamais tronqué en place, jamais transmis même partiellement) :
  - prompts complets (clés connues : ``system_prompt``, ``prompt``,
    ``messages``...) ;
  - texte OCR complet (``full_text``, ``raw_text``, ``ocr_text``,
    ``extracted_text``...) ;
  - secrets/clés/tokens (motif ``api_key``/``secret``/``password``/
    ``token``/``bearer``) ;
  - tout texte libre long (> ``MAX_SHORT_TEXT_LENGTH``), qu'il s'agisse de
    texte médical, d'un prompt non nommé explicitement ou de tout autre
    contenu volumineux non structuré — la longueur seule suffit à retirer
    un champ, indépendamment de son nom.

Conserve : identifiants (``case_id``, ``event_id``, ``entry_id``,
``agent_name``, ``actor``...), preuves déjà minimisées (``evidence_ids``,
codes, listes courtes), champs courts (<= ``MAX_SHORT_TEXT_LENGTH``
caractères, sans motif de secret) et empreintes SHA-256 hexadécimales
(toujours conservées, jamais réversibles).

``redact_audit_payload`` calcule et ajoute lui-même ``redaction_status``
(valeur de ``schemas.audit.RedactionStatus`` — jamais une nouvelle
énumération dupliquée) : ``not_redacted`` si rien n'a été retiré,
``partially_redacted`` si au moins un champ a été retiré alors que d'autres
champs exploitables subsistent, ``fully_redacted`` si tout le contenu autre
que les identifiants a dû être retiré. Ce statut n'est jamais laissé à la
seule appréciation du LLM normalizer en aval.
"""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from schemas.audit import RedactionStatus

MAX_SHORT_TEXT_LENGTH = 300
"""Longueur maximale d'un champ texte conservé tel quel. Un champ texte plus
long est toujours retiré, quel que soit son nom — c'est ce seuil qui couvre
le texte médical long et les prompts/OCR non nommés explicitement."""

MAX_LIST_ITEMS = 20
"""Nombre maximal d'éléments conservés dans une liste — même convention que
``agents.audit_agent.agent._compact_value``."""

_HEX64_RE = re.compile(r"^[0-9a-fA-F]{64}$")

_SECRET_HINT_RE = re.compile(
    r"(?:api[_-]?key|secret\s*[:=]|password\s*[:=]|token\s*[:=]|bearer\s+[a-z0-9._~+/=-]+)",
    re.IGNORECASE,
)

ALWAYS_DROPPED_KEYS = frozenset(
    {
        "system_prompt",
        "prompt",
        "prompt_text",
        "system_message",
        "messages",
        "full_text",
        "raw_text",
        "ocr_text",
        "extracted_text",
        "document_text",
        "raw_content",
        "html_content",
        "clinical_context",
        "medical_notes",
    }
)
"""Noms de champ toujours retirés, indépendamment de leur longueur — un
prompt ou un OCR complet reste dangereux même s'il tenait sous le seuil de
longueur générique dans un cas particulier."""

_IDENTIFIER_FIELDS = frozenset(
    {
        "case_id",
        "event_id",
        "entry_id",
        "actor",
        "agent_name",
        "affected_agent",
        "timestamp",
        "evaluated_at",
        "observed_at",
        "redaction_status",
    }
)
"""Champs considérés comme de purs identifiants/métadonnées structurelles —
exclus du calcul de ``fully_redacted``. Volontairement restreint aux seuls
champs qui ne constituent jamais, à eux seuls, un contenu d'audit
exploitable (identité du dossier/de l'événement/de l'acteur, horodatage) —
``outcome``/``action``/``event_type``/``policy_applied``/``reason``... sont
au contraire le contenu même de l'audit et comptent comme du contenu
exploitable s'ils survivent à la rédaction."""


class _Drop:
    """Sentinel distinct de ``None`` — signifie « retirer cette clé »,
    jamais confondu avec une valeur ``None`` légitime déjà présente dans le
    payload d'origine (ex. ``target_node: None``), qui doit rester visible."""

    def __repr__(self) -> str:  # pragma: no cover - confort de debug uniquement
        return "<DROP>"


_DROP = _Drop()


def _is_hash(value: str) -> bool:
    return bool(_HEX64_RE.match(value))


def _redact_string(value: str) -> tuple[Any, bool]:
    """Retourne ``(valeur conservée ou _DROP, un retrait a-t-il eu lieu ?)``."""
    if _is_hash(value):
        return value, False
    if _SECRET_HINT_RE.search(value):
        return _DROP, True
    if len(value) > MAX_SHORT_TEXT_LENGTH:
        return _DROP, True
    return value, False


def _redact_value(key: str, value: Any) -> tuple[Any, bool]:
    """Retourne ``(valeur à conserver ou _DROP, un retrait a-t-il eu lieu
    quelque part dans cette valeur ?)``.

    Le second élément est purement informatif (alimente le calcul global de
    ``redaction_status``) : un conteneur imbriqué peut très bien survivre
    (valeur conservée non ``_DROP``) tout en signalant qu'un retrait a eu
    lieu à l'intérieur.
    """
    # Le nom du champ prime sur son type et sur son contenu : un prompt ou
    # un OCR complet reste dangereux quelle que soit sa forme (texte, liste
    # de messages, dict imbriqué) — jamais transmis, même partiellement.
    if key in ALWAYS_DROPPED_KEYS:
        return _DROP, True
    if value is None or isinstance(value, (bool, int, float)):
        return value, False
    if isinstance(value, str):
        return _redact_string(value)
    if isinstance(value, Mapping):
        nested, dropped_any = _redact_mapping(value)
        if not nested and value:
            # Tout le contenu du dict imbriqué a été retiré : la clé
            # elle-même disparaît plutôt que de conserver un dict vide.
            return _DROP, True
        return nested, dropped_any
    if isinstance(value, (list, tuple)):
        return _redact_sequence(key, value)
    # Type non structuré (objet Python quelconque) : jamais transmis tel quel.
    return _DROP, True


def _redact_sequence(key: str, values: Sequence[Any]) -> tuple[Any, bool]:
    dropped_any = len(values) > MAX_LIST_ITEMS
    kept: list[Any] = []
    for item in list(values)[:MAX_LIST_ITEMS]:
        redacted_item, item_dropped = _redact_value(key, item)
        if item_dropped:
            dropped_any = True
        if redacted_item is _DROP:
            continue
        kept.append(redacted_item)
    if not kept and values:
        # Tout le contenu de la liste a été retiré : la clé elle-même est
        # retirée plutôt que de conserver une liste vide sans valeur.
        return _DROP, True
    return kept, dropped_any


def _redact_mapping(payload: Mapping[str, Any]) -> tuple[dict[str, Any], bool]:
    dropped_any = False
    kept: dict[str, Any] = {}
    for key, value in payload.items():
        key_str = str(key)
        redacted_value, value_dropped = _redact_value(key_str, value)
        if value_dropped:
            dropped_any = True
        if redacted_value is _DROP:
            continue
        kept[key_str] = redacted_value
    return kept, dropped_any


def redact_audit_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Retourne une copie minimisée et sûre de ``payload``.

    Toujours un nouveau dict (jamais une mutation de ``payload``). Ajoute
    ``redaction_status`` (chaîne, valeur de ``schemas.audit.RedactionStatus``)
    calculé à partir de ce qui a réellement été retiré.
    """
    redacted, dropped_any = _redact_mapping(payload)

    meaningful_kept_keys = [key for key in redacted if key not in _IDENTIFIER_FIELDS]

    if not dropped_any:
        status = RedactionStatus.NOT_REDACTED
    elif meaningful_kept_keys:
        status = RedactionStatus.PARTIALLY_REDACTED
    else:
        status = RedactionStatus.FULLY_REDACTED

    redacted["redaction_status"] = status.value
    return redacted


__all__ = [
    "ALWAYS_DROPPED_KEYS",
    "MAX_LIST_ITEMS",
    "MAX_SHORT_TEXT_LENGTH",
    "redact_audit_payload",
]
