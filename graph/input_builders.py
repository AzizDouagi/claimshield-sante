"""Construction minimale des entrées `ocr_input`/`fhir_input`/
`identity_coverage_input`/`coding_input` depuis le manifest produit par
`claim_intake_agent` — comble le gap documenté dans `CLAUDE.md` (rien en
production ne peuplait ces clés jusqu'ici, toujours consommées puis vidées
par leurs agents respectifs sans jamais être construites).

**Limite MVP assumée** (décision AZIZ) : `ClaimState.ocr_result` est un
champ singulier (pas une liste) et `document_ocr_agent` ne traite qu'**un
seul document par exécution** — alors qu'un vrai dossier contient plusieurs
documents (facture, ordonnance, demande). Ce module choisit donc **un seul**
document par dossier pour l'OCR (heuristique de nom de fichier, voir
`build_ocr_input`), jamais plusieurs. Le vrai fan-out multi-documents
(`langgraph.types.Send`, jamais utilisé dans ce projet) est hors périmètre —
chantier séparé, plus lourd, touchant `ClaimState`/la topologie du
graphe/tous les agents avals.

Fonctions pures : aucun accès disque, aucun appel LLM, aucune mutation de
l'objet `state` reçu — chaque fonction retourne un nouveau dict (ou `None`
si les données nécessaires sont absentes/invalides, auquel cas l'agent
concerné retombe sur son comportement déjà testé « entrée absente »).
"""
from __future__ import annotations

from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from schemas.domain import FileStatus
from schemas.results import ClaimIntakeResult, InspectedFile, SecurityGateResult
from state.claim_state import ClaimState

_M = TypeVar("_M", bound=BaseModel)


def _coerce(value: Any, model: type[_M]) -> _M | None:
    """Normalise une valeur de state en instance validée de ``model``.

    Accepte aussi bien une instance Pydantic déjà construite (exécution
    normale du graphe) qu'un dict sérialisé (state reconstruit depuis un
    checkpoint désérialisé, ou state de test construit à la main) — les deux
    formes sont légitimes. Ne lève jamais : retourne ``None`` si la valeur
    est absente, d'un type inattendu, ou invalide selon ``model``.
    """
    if value is None:
        return None
    if isinstance(value, model):
        return value
    if isinstance(value, dict):
        try:
            return model.model_validate(value)
        except ValidationError:
            return None
    return None


def _accepted_candidates(files: list[InspectedFile]) -> list[tuple[int, InspectedFile]]:
    """Fichiers du manifest exploitables : acceptés, avec chemin de stockage
    et hash calculés — jamais un fichier en quarantaine/bloqué/en erreur."""
    return [
        (idx, f)
        for idx, f in enumerate(files)
        if f.status is FileStatus.ACCEPTED
        and f.relative_storage_path is not None
        and f.sha256 is not None
    ]


def _strip_incoming_prefix(path: str) -> str:
    """`fhir_validator_agent`/`identity_coverage_agent` résolvent
    `fhir_bundle_path` relatif à `storage/incoming/` (ex.
    `agents/fhir_validator_agent/agent.py::_resolve_bundle_path` :
    `Path("storage") / "incoming" / relative_path`) — contrairement à
    `DocumentOcrInput.sanitized_path`, qui inclut lui le préfixe `incoming/`
    (résolu relatif à `storage/` seul). `InspectedFile.relative_storage_path`
    inclut toujours `incoming/` (ou `quarantine/`) — à retirer pour ces deux
    agents précisément, sous peine d'un chemin doublé (`storage/incoming/incoming/...`)
    introuvable."""
    prefix = "incoming/"
    return path[len(prefix):] if path.startswith(prefix) else path


def _is_fhir_bundle_file(f: InspectedFile) -> bool:
    """Critère double — mime JSON ET "fhir" dans le nom — jamais un seul des
    deux : plusieurs fichiers `application/json` peuvent coexister dans un
    même dossier (métadonnées diverses), seul le nom distingue le bundle."""
    return f.detected_mime_type.lower() == "application/json" and "fhir" in f.original_name.lower()


def _find_fhir_bundle_candidate(
    files: list[InspectedFile],
) -> tuple[InspectedFile | None, bool]:
    """Retourne ``(candidat, ambigu)``. Zéro candidat -> ``(None, False)``
    (bundle non fourni, légitime). Plus d'un candidat -> ``(None, True)``
    (ambiguïté — jamais un choix arbitraire silencieux)."""
    candidates = [f for _, f in _accepted_candidates(files) if _is_fhir_bundle_file(f)]
    if len(candidates) == 1:
        return candidates[0], False
    if len(candidates) == 0:
        return None, False
    return None, True


def build_ocr_input(state: ClaimState) -> dict | None:
    """Construit ``ocr_input`` pour un seul document (voir limite MVP en
    tête de module) : préférence au fichier dont le nom contient "facture"
    (insensible à la casse), sinon premier candidat accepté par position
    dans le manifest. Exclut toujours le fichier bundle FHIR (réservé à
    `fhir_validator_agent`). Retourne ``None`` si aucune donnée exploitable
    (dossier/sécurité absents, ou aucun candidat) — l'agent retombe alors
    sur son comportement déjà testé « ocr_input absent »."""
    case_id = state.get("case_id")
    intake = _coerce(state.get("intake_result"), ClaimIntakeResult)
    security = _coerce(state.get("security_result"), SecurityGateResult)
    if not case_id or intake is None or security is None:
        return None

    non_bundle = [
        (idx, f)
        for idx, f in _accepted_candidates(intake.manifest.files)
        if not _is_fhir_bundle_file(f)
    ]
    if not non_bundle:
        return None

    chosen = next(
        ((idx, f) for idx, f in non_bundle if "facture" in f.original_name.lower()),
        non_bundle[0],
    )
    idx, f = chosen

    return {
        "claim_id": str(case_id),
        "document_id": f"{case_id}-doc-{idx}",
        "filename": f.original_name,
        "mime_type": f.detected_mime_type,
        "sha256": f.sha256,
        "sanitized_path": f.relative_storage_path,
        "security_decision": security.decision.value,
        "file_index": idx,
    }


def build_fhir_input(state: ClaimState) -> dict | None:
    """Construit ``fhir_input`` — bundle détecté par `_find_fhir_bundle_candidate`.
    Zéro candidat -> ``bundle_expected=False``/``fhir_bundle_path=None``
    (chemin ``NOT_PROVIDED`` déjà géré par `fhir_validator_agent`, jamais un
    FAIL). Ambiguïté -> ``None`` (repli sur le comportement « entrée
    absente » déjà testé, jamais un choix arbitraire)."""
    case_id = state.get("case_id")
    intake = _coerce(state.get("intake_result"), ClaimIntakeResult)
    if not case_id or intake is None:
        return None

    candidate, ambiguous = _find_fhir_bundle_candidate(intake.manifest.files)
    if ambiguous:
        return None

    return {
        "case_id": str(case_id),
        "fhir_bundle_path": (
            _strip_incoming_prefix(candidate.relative_storage_path) if candidate else None
        ),
        "bundle_expected": candidate is not None,
    }


def build_identity_coverage_input(state: ClaimState) -> dict | None:
    """Construit ``identity_coverage_input`` — ``case_id`` + ``fhir_bundle_path``
    optionnel (même détection que `build_fhir_input`). Jamais ``None`` dès
    que ``case_id`` existe (contrairement à OCR/FHIR) : les données
    patient/couverture sont lues séparément par `identity_coverage_agent`
    depuis ``state["ocr_result"]``, jamais dupliquées ici."""
    case_id = state.get("case_id")
    if not case_id:
        return None

    fhir_bundle_path: str | None = None
    intake = _coerce(state.get("intake_result"), ClaimIntakeResult)
    if intake is not None:
        candidate, ambiguous = _find_fhir_bundle_candidate(intake.manifest.files)
        if candidate is not None and not ambiguous:
            fhir_bundle_path = _strip_incoming_prefix(candidate.relative_storage_path)

    return {"case_id": str(case_id), "fhir_bundle_path": fhir_bundle_path}


def build_coding_input(state: ClaimState) -> dict | None:
    """Construit ``coding_input`` — ``procedures``/``medications`` toujours
    des listes vides. Aucune extraction fiable acte/médicament n'existe
    encore (``MedicalItem`` n'a pas de discriminant type, alors que
    `medical_coding_agent` route vers deux tables de référence distinctes) —
    inventer une répartition heuristique produirait un codage silencieusement
    erroné. Limitation assumée, pas une invention de logique métier."""
    case_id = state.get("case_id")
    if not case_id:
        return None
    return {"case_id": str(case_id), "procedures": [], "medications": []}
