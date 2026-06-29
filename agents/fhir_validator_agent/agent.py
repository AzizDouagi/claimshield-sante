"""FHIR Validator Agent — validation structurelle du bundle FHIR R4.

Agent purement déterministe — aucun appel LLM.

Pipeline (5 étapes) :
  1. Lecture de fhir_input depuis le state (dict → FhirValidatorInput)
  2. Appel validate_fhir_bundle(fhir_bundle_path, bundle_expected, rules)
  3. Construction de FhirValidatorResult
  4. Appel validate_state_update() — garde-fou anti-secrets/absolus
  5. Retour des mises à jour du state (fhir_input vidé à None)

Interdictions strictes :
  - Aucune décision médicale ou de remboursement.
  - Aucun accès au contenu brut des ressources FHIR.
  - Aucun secret, token ou chemin absolu dans le résultat.
  - Aucune modification de l'objet source.

Résolution du chemin :
  - Si fhir_bundle_path est None → pas de bundle à valider.
  - Si fhir_bundle_path est relatif → résolu depuis la racine du projet
    (Path(fhir_bundle_path) en mode tests sur fixtures, ou
    Path("storage/incoming") / fhir_bundle_path en production).
    La logique de résolution essaie d'abord le chemin direct (fixtures),
    puis le chemin sous storage/incoming/.
"""
from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError

from agents.fhir_validator_agent.schemas import FhirValidatorInput
from schemas.domain import VerificationStatus
from schemas.results import FhirValidatorResult
from state.claim_state import ClaimState, validate_state_update
from tools.fhir_validation import validate_fhir_bundle
from tools.fhir_validation import extract_resource_types, load_fhir_bundle
from tools.rule_loader import get_rule_version, load_rules


# ── Résolution du chemin du bundle ───────────────────────────────────────────


def _resolve_bundle_path(relative_path: str) -> str:
    """Résout un chemin relatif de bundle en chemin utilisable.

    Stratégie :
      1. Chemin direct (pour les fixtures de test)
      2. Chemin sous storage/incoming/ (pour la production)

    Retourne le chemin sous forme de chaîne (str) pour conserver
    la portabilité et permettre validate_fhir_bundle de gérer les erreurs.
    """
    direct = Path(relative_path)
    if direct.exists():
        return str(direct)

    under_storage = Path("storage") / "incoming" / relative_path
    if under_storage.exists():
        return str(under_storage)

    # Retourner le chemin direct même s'il n'existe pas —
    # load_fhir_bundle renverra une erreur explicite.
    return str(direct)


# ── Fonction principale ───────────────────────────────────────────────────────


def run(
    case_id: str,
    fhir_bundle_path: str | None = None,
    *,
    bundle_expected: bool = True,
) -> FhirValidatorResult:
    """Valide le bundle FHIR R4 associé au dossier.

    Args:
        case_id: Identifiant du dossier de remboursement.
        fhir_bundle_path: Chemin relatif vers le bundle FHIR.
                          None si aucun bundle n'a été fourni.
        bundle_expected: True si un bundle est attendu pour ce dossier.

    Returns:
        FhirValidatorResult avec statut PASS / NEEDS_REVIEW / FAIL.
    """
    # Chargement des règles de validation
    rules = load_rules("fhir_rules.yaml")
    rule_version = get_rule_version("fhir_rules.yaml")

    # Résolution du chemin si fourni
    resolved_path: str | None = None
    if fhir_bundle_path is not None:
        resolved_path = _resolve_bundle_path(fhir_bundle_path)

    # Validation du bundle
    status, errors, warnings, profile_checked = validate_fhir_bundle(
        resolved_path,
        bundle_expected=bundle_expected,
        rules=rules,
    )

    # Construction des raisons synthétiques
    reasons: list[str] = []
    if status == VerificationStatus.PASS:
        reasons.append(
            f"Bundle FHIR {profile_checked or 'R4'} valide — "
            "structure et ressources conformes"
        )
    elif status == VerificationStatus.NEEDS_REVIEW:
        reasons.append(
            "Bundle FHIR chargé avec avertissements — revue recommandée"
        )
        reasons.extend(warnings[:5])
    else:
        reasons.append("Validation FHIR échouée — voir la liste des erreurs")
        reasons.extend(errors[:5])
    reasons.append(
        "Validation structurelle uniquement : la conformité FHIR ne prouve pas "
        "la vérité médicale ni la cohérence clinique du contenu."
    )

    resource_types: list[str] = []
    resource_count = 0
    if resolved_path is not None and not errors:
        bundle, load_errors = load_fhir_bundle(resolved_path)
        if bundle is not None and not load_errors:
            resource_types = extract_resource_types(bundle)
            resource_count = len(resource_types)

    return FhirValidatorResult(
        case_id=case_id,
        status=status,
        bundle_expected=bundle_expected,
        profile_checked=profile_checked,
        rule_version=rule_version,
        resource_types=resource_types,
        resource_count=resource_count,
        references_checked=resolved_path is not None and status != VerificationStatus.FAIL,
        errors=errors,
        warnings=warnings,
        reasons=reasons,
    )


# ── Nœud LangGraph ───────────────────────────────────────────────────────────


def node(state: ClaimState) -> dict:
    """Nœud LangGraph du FHIR Validator Agent.

    Lit fhir_input depuis le state, exécute la validation, écrit fhir_result.
    Vide fhir_input à None après traitement (consommation du champ d'entrée).

    Args:
        state: État partagé du workflow LangGraph.

    Returns:
        Dictionnaire de mise à jour du state.
    """
    fhir_input_raw: dict | None = state.get("fhir_input")

    # Cas où fhir_input est absent ou None → FAIL propre
    if fhir_input_raw is None:
        case_id = state.get("case_id", "UNKNOWN")
        result = FhirValidatorResult(
            case_id=case_id,
            status=VerificationStatus.FAIL,
            bundle_expected=True,
            profile_checked=None,
            rule_version=get_rule_version("fhir_rules.yaml"),
            errors=["fhir_input absent du state — impossible d'exécuter la validation FHIR"],
            warnings=[],
            reasons=["Entrée FHIR manquante dans le state"],
        )
        updates: dict = {
            "fhir_input": None,
            "fhir_result": result,
            "completed_steps": ["fhir_validation"],
            "current_step": "fhir_validation",
        }
        validate_state_update(updates)
        return updates

    # Validation Pydantic de l'entrée
    try:
        fhir_input = FhirValidatorInput(**fhir_input_raw)
    except (ValidationError, TypeError) as exc:
        case_id = fhir_input_raw.get("case_id", state.get("case_id", "UNKNOWN"))
        result = FhirValidatorResult(
            case_id=str(case_id),
            status=VerificationStatus.FAIL,
            bundle_expected=bool(fhir_input_raw.get("bundle_expected", True)),
            profile_checked=None,
            rule_version=get_rule_version("fhir_rules.yaml"),
            errors=[f"fhir_input invalide : {exc}"],
            warnings=[],
            reasons=["Entrée FHIR invalide — validation Pydantic échouée"],
        )
        updates = {
            "fhir_input": None,
            "fhir_result": result,
            "completed_steps": ["fhir_validation"],
            "current_step": "fhir_validation",
        }
        validate_state_update(updates)
        return updates

    # Exécution de la validation
    result = run(
        case_id=fhir_input.case_id,
        fhir_bundle_path=fhir_input.fhir_bundle_path,
        bundle_expected=fhir_input.bundle_expected,
    )

    updates = {
        "fhir_input": None,
        "fhir_result": result,
        "completed_steps": ["fhir_validation"],
        "current_step": "fhir_validation",
    }

    validate_state_update(updates)
    return updates
