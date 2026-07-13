"""Service de confidentialité déterministe — services/privacy_service.py (V2).

Remplace `agents/privacy_agent` (V1) pour la V2 : mêmes garanties RBAC
DENY-by-default et anti-fuite (`security.access_policies`,
`tools.pseudonymize`, builders de vue déjà purs d'`agents.privacy_agent`),
mais exposé comme un service appelé par `document_understanding_agent`
plutôt que comme un agent LangGraph séparé.

Différences volontaires avec `agents/privacy_agent` (V1) — voir plan V2 §5 :
  - aucun appel LLM (la justification d'audit textuelle du Privacy Agent
    V1 n'existe plus — ce service ne produit aucun texte libre) ;
  - aucune précondition Security Gate (le graphe V2 garantit déjà l'ordre
    `intake_safety → document_understanding`, voir `graph/workflow_v2.py`) ;
  - aucun `ClaimState`/`PrivacyResult`/`PrivacyAuditEntry` — un résultat
    local (`PrivacyViewResult`) suffit à l'usage interne V2.

Ne modifie aucun fichier V1 — importe uniquement des modules déjà partagés
(`schemas.domain`, `security.access_policies`, `tools.pseudonymize`) et les
builders de vue déjà purs d'`agents.privacy_agent.views`/`schemas`, qui ne
connaissent ni LangGraph ni le LLM (vérifié par lecture directe du code
avant réutilisation).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import ValidationError

from agents.privacy_agent.views import build_view
from schemas.domain import PrivacyCode, ReaderRole, VerificationStatus
from security.access_policies import compute_masked_fields, verify_view_privacy
from tools.pseudonymize import pseudonymization_key_is_available

__all__ = ["PrivacyService", "PrivacyViewResult"]


@dataclass(frozen=True)
class PrivacyViewResult:
    """Résultat de la construction d'une vue minimisée — sans LLM ni ClaimState.

    ``status`` reflète uniquement des faits déterministes (clé de
    pseudonymisation disponible, vue valide, aucune fuite détectée après
    construction, présence de données personnelles réelles) — jamais une
    appréciation d'un LLM, qui n'existe pas dans ce service.
    """

    status: VerificationStatus
    role: ReaderRole
    view: dict | None
    redacted_fields: list[str] = field(default_factory=list)
    reason_codes: list[PrivacyCode] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class PrivacyService:
    """Service déterministe RBAC + pseudonymisation, injectable, sans état partagé.

    Aucune instance globale cachée — à instancier explicitement, même
    convention que ``services.duplicate_index.DuplicateIndex``/
    ``services.audit_store.AuditStore``.
    """

    def build_view(
        self,
        *,
        case_id: str,
        role: ReaderRole,
        claim_data: dict,
        fields_to_evaluate: list[str] | None = None,
        contains_real_personal_data: bool = False,
    ) -> PrivacyViewResult:
        """Construit la vue minimisée adaptée au rôle — DENY-by-default.

        Retourne toujours un ``PrivacyViewResult`` (jamais une exception) :
        toute défaillance (clé de pseudonymisation absente, vue invalide,
        identifiant brut détecté après construction) produit un statut
        ``FAIL`` avec un motif structuré — jamais une vue partiellement
        exposée ni un statut optimiste par défaut.
        """
        redacted_fields = compute_masked_fields(role, fields_to_evaluate)

        if not pseudonymization_key_is_available():
            return PrivacyViewResult(
                status=VerificationStatus.FAIL,
                role=role,
                view=None,
                redacted_fields=redacted_fields,
                reason_codes=[PrivacyCode.MISSING_PSEUDONYMIZATION_KEY],
                errors=["Clé de pseudonymisation indisponible — construction de vue refusée."],
            )

        try:
            view = build_view(role, case_id, claim_data)
        except ValidationError as exc:
            return PrivacyViewResult(
                status=VerificationStatus.FAIL,
                role=role,
                view=None,
                redacted_fields=redacted_fields,
                reason_codes=[PrivacyCode.INVALID_PRIVACY_OUTPUT],
                errors=[f"{len(exc.errors())} erreur(s) de validation de la vue minimisée."],
            )
        except Exception as exc:  # pseudonymisation/masquage — erreur technique isolée
            return PrivacyViewResult(
                status=VerificationStatus.FAIL,
                role=role,
                view=None,
                redacted_fields=redacted_fields,
                reason_codes=[PrivacyCode.PSEUDONYMIZATION_ERROR],
                errors=[f"Erreur technique de pseudonymisation : {type(exc).__name__}"],
            )

        violations = verify_view_privacy(view)
        if violations:
            codes = sorted(
                {
                    PrivacyCode.FORBIDDEN_FIELD_EXPOSED
                    if v.reason_code == "SECRET_FIELD_IN_VIEW"
                    else PrivacyCode.UNMASKED_IDENTIFIER
                    for v in violations
                },
                key=lambda c: c.value,
            )
            return PrivacyViewResult(
                status=VerificationStatus.FAIL,
                role=role,
                view=None,
                redacted_fields=redacted_fields,
                reason_codes=codes,
                errors=[v.message for v in violations],
            )

        status = (
            VerificationStatus.NEEDS_REVIEW
            if contains_real_personal_data
            else VerificationStatus.PASS
        )
        return PrivacyViewResult(
            status=status,
            role=role,
            view=view,
            redacted_fields=redacted_fields,
            reason_codes=[],
            errors=[],
        )
