"""Politiques d'accès par rôle (RBAC) — modèle DENY-by-default.

Principe fondamental :
  Un champ est refusé sauf s'il figure explicitement dans l'allowlist du rôle.
  Aucun rôle n'a accès à l'intégralité des champs connus.
  Tout rôle inconnu est rejeté avant l'évaluation.

Les quatre rôles stables :
  ADMINISTRATIVE_MANAGER — traitement administratif et financier des dossiers.
  MEDICAL_REVIEWER       — examen médical (actes, diagnostics, médicaments).
  FRAUD_ANALYST          — analyse des anomalies financières et temporelles.
  AUDITOR                — audit interne minimal (référence + horodatages).

Champs couverts par ALL_KNOWN_FIELDS :
  Identification personnelle, données financières, données médicales,
  horodatages logistiques.

Classification des champs (taxonomie) :
  IDENTITY_FIELDS   — identifiants personnels nominatifs.
  FINANCIAL_FIELDS  — facturation et couverture.
  MEDICAL_FIELDS    — actes, prescriptions, diagnostics.
  FRAUD_FIELDS      — champs utilisés pour la détection de fraude.
  AUDIT_FIELDS      — accès minimal pour l'auditeur interne.
  SECRET_FIELDS     — champs système bloqués pour tous les rôles.
"""
from __future__ import annotations

from dataclasses import dataclass

from schemas.domain import ReaderRole

# ── Version de la politique d'accès ──────────────────────────────────────────

POLICY_VERSION: str = "1.1.0"

# ── Classification des champs par famille ─────────────────────────────────────

#: Champs d'identification personnelle — jamais exposés directement dans une vue.
IDENTITY_FIELDS: frozenset[str] = frozenset({
    "patient_name",
    "patient_id",
    "birth_date",
    "gender",
})

#: Champs financiers et de facturation.
FINANCIAL_FIELDS: frozenset[str] = frozenset({
    "total_billed",
    "amount_requested",
    "patient_share",
    "coverage_rate",
    "payer_name",
    "invoice_number",
    "prescription_number",
    "claim_reference",
})

#: Champs médicaux et cliniques.
MEDICAL_FIELDS: frozenset[str] = frozenset({
    "procedures",
    "prescriptions",
    "diagnosis_codes",
    "encounter_class",
    "provider_id",
    "organization_id",
})

#: Champs pertinents pour la détection de fraude (recoupement financier et temporel).
FRAUD_FIELDS: frozenset[str] = frozenset({
    "claim_reference",
    "invoice_number",
    "total_billed",
    "amount_requested",
    "patient_share",
    "service_date",
    "submitted_at",
})

#: Champs visibles par l'auditeur interne (accès minimal — traçabilité uniquement).
AUDIT_FIELDS: frozenset[str] = frozenset({
    "claim_reference",
    "service_date",
    "submitted_at",
})

#: Champs système bloqués pour tous les rôles — doivent être absents de toute vue.
#: Ces champs ne figurent pas dans ALL_KNOWN_FIELDS ; leur présence dans une vue est
#: une violation de politique indépendamment du rôle demandeur.
SECRET_FIELDS: frozenset[str] = frozenset({
    "raw_ocr_text",
    "system_prompt",
    "api_key",
    "storage_path",
    "file_path",
    "password",
    "token",
    "private_key",
})

#: Champs dont la présence en clair comme clé d'une vue constitue toujours une violation.
#: Regroupe les identifiants personnels bruts (doivent être pseudonymisés) et les secrets.
ALWAYS_BLOCKED_FIELDS: frozenset[str] = IDENTITY_FIELDS | SECRET_FIELDS

# ── Univers complet des champs évalués par le Privacy Agent ──────────────────

ALL_KNOWN_FIELDS: frozenset[str] = frozenset({
    # Identification personnelle
    "patient_name",
    "patient_id",
    "birth_date",
    "gender",
    # Financier et facturation
    "total_billed",
    "amount_requested",
    "patient_share",
    "coverage_rate",
    "payer_name",
    "invoice_number",
    "prescription_number",
    "claim_reference",
    # Médical et clinique
    "procedures",
    "prescriptions",
    "diagnosis_codes",
    "encounter_class",
    "provider_id",
    "organization_id",
    # Temporel et logistique
    "service_date",
    "submitted_at",
})

# Cohérence interne : ALL_KNOWN_FIELDS = classifications connues + temporels
assert ALL_KNOWN_FIELDS == IDENTITY_FIELDS | FINANCIAL_FIELDS | MEDICAL_FIELDS | {
    "service_date",
    "submitted_at",
}, "ALL_KNOWN_FIELDS est incohérent avec les classifications par famille"

# SECRET_FIELDS doit rester disjoint de ALL_KNOWN_FIELDS
assert SECRET_FIELDS.isdisjoint(ALL_KNOWN_FIELDS), (
    "SECRET_FIELDS ne doit pas contenir de champ métier connu"
)


# ── Politique par rôle ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RoleAccessPolicy:
    """Politique d'accès pour un rôle donné — modèle DENY-by-default.

    `allowed_fields` est la liste explicite des champs visibles.
    Tout champ absent de cette liste est automatiquement refusé (DENY).
    Aucune politique ne contient l'intégralité de ALL_KNOWN_FIELDS.
    """

    role: ReaderRole
    allowed_fields: frozenset[str]


ROLE_POLICIES: dict[ReaderRole, RoleAccessPolicy] = {
    ReaderRole.ADMINISTRATIVE_MANAGER: RoleAccessPolicy(
        role=ReaderRole.ADMINISTRATIVE_MANAGER,
        allowed_fields=frozenset({
            # Références administratives
            "claim_reference",
            "invoice_number",
            "prescription_number",
            # Couverture et montants
            "payer_name",
            "coverage_rate",
            "total_billed",
            "amount_requested",
            "patient_share",
            # Horodatages
            "service_date",
            "submitted_at",
            # Aucun identifiant personnel (patient_name, patient_id, birth_date, gender)
            # Aucun champ médical (procedures, prescriptions, diagnosis_codes…)
        }),
    ),
    ReaderRole.MEDICAL_REVIEWER: RoleAccessPolicy(
        role=ReaderRole.MEDICAL_REVIEWER,
        allowed_fields=frozenset({
            # Contenu médical
            "procedures",
            "prescriptions",
            "diagnosis_codes",
            "encounter_class",
            "provider_id",
            "organization_id",
            # Contexte démographique anonyme (sans nom ni identifiant nominatif)
            "birth_date",
            "gender",
            # Horodatage clinique
            "service_date",
            # Aucun identifiant personnel nominatif (patient_name, patient_id)
            # Aucun champ financier (total_billed, invoice_number…)
        }),
    ),
    ReaderRole.FRAUD_ANALYST: RoleAccessPolicy(
        role=ReaderRole.FRAUD_ANALYST,
        allowed_fields=frozenset({
            # Références pour recoupement
            "claim_reference",
            "invoice_number",
            # Montants analysés
            "total_billed",
            "amount_requested",
            "patient_share",
            # Horodatages pour analyse des patterns
            "service_date",
            "submitted_at",
            # Aucune donnée personnelle ni médicale
        }),
    ),
    ReaderRole.AUDITOR: RoleAccessPolicy(
        role=ReaderRole.AUDITOR,
        allowed_fields=frozenset({
            # Accès minimal — existence et traçabilité uniquement
            "claim_reference",
            "service_date",
            "submitted_at",
            # Aucune donnée personnelle, médicale ni financière individuelle
        }),
    ),
}

# Garde runtime : toutes les valeurs de ReaderRole doivent avoir une politique définie.
_missing_policies = [role for role in ReaderRole if role not in ROLE_POLICIES]
if _missing_policies:
    raise RuntimeError(
        "Politiques d'accès manquantes — chaque ReaderRole doit avoir une entrée dans "
        f"ROLE_POLICIES : {[r.value for r in _missing_policies]}"
    )


# ── PolicyViolation — motif structuré ─────────────────────────────────────────


@dataclass(frozen=True)
class PolicyViolation:
    """Motif structuré retourné lors d'une violation de politique de confidentialité.

    Attributes:
        field       : nom du champ en cause.
        reason_code : code stable identifiant la catégorie de violation.
        message     : description lisible, sans donnée personnelle.

    Codes définis :
        SECRET_FIELD_IN_VIEW      — champ système secret présent comme clé de vue.
        RAW_IDENTITY_IN_VIEW      — identifiant personnel brut visible comme clé de vue.
        INVALID_PSEUDONYM_FORMAT  — valeur d'un champ pseudonyme sans le bon préfixe.
    """

    field: str
    reason_code: str
    message: str


# ── Vérification post-vue ─────────────────────────────────────────────────────

# Préfixes attendus pour les champs de pseudonymes dans les vues minimisées.
_PSEUDONYM_PREFIXES: dict[str, str] = {
    "patient_pseudonym": "PAT-",
    "provider_pseudonym": "PRV-",
    "provider_reference": "PRV-",
}


def verify_view_privacy(view: dict) -> list[PolicyViolation]:
    """Vérifie qu'aucun identifiant brut ne survit dans la vue minimisée.

    Contrôles effectués (défense en profondeur après les validators Pydantic) :
      1. Aucun champ de SECRET_FIELDS ne doit apparaître comme clé de la vue.
      2. Aucun identifiant personnel brut (IDENTITY_FIELDS) ne doit apparaître
         comme clé de la vue — les vues doivent utiliser `patient_pseudonym`, etc.
      3. Les champs de pseudonymes présents doivent respecter leur préfixe attendu.

    Args:
        view : dict JSON-sérialisable produit par un builder de vue.

    Returns:
        Liste vide si la vue est conforme ; liste de PolicyViolation sinon.
    """
    violations: list[PolicyViolation] = []

    for field in SECRET_FIELDS:
        if field in view:
            violations.append(PolicyViolation(
                field=field,
                reason_code="SECRET_FIELD_IN_VIEW",
                message=(
                    f"Champ système '{field}' présent dans la vue — "
                    "bloqué pour tous les rôles"
                ),
            ))

    for field in IDENTITY_FIELDS:
        if field in view:
            violations.append(PolicyViolation(
                field=field,
                reason_code="RAW_IDENTITY_IN_VIEW",
                message=(
                    f"Identifiant personnel brut '{field}' visible comme clé de vue — "
                    "masquage ou pseudonymisation manquant"
                ),
            ))

    for field, prefix in _PSEUDONYM_PREFIXES.items():
        value = view.get(field)
        if value is not None and not str(value).startswith(prefix):
            violations.append(PolicyViolation(
                field=field,
                reason_code="INVALID_PSEUDONYM_FORMAT",
                message=(
                    f"'{field}' ne respecte pas le préfixe '{prefix}' "
                    f"— identifiant potentiellement brut"
                ),
            ))

    return violations


# ── Helpers ───────────────────────────────────────────────────────────────────


def get_role_policy(role: ReaderRole) -> RoleAccessPolicy:
    """Retourne la politique d'accès pour un rôle donné.

    Lève KeyError si le rôle n'a pas de politique — ce cas ne peut arriver
    qu'en cas d'incohérence interne (la garde runtime protège à l'import).
    """
    return ROLE_POLICIES[role]


def compute_masked_fields(
    role: ReaderRole,
    fields_to_evaluate: list[str] | None = None,
) -> list[str]:
    """Retourne la liste triée des champs REFUSÉS pour ce rôle (DENY-by-default).

    Logique :
      - L'univers d'évaluation est `fields_to_evaluate` si fourni, sinon ALL_KNOWN_FIELDS.
      - Champ refusé = champ présent dans l'univers ET absent de l'allowlist du rôle.
      - Tout champ inconnu (hors ALL_KNOWN_FIELDS) passé dans `fields_to_evaluate`
        est également refusé car absent de l'allowlist (politique DENY-by-default).

    Args:
        role               : rôle du lecteur demandeur.
        fields_to_evaluate : sous-ensemble de champs à évaluer, ou None pour ALL_KNOWN_FIELDS.

    Returns:
        Liste triée des champs refusés pour ce rôle.
    """
    policy = get_role_policy(role)
    universe: frozenset[str] = (
        frozenset(fields_to_evaluate) if fields_to_evaluate is not None else ALL_KNOWN_FIELDS
    )
    denied = universe - policy.allowed_fields
    return sorted(denied)
