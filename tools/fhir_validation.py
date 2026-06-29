"""Outils de validation bundle FHIR R4 — ClaimShield Santé.

Fonctions déterministes pures — aucun appel LLM, aucun effet de bord.
Utilisées par le FHIR Validator Agent pour vérifier l'intégrité structurelle
d'un bundle FHIR Synthea avant toute exploitation médicale ou administrative.

Logique de statut retourné par validate_fhir_bundle :
  - path None et bundle_expected False → PASS (pas de bundle attendu)
  - path None et bundle_expected True  → FAIL (bundle manquant)
  - path fourni mais fichier inexistant → FAIL
  - bundle chargé mais structure invalide → FAIL
  - bundle valide mais ressource obligatoire absente → FAIL
  - bundle valide mais ressource optionnelle absente → NEEDS_REVIEW (warning)
  - bundle valide et complet → PASS
"""
from __future__ import annotations

import json
from pathlib import Path

from schemas.domain import VerificationStatus

# Règles par défaut utilisées si aucune règle n'est passée en paramètre
_DEFAULT_REQUIRED_TYPES: list[str] = ["Patient"]
_DEFAULT_BUNDLE_TYPES: list[str] = ["collection", "transaction", "document", "batch"]
_DEFAULT_PROFILE = "R4"


# ── Chargement du bundle ──────────────────────────────────────────────────────


def load_fhir_bundle(path: str) -> tuple[dict | None, list[str]]:
    """Charge et parse le JSON FHIR depuis le chemin donné.

    Args:
        path: Chemin vers le fichier JSON (relatif ou absolu).

    Returns:
        (bundle_dict, errors) — errors est vide si le chargement réussit.
        bundle_dict est None si une erreur survient.
    """
    file_path = Path(path)
    if not file_path.exists():
        return None, [f"Fichier bundle FHIR introuvable : {path}"]
    if not file_path.is_file():
        return None, [f"Le chemin ne désigne pas un fichier régulier : {path}"]

    try:
        raw = file_path.read_text(encoding="utf-8")
    except OSError as exc:
        return None, [f"Impossible de lire le fichier bundle FHIR : {exc}"]

    if not raw.strip():
        return None, ["Le fichier bundle FHIR est vide"]

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, [f"Bundle FHIR invalide (JSON malformé) : {exc}"]

    if not isinstance(data, dict):
        return None, ["Bundle FHIR invalide : la racine doit être un objet JSON"]

    return data, []


# ── Validation de la structure de base ───────────────────────────────────────


def validate_bundle_structure(
    bundle: dict,
    *,
    accepted_types: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Valide la structure de base d'un bundle FHIR.

    Vérifie :
      - resourceType == "Bundle"
      - Présence du champ "type"
      - Valeur de "type" dans la liste des types acceptés
      - Présence du champ "entry" (liste)

    Args:
        bundle: Dictionnaire représentant le bundle FHIR.
        accepted_types: Types de bundle acceptés. Si None, utilise les valeurs par défaut.

    Returns:
        (errors, warnings) — deux listes de chaînes décrivant les problèmes détectés.
    """
    errors: list[str] = []
    warnings: list[str] = []
    types_ok = accepted_types if accepted_types is not None else _DEFAULT_BUNDLE_TYPES

    # Vérification resourceType
    resource_type = bundle.get("resourceType")
    if resource_type != "Bundle":
        errors.append(
            f"resourceType attendu : 'Bundle', reçu : {resource_type!r}"
        )

    # Vérification du type de bundle
    bundle_type = bundle.get("type")
    if bundle_type is None:
        errors.append("Champ 'type' absent du bundle FHIR")
    elif bundle_type not in types_ok:
        errors.append(
            f"Type de bundle non supporté : {bundle_type!r} "
            f"(valeurs acceptées : {types_ok})"
        )

    # Vérification de la présence du champ entry
    if "entry" not in bundle:
        warnings.append("Champ 'entry' absent du bundle — aucune ressource disponible")
    elif not isinstance(bundle["entry"], list):
        errors.append("Champ 'entry' invalide : doit être une liste")
    elif len(bundle["entry"]) == 0:
        warnings.append("Bundle FHIR sans entrée (entry vide)")

    return errors, warnings


# ── Extraction des types de ressources ───────────────────────────────────────


def extract_resource_types(bundle: dict) -> list[str]:
    """Retourne la liste des resourceType présents dans bundle['entry'].

    Args:
        bundle: Dictionnaire représentant le bundle FHIR.

    Returns:
        Liste des types de ressources présents (avec doublons si plusieurs
        ressources du même type sont présentes).
    """
    types: list[str] = []
    for entry in bundle.get("entry", []):
        resource = entry.get("resource", {})
        if isinstance(resource, dict):
            rt = resource.get("resourceType")
            if rt:
                types.append(rt)
    return types


def _iter_resources(bundle: dict) -> list[dict]:
    resources: list[dict] = []
    for entry in bundle.get("entry", []):
        if not isinstance(entry, dict):
            continue
        resource = entry.get("resource")
        if isinstance(resource, dict):
            resources.append(resource)
    return resources


def check_entry_resources(bundle: dict, rules: dict) -> tuple[list[str], list[str]]:
    """Vérifie que chaque entry contient une ressource typée et supportée localement."""
    errors: list[str] = []
    warnings: list[str] = []
    supported = set(rules.get("supported_resource_types", []))

    for index, entry in enumerate(bundle.get("entry", [])):
        if not isinstance(entry, dict):
            errors.append(f"Entrée bundle invalide à l'index {index} : objet attendu")
            continue
        resource = entry.get("resource")
        if not isinstance(resource, dict):
            errors.append(f"Entrée bundle sans ressource exploitable à l'index {index}")
            continue
        resource_type = resource.get("resourceType")
        if not isinstance(resource_type, str) or not resource_type:
            errors.append(f"resourceType absent dans entry[{index}].resource")
        elif supported and resource_type not in supported:
            warnings.append(f"resourceType non supporté localement : {resource_type}")

    return errors, warnings


def check_min_cardinalities(bundle: dict, rules: dict) -> list[str]:
    """Vérifie les cardinalités minimales par resourceType."""
    errors: list[str] = []
    counts: dict[str, int] = {}
    for resource_type in extract_resource_types(bundle):
        counts[resource_type] = counts.get(resource_type, 0) + 1

    for resource_type, minimum in rules.get("min_cardinalities", {}).items():
        try:
            min_count = int(minimum)
        except (TypeError, ValueError):
            errors.append(f"Cardinalité minimale invalide pour {resource_type} : {minimum!r}")
            continue
        actual = counts.get(resource_type, 0)
        if actual < min_count:
            errors.append(
                f"Cardinalité minimale non respectée : {resource_type} "
                f"attendu >= {min_count}, reçu {actual}"
            )
    return errors


def _collect_reference_targets(bundle: dict) -> set[str]:
    targets: set[str] = set()
    for entry in bundle.get("entry", []):
        if not isinstance(entry, dict):
            continue
        full_url = entry.get("fullUrl")
        if isinstance(full_url, str) and full_url:
            targets.add(full_url)
        resource = entry.get("resource")
        if not isinstance(resource, dict):
            continue
        resource_type = resource.get("resourceType")
        resource_id = resource.get("id")
        if isinstance(resource_type, str) and isinstance(resource_id, str) and resource_id:
            targets.add(resource_id)
            targets.add(f"{resource_type}/{resource_id}")
            if full_url and isinstance(full_url, str):
                targets.add(f"{resource_type}/{full_url.rsplit('/', 1)[-1]}")
    return targets


def _get_nested(resource: dict, dotted_path: str) -> object | None:
    current: object = resource
    for part in dotted_path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def check_internal_references(bundle: dict, rules: dict) -> list[str]:
    """Vérifie que les références internes configurées pointent vers une ressource présente."""
    errors: list[str] = []
    targets = _collect_reference_targets(bundle)
    reference_rules: dict = rules.get("reference_fields", {})

    for resource in _iter_resources(bundle):
        resource_type = resource.get("resourceType")
        resource_id = resource.get("id", "<sans-id>")
        for path in reference_rules.get(resource_type, []):
            reference = _get_nested(resource, path)
            if reference in (None, ""):
                continue
            if not isinstance(reference, str):
                errors.append(
                    f"Référence invalide {resource_type}/{resource_id}.{path} : chaîne attendue"
                )
                continue
            if reference.startswith("#"):
                continue
            if reference not in targets:
                errors.append(
                    f"Référence interne non résolue {resource_type}/{resource_id}.{path} -> {reference}"
                )
    return errors


def check_supported_profiles(bundle: dict, rules: dict) -> list[str]:
    """Signale les profils déclarés non supportés par les règles locales."""
    warnings: list[str] = []
    supported = set(rules.get("supported_profiles", [_DEFAULT_PROFILE]))
    if not supported:
        return warnings

    for resource in _iter_resources(bundle):
        resource_type = resource.get("resourceType", "<unknown>")
        resource_id = resource.get("id", "<sans-id>")
        profiles = resource.get("meta", {}).get("profile", [])
        if isinstance(profiles, str):
            profiles = [profiles]
        if not isinstance(profiles, list):
            continue
        for profile in profiles:
            profile_text = str(profile)
            if not any(supported_profile in profile_text for supported_profile in supported):
                warnings.append(
                    f"Profil FHIR non supporté localement : {resource_type}/{resource_id} -> {profile_text}"
                )
    return warnings


# ── Vérification des ressources obligatoires ─────────────────────────────────


def check_required_resources(
    bundle: dict,
    required_types: list[str],
) -> tuple[list[str], list[str]]:
    """Vérifie la présence des ressources obligatoires dans le bundle.

    Args:
        bundle: Dictionnaire représentant le bundle FHIR.
        required_types: Liste des resourceType obligatoires.

    Returns:
        (errors, warnings) — errors contient les ressources manquantes obligatoires.
    """
    errors: list[str] = []
    warnings: list[str] = []
    present = set(extract_resource_types(bundle))

    for resource_type in required_types:
        if resource_type not in present:
            errors.append(
                f"Ressource obligatoire absente du bundle : {resource_type}"
            )

    return errors, warnings


# ── Vérification des champs obligatoires par ressource ───────────────────────


def check_resource_fields(bundle: dict, rules: dict) -> list[str]:
    """Vérifie les champs obligatoires par type de ressource selon les règles YAML.

    Les règles sont lues depuis la clé 'resource_required_fields' du dictionnaire
    de règles. Chaque sous-clé est un resourceType et sa valeur est la liste des
    champs obligatoires pour ce type.

    Args:
        bundle: Dictionnaire représentant le bundle FHIR.
        rules: Dictionnaire de règles chargé depuis fhir_rules.yaml.

    Returns:
        Liste des erreurs de validation des champs (vide si tout est conforme).
    """
    errors: list[str] = []
    field_rules: dict = rules.get("resource_required_fields", {})
    if not field_rules:
        return errors

    for entry in bundle.get("entry", []):
        resource = entry.get("resource", {})
        if not isinstance(resource, dict):
            continue
        resource_type = resource.get("resourceType")
        if resource_type not in field_rules:
            continue
        required_fields = field_rules[resource_type]
        for field in required_fields:
            if field not in resource:
                errors.append(
                    f"Champ obligatoire '{field}' absent "
                    f"dans la ressource {resource_type}"
                )

    return errors


# ── Vérification des valeurs de statut Coverage ──────────────────────────────


def _check_coverage_status(bundle: dict, rules: dict) -> list[str]:
    """Vérifie que le statut de chaque ressource Coverage est dans la liste autorisée."""
    errors: list[str] = []
    accepted_statuses: list[str] = rules.get(
        "coverage_status_values",
        ["active", "cancelled", "draft", "entered-in-error"],
    )
    for entry in bundle.get("entry", []):
        resource = entry.get("resource", {})
        if not isinstance(resource, dict):
            continue
        if resource.get("resourceType") != "Coverage":
            continue
        status = resource.get("status")
        if status is not None and status not in accepted_statuses:
            errors.append(
                f"Statut Coverage invalide : {status!r} "
                f"(valeurs acceptées : {accepted_statuses})"
            )
    return errors


# ── Point d'entrée principal ──────────────────────────────────────────────────


def validate_fhir_bundle(
    path: str | None,
    *,
    bundle_expected: bool = True,
    rules: dict | None = None,
) -> tuple[VerificationStatus, list[str], list[str], str | None]:
    """Valide un bundle FHIR R4 selon les règles YAML configurées.

    Args:
        path: Chemin vers le fichier bundle FHIR (relatif au projet).
              None si aucun bundle n'a été fourni.
        bundle_expected: True si un bundle est attendu pour ce dossier.
        rules: Dictionnaire de règles (chargé depuis fhir_rules.yaml).
               Si None, utilise les règles minimales par défaut.

    Returns:
        (status, errors, warnings, profile_checked)
        - status: PASS / NEEDS_REVIEW / FAIL
        - errors: liste de messages d'erreur
        - warnings: liste de messages d'avertissement
        - profile_checked: identifiant du profil validé (ex. "R4"), ou None si FAIL avant validation
    """
    effective_rules = rules or {}
    profile = effective_rules.get("profile", _DEFAULT_PROFILE)
    required_types: list[str] = effective_rules.get(
        "required_resource_types", _DEFAULT_REQUIRED_TYPES
    )
    optional_types: list[str] = effective_rules.get("optional_resource_types", [])
    accepted_bundle_types: list[str] = effective_rules.get(
        "bundle_types_accepted", _DEFAULT_BUNDLE_TYPES
    )

    # Cas : aucun bundle attendu
    if path is None and not bundle_expected:
        return VerificationStatus.PASS, [], [], None

    # Cas : bundle attendu mais absent
    if path is None and bundle_expected:
        return (
            VerificationStatus.FAIL,
            ["Bundle FHIR attendu mais aucun chemin fourni"],
            [],
            None,
        )

    # Chargement du fichier
    bundle, load_errors = load_fhir_bundle(path)
    if load_errors:
        return VerificationStatus.FAIL, load_errors, [], None

    all_errors: list[str] = []
    all_warnings: list[str] = []

    # Validation de la structure de base
    struct_errors, struct_warnings = validate_bundle_structure(
        bundle, accepted_types=accepted_bundle_types
    )
    all_errors.extend(struct_errors)
    all_warnings.extend(struct_warnings)

    # Si la structure est invalide, on arrête ici
    if all_errors:
        return VerificationStatus.FAIL, all_errors, all_warnings, None

    # Vérification des entrées et resourceType
    entry_errors, entry_warnings = check_entry_resources(bundle, effective_rules)
    all_errors.extend(entry_errors)
    all_warnings.extend(entry_warnings)

    # Vérification des ressources obligatoires
    req_errors, req_warnings = check_required_resources(bundle, required_types)
    all_errors.extend(req_errors)
    all_warnings.extend(req_warnings)

    # Vérification des cardinalités minimales
    cardinality_errors = check_min_cardinalities(bundle, effective_rules)
    all_errors.extend(cardinality_errors)

    # Vérification des champs obligatoires par ressource
    field_errors = check_resource_fields(bundle, effective_rules)
    all_errors.extend(field_errors)

    # Vérification des statuts Coverage
    coverage_errors = _check_coverage_status(bundle, effective_rules)
    all_errors.extend(coverage_errors)

    # Vérification des références internes
    reference_errors = check_internal_references(bundle, effective_rules)
    all_errors.extend(reference_errors)

    # Vérification des profils supportés localement
    profile_warnings = check_supported_profiles(bundle, effective_rules)
    all_warnings.extend(profile_warnings)

    # Vérification des ressources optionnelles (warning si absentes)
    if optional_types:
        present = set(extract_resource_types(bundle))
        for opt_type in optional_types:
            if opt_type not in present:
                all_warnings.append(
                    f"Ressource optionnelle absente du bundle : {opt_type}"
                )

    # Détermination du statut final
    if all_errors:
        return VerificationStatus.FAIL, all_errors, all_warnings, profile

    if all_warnings:
        return VerificationStatus.NEEDS_REVIEW, all_errors, all_warnings, profile

    return VerificationStatus.PASS, [], all_warnings, profile
