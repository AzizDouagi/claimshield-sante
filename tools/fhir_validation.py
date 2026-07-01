"""Outils de validation bundle FHIR R4 — ClaimShield Santé.

Fonctions déterministes pures — aucun appel LLM, aucun effet de bord.
Le bundle source n'est jamais modifié. Aucune conclusion clinique n'est tirée.

Logique de statut retourné par validate_fhir_bundle :
  - path None et bundle_expected False → PASS (pas de bundle attendu)
  - path None et bundle_expected True  → FAIL (bundle manquant)
  - path fourni mais fichier inexistant → FAIL
  - JSON malformé ou racine non-dict    → FAIL
  - resourceType != "Bundle"            → FAIL
  - bundle.entry absent ou non-liste    → FAIL / warning selon cas
  - ressource obligatoire absente       → FAIL
  - ressource optionnelle absente       → NEEDS_REVIEW (warning)
  - bundle valide et complet            → PASS

Chaque message d'erreur ou d'avertissement contient un emplacement précis
au format FHIRPath-style : "entry[N] (ResourceType/id) .champ.imbriqué"
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import ValidationError as _PydanticValidationError

from schemas.domain import VerificationStatus

# Import optionnel — la validation schéma est désactivée si la lib est absente.
try:
    from fhir.resources import get_fhir_model_class as _fhir_get_model
    _FHIR_LIB_AVAILABLE: bool = True
except ImportError:
    _FHIR_LIB_AVAILABLE = False
    _fhir_get_model = None  # type: ignore[assignment]

_DEFAULT_REQUIRED_TYPES: list[str] = ["Patient"]
_DEFAULT_BUNDLE_TYPES: list[str] = ["collection", "transaction", "document", "batch"]
_DEFAULT_PROFILE = "R4"


# ── Index structural du bundle ────────────────────────────────────────────────


@dataclass
class BundleIndex:
    """Index d'un bundle FHIR par fullUrl, (type, id) et type seul.

    Construit uniquement à partir de bundle['entry'] — jamais depuis le
    contenu métier des ressources. Le bundle source n'est jamais modifié.

    Attributs :
        by_full_url     : fullUrl → ressource (dict)
        by_type_id      : (resourceType, id) → ressource (dict)
        by_type         : resourceType → liste ordonnée des ressources
        all_targets     : ensemble de tous les identifiants adressables dans le bundle
                          (fullUrl, "ResourceType/id", id brut)
        entry_positions : index d'entrée (int) → ressource (dict)
    """

    by_full_url: dict[str, dict] = field(default_factory=dict)
    by_type_id: dict[tuple[str, str], dict] = field(default_factory=dict)
    by_type: dict[str, list[dict]] = field(default_factory=dict)
    all_targets: set[str] = field(default_factory=set)
    entry_positions: dict[int, dict] = field(default_factory=dict)


def build_bundle_index(bundle: dict) -> BundleIndex:
    """Construit un BundleIndex à partir de bundle['entry'].

    Indexe chaque entrée selon :
      - son fullUrl (URN UUID ou URL absolue)
      - son couple (resourceType, id) → référence relative "ResourceType/id"
      - son resourceType seul

    Le bundle source n'est pas modifié. Les entrées non-dict ou sans ressource
    sont silencieusement ignorées (elles seront signalées par check_entry_resources).

    Args:
        bundle: Dictionnaire représentant le bundle FHIR (lecture seule).

    Returns:
        BundleIndex peuplé — jamais None.
    """
    index = BundleIndex()

    for position, entry in enumerate(bundle.get("entry", [])):
        if not isinstance(entry, dict):
            continue

        resource = entry.get("resource")
        if not isinstance(resource, dict):
            continue

        resource_type = resource.get("resourceType")
        resource_id = resource.get("id")
        full_url = entry.get("fullUrl")

        # Enregistrement par position d'entrée (pour les messages localisés)
        index.entry_positions[position] = resource

        # Indexation par fullUrl
        if isinstance(full_url, str) and full_url:
            index.by_full_url[full_url] = resource
            index.all_targets.add(full_url)

        # Indexation par (resourceType, id) et référence relative
        if isinstance(resource_type, str) and isinstance(resource_id, str) and resource_id:
            index.by_type_id[(resource_type, resource_id)] = resource
            index.all_targets.add(f"{resource_type}/{resource_id}")
            index.all_targets.add(resource_id)

        # Indexation par type seul
        if isinstance(resource_type, str) and resource_type:
            index.by_type.setdefault(resource_type, []).append(resource)

    return index


# ── Helpers de localisation ───────────────────────────────────────────────────


def _entry_prefix(position: int, resource: dict | None = None) -> str:
    """Préfixe FHIRPath-style pour les messages d'erreur : 'entry[N] (Type/id)'."""
    if resource is None:
        return f"entry[{position}]"
    rt = resource.get("resourceType")
    rid = resource.get("id")
    if rt and rid:
        return f"entry[{position}] ({rt}/{rid})"
    if rt:
        return f"entry[{position}] ({rt})"
    return f"entry[{position}]"


def _get_nested(obj: dict, dotted_path: str) -> object | None:
    """Accès en lecture seule à un champ imbriqué par chemin pointé.

    Ex. : _get_nested(resource, "patient.reference") → valeur ou None.
    Le dict source n'est jamais modifié.
    """
    current: object = obj
    for part in dotted_path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


# ── Chargement du bundle ──────────────────────────────────────────────────────


def load_fhir_bundle(path: str) -> tuple[dict | None, list[str]]:
    """Charge et parse le fichier bundle FHIR depuis le chemin donné.

    Vérifie successivement :
      1. Existence et type régulier du fichier
      2. Lecture UTF-8
      3. Contenu non vide
      4. JSON valide
      5. Racine de type dict

    Args:
        path: Chemin vers le fichier JSON (relatif au projet ou absolu).

    Returns:
        (bundle_dict, errors) — bundle_dict est None si une erreur survient.
    """
    file_path = Path(path)

    if not file_path.exists():
        return None, [f"bundle.file: fichier FHIR introuvable — {path!r}"]
    if not file_path.is_file():
        return None, [f"bundle.file: le chemin ne désigne pas un fichier régulier — {path!r}"]

    try:
        raw = file_path.read_text(encoding="utf-8")
    except OSError as exc:
        return None, [f"bundle.file: lecture impossible — {exc}"]

    if not raw.strip():
        return None, ["bundle.file: fichier vide"]

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, [f"bundle: JSON malformé — {exc}"]

    if not isinstance(data, dict):
        return None, [
            f"bundle: la racine doit être un objet JSON (dict), "
            f"reçu {type(data).__name__!r}"
        ]

    return data, []


# ── Validation de la structure de base ───────────────────────────────────────


def validate_bundle_structure(
    bundle: dict,
    *,
    accepted_types: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Valide la structure minimale d'un bundle FHIR.

    Vérifie :
      - bundle.resourceType == "Bundle"
      - bundle.type présent et dans les types acceptés
      - bundle.entry présent et de type liste

    Chaque message inclut le chemin JSON concerné.

    Returns:
        (errors, warnings)
    """
    errors: list[str] = []
    warnings: list[str] = []
    types_ok = accepted_types if accepted_types is not None else _DEFAULT_BUNDLE_TYPES

    resource_type = bundle.get("resourceType")
    if resource_type != "Bundle":
        errors.append(
            f"bundle.resourceType: attendu 'Bundle', reçu {resource_type!r}"
        )

    bundle_type = bundle.get("type")
    if bundle_type is None:
        errors.append("bundle.type: champ obligatoire absent")
    elif bundle_type not in types_ok:
        errors.append(
            f"bundle.type: valeur non supportée {bundle_type!r} "
            f"(valeurs acceptées : {types_ok})"
        )

    entry_value = bundle.get("entry", _SENTINEL := object())
    if entry_value is _SENTINEL:
        warnings.append("bundle.entry: champ absent — aucune ressource disponible")
    elif not isinstance(bundle.get("entry"), list):
        errors.append(
            f"bundle.entry: doit être une liste, "
            f"reçu {type(bundle.get('entry')).__name__!r}"
        )
    elif len(bundle["entry"]) == 0:
        warnings.append("bundle.entry: liste vide — bundle sans ressource")

    return errors, warnings


# ── Extraction des types de ressources ───────────────────────────────────────


def extract_resource_types(bundle: dict) -> list[str]:
    """Retourne la liste (avec doublons) des resourceType présents dans bundle['entry']."""
    types: list[str] = []
    for entry in bundle.get("entry", []):
        resource = entry.get("resource", {}) if isinstance(entry, dict) else {}
        if isinstance(resource, dict):
            rt = resource.get("resourceType")
            if isinstance(rt, str) and rt:
                types.append(rt)
    return types


def _iter_indexed_resources(bundle: dict) -> list[tuple[int, dict]]:
    """Itère sur les ressources valides du bundle avec leur index d'entrée."""
    result: list[tuple[int, dict]] = []
    for position, entry in enumerate(bundle.get("entry", [])):
        if not isinstance(entry, dict):
            continue
        resource = entry.get("resource")
        if isinstance(resource, dict):
            result.append((position, resource))
    return result


# ── Vérification des entrées ──────────────────────────────────────────────────


def check_entry_resources(bundle: dict, rules: dict) -> tuple[list[str], list[str]]:
    """Vérifie que chaque entry contient une ressource typée et supportée localement.

    Messages incluent l'index d'entrée : "entry[N].resource.resourceType: …"
    """
    errors: list[str] = []
    warnings: list[str] = []
    supported = set(rules.get("supported_resource_types", []))

    for position, entry in enumerate(bundle.get("entry", [])):
        if not isinstance(entry, dict):
            errors.append(
                f"entry[{position}]: objet JSON attendu, reçu {type(entry).__name__!r}"
            )
            continue

        resource = entry.get("resource")
        if not isinstance(resource, dict):
            errors.append(
                f"entry[{position}].resource: ressource exploitable absente ou invalide"
            )
            continue

        resource_type = resource.get("resourceType")
        if not isinstance(resource_type, str) or not resource_type:
            errors.append(
                f"entry[{position}].resource.resourceType: champ obligatoire absent"
            )
        elif supported and resource_type not in supported:
            warnings.append(
                f"entry[{position}] ({resource_type}): "
                "type de ressource non supporté localement"
            )

    return errors, warnings


# ── Ressources obligatoires ───────────────────────────────────────────────────


def check_required_resources(
    bundle: dict,
    required_types: list[str],
) -> tuple[list[str], list[str]]:
    """Vérifie la présence de chaque type de ressource obligatoire.

    Returns:
        (errors, warnings) — errors contient un message par type manquant.
    """
    errors: list[str] = []
    warnings: list[str] = []
    present = set(extract_resource_types(bundle))

    for resource_type in required_types:
        if resource_type not in present:
            errors.append(
                f"bundle.entry: ressource obligatoire absente — {resource_type!r}"
            )

    return errors, warnings


# ── Cardinalités minimales ────────────────────────────────────────────────────


def check_min_cardinalities(bundle: dict, rules: dict) -> list[str]:
    """Vérifie les cardinalités minimales déclarées dans les règles.

    Messages : "bundle.entry (ResourceType): cardinalité minimale — attendu >= N, présent M"
    """
    errors: list[str] = []
    counts: dict[str, int] = {}
    for rt in extract_resource_types(bundle):
        counts[rt] = counts.get(rt, 0) + 1

    for resource_type, minimum in rules.get("min_cardinalities", {}).items():
        try:
            min_count = int(minimum)
        except (TypeError, ValueError):
            errors.append(
                f"bundle.entry ({resource_type}): "
                f"cardinalité minimale invalide dans les règles — {minimum!r}"
            )
            continue
        actual = counts.get(resource_type, 0)
        if actual < min_count:
            errors.append(
                f"bundle.entry ({resource_type}): cardinalité minimale non respectée "
                f"— attendu >= {min_count}, présent {actual}"
            )

    return errors


# ── Champs obligatoires par ressource ────────────────────────────────────────


def check_resource_fields(bundle: dict, rules: dict) -> list[str]:
    """Vérifie les champs obligatoires par type de ressource selon les règles YAML.

    Supporte les chemins imbriqués via notation pointée (ex. "name.family").
    Messages : "entry[N] (ResourceType/id) .champ: champ obligatoire absent"
    """
    errors: list[str] = []
    field_rules: dict = rules.get("resource_required_fields", {})
    if not field_rules:
        return errors

    for position, resource in _iter_indexed_resources(bundle):
        resource_type = resource.get("resourceType")
        required_fields = field_rules.get(resource_type)
        if not required_fields:
            continue

        prefix = _entry_prefix(position, resource)
        for required_field in required_fields:
            if _get_nested(resource, required_field) is None:
                errors.append(
                    f"{prefix} .{required_field}: champ obligatoire absent"
                )

    return errors


# ── Statut des ressources Coverage ───────────────────────────────────────────


def _check_coverage_status(bundle: dict, rules: dict) -> list[str]:
    """Vérifie que chaque ressource Coverage a un statut dans la liste autorisée.

    Messages : "entry[N] (Coverage/id) .status: valeur non autorisée …"
    """
    errors: list[str] = []
    accepted: list[str] = rules.get(
        "coverage_status_values",
        ["active", "cancelled", "draft", "entered-in-error"],
    )
    for position, resource in _iter_indexed_resources(bundle):
        if resource.get("resourceType") != "Coverage":
            continue
        status = resource.get("status")
        if status is not None and status not in accepted:
            prefix = _entry_prefix(position, resource)
            errors.append(
                f"{prefix} .status: valeur non autorisée {status!r} "
                f"(valeurs acceptées : {accepted})"
            )

    return errors


# ── Références internes non résolues ─────────────────────────────────────────


def check_internal_references(bundle: dict, rules: dict) -> list[str]:
    """Vérifie que les références internes pointent vers une ressource du bundle.

    Utilise BundleIndex pour la résolution des cibles.
    Ignore les références contenues (#...) et les URL absolues (http/https).

    Messages : "entry[N] (Type/id) .path.to.reference: référence non résolue → 'target'"
    """
    errors: list[str] = []
    reference_rules: dict = rules.get("reference_fields", {})
    if not reference_rules:
        return errors

    index = build_bundle_index(bundle)

    for position, resource in _iter_indexed_resources(bundle):
        resource_type = resource.get("resourceType")
        paths_to_check: list[str] = reference_rules.get(resource_type, [])
        if not paths_to_check:
            continue

        prefix = _entry_prefix(position, resource)

        for path in paths_to_check:
            reference = _get_nested(resource, path)

            if reference is None or reference == "":
                continue

            if not isinstance(reference, str):
                errors.append(
                    f"{prefix} .{path}: référence doit être une chaîne, "
                    f"reçu {type(reference).__name__!r}"
                )
                continue

            # Référence vers une ressource contenue (#fragment) — hors bundle, admis
            if reference.startswith("#"):
                continue

            # URL absolue (serveur externe) — pas de vérification locale
            if reference.startswith(("http://", "https://")):
                continue

            if reference not in index.all_targets:
                errors.append(
                    f"{prefix} .{path}: référence interne non résolue → {reference!r}"
                )

    return errors


# ── Profils FHIR supportés ────────────────────────────────────────────────────


def check_supported_profiles(bundle: dict, rules: dict) -> list[str]:
    """Signale les profils déclarés dans meta.profile non supportés localement.

    Messages : "entry[N] (Type/id) .meta.profile: profil non supporté → 'url'"
    """
    warnings: list[str] = []
    supported = set(rules.get("supported_profiles", [_DEFAULT_PROFILE]))
    if not supported:
        return warnings

    for position, resource in _iter_indexed_resources(bundle):
        prefix = _entry_prefix(position, resource)
        profiles = resource.get("meta", {}).get("profile", [])
        if isinstance(profiles, str):
            profiles = [profiles]
        if not isinstance(profiles, list):
            continue
        for profile in profiles:
            profile_text = str(profile)
            if not any(s in profile_text for s in supported):
                warnings.append(
                    f"{prefix} .meta.profile: profil non supporté localement "
                    f"→ {profile_text!r}"
                )

    return warnings


# ── Validation schéma via fhir.resources ─────────────────────────────────────

_SUPPORTED_FHIR_LIB_VERSIONS: frozenset[str] = frozenset({"R4", "R4B"})


def _is_rule_enabled(rules: dict, rule_id: str) -> bool:
    """Vérifie qu'une règle est présente dans les règles chargées.

    Toutes les règles retournées par load_rules sont déjà activées (le loader
    rejette les règles avec enabled=False). Il suffit de vérifier la présence.
    """
    for rule in rules.get("rules", ()):
        if hasattr(rule, "get") and rule.get("id") == rule_id:
            return True
    return False


def validate_resource_schema(
    bundle: dict,
    *,
    fhir_version: str = "R4",
) -> tuple[list[str], list[str]]:
    """Valide chaque ressource du bundle contre les modèles Pydantic de fhir.resources.

    La bibliothèque fhir.resources v8+ implémente FHIR R4B. Des champs R4 absents
    de R4B (ex. Procedure.performedPeriod, MedicationRequest.medicationCodeableConcept)
    produisent des avertissements — pas des erreurs bloquantes — car ils reflètent un
    écart de version, pas une anomalie de données.

    Cas traités :
      - Bibliothèque absente              → warning, retour anticipé
      - Version FHIR non prise en charge  → warning [FHIR_RESOURCE_SCHEMA_VALID], retour anticipé
      - Type inconnu de la bibliothèque   → warning [FHIR_RESOURCE_TYPE_SUPPORTED]
      - Ressource mal formée              → warning [FHIR_RESOURCE_SCHEMA_VALID]
                                            (seuls les chemins de champs sont inclus, jamais les valeurs)
      - Profil non validable localement   → note implicite (validation structurelle uniquement)
      - Erreur interne du validateur      → warning [FHIR_RESOURCE_SCHEMA_VALID]

    Le bundle source n'est jamais modifié. Aucune valeur de champ (pouvant contenir
    des données personnelles) n'est copiée dans les résultats.

    Args:
        bundle      : dictionnaire du bundle FHIR (lecture seule).
        fhir_version: version FHIR déclarée dans les règles ("R4" ou "R4B").

    Returns:
        (errors, warnings) — errors reste toujours vide (toutes anomalies → warnings).
    """
    warnings: list[str] = []

    if not _FHIR_LIB_AVAILABLE:
        warnings.append(
            "[FHIR_RESOURCE_SCHEMA_VALID] bundle: "
            "bibliothèque fhir.resources absente — validation schéma ignorée"
        )
        return [], warnings

    if fhir_version.upper() not in _SUPPORTED_FHIR_LIB_VERSIONS:
        warnings.append(
            f"[FHIR_RESOURCE_SCHEMA_VALID] bundle: version FHIR non prise en charge "
            f"par le validateur local — {fhir_version!r} "
            f"(supportées : {sorted(_SUPPORTED_FHIR_LIB_VERSIONS)})"
        )
        return [], warnings

    for position, resource in _iter_indexed_resources(bundle):
        resource_type = resource.get("resourceType")
        prefix = _entry_prefix(position, resource)

        if not isinstance(resource_type, str) or not resource_type:
            continue  # déjà signalé par check_entry_resources

        # Résolution du modèle Pydantic fhir.resources
        try:
            model_class = _fhir_get_model(resource_type)
        except ValueError:
            warnings.append(
                f"[FHIR_RESOURCE_TYPE_SUPPORTED] {prefix}: "
                f"type non reconnu par fhir.resources — {resource_type!r}"
            )
            continue
        except Exception as exc:  # noqa: BLE001
            warnings.append(
                f"[FHIR_RESOURCE_SCHEMA_VALID] {prefix}: "
                f"erreur interne de résolution du modèle — {type(exc).__name__}"
            )
            continue

        # Validation structurelle (jamais de profil réseau)
        try:
            model_class.model_validate(resource)
        except _PydanticValidationError as exc:
            issue_count = len(exc.errors())
            # Seuls les chemins de champs (loc) sont extraits — jamais les valeurs (input)
            field_paths = [
                ".".join(str(p) for p in err["loc"])
                for err in exc.errors()[:4]
            ]
            warnings.append(
                f"[FHIR_RESOURCE_SCHEMA_VALID] {prefix}: "
                f"{issue_count} problème(s) de schéma R4B "
                f"(champ(s) : {', '.join(field_paths)})"
            )
        except Exception as exc:  # noqa: BLE001
            warnings.append(
                f"[FHIR_RESOURCE_SCHEMA_VALID] {prefix}: "
                f"erreur interne du validateur — {type(exc).__name__}"
            )

    return [], warnings


# ── Point d'entrée principal ──────────────────────────────────────────────────


def validate_fhir_bundle(
    path: str | None,
    *,
    bundle_expected: bool = True,
    rules: dict | None = None,
) -> tuple[VerificationStatus, list[str], list[str], str | None]:
    """Valide un bundle FHIR R4 selon les règles YAML configurées.

    Le bundle source n'est jamais modifié. Aucune conclusion clinique n'est tirée.

    Pipeline :
      1. Vérification de présence (path / bundle_expected)
      2. Chargement et parsing JSON
      3. Validation structure de base (resourceType, type, entry)
      4. Vérification des entrées (resourceType de chaque ressource)
      5. Ressources obligatoires
      6. Cardinalités minimales
      7. Champs obligatoires par ressource
      8. Statuts Coverage
      9. Références internes via BundleIndex
     10. Profils supportés (warnings)
     11. Ressources optionnelles absentes (warnings)

    Args:
        path           : chemin vers le fichier bundle FHIR. None si absent.
        bundle_expected: True si un bundle est attendu pour ce dossier.
        rules          : règles de validation (depuis fhir_rules.yaml).
                         None → règles minimales par défaut.

    Returns:
        (status, errors, warnings, profile_checked)
          - status          : PASS / NEEDS_REVIEW / FAIL
          - errors          : messages d'erreur avec emplacement FHIRPath-style
          - warnings        : avertissements non bloquants avec emplacement
          - profile_checked : profil validé (ex. "R4"), None si échec avant validation
    """
    effective_rules = rules or {}
    profile: str = effective_rules.get("profile", _DEFAULT_PROFILE)
    required_types: list[str] = effective_rules.get(
        "required_resource_types", _DEFAULT_REQUIRED_TYPES
    )
    optional_types: list[str] = effective_rules.get("optional_resource_types", [])
    accepted_bundle_types: list[str] = effective_rules.get(
        "bundle_types_accepted", _DEFAULT_BUNDLE_TYPES
    )

    # Étape 1 — vérification de présence
    if path is None and not bundle_expected:
        return VerificationStatus.PASS, [], [], None

    if path is None:
        return (
            VerificationStatus.FAIL,
            ["bundle: bundle FHIR attendu mais aucun chemin fourni"],
            [],
            None,
        )

    # Étape 2 — chargement et parsing JSON
    bundle, load_errors = load_fhir_bundle(path)
    if load_errors:
        return VerificationStatus.FAIL, load_errors, [], None
    assert bundle is not None  # garanti par load_fhir_bundle quand load_errors est vide

    all_errors: list[str] = []
    all_warnings: list[str] = []

    # Étape 3 — structure de base
    struct_errors, struct_warnings = validate_bundle_structure(
        bundle, accepted_types=accepted_bundle_types
    )
    all_errors.extend(struct_errors)
    all_warnings.extend(struct_warnings)

    # Arrêt précoce si la structure est invalide (les étapes suivantes dépendent d'elle)
    if all_errors:
        return VerificationStatus.FAIL, all_errors, all_warnings, None

    # Étape 4 — vérification des entrées
    entry_errors, entry_warnings = check_entry_resources(bundle, effective_rules)
    all_errors.extend(entry_errors)
    all_warnings.extend(entry_warnings)

    # Étape 5 — ressources obligatoires
    req_errors, _ = check_required_resources(bundle, required_types)
    all_errors.extend(req_errors)

    # Étape 6 — cardinalités minimales
    all_errors.extend(check_min_cardinalities(bundle, effective_rules))

    # Étape 7 — champs obligatoires par ressource
    all_errors.extend(check_resource_fields(bundle, effective_rules))

    # Étape 8 — statuts Coverage
    all_errors.extend(_check_coverage_status(bundle, effective_rules))

    # Étape 9 — références internes via BundleIndex
    all_errors.extend(check_internal_references(bundle, effective_rules))

    # Étape 10 — profils supportés (warnings uniquement)
    all_warnings.extend(check_supported_profiles(bundle, effective_rules))

    # Étape 11 — validation schéma par ressource via fhir.resources (si règle présente)
    if _is_rule_enabled(effective_rules, "FHIR_RESOURCE_SCHEMA_VALID"):
        _, schema_warnings = validate_resource_schema(bundle, fhir_version=profile)
        all_warnings.extend(schema_warnings)

    # Étape 12 — ressources optionnelles absentes (warnings)
    if optional_types:
        present = set(extract_resource_types(bundle))
        for opt_type in optional_types:
            if opt_type not in present:
                all_warnings.append(
                    f"bundle.entry: ressource optionnelle absente — {opt_type!r}"
                )

    if all_errors:
        return VerificationStatus.FAIL, all_errors, all_warnings, profile

    if all_warnings:
        return VerificationStatus.NEEDS_REVIEW, [], all_warnings, profile

    return VerificationStatus.PASS, [], [], profile
