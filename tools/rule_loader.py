"""Chargeur de règles YAML versionnées — partagé par tous les agents.

Frontière de sécurité :
- seuls les fichiers sous config/rules/ sont autorisés ;
- les chemins absolus externes et les traversées ``..`` sont refusés ;
- le YAML est lu avec ``safe_load`` et jamais exécuté ;
- le contenu est validé par Pydantic ;
- le résultat est immuable pendant l'exécution.
"""
from __future__ import annotations

import hashlib
from enum import Enum
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

RULES_DIR = Path("config/rules")


class RuleLoaderCode(str, Enum):
    RULE_FILE_NOT_FOUND = "RULE_FILE_NOT_FOUND"
    RULE_FILE_INVALID = "RULE_FILE_INVALID"
    RULE_VERSION_MISSING = "RULE_VERSION_MISSING"
    RULE_ID_DUPLICATE = "RULE_ID_DUPLICATE"
    RULE_DISABLED = "RULE_DISABLED"
    RULE_PATH_NOT_ALLOWED = "RULE_PATH_NOT_ALLOWED"


class RuleLoaderError(Exception):
    """Erreur contrôlée du chargeur de règles."""

    def __init__(self, code: RuleLoaderCode, message: str) -> None:
        self.code = code.value
        super().__init__(f"{code.value}: {message}")


class RuleFileNotFoundError(RuleLoaderError, FileNotFoundError):
    pass


class RuleFileInvalidError(RuleLoaderError, ValueError):
    pass


class RuleVersionMissingError(RuleFileInvalidError):
    pass


class RuleIdDuplicateError(RuleFileInvalidError):
    pass


class RuleDisabledError(RuleFileInvalidError):
    pass


class RulePathNotAllowedError(RuleLoaderError, ValueError):
    pass


_VALID_RULESET_STATUSES = {"active", "inactive"}

_KNOWN_RULE_IDS: dict[str, set[str]] = {
    "authorization_rules.yaml": {
        "PREAUTH_REQUIRED_BY_AMOUNT",
        "PREAUTH_REQUIRED_BY_VOLUME",
        "DENIAL_CODE_ALLOWED",
    },
    "coverage_rules.yaml": {
        "CONTRACT_EXISTS",
        "CONTRACT_ACTIVE",
        "SERVICE_DATE_ON_OR_AFTER_START",
        "SERVICE_DATE_ON_OR_BEFORE_END",
        "CURRENCY_MATCHES_CONTRACT",
        "REQUESTED_AMOUNT_POSITIVE",
        "REQUESTED_NOT_GREATER_THAN_TOTAL",
        "REQUESTED_NOT_GREATER_THAN_AVAILABLE_LIMIT",
        "PROCEDURE_CODE_COVERED",
        "PROCEDURE_CODE_NOT_EXCLUDED",
        "PREAUTHORIZATION_PRESENT_WHEN_REQUIRED",
    },
    "fhir_rules.yaml": {
        "FHIR_BUNDLE_STRUCTURE_VALID",
        "FHIR_RESOURCE_TYPE_SUPPORTED",
        "FHIR_REQUIRED_RESOURCES_PRESENT",
        "FHIR_MIN_CARDINALITY",
        "FHIR_REQUIRED_FIELDS_PRESENT",
        "FHIR_INTERNAL_REFERENCES_RESOLVE",
        "FHIR_PROFILE_SUPPORTED",
        "FHIR_COVERAGE_STATUS_ALLOWED",
    },
    "identity_rules.yaml": {
        "IDENTITY_PATIENT_ID_MATCH",
        "IDENTITY_PATIENT_NAME_PRESENT",
        "IDENTITY_MIN_SOURCES_AGREE",
    },
    "medical_codes.yaml": {
        "MEDICAL_CODE_EXACT_MATCH",
        "MEDICAL_CODE_KEYWORD_MATCH_REVIEW",
        "MEDICAL_CODE_UNKNOWN_REVIEW",
    },
}


class RuleDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., min_length=1)
    enabled: bool
    severity: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    operator: str | None = None
    value: Any | None = None
    left: str | None = None
    right: str | None = None

    @field_validator("id", "severity", "description")
    @classmethod
    def _strip_non_empty(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("champ vide interdit")
        return stripped


class RuleSetMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1)
    version: str = Field(..., min_length=1)
    effective_from: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    parameters: dict[str, Any] = Field(default_factory=dict)
    thresholds: dict[str, Any] = Field(default_factory=dict)
    result_codes: list[str] = Field(default_factory=list)
    status: str = Field(..., min_length=1)

    @field_validator("name", "version", "effective_from", "description", "status")
    @classmethod
    def _strip_non_empty(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("champ vide interdit")
        return stripped

    @field_validator("status")
    @classmethod
    def _known_status(cls, value: str) -> str:
        status = value.casefold()
        if status not in _VALID_RULESET_STATUSES:
            raise ValueError(f"statut inconnu : {value!r}")
        return status


class RuleFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ruleset: RuleSetMetadata
    rules: list[RuleDefinition] = Field(min_length=1)


def _error(code: RuleLoaderCode, message: str) -> RuleLoaderError:
    if code == RuleLoaderCode.RULE_FILE_NOT_FOUND:
        return RuleFileNotFoundError(code, message)
    if code == RuleLoaderCode.RULE_VERSION_MISSING:
        return RuleVersionMissingError(code, message)
    if code == RuleLoaderCode.RULE_ID_DUPLICATE:
        return RuleIdDuplicateError(code, message)
    if code == RuleLoaderCode.RULE_DISABLED:
        return RuleDisabledError(code, message)
    if code == RuleLoaderCode.RULE_PATH_NOT_ALLOWED:
        return RulePathNotAllowedError(code, message)
    return RuleFileInvalidError(code, message)


def _resolve_rule_path(filename: str) -> Path:
    """Résout filename sous RULES_DIR et refuse toute sortie de ce répertoire."""
    requested = Path(filename)
    if requested.is_absolute() or ".." in requested.parts or len(requested.parts) != 1:
        raise _error(
            RuleLoaderCode.RULE_PATH_NOT_ALLOWED,
            f"chemin de règle interdit : {filename!r}",
        )

    rules_root = RULES_DIR.resolve()
    path = (rules_root / requested).resolve()
    try:
        path.relative_to(rules_root)
    except ValueError as exc:
        raise _error(
            RuleLoaderCode.RULE_PATH_NOT_ALLOWED,
            f"chemin résolu hors config/rules : {filename!r}",
        ) from exc
    return path


def _read_yaml(path: Path) -> tuple[dict[str, Any], str]:
    if not path.exists() or not path.is_file():
        raise _error(
            RuleLoaderCode.RULE_FILE_NOT_FOUND,
            f"fichier de règles introuvable : {path.name}",
        )
    raw = path.read_bytes()
    file_hash = hashlib.sha256(raw).hexdigest()
    try:
        data = yaml.safe_load(raw.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise _error(
            RuleLoaderCode.RULE_FILE_INVALID,
            f"YAML invalide pour {path.name} : {exc}",
        ) from exc
    if not isinstance(data, dict):
        raise _error(
            RuleLoaderCode.RULE_FILE_INVALID,
            f"contenu invalide pour {path.name} : objet YAML attendu",
        )
    return data, file_hash


def _validate_rule_file(path: Path, data: dict[str, Any]) -> RuleFile:
    ruleset = data.get("ruleset")
    if not isinstance(ruleset, dict):
        raise _error(
            RuleLoaderCode.RULE_FILE_INVALID,
            f"ruleset manquant ou invalide dans {path.name}",
        )
    if "version" not in ruleset or not str(ruleset.get("version", "")).strip():
        raise _error(
            RuleLoaderCode.RULE_VERSION_MISSING,
            f"ruleset.version manquant dans {path.name}",
        )

    try:
        parsed = RuleFile.model_validate(data)
    except ValidationError as exc:
        raise _error(
            RuleLoaderCode.RULE_FILE_INVALID,
            f"format de règles invalide pour {path.name} : {exc}",
        ) from exc

    if parsed.ruleset.status != "active":
        raise _error(
            RuleLoaderCode.RULE_DISABLED,
            f"jeu de règles inactif : {parsed.ruleset.name}",
        )

    seen: set[str] = set()
    known_ids = _KNOWN_RULE_IDS.get(path.name, set())
    for rule in parsed.rules:
        if rule.id in seen:
            raise _error(
                RuleLoaderCode.RULE_ID_DUPLICATE,
                f"identifiant de règle dupliqué dans {path.name} : {rule.id}",
            )
        seen.add(rule.id)

        if known_ids and rule.id not in known_ids:
            raise _error(
                RuleLoaderCode.RULE_FILE_INVALID,
                f"règle inconnue dans {path.name} : {rule.id}",
            )
        if not rule.enabled:
            raise _error(
                RuleLoaderCode.RULE_DISABLED,
                f"règle désactivée dans {path.name} : {rule.id}",
            )

    return parsed


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(v) for v in value)
    return value


@lru_cache(maxsize=16)
def load_rules(filename: str) -> Mapping[str, Any]:
    """Charge un fichier YAML depuis config/rules/.

    Retourne un mapping immuable. Toute erreur levée est contrôlée et porte un
    code stable via ``exc.code``.
    """
    path = _resolve_rule_path(filename)
    data, file_hash = _read_yaml(path)
    parsed = _validate_rule_file(path, data)

    normalized: dict[str, Any] = parsed.model_dump()
    normalized["version"] = parsed.ruleset.version
    normalized.update(parsed.ruleset.parameters)
    normalized.update(parsed.ruleset.thresholds)
    normalized["result_codes"] = parsed.ruleset.result_codes
    normalized["ruleset_status"] = parsed.ruleset.status
    normalized["rule_file_hash"] = file_hash
    return _freeze(normalized)


def get_rule_version(filename: str) -> str:
    """Retourne la version du fichier de règles."""
    return str(load_rules(filename)["version"])
