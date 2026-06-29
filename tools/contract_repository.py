"""Référentiel local de contrats synthétiques.

Le référentiel complet reste côté outil. Les agents ne reçoivent qu'un snapshot
du contrat demandé par numéro de police.
"""
from __future__ import annotations

import json
from datetime import date
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import ConfigDict, Field, ValidationError, field_validator, model_validator

from schemas.domain import StrictModel

CONTRACTS_PATH = Path("datasets/reference/contracts.json")


class ContractRepositoryError(ValueError):
    """Erreur contrôlée de chargement du référentiel de contrats."""


class SyntheticContract(StrictModel):
    """Contrat synthétique validé localement."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True, frozen=True)

    policy_number: str = Field(..., min_length=1)
    patient_pseudonym: str = Field(..., min_length=1)
    start_date: date
    end_date: date
    status: str = Field(..., min_length=1)
    currency: str = Field(..., min_length=3, max_length=3)
    annual_limit: Decimal = Field(..., ge=Decimal("0"))
    covered_procedure_codes: tuple[str, ...] = Field(default_factory=tuple)
    excluded_procedure_codes: tuple[str, ...] = Field(default_factory=tuple)
    preauthorization_required_for: tuple[str, ...] = Field(default_factory=tuple)
    contract_version: str = Field(..., min_length=1)

    @field_validator(
        "covered_procedure_codes",
        "excluded_procedure_codes",
        "preauthorization_required_for",
        mode="before",
    )
    @classmethod
    def _list_to_tuple(cls, value: object) -> tuple[str, ...]:
        if value is None:
            return tuple()
        if not isinstance(value, list | tuple):
            raise ValueError("liste de codes attendue")
        return tuple(str(item).strip() for item in value if str(item).strip())

    @field_validator("status")
    @classmethod
    def _known_status(cls, value: str) -> str:
        status = value.strip().casefold()
        if status not in {"active", "inactive"}:
            raise ValueError("status doit être active ou inactive")
        return status

    @field_validator("currency")
    @classmethod
    def _currency_upper(cls, value: str) -> str:
        return value.strip().upper()

    @model_validator(mode="after")
    def _dates_are_ordered(self) -> "SyntheticContract":
        if self.end_date < self.start_date:
            raise ValueError("end_date ne peut pas précéder start_date")
        return self

    def to_agent_snapshot(self) -> dict[str, Any]:
        """Retourne uniquement les champs utiles à l'agent pour ce contrat."""
        return {
            "policy_number": self.policy_number,
            "patient_id": self.patient_pseudonym,
            "coverage_start_date": self.start_date.isoformat(),
            "coverage_end_date": self.end_date.isoformat(),
            "policy_active": self.status == "active",
            "currency": self.currency,
            "ceiling_amount": str(self.annual_limit),
            "covered_procedure_codes": list(self.covered_procedure_codes),
            "excluded_procedure_codes": list(self.excluded_procedure_codes),
            "preauthorization_required_for": list(self.preauthorization_required_for),
            "contract_version": self.contract_version,
        }


@lru_cache(maxsize=1)
def load_contracts(path: str | Path = CONTRACTS_PATH) -> tuple[SyntheticContract, ...]:
    """Charge et valide le référentiel synthétique local."""
    contracts_path = Path(path)
    if not contracts_path.exists():
        raise ContractRepositoryError(f"Référentiel de contrats introuvable : {contracts_path}")

    try:
        raw = json.loads(contracts_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ContractRepositoryError(f"JSON contrats invalide : {exc}") from exc

    if not isinstance(raw, list):
        raise ContractRepositoryError("Le référentiel de contrats doit être une liste")

    contracts: list[SyntheticContract] = []
    seen_policy_numbers: set[str] = set()
    for index, item in enumerate(raw):
        try:
            contract = SyntheticContract.model_validate(item)
        except ValidationError as exc:
            raise ContractRepositoryError(f"Contrat invalide à l'index {index} : {exc}") from exc
        if contract.policy_number in seen_policy_numbers:
            raise ContractRepositoryError(
                f"Numéro de contrat dupliqué : {contract.policy_number}"
            )
        seen_policy_numbers.add(contract.policy_number)
        contracts.append(contract)

    return tuple(contracts)


def get_contract(policy_number: str, path: str | Path = CONTRACTS_PATH) -> SyntheticContract | None:
    """Retourne un contrat synthétique par numéro, ou None s'il est absent."""
    policy = policy_number.strip()
    for contract in load_contracts(path):
        if contract.policy_number == policy:
            return contract
    return None


def get_contract_snapshot(policy_number: str, path: str | Path = CONTRACTS_PATH) -> dict[str, Any] | None:
    """Retourne un snapshot agent pour un seul contrat, jamais le référentiel complet."""
    contract = get_contract(policy_number, path)
    return contract.to_agent_snapshot() if contract is not None else None
