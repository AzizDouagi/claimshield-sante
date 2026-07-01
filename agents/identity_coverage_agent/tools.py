"""Outils @tool du identity_coverage_agent — wrappers read-only sur les règles locales."""
from __future__ import annotations

from langchain_core.tools import tool

from tools.contract_repository import get_contract
from tools.coverage_rules import evaluate_coverage_contract
from tools.identity_matching import match_identity_contract
from tools.rule_loader import get_rule_version


@tool
def charger_contrat(policy_number: str) -> dict:
    """Charge un snapshot minimal de contrat depuis le référentiel local."""
    contract = get_contract(policy_number)
    if contract is None:
        return {"found": False, "policy_number": policy_number}
    snapshot = contract.to_agent_snapshot()
    return {
        "found": True,
        "policy_number": snapshot.get("policy_number"),
        "status": snapshot.get("status"),
        "coverage_start_date": snapshot.get("start_date"),
        "coverage_end_date": snapshot.get("end_date"),
        "covered_procedure_codes": snapshot.get("covered_procedure_codes", []),
    }


@tool
def verifier_identite_contrat(patient_pseudonym: str, policy_number: str) -> dict:
    """Vérifie la concordance patient/contrat sans exposer d'identité brute."""
    contract = get_contract(policy_number)
    candidates = [contract.to_agent_snapshot()] if contract is not None else []
    result = match_identity_contract(
        dossier_patient_pseudonym=patient_pseudonym,
        policy_number=policy_number,
        candidate_contracts=candidates,
        rule_version=get_rule_version("identity_rules.yaml"),
    )
    return result.model_dump(mode="json")


@tool
def verifier_couverture_contrat(
    policy_number: str,
    service_date: str,
    requested_amount: str,
    total_amount: str,
    currency: str,
    procedure_codes: list[str],
    preauthorization_reference: str | None = None,
) -> dict:
    """Évalue les règles de couverture active/inactive depuis le référentiel local."""
    contract = get_contract(policy_number)
    snapshot = contract.to_agent_snapshot() if contract is not None else None
    result = evaluate_coverage_contract(
        contract=snapshot,
        service_date=service_date,
        requested_amount=requested_amount,
        total_amount=total_amount,
        currency=currency,
        procedure_codes=procedure_codes,
        preauthorization_reference=preauthorization_reference,
    )
    return result.model_dump(mode="json")
