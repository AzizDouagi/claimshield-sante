"""Tests d'interruption/reprise HITL — ``human_review/``.

Construit un graphe LangGraph **minimal et autonome**, distinct du graphe de
production (``graph/workflow.py``/``graph/technical_nodes.py``). Les deux
graphes partagent désormais littéralement les mêmes noms d'action
(``APPROVE``/``MODIFY``/``REJECT``/``RETRY``, depuis le renommage de
``NEEDS_MORE_INFO`` en ``RETRY`` côté production) mais restent deux
mécanismes de validation distincts : ce graphe de test valide via
``human_review.service.validate_and_audit_human_decision`` (Pydantic strict,
justification toujours obligatoire), tandis que le graphe de production
valide via ``graph.technical_nodes._validate_human_decision`` (``TypedDict``,
``comment`` optionnel). Ce graphe de test exerce le cycle
``interrupt()``/``Command(resume=...)`` avec les 4 actions de
``human_review.models.ReviewAction`` — il prouve que le contrat
``human_review/`` est prêt pour un câblage complet dans le graphe de
production (étape 14, pas encore réalisée) sans modifier ce dernier.

La « modification d'un montant » (action ``MODIFY``) n'est jamais un champ de
``HumanDecision`` (schéma ``extra="forbid"``, volontairement minimal et
générique — voir ``human_review/models.py``) : le nœud de test extrait une clé
``modification`` du payload de reprise brut *avant* validation de la décision,
puis ne l'applique qu'après acceptation de celle-ci (justification obligatoire
déjà garantie par ``HumanDecision.justification``). Une décision refusée
(justification absente) ne modifie donc jamais le montant.
"""
from __future__ import annotations

from typing import Any, TypedDict

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from human_review.models import ReviewAction
from human_review.service import (
    HumanDecisionValidationError,
    build_human_review_payload,
    validate_and_audit_human_decision,
)

# ── Graphe de test minimal ────────────────────────────────────────────────────


class _TestState(TypedDict, total=False):
    case_id: str
    amount_requested: str
    alerts: list[str]
    errors: list[str]
    human_decision: dict
    audit_trail: list
    # ocr_result : uniquement utilisé par TestMinimalStateNoRawContent, pour
    # prouver que le texte OCR complet n'est jamais reproduit ailleurs dans
    # le state — build_human_review_payload n'en extrait que le statut.
    ocr_result: Any


def _node_await_human_review(state: _TestState) -> dict:
    """Suspend via ``interrupt()`` puis valide/audite la décision de reprise.

    Sépare strictement deux préoccupations : la décision elle-même (validée
    contre ``HumanDecision``, jamais de champ métier additionnel) et la
    modification demandée (``modification``, appliquée seulement après
    acceptation de la décision — jamais avant).
    """
    payload = build_human_review_payload(state)
    raw = interrupt(payload.model_dump(mode="json"))

    raw_mapping = dict(raw) if isinstance(raw, dict) else {}
    modification = raw_mapping.pop("modification", None)
    decision, audit_event = validate_and_audit_human_decision(raw_mapping)

    updates: dict[str, Any] = {
        "human_decision": decision.model_dump(mode="json"),
        "audit_trail": [audit_event],
    }
    if decision.action is ReviewAction.MODIFY and modification:
        updates.update(modification)
    return updates


def _build_app():
    graph = StateGraph(_TestState)
    graph.add_node("await_human_review", _node_await_human_review)
    graph.add_edge(START, "await_human_review")
    graph.add_edge("await_human_review", END)
    return graph.compile(checkpointer=InMemorySaver())


def _initial_state(case_id: str) -> dict:
    return {
        "case_id": case_id,
        "amount_requested": "500.00",
        "alerts": ["Revue humaine requise — écart de montant détecté."],
        "errors": [],
    }


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}


def _resume(case_id: str, action: str, **extra: Any) -> dict:
    return {"case_id": case_id, "actor": "reviewer@example.com", "action": action, **extra}


# ── Interruption ──────────────────────────────────────────────────────────────


class TestInterruptOccurs:
    def test_pipeline_interrupts_with_payload(self):
        app = _build_app()
        config = _config("CLM-9001")

        result = app.invoke(_initial_state("CLM-9001"), config=config)

        assert "__interrupt__" in result
        payload = result["__interrupt__"][0].value
        assert payload["case_id"] == "CLM-9001"
        assert set(payload["options"]) == {action.value for action in ReviewAction}
        assert "Revue humaine requise" in payload["summary"][0]


# ── APPROVE ───────────────────────────────────────────────────────────────────


class TestApprove:
    def test_approve_resumes_and_produces_audit(self):
        app = _build_app()
        config = _config("CLM-9010")
        app.invoke(_initial_state("CLM-9010"), config=config)

        result = app.invoke(
            Command(resume=_resume("CLM-9010", "APPROVE", justification="Dossier conforme.")),
            config=config,
        )

        assert "__interrupt__" not in result
        assert result["human_decision"]["action"] == "APPROVE"
        assert result["human_decision"]["justification"] == "Dossier conforme."
        assert result["audit_trail"][0].outcome == "APPROVE"
        assert result["audit_trail"][0].case_id == "CLM-9010"


# ── MODIFY ────────────────────────────────────────────────────────────────────


class TestModify:
    def test_modify_applies_amount_change_with_justification(self):
        app = _build_app()
        config = _config("CLM-9020")
        app.invoke(_initial_state("CLM-9020"), config=config)

        result = app.invoke(
            Command(
                resume=_resume(
                    "CLM-9020",
                    "MODIFY",
                    justification="Montant corrigé après vérification de la facture.",
                    modification={"amount_requested": "450.00"},
                )
            ),
            config=config,
        )

        assert "__interrupt__" not in result
        assert result["amount_requested"] == "450.00"
        assert result["human_decision"]["action"] == "MODIFY"
        assert (
            result["audit_trail"][0].details["justification"]
            == "Montant corrigé après vérification de la facture."
        )

    def test_modify_without_justification_is_refused_and_amount_unchanged(self):
        """Refus si justification absente : le montant n'est jamais modifié
        sans justification valide, même si la modification est fournie."""
        app = _build_app()
        config = _config("CLM-9021")
        app.invoke(_initial_state("CLM-9021"), config=config)

        with pytest.raises(HumanDecisionValidationError):
            app.invoke(
                Command(
                    resume={
                        "case_id": "CLM-9021",
                        "actor": "reviewer@example.com",
                        "action": "MODIFY",
                        "modification": {"amount_requested": "1.00"},
                    }
                ),
                config=config,
            )

        state = app.get_state(config)
        assert state.values.get("amount_requested") == "500.00"
        assert "human_decision" not in state.values

    def test_modify_with_empty_justification_is_refused(self):
        app = _build_app()
        config = _config("CLM-9022")
        app.invoke(_initial_state("CLM-9022"), config=config)

        with pytest.raises(HumanDecisionValidationError):
            app.invoke(
                Command(
                    resume=_resume(
                        "CLM-9022",
                        "MODIFY",
                        justification="",
                        modification={"amount_requested": "1.00"},
                    )
                ),
                config=config,
            )

        state = app.get_state(config)
        assert state.values.get("amount_requested") == "500.00"


# ── REJECT ────────────────────────────────────────────────────────────────────


class TestReject:
    def test_reject_resumes_and_produces_audit(self):
        app = _build_app()
        config = _config("CLM-9030")
        app.invoke(_initial_state("CLM-9030"), config=config)

        result = app.invoke(
            Command(resume=_resume("CLM-9030", "REJECT", justification="Preuves insuffisantes.")),
            config=config,
        )

        assert "__interrupt__" not in result
        assert result["human_decision"]["action"] == "REJECT"
        assert result["audit_trail"][0].outcome == "REJECT"

    def test_reject_without_justification_is_refused(self):
        app = _build_app()
        config = _config("CLM-9031")
        app.invoke(_initial_state("CLM-9031"), config=config)

        with pytest.raises(HumanDecisionValidationError):
            app.invoke(
                Command(resume={"case_id": "CLM-9031", "actor": "reviewer@example.com", "action": "REJECT"}),
                config=config,
            )


# ── RETRY ─────────────────────────────────────────────────────────────────────


class TestRetry:
    def test_retry_resumes_with_target_node_and_audit(self):
        app = _build_app()
        config = _config("CLM-9040")
        app.invoke(_initial_state("CLM-9040"), config=config)

        result = app.invoke(
            Command(
                resume=_resume(
                    "CLM-9040",
                    "RETRY",
                    justification="Pièce manquante à demander à nouveau.",
                    target_node="document_ocr",
                )
            ),
            config=config,
        )

        assert "__interrupt__" not in result
        assert result["human_decision"]["action"] == "RETRY"
        assert result["human_decision"]["target_node"] == "document_ocr"
        assert result["audit_trail"][0].details["target_node"] == "document_ocr"

    def test_retry_without_target_node_is_refused(self):
        app = _build_app()
        config = _config("CLM-9041")
        app.invoke(_initial_state("CLM-9041"), config=config)

        with pytest.raises(HumanDecisionValidationError):
            app.invoke(
                Command(resume=_resume("CLM-9041", "RETRY", justification="Pièce manquante.")),
                config=config,
            )

    def test_retry_without_justification_is_refused(self):
        app = _build_app()
        config = _config("CLM-9042")
        app.invoke(_initial_state("CLM-9042"), config=config)

        with pytest.raises(HumanDecisionValidationError):
            app.invoke(
                Command(resume=_resume("CLM-9042", "RETRY", target_node="document_ocr")),
                config=config,
            )


# ── Reprise avec le même thread_id ────────────────────────────────────────────


class TestResumeThreadId:
    def test_resume_with_same_thread_id_completes(self):
        app = _build_app()
        config = _config("CLM-9050")
        first = app.invoke(_initial_state("CLM-9050"), config=config)
        assert "__interrupt__" in first

        result = app.invoke(
            Command(resume=_resume("CLM-9050", "APPROVE", justification="Conforme.")),
            config=config,
        )

        assert "__interrupt__" not in result
        assert result["case_id"] == "CLM-9050"
        assert result["human_decision"]["action"] == "APPROVE"

    def test_resume_with_different_thread_id_does_not_resume(self):
        """Un thread_id différent ne retrouve aucun checkpoint en attente :
        LangGraph redémarre une exécution indépendante depuis START, qui
        réinterrompt aussitôt sur un state vierge — la décision fournie
        n'est jamais appliquée au dossier interrompu."""
        app = _build_app()
        config = _config("CLM-9051")
        app.invoke(_initial_state("CLM-9051"), config=config)

        other_config = _config("CLM-9052")
        result = app.invoke(
            Command(resume=_resume("CLM-9051", "APPROVE", justification="Conforme.")),
            config=other_config,
        )

        assert "__interrupt__" in result
        assert "human_decision" not in result


# ── Refus générique si justification absente (toutes actions) ───────────────


class TestMissingJustificationRefusedForEveryAction:
    @pytest.mark.parametrize(
        ("case_id", "action", "extra"),
        [
            ("CLM-9060", "APPROVE", {}),
            ("CLM-9061", "MODIFY", {"modification": {"amount_requested": "1.00"}}),
            ("CLM-9062", "REJECT", {}),
            ("CLM-9063", "RETRY", {"target_node": "document_ocr"}),
        ],
    )
    def test_refused_without_justification(self, case_id, action, extra):
        app = _build_app()
        config = _config(case_id)
        app.invoke(_initial_state(case_id), config=config)

        with pytest.raises(HumanDecisionValidationError):
            app.invoke(
                Command(resume={"case_id": case_id, "actor": "reviewer@example.com", "action": action, **extra}),
                config=config,
            )

        state = app.get_state(config)
        assert "human_decision" not in state.values
        assert "audit_trail" not in state.values or state.values["audit_trail"] == []


# ── Auto-approbation interdite ────────────────────────────────────────────────


class TestAutoApprovalForbidden:
    """Aucune décision — y compris APPROVE — ne peut jamais être appliquée
    sans validation complète (justification obligatoire, action reconnue) :
    aucun raccourci ni valeur par défaut ne permet une approbation
    automatique sans ``HumanDecision`` valide."""

    def test_approve_without_justification_is_not_auto_accepted(self):
        app = _build_app()
        config = _config("CLM-9070")
        app.invoke(_initial_state("CLM-9070"), config=config)

        with pytest.raises(HumanDecisionValidationError):
            app.invoke(
                Command(resume={"case_id": "CLM-9070", "actor": "reviewer@example.com", "action": "APPROVE"}),
                config=config,
            )

        state = app.get_state(config)
        assert "human_decision" not in state.values

    def test_no_default_action_exists_without_an_explicit_decision(self):
        """Un payload de reprise sans champ ``action`` est refusé — aucune
        valeur par défaut ne peut jamais représenter une décision."""
        app = _build_app()
        config = _config("CLM-9071")
        app.invoke(_initial_state("CLM-9071"), config=config)

        with pytest.raises(HumanDecisionValidationError):
            app.invoke(
                Command(resume={"case_id": "CLM-9071", "actor": "reviewer@example.com"}),
                config=config,
            )

    def test_unknown_action_value_is_never_treated_as_approval(self):
        app = _build_app()
        config = _config("CLM-9072")
        app.invoke(_initial_state("CLM-9072"), config=config)

        with pytest.raises(HumanDecisionValidationError):
            app.invoke(
                Command(
                    resume={
                        "case_id": "CLM-9072",
                        "actor": "reviewer@example.com",
                        "action": "AUTO_APPROVE",
                        "justification": "Tentative de contournement.",
                    }
                ),
                config=config,
            )

        state = app.get_state(config)
        assert "human_decision" not in state.values


# ── State minimal — jamais de document brut ni de texte OCR complet ─────────


class TestMinimalStateNoRawContent:
    def test_full_ocr_text_present_in_state_never_leaks_into_payload_decision_or_audit(self):
        """Un ``ocr_result`` réel portant un texte OCR complet (potentiel
        secret inclus) reste présent dans le ``ClaimState`` — mais ni le
        payload d'interruption, ni la décision validée, ni l'événement
        d'audit ne le reproduisent jamais."""
        from schemas.domain import DocumentType, ExtractionStatus, OcrSource, VerificationStatus
        from schemas.results import DocumentOcrResult

        secret_text = "password=hunter2 — texte OCR complet non destiné à l'audit"
        ocr_result = DocumentOcrResult(
            claim_id="CLM-9080",
            file_path="incoming/CLM-9080/facture.pdf",
            sha256="a" * 64,
            mime_type="application/pdf",
            extraction_status=ExtractionStatus.SUCCESS,
            status=VerificationStatus.NEEDS_REVIEW,
            document_type=DocumentType.INVOICE,
            ocr_source=OcrSource.PDF_TEXT,
            full_text=secret_text,
        )
        state = _initial_state("CLM-9080")
        state["ocr_result"] = ocr_result
        app = _build_app()
        config = _config("CLM-9080")

        first = app.invoke(state, config=config)
        assert "__interrupt__" in first
        payload = first["__interrupt__"][0].value
        assert secret_text not in str(payload)
        assert payload["evidence"].get("ocr_result") == "NEEDS_REVIEW"

        result = app.invoke(
            Command(resume=_resume("CLM-9080", "APPROVE", justification="Conforme.")),
            config=config,
        )

        assert secret_text not in str(result["human_decision"])
        assert secret_text not in str(result["audit_trail"][0].details)
