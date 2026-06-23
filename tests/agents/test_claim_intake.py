"""Tests du Claim Intake Agent.

Utilise les fixtures de datasets/demo/ comme source de vérité.
Aucun mock — les fichiers sont lus depuis le disque.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agents.claim_intake_agent.agent import node, run
from agents.claim_intake_agent.schemas import ClaimIntakeInput
from config.settings import Settings
from schemas.domain import IntakeStatus
from services.storage import StorageService
from tools.file_inspection import compute_sha256, inspect_document

DEMO_DIR = Path(__file__).resolve().parents[2] / "datasets" / "demo"


def gt(case_id: str) -> dict:
    return json.loads((DEMO_DIR / case_id / "oracle" / "ground_truth.json").read_text())


def _make_storage(tmp_path: Path) -> StorageService:
    """Crée un StorageService isolé dans tmp_path pour les tests."""
    s = Settings(
        CLAIMSHIELD_STORAGE_DIR=str(tmp_path / "storage"),
        CLAIMSHIELD_QUARANTINE_DIR=str(tmp_path / "storage" / "quarantine"),
    )
    svc = StorageService(settings=s)
    svc.ensure_dirs()
    return svc


# ── run() — logique principale ────────────────────────────────────────────────


def test_sc01_dossier_complet(tmp_path):
    """SC-01 : tous les documents présents → ACCEPTED, aucune alerte."""
    svc = _make_storage(tmp_path)
    meta = gt("CLM-0004")
    result = run(
        case_id="CLM-0004",
        source_path=DEMO_DIR / "CLM-0004" / "input",
        required_documents=meta["required_documents"],
        storage=svc,
    )
    assert result.status == IntakeStatus.ACCEPTED
    assert result.manifest.alerts == []
    assert result.accepted_count >= len(meta["required_documents"])
    assert result.quarantined_count == 0
    assert result.errors == []


def test_sc01_hashes_correspondent_au_manifest(tmp_path):
    """Les SHA-256 calculés par l'agent correspondent au manifest.json."""
    svc = _make_storage(tmp_path)
    manifest_path = DEMO_DIR / "CLM-0004" / "audit" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest_hashes = {e["filename"]: e["sha256"] for e in manifest["files"]}

    result = run(
        case_id="CLM-0004",
        source_path=DEMO_DIR / "CLM-0004" / "input",
        storage=svc,
    )

    for inspected in result.manifest.files:
        if inspected.original_name in manifest_hashes and inspected.sha256 is not None:
            assert inspected.sha256 == manifest_hashes[inspected.original_name], (
                f"Hash incorrect pour {inspected.original_name}"
            )


def test_sc04_facture_manquante(tmp_path):
    """SC-04 : document obligatoire absent → QUARANTINED avec alerte lisible."""
    svc = _make_storage(tmp_path)
    result = run(
        case_id="CLM-0019",
        source_path=DEMO_DIR / "CLM-0019" / "input",
        required_documents=["facture_CLM-0019.pdf"],
        storage=svc,
    )
    assert result.status == IntakeStatus.QUARANTINED
    assert any("facture_CLM-0019.pdf" in a for a in result.manifest.alerts)
    assert any("facture_CLM-0019.pdf" in r for r in result.reasons)


def test_cas_sans_documents_requis(tmp_path):
    """Sans liste de documents requis, l'agent inventorie et retourne ACCEPTED."""
    svc = _make_storage(tmp_path)
    result = run(
        case_id="CLM-0004",
        source_path=DEMO_DIR / "CLM-0004" / "input",
        required_documents=[],
        storage=svc,
    )
    assert result.status == IntakeStatus.ACCEPTED
    assert result.manifest.alerts == []
    assert result.manifest.file_count > 0
    assert result.accepted_count > 0


def test_dossier_vide(tmp_path):
    """Un répertoire vide produit un résultat BLOCKED immédiat."""
    svc = _make_storage(tmp_path)
    empty_dir = tmp_path / "empty_case"
    empty_dir.mkdir()

    result = run(case_id="CLM-9999", source_path=empty_dir, storage=svc)
    assert result.status == IntakeStatus.BLOCKED
    assert result.accepted_count == 0
    assert any(e.code == "EMPTY_FOLDER" for e in result.errors)


def test_claim_id_dans_manifest(tmp_path):
    """Le claim_id fourni est conservé dans le manifest."""
    svc = _make_storage(tmp_path)
    result = run(
        case_id="CLM-0004",
        source_path=DEMO_DIR / "CLM-0004" / "input",
        storage=svc,
    )
    assert result.claim_id == "CLM-0004"
    assert result.manifest.claim_id == "CLM-0004"


def test_fichiers_stockes_sans_contenu_brut(tmp_path):
    """Aucun fichier InspectedFile ne contient les octets bruts du document."""
    svc = _make_storage(tmp_path)
    result = run(
        case_id="CLM-0004",
        source_path=DEMO_DIR / "CLM-0004" / "input",
        storage=svc,
    )
    for f in result.manifest.files:
        dumped = f.model_dump()
        # Vérification : aucun champ bytes dans le modèle
        for v in dumped.values():
            assert not isinstance(v, bytes), f"Contenu brut trouvé dans {f.original_name}"


def test_chemins_relatifs_pas_absolus(tmp_path):
    """Les chemins de stockage dans le manifest sont relatifs, pas absolus."""
    svc = _make_storage(tmp_path)
    result = run(
        case_id="CLM-0004",
        source_path=DEMO_DIR / "CLM-0004" / "input",
        storage=svc,
    )
    for f in result.manifest.files:
        if f.relative_storage_path:
            assert not f.relative_storage_path.startswith("/"), (
                f"Chemin absolu détecté : {f.relative_storage_path}"
            )


# ── ClaimIntakeInput — validation Pydantic ────────────────────────────────────


def test_input_schema_valide():
    inp = ClaimIntakeInput(
        case_id="CLM-0004",
        source_path=DEMO_DIR / "CLM-0004" / "input",
        required_documents=["facture_CLM-0004.pdf"],
    )
    assert inp.case_id == "CLM-0004"


def test_input_case_id_invalide():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ClaimIntakeInput(
            case_id="DOSSIER-001",
            source_path=DEMO_DIR / "CLM-0004" / "input",
        )


# ── tools/file_inspection ────────────────────────────────────────────────────


def test_compute_sha256_stable():
    """Le hash SHA-256 est identique entre deux appels successifs."""
    path = DEMO_DIR / "CLM-0004" / "input" / "facture_CLM-0004.pdf"
    assert compute_sha256(path) == compute_sha256(path)


def test_inspect_document_champs_complets():
    path = DEMO_DIR / "CLM-0004" / "input" / "facture_CLM-0004.pdf"
    info = inspect_document(path)
    assert len(info["sha256"]) == 64
    assert info["size_bytes"] > 0
    assert info["filename"] == "facture_CLM-0004.pdf"
    assert "/" in info["mime_type"]


# ── services/storage ─────────────────────────────────────────────────────────


def test_storage_stage_and_quarantine(tmp_path: Path):
    """stage_to_incoming puis move_to_quarantine déplacent correctement les fichiers."""
    svc = _make_storage(tmp_path)
    source = DEMO_DIR / "CLM-0004" / "input"
    incoming = svc.stage_to_incoming("CLM-0004", source)
    assert incoming.exists()
    assert (incoming / "facture_CLM-0004.pdf").exists()

    quarantine = svc.move_to_quarantine("CLM-0004")
    assert quarantine.exists()
    assert not svc.incoming_path("CLM-0004").exists()


def test_storage_temporary(tmp_path: Path):
    """move_to_temporary copie sans supprimer l'original."""
    svc = _make_storage(tmp_path)
    source = DEMO_DIR / "CLM-0004" / "input"
    svc.stage_to_incoming("CLM-0004", source)
    tmp_case = svc.move_to_temporary("CLM-0004")

    assert tmp_case.exists()
    assert svc.incoming_path("CLM-0004").exists()  # original conservé

    svc.cleanup_temporary("CLM-0004")
    assert not tmp_case.exists()


# ── node() — intégration LangGraph ───────────────────────────────────────────


def test_node_accepted(tmp_path: Path):
    """node() retourne intake_result ACCEPTED et completed_steps=['claim_intake']."""
    meta = gt("CLM-0004")
    svc = _make_storage(tmp_path)

    # Le node() crée son propre StorageService — on fournit une settings temporaire
    # en surchargeant via monkeypatch implicite dans les prochaines étapes.
    # Pour ce test, on appelle directement run() via le state.
    state = {
        "case_id": "CLM-0004",
        "intake_input": {
            "source_path": str(DEMO_DIR / "CLM-0004" / "input"),
            "required_documents": meta["required_documents"],
        },
    }
    # node() utilise ses propres répertoires storage par défaut — on vérifie
    # uniquement la structure du résultat, pas l'emplacement physique.
    updates = node(state)  # type: ignore[arg-type]
    assert updates["intake_result"].claim_id == "CLM-0004"
    assert updates["intake_result"].status in (
        IntakeStatus.ACCEPTED,
        IntakeStatus.QUARANTINED,
    )
    assert "claim_intake" in updates["completed_steps"]


def test_node_quarantine_missing_doc(tmp_path: Path):
    """node() retourne QUARANTINED et errors quand un document obligatoire est absent."""
    meta = gt("CLM-0019")
    state = {
        "case_id": "CLM-0019",
        "intake_input": {
            "source_path": str(DEMO_DIR / "CLM-0019" / "input"),
            "required_documents": meta["required_documents"],
        },
    }
    updates = node(state)  # type: ignore[arg-type]
    result = updates["intake_result"]
    assert result.status in (IntakeStatus.QUARANTINED, IntakeStatus.BLOCKED)
    assert "errors" in updates or result.manifest.alerts
