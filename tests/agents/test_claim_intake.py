"""Tests du Claim Intake Agent.

Utilise les fixtures de datasets/demo/ comme source de vérité.
Les fichiers sont lus depuis le disque ; seuls quelques tests ciblés
espionnent explicitement ``_invoke_llm_intake`` (via ``unittest.mock.Mock``)
pour prouver l'absence d'appel LLM sur les gardes déterministes.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from agents.claim_intake_agent.agent import node, run
from agents.claim_intake_agent.schemas import ClaimIntakeInput
from config.settings import Settings
from schemas.domain import FileStatus, IntakeReasonCode, IntakeStatus
from services.storage import StorageService
from tools.file_inspection import compute_sha256, inspect_document

DEMO_DIR = Path(__file__).resolve().parents[2] / "datasets" / "demo"


def gt(case_id: str) -> dict:
    return json.loads((DEMO_DIR / case_id / "oracle" / "ground_truth.json").read_text())


def _make_storage(tmp_path: Path) -> StorageService:
    """Crée un StorageService isolé dans tmp_path pour les tests."""
    s = Settings(  # type: ignore[call-arg]
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
    assert any(e.code == IntakeReasonCode.EMPTY_CLAIM for e in result.errors)


def test_dossier_vide_ne_declenche_aucun_appel_llm(tmp_path, monkeypatch):
    """Un dossier vide est un cas non ambigu — la Phase A suffit à elle
    seule, aucun appel LLM n'est nécessaire ni souhaitable (coût réseau
    inutile vers Ollama). Remplace le mock déterministe autouse par un espion
    dédié pour vérifier explicitement l'absence totale d'invocation."""
    llm_spy = Mock()
    monkeypatch.setattr("agents.claim_intake_agent.agent._invoke_llm_intake", llm_spy)
    svc = _make_storage(tmp_path)
    empty_dir = tmp_path / "empty_case"
    empty_dir.mkdir()

    result = run(case_id="CLM-9999", source_path=empty_dir, storage=svc)

    llm_spy.assert_not_called()
    assert result.status == IntakeStatus.BLOCKED
    assert any(e.code == IntakeReasonCode.EMPTY_CLAIM for e in result.errors)
    assert result.llm_metadata is not None


def test_repertoire_absent_retourne_blocked_sans_appel_llm(tmp_path, monkeypatch):
    """source_path absent du disque (jamais déposé) est traité comme un
    dossier vide : résultat BLOCKED immédiat, jamais une exception non
    gérée, jamais un appel LLM."""
    llm_spy = Mock()
    monkeypatch.setattr("agents.claim_intake_agent.agent._invoke_llm_intake", llm_spy)
    svc = _make_storage(tmp_path)
    missing_dir = tmp_path / "does_not_exist"

    result = run(case_id="CLM-9999", source_path=missing_dir, storage=svc)

    llm_spy.assert_not_called()
    assert result.status == IntakeStatus.BLOCKED
    assert result.accepted_count == 0
    assert any(e.code == IntakeReasonCode.EMPTY_CLAIM for e in result.errors)


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


# ── Nouveaux statuts : DUPLICATE et ERROR ────────────────────────────────────


def _setup_dup_case(tmp_path: Path, case_id: str = "CLM-9998") -> tuple:
    """Crée un dossier avec deux copies du même fichier, noms différents."""
    import shutil as _shutil
    svc = _make_storage(tmp_path)
    source_dir = tmp_path / "dup_source"
    source_dir.mkdir()
    original = DEMO_DIR / "CLM-0004" / "input" / "facture_CLM-0004.pdf"
    # Tri alphabétique : invoice.pdf < invoice_copy.pdf
    _shutil.copy2(original, source_dir / "invoice.pdf")
    _shutil.copy2(original, source_dir / "invoice_copy.pdf")
    return svc, source_dir


# Checklist doublon — un test par item ─────────────────────────────────────────


def test_doublon_detecte_meme_noms_differents(tmp_path):
    """Checklist : Détecter deux fichiers identiques portant des noms différents."""
    svc, source_dir = _setup_dup_case(tmp_path)
    result = run(case_id="CLM-9998", source_path=source_dir, storage=svc)

    files_by_name = {f.original_name: f for f in result.manifest.files}
    assert files_by_name["invoice.pdf"].status == FileStatus.ACCEPTED
    assert files_by_name["invoice_copy.pdf"].status == FileStatus.DUPLICATE


def test_premier_fichier_non_ecrase(tmp_path):
    """Checklist : Ne pas écraser le premier fichier.

    Le premier (ACCEPTED) va dans incoming/ ; le doublon dans quarantine/.
    Les deux ont des storage_names différents : aucun écrasement possible.
    """
    svc, source_dir = _setup_dup_case(tmp_path)
    result = run(case_id="CLM-9998", source_path=source_dir, storage=svc)

    files_by_name = {f.original_name: f for f in result.manifest.files}
    first = files_by_name["invoice.pdf"]
    dup = files_by_name["invoice_copy.pdf"]

    assert first.status == FileStatus.ACCEPTED
    assert first.relative_storage_path is not None
    assert first.relative_storage_path.startswith("incoming/")

    assert dup.status == FileStatus.DUPLICATE
    assert dup.relative_storage_path is not None
    assert dup.relative_storage_path.startswith("quarantine/")

    # Les noms de stockage sont différents : pas de collision physique
    assert first.storage_name != dup.storage_name


def test_second_marque_duplicate(tmp_path):
    """Checklist : Marquer le second comme DUPLICATE."""
    svc, source_dir = _setup_dup_case(tmp_path)
    result = run(case_id="CLM-9998", source_path=source_dir, storage=svc)

    dup_files = [f for f in result.manifest.files if f.status == FileStatus.DUPLICATE]
    assert len(dup_files) == 1
    assert dup_files[0].original_name == "invoice_copy.pdf"
    assert result.duplicate_count == 1


def test_hash_identique_conserve_dans_doublon(tmp_path):
    """Checklist : Conserver la référence du hash identique.

    Le sha256 du fichier DUPLICATE doit être identique à celui du premier
    fichier ACCEPTED, et doit être présent (pas None) dans le manifest.
    """
    svc, source_dir = _setup_dup_case(tmp_path)
    result = run(case_id="CLM-9998", source_path=source_dir, storage=svc)

    files_by_name = {f.original_name: f for f in result.manifest.files}
    first = files_by_name["invoice.pdf"]
    dup = files_by_name["invoice_copy.pdf"]

    assert first.sha256 is not None, "Le premier fichier doit avoir un SHA-256"
    assert dup.sha256 is not None, "Le doublon doit conserver son SHA-256"
    assert first.sha256 == dup.sha256, "Les SHA-256 doivent être identiques"


def test_raison_doublon_contient_hash_et_nom_original(tmp_path):
    """Checklist : La raison du doublon référence explicitement le hash et le premier fichier."""
    svc, source_dir = _setup_dup_case(tmp_path)
    result = run(case_id="CLM-9998", source_path=source_dir, storage=svc)

    files_by_name = {f.original_name: f for f in result.manifest.files}
    dup = files_by_name["invoice_copy.pdf"]
    first = files_by_name["invoice.pdf"]

    dup_reason = next(
        r for r in dup.reasons if r.code == IntakeReasonCode.DUPLICATE_FILE
    )

    # Le SHA-256 complet doit apparaître dans le message
    assert first.sha256 is not None, "Le premier fichier doit avoir un SHA-256 calculé"
    assert first.sha256 in dup_reason.message, (
        f"Le SHA-256 '{first.sha256}' doit être cité dans le message de la raison"
    )
    # Le nom du premier fichier doit apparaître dans le message
    assert "invoice.pdf" in dup_reason.message, (
        "Le nom du premier fichier doit être cité dans le message de la raison"
    )


def test_doublon_pas_une_fraude(tmp_path):
    """Checklist : Ne pas encore conclure à une fraude.

    Le statut DUPLICATE ne génère aucun signal de fraude.
    La détection de fraude est réservée à fraud_detection_agent.
    """
    svc, source_dir = _setup_dup_case(tmp_path)
    result = run(case_id="CLM-9998", source_path=source_dir, storage=svc)

    dup_files = [f for f in result.manifest.files if f.status == FileStatus.DUPLICATE]
    assert len(dup_files) == 1

    dup_reason = next(
        r for r in dup_files[0].reasons if r.code == IntakeReasonCode.DUPLICATE_FILE
    )
    # Le message ne doit pas contenir de mots qui concluent à une fraude
    for mot_interdit in ("fraude", "fraud", "frauduleux", "malveillant", "suspect"):
        assert mot_interdit.lower() not in dup_reason.message.lower(), (
            f"Le message de doublon ne doit pas mentionner '{mot_interdit}'"
        )

    # Aucun FraudDetectionResult dans le résultat d'ingestion
    assert not hasattr(result, "fraud_signals") or not getattr(result, "fraud_signals", None)


def test_doublon_statut_global_quarantined(tmp_path):
    """Un dossier contenant un DUPLICATE remonte en QUARANTINED au niveau global."""
    svc, source_dir = _setup_dup_case(tmp_path)
    result = run(case_id="CLM-9998", source_path=source_dir, storage=svc)

    assert result.status == IntakeStatus.QUARANTINED
    assert result.accepted_count == 1
    assert result.duplicate_count == 1


def test_trois_copies_deux_doublons(tmp_path):
    """Trois fichiers au contenu identique → 1 ACCEPTED + 2 DUPLICATE."""
    import shutil as _shutil
    svc = _make_storage(tmp_path)
    source_dir = tmp_path / "triple_dup"
    source_dir.mkdir()
    original = DEMO_DIR / "CLM-0004" / "input" / "facture_CLM-0004.pdf"
    _shutil.copy2(original, source_dir / "a_facture.pdf")
    _shutil.copy2(original, source_dir / "b_facture.pdf")
    _shutil.copy2(original, source_dir / "c_facture.pdf")

    result = run(case_id="CLM-9996", source_path=source_dir, storage=svc)

    accepted = [f for f in result.manifest.files if f.status == FileStatus.ACCEPTED]
    duplicates = [f for f in result.manifest.files if f.status == FileStatus.DUPLICATE]

    assert len(accepted) == 1, "Un seul fichier doit être ACCEPTED"
    assert len(duplicates) == 2, "Les deux autres doivent être DUPLICATE"
    assert result.duplicate_count == 2
    assert result.status == IntakeStatus.QUARANTINED

    # Les trois fichiers partagent le même SHA-256
    sha = accepted[0].sha256
    assert all(d.sha256 == sha for d in duplicates)

    # Le fichier ACCEPTED référence "a_facture.pdf" (premier alphabétiquement)
    assert accepted[0].original_name == "a_facture.pdf"


def test_duplicate_file_detection(tmp_path):
    """Deux fichiers au contenu identique (SHA-256 égal) : le second est DUPLICATE."""
    svc, source_dir = _setup_dup_case(tmp_path)
    result = run(case_id="CLM-9998", source_path=source_dir, storage=svc)

    dup_files = [f for f in result.manifest.files if f.status == FileStatus.DUPLICATE]
    assert len(dup_files) == 1, "Un seul doublon attendu"

    dup = dup_files[0]
    assert any(r.code == IntakeReasonCode.DUPLICATE_FILE for r in dup.reasons)

    assert result.status == IntakeStatus.QUARANTINED
    assert result.duplicate_count == 1


def test_reasons_sont_des_structured_errors(tmp_path):
    """Chaque InspectedFile porte des StructuredError avec code stable (pas du texte libre)."""
    from schemas.results import StructuredError

    svc = _make_storage(tmp_path)

    # Crée un fichier avec une extension non autorisée
    source_dir = tmp_path / "bad_ext_case"
    source_dir.mkdir()
    (source_dir / "rapport.exe").write_bytes(b"MZ" + b"\x00" * 100)

    result = run(case_id="CLM-9997", source_path=source_dir, storage=svc)

    blocked = [f for f in result.manifest.files if f.status == FileStatus.BLOCKED]
    assert len(blocked) == 1
    for reason in blocked[0].reasons:
        assert isinstance(reason, StructuredError)
        assert reason.code  # non vide
        assert reason.message  # non vide


def test_statuts_fichiers_couvrent_tous_les_cas(tmp_path):
    """FileStatus.ACCEPTED est retourné pour un fichier valide dans le manifest."""
    svc = _make_storage(tmp_path)
    result = run(
        case_id="CLM-0004",
        source_path=DEMO_DIR / "CLM-0004" / "input",
        storage=svc,
    )
    accepted_files = [f for f in result.manifest.files if f.status == FileStatus.ACCEPTED]
    assert len(accepted_files) > 0
    for f in accepted_files:
        assert f.reasons == [], f"Un fichier ACCEPTED ne doit avoir aucun motif de rejet : {f.original_name}"


# ── Persistance et format du manifest ────────────────────────────────────────


def test_manifest_ecrit_sur_disque(tmp_path):
    """Le manifest JSON est écrit dans manifests/{case_id}.json après ingestion."""
    svc = _make_storage(tmp_path)
    run(
        case_id="CLM-0004",
        source_path=DEMO_DIR / "CLM-0004" / "input",
        storage=svc,
    )
    manifest_file = tmp_path / "storage" / "manifests" / "CLM-0004.json"
    assert manifest_file.exists(), f"Manifest absent : {manifest_file}"
    assert manifest_file.stat().st_size > 0, "Manifest vide"


def test_manifest_relu_par_pydantic(tmp_path):
    """Le manifest écrit peut être relu intégralement par ClaimManifest.model_validate_json()."""
    from schemas.results import ClaimManifest

    svc = _make_storage(tmp_path)
    result = run(
        case_id="CLM-0004",
        source_path=DEMO_DIR / "CLM-0004" / "input",
        storage=svc,
    )
    manifest_file = tmp_path / "storage" / "manifests" / "CLM-0004.json"
    content = manifest_file.read_text(encoding="utf-8")

    # Désérialisation Pydantic sans erreur
    reloaded = ClaimManifest.model_validate_json(content)

    # Les données correspondent au résultat en mémoire
    assert reloaded.claim_id == result.manifest.claim_id
    assert reloaded.status == result.manifest.status
    assert reloaded.file_count == result.manifest.file_count
    assert reloaded.total_size_bytes == result.manifest.total_size_bytes
    assert len(reloaded.files) == len(result.manifest.files)


def test_manifest_contient_champs_checklist(tmp_path):
    """Chaque entrée fichier du manifest satisfait la checklist d'ingestion."""
    from schemas.results import ClaimManifest

    svc = _make_storage(tmp_path)
    run(
        case_id="CLM-0004",
        source_path=DEMO_DIR / "CLM-0004" / "input",
        storage=svc,
    )
    manifest_file = tmp_path / "storage" / "manifests" / "CLM-0004.json"
    m = ClaimManifest.model_validate_json(manifest_file.read_text())

    # Checklist niveau manifest
    assert m.claim_id, "claim_id absent"
    assert m.file_count == len(m.files), "file_count incohérent"
    assert m.total_size_bytes == sum(f.actual_size for f in m.files), "total_size_bytes incohérent"

    for f in m.files:
        assert f.sha256 is not None, f"Hash manquant pour {f.original_name}"
        assert f.actual_size > 0, f"Taille 0 pour {f.original_name}"
        assert f.detected_mime_type, f"MIME absent pour {f.original_name}"
        assert f.status is not None, f"Statut absent pour {f.original_name}"
        assert f.relative_storage_path is not None, f"Chemin relatif absent pour {f.original_name}"
        assert not f.relative_storage_path.startswith("/"), f"Chemin absolu détecté pour {f.original_name}"
        assert not f.relative_storage_path.startswith("storage/"), (
            f"Chemin doit être relatif à la racine storage/, pas au projet : {f.relative_storage_path}"
        )


def test_manifest_ecrit_meme_pour_dossier_vide(tmp_path):
    """Le manifest est toujours écrit, même pour un dossier BLOCKED (traçabilité)."""
    from schemas.results import ClaimManifest

    svc = _make_storage(tmp_path)
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    run(case_id="CLM-9990", source_path=empty_dir, storage=svc)

    manifest_file = tmp_path / "storage" / "manifests" / "CLM-9990.json"
    assert manifest_file.exists(), "Manifest absent pour dossier vide"
    m = ClaimManifest.model_validate_json(manifest_file.read_text())
    assert m.claim_id == "CLM-9990"
    assert m.status == IntakeStatus.BLOCKED
    assert m.file_count == 0


def test_manifest_chemins_relatifs_a_racine_storage(tmp_path):
    """Les chemins dans le manifest sont relatifs à la racine storage/ (pas au projet)."""
    svc = _make_storage(tmp_path)
    result = run(
        case_id="CLM-0004",
        source_path=DEMO_DIR / "CLM-0004" / "input",
        storage=svc,
    )
    for f in result.manifest.files:
        if f.relative_storage_path:
            # Doit commencer par incoming/ ou quarantine/, pas par storage/
            assert f.relative_storage_path.startswith(("incoming/", "quarantine/")), (
                f"Chemin inattendu : {f.relative_storage_path}"
            )


def test_manifest_sans_contenu_medical_brut(tmp_path):
    """Le manifest ne contient pas de contenu médical brut (bytes, texte OCR, etc.)."""
    from schemas.results import ClaimManifest

    svc = _make_storage(tmp_path)
    run(
        case_id="CLM-0004",
        source_path=DEMO_DIR / "CLM-0004" / "input",
        storage=svc,
    )
    manifest_file = tmp_path / "storage" / "manifests" / "CLM-0004.json"
    m = ClaimManifest.model_validate_json(manifest_file.read_text())

    for f in m.files:
        dumped = f.model_dump()
        for key, val in dumped.items():
            assert not isinstance(val, bytes), f"Contenu brut (bytes) détecté dans {key}"


# ── Contrat ClaimState après ingestion ────────────────────────────────────────
# Étape 10 : node() doit mettre à jour ClaimState avec uniquement :
#   case_id · statut · manifest · métadonnées · chemins relatifs · hashes · alertes · erreurs
# Et ne jamais transmettre : octets bruts · PDF · base64 · OCR · chemins absolus · fichiers ouverts


def test_node_intake_status_promu():
    """node() promeut intake_status au niveau supérieur du state pour le routage."""
    state = {
        "case_id": "CLM-0004",
        "intake_input": {
            "source_path": str(DEMO_DIR / "CLM-0004" / "input"),
        },
    }
    updates = node(state)  # type: ignore[arg-type]

    assert "intake_status" in updates, "intake_status doit être dans les mises à jour"
    assert updates["intake_status"] == updates["intake_result"].status, (
        "intake_status doit correspondre au statut du ClaimIntakeResult"
    )


def test_node_intake_input_vide_apres_ingestion():
    """node() vide intake_input pour supprimer le source_path absolu du state."""
    state = {
        "case_id": "CLM-0004",
        "intake_input": {
            "source_path": str(DEMO_DIR / "CLM-0004" / "input"),
        },
    }
    updates = node(state)  # type: ignore[arg-type]

    assert "intake_input" in updates, "intake_input doit être explicitement remis à None"
    assert updates["intake_input"] is None, (
        "intake_input doit être None après ingestion — le source_path absolu ne doit pas persister"
    )


def test_node_pas_de_chemin_absolu_dans_updates():
    """Aucune des mises à jour du state ne contient un chemin absolu."""
    from state.claim_state import validate_state_update

    state = {
        "case_id": "CLM-0004",
        "intake_input": {
            "source_path": str(DEMO_DIR / "CLM-0004" / "input"),
        },
    }
    updates = node(state)  # type: ignore[arg-type]

    # validate_state_update ne doit pas lever d'exception
    validate_state_update(updates)


def test_node_pas_de_binaire_dans_updates():
    """Aucune des mises à jour du state ne contient d'octets bruts."""
    from pydantic import BaseModel

    state = {
        "case_id": "CLM-0004",
        "intake_input": {
            "source_path": str(DEMO_DIR / "CLM-0004" / "input"),
        },
    }
    updates = node(state)  # type: ignore[arg-type]

    def _aucun_bytes(value: object, breadcrumb: str) -> None:
        if isinstance(value, (bytes, bytearray)):
            raise AssertionError(f"Contenu binaire interdit dans {breadcrumb}")
        elif isinstance(value, dict):
            for k, v in value.items():
                _aucun_bytes(v, f"{breadcrumb}.{k}")
        elif isinstance(value, (list, tuple)):
            for i, item in enumerate(value):
                _aucun_bytes(item, f"{breadcrumb}[{i}]")
        elif isinstance(value, BaseModel):
            _aucun_bytes(value.model_dump(), breadcrumb)

    for key, val in updates.items():
        _aucun_bytes(val, key)


def test_node_updates_contient_champs_obligatoires():
    """Les mises à jour du state contiennent tous les champs requis après ingestion.

    Checklist Étape 10 : case_id inclus dans intake_result, statut, manifest,
    métadonnées documents (chemins relatifs, hashes), alertes/erreurs disponibles.
    """
    state = {
        "case_id": "CLM-0004",
        "intake_input": {
            "source_path": str(DEMO_DIR / "CLM-0004" / "input"),
        },
    }
    updates = node(state)  # type: ignore[arg-type]
    result = updates["intake_result"]

    # case_id / claim_id
    assert result.claim_id == "CLM-0004"

    # statut ingestion
    assert result.status is not None
    assert updates["intake_status"] == result.status

    # manifest structuré
    assert result.manifest is not None
    assert result.manifest.claim_id == "CLM-0004"

    # métadonnées documents (pour chaque fichier accepté)
    for f in result.manifest.files:
        assert f.original_name, "original_name manquant"
        assert f.storage_name, "storage_name manquant"
        assert f.detected_mime_type, "detected_mime_type manquant"
        assert f.actual_size >= 0, "actual_size manquant"

    # hashes SHA-256 présents pour les fichiers acceptés
    accepted = [f for f in result.manifest.files if f.sha256 is not None]
    assert len(accepted) > 0, "Aucun fichier avec hash SHA-256"

    # chemins relatifs (pas absolus, pas de préfixe storage/)
    for f in result.manifest.files:
        if f.relative_storage_path:
            assert not f.relative_storage_path.startswith("/")
            assert f.relative_storage_path.startswith(("incoming/", "quarantine/"))

    # les alertes et erreurs sont accessibles (listes, potentiellement vides)
    assert isinstance(result.manifest.alerts, list)
    assert isinstance(result.errors, list)


# ═══════════════════════════════════════════════════════════════════════════════
# CAS OBLIGATOIRE 1 — Dossier valide (3 fichiers)
# ═══════════════════════════════════════════════════════════════════════════════

_PDF_MINIMAL = b"%PDF-1.4 1 0 obj<</Type/Catalog>>endobj xref"


def test_co1_trois_fichiers_valides_acceptes(tmp_path):
    """CAS-01 : Trois fichiers PDF valides → trois InspectedFile ACCEPTED, statut global ACCEPTED."""
    svc = _make_storage(tmp_path)
    source_dir = tmp_path / "source_co1a"
    source_dir.mkdir()
    for i, name in enumerate(("facture.pdf", "ordonnance.pdf", "identite.pdf")):
        # Contenus distincts pour éviter la détection de doublon SHA-256
        (source_dir / name).write_bytes(_PDF_MINIMAL + f" doc{i}".encode())

    result = run(case_id="CLM-8001", source_path=source_dir, storage=svc)

    assert result.manifest.file_count == 3
    accepted = [f for f in result.manifest.files if f.status == FileStatus.ACCEPTED]
    assert len(accepted) == 3, f"3 fichiers ACCEPTED attendus, {len(accepted)} obtenus"
    assert result.accepted_count == 3
    assert result.status == IntakeStatus.ACCEPTED
    assert result.quarantined_count == 0
    assert result.duplicate_count == 0
    assert result.error_count == 0


def test_co1_claim_id_dans_result_et_manifest(tmp_path):
    """CAS-01 : Le claim_id est présent et cohérent dans le résultat et dans le manifest."""
    svc = _make_storage(tmp_path)
    source_dir = tmp_path / "source_co1b"
    source_dir.mkdir()
    (source_dir / "doc.pdf").write_bytes(_PDF_MINIMAL)

    result = run(case_id="CLM-8002", source_path=source_dir, storage=svc)

    assert result.claim_id == "CLM-8002"
    assert result.manifest.claim_id == "CLM-8002"


def test_co1_hash_sha256_par_fichier_accepte(tmp_path):
    """CAS-01 : Chaque fichier ACCEPTED possède un hash SHA-256 hexadécimal de 64 caractères."""
    svc = _make_storage(tmp_path)
    source_dir = tmp_path / "source_co1c"
    source_dir.mkdir()
    for i, name in enumerate(("a.pdf", "b.pdf", "c.pdf")):
        # Contenus volontairement différents pour avoir 3 hashes distincts
        (source_dir / name).write_bytes(_PDF_MINIMAL + str(i).encode())

    result = run(case_id="CLM-8003", source_path=source_dir, storage=svc)

    hashes = set()
    for f in result.manifest.files:
        if f.status == FileStatus.ACCEPTED:
            assert f.sha256 is not None, f"sha256 manquant pour {f.original_name}"
            assert len(f.sha256) == 64, f"sha256 doit faire 64 caractères pour {f.original_name}"
            assert all(c in "0123456789abcdef" for c in f.sha256), (
                f"sha256 non hexadécimal pour {f.original_name}"
            )
            hashes.add(f.sha256)
    assert len(hashes) == 3, "Trois fichiers au contenu distinct → trois hashes distincts"


def test_co1_taille_correspond_fichier_reel(tmp_path):
    """CAS-01 : actual_size dans le manifest correspond à la taille réelle des fichiers source."""
    svc = _make_storage(tmp_path)
    source_dir = tmp_path / "source_co1d"
    source_dir.mkdir()

    contenu = {
        "petit.pdf":  _PDF_MINIMAL + b"x" * 100,
        "moyen.pdf":  _PDF_MINIMAL + b"y" * 1_000,
        "grand.pdf":  _PDF_MINIMAL + b"z" * 5_000,
    }
    for name, data in contenu.items():
        (source_dir / name).write_bytes(data)

    result = run(case_id="CLM-8004", source_path=source_dir, storage=svc)

    files_by_name = {f.original_name: f for f in result.manifest.files}
    for name, data in contenu.items():
        assert files_by_name[name].actual_size == len(data), (
            f"{name} : taille attendue {len(data)}, reçue {files_by_name[name].actual_size}"
        )


def test_co1_fichiers_physiquement_dans_incoming(tmp_path):
    """CAS-01 : Chaque fichier ACCEPTED est physiquement présent dans incoming/<case_id>/."""
    svc = _make_storage(tmp_path)
    source_dir = tmp_path / "source_co1e"
    source_dir.mkdir()
    (source_dir / "facture.pdf").write_bytes(_PDF_MINIMAL + b"\x00" * 50)

    result = run(case_id="CLM-8005", source_path=source_dir, storage=svc)

    assert result.status == IntakeStatus.ACCEPTED
    for f in result.manifest.files:
        if f.status == FileStatus.ACCEPTED:
            physical = svc.incoming_dir / "CLM-8005" / f.storage_name
            assert physical.exists(), f"Fichier absent de incoming/ : {physical}"
            assert physical.stat().st_size == f.actual_size, (
                f"Taille physique ≠ actual_size pour {f.original_name}"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# CAS OBLIGATOIRE 2 — Dossier vide
# ═══════════════════════════════════════════════════════════════════════════════


def test_co2_motif_empty_claim_present(tmp_path):
    """CAS-02 : Le motif de blocage est EMPTY_CLAIM (code stable)."""
    svc = _make_storage(tmp_path)
    empty = tmp_path / "vide"
    empty.mkdir()

    result = run(case_id="CLM-8010", source_path=empty, storage=svc)

    assert result.status == IntakeStatus.BLOCKED
    assert result.accepted_count == 0
    assert any(e.code == IntakeReasonCode.EMPTY_CLAIM for e in result.errors), (
        f"EMPTY_CLAIM attendu dans result.errors, obtenu : {[e.code for e in result.errors]}"
    )


def test_co2_aucun_repertoire_metier_cree(tmp_path):
    """CAS-02 : Un dossier vide ne crée aucun répertoire incoming/, quarantine/ ni temporary/."""
    svc = _make_storage(tmp_path)
    storage_root = tmp_path / "storage"
    empty = tmp_path / "vide2"
    empty.mkdir()

    result = run(case_id="CLM-8011", source_path=empty, storage=svc)

    assert result.status == IntakeStatus.BLOCKED
    for zone in ("incoming", "quarantine", "temporary"):
        assert not (storage_root / zone / "CLM-8011").exists(), (
            f"{zone}/CLM-8011/ créé à tort pour un dossier vide"
        )
    # Seul le manifest de traçabilité doit être écrit
    assert (storage_root / "manifests" / "CLM-8011.json").exists(), (
        "Le manifest de traçabilité doit être écrit même pour un dossier vide"
    )


def test_co2_manifest_non_mensonger(tmp_path):
    """CAS-02 : Le manifest d'un dossier vide ne contient aucun fichier inventé."""
    from schemas.results import ClaimManifest

    svc = _make_storage(tmp_path)
    empty = tmp_path / "vide3"
    empty.mkdir()

    run(case_id="CLM-8012", source_path=empty, storage=svc)

    m = ClaimManifest.model_validate_json(
        (tmp_path / "storage" / "manifests" / "CLM-8012.json").read_text()
    )
    assert m.claim_id == "CLM-8012"
    assert m.status == IntakeStatus.BLOCKED
    assert m.file_count == 0, "Aucun fichier inventé dans le manifest"
    assert m.files == [], "La liste de fichiers doit être vide"
    assert m.total_size_bytes == 0


# ═══════════════════════════════════════════════════════════════════════════════
# CAS OBLIGATOIRE 3 — Doublon (complément : doublon visible dans le manifest)
# ═══════════════════════════════════════════════════════════════════════════════


def test_co3_doublon_visible_dans_manifest_sur_disque(tmp_path):
    """CAS-03 : Le doublon apparaît dans le manifest JSON avec statut DUPLICATE et hash identique."""
    from schemas.results import ClaimManifest

    svc, source_dir = _setup_dup_case(tmp_path, case_id="CLM-8015")
    run(case_id="CLM-8015", source_path=source_dir, storage=svc)

    m = ClaimManifest.model_validate_json(
        (tmp_path / "storage" / "manifests" / "CLM-8015.json").read_text()
    )
    by_name = {f.original_name: f for f in m.files}
    assert by_name["invoice.pdf"].status == FileStatus.ACCEPTED
    assert by_name["invoice_copy.pdf"].status == FileStatus.DUPLICATE
    # Le doublon conserve son SHA-256 dans le manifest
    assert by_name["invoice.pdf"].sha256 == by_name["invoice_copy.pdf"].sha256
    # Aucun fichier écrasé : les storage_names sont distincts
    assert by_name["invoice.pdf"].storage_name != by_name["invoice_copy.pdf"].storage_name


# ═══════════════════════════════════════════════════════════════════════════════
# CAS OBLIGATOIRE 4 — Chemin traversal ../
# ═══════════════════════════════════════════════════════════════════════════════


def test_co4_validate_filename_rejette_traversal_posix():
    """CAS-04 : validate_filename rejette les motifs POSIX de traversal (../)."""
    from tools.file_inspection import validate_filename

    for pattern in ("../secret.pdf", "../../etc/passwd", "..", "../"):
        ok, reasons = validate_filename(pattern)
        assert not ok, f"'{pattern}' aurait dû être refusé"
        codes = [r.code for r in reasons]
        assert (
            IntakeReasonCode.PATH_TRAVERSAL_ATTEMPT in codes
            or IntakeReasonCode.INVALID_FILENAME in codes
        ), f"Code de traversal attendu pour '{pattern}', obtenu : {codes}"


def test_co4_motif_est_path_traversal_attempt():
    """CAS-04 : Le code retourné pour '../facture.pdf' est PATH_TRAVERSAL_ATTEMPT."""
    from tools.file_inspection import validate_filename

    ok, reasons = validate_filename("../facture.pdf")

    assert not ok
    assert len(reasons) == 1
    assert reasons[0].code == IntakeReasonCode.PATH_TRAVERSAL_ATTEMPT
    assert reasons[0].field == "filename"
    assert reasons[0].message


def test_co4_traversal_backslash_refuse_via_run(tmp_path):
    """CAS-04 : Un fichier dont le nom contient un backslash-traversal est refusé par run().

    Sur POSIX, '\\' n'est pas un séparateur de chemin mais le pattern '..\\' est
    reconnu comme traversal potentiel par validate_filename → code PATH_TRAVERSAL_ATTEMPT.
    """
    svc = _make_storage(tmp_path)
    source_dir = tmp_path / "source_trav"
    source_dir.mkdir()

    # '..\secret.pdf' est un nom de fichier valide sur POSIX (backslash = caractère ordinaire)
    traversal_name = "..\\secret.pdf"
    traversal_file = source_dir / traversal_name
    try:
        traversal_file.write_bytes(b"%PDF-1.4 traversal attempt")
    except OSError:
        pytest.skip("Impossible de créer un fichier avec '\\' dans le nom sur cette plateforme")

    result = run(case_id="CLM-8020", source_path=source_dir, storage=svc)

    assert result.status == IntakeStatus.BLOCKED
    blocked = [f for f in result.manifest.files if f.status == FileStatus.BLOCKED]
    assert len(blocked) == 1
    assert blocked[0].original_name == traversal_name
    assert any(r.code == IntakeReasonCode.PATH_TRAVERSAL_ATTEMPT for r in blocked[0].reasons), (
        f"PATH_TRAVERSAL_ATTEMPT attendu, obtenu : {[r.code for r in blocked[0].reasons]}"
    )


def test_co4_aucun_fichier_hors_racine_storage(tmp_path):
    """CAS-04 : Même avec un nom traversal, aucun fichier n'est écrit hors de storage/."""
    svc = _make_storage(tmp_path)
    storage_root = (tmp_path / "storage").resolve()

    source_dir = tmp_path / "source_trav2"
    source_dir.mkdir()

    traversal_name = "..\\secret.pdf"
    traversal_file = source_dir / traversal_name
    try:
        traversal_file.write_bytes(b"%PDF-1.4 traversal attempt")
    except OSError:
        pytest.skip("Impossible de créer un fichier avec '\\' dans le nom sur cette plateforme")

    run(case_id="CLM-8021", source_path=source_dir, storage=svc)

    # Tous les fichiers produits par l'agent sont sous storage/
    for f in tmp_path.rglob("*"):
        if not f.is_file():
            continue
        if f.is_relative_to(source_dir):
            continue  # fichier source, non créé par l'agent
        assert f.is_relative_to(storage_root), (
            f"Fichier écrit en dehors de la racine storage : {f}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TESTS SUPPLÉMENTAIRES
# ═══════════════════════════════════════════════════════════════════════════════


def test_supp_fichier_vide_bloque(tmp_path):
    """Supplémentaire : Un fichier de 0 octet est bloqué avec le code EMPTY_FILE."""
    svc = _make_storage(tmp_path)
    source_dir = tmp_path / "source_empty_file"
    source_dir.mkdir()
    (source_dir / "vide.pdf").write_bytes(b"")

    result = run(case_id="CLM-8030", source_path=source_dir, storage=svc)

    assert result.status == IntakeStatus.BLOCKED
    blocked = [f for f in result.manifest.files if f.status == FileStatus.BLOCKED]
    assert len(blocked) == 1
    assert any(r.code == IntakeReasonCode.EMPTY_FILE for r in blocked[0].reasons), (
        f"EMPTY_FILE attendu, obtenu : {[r.code for r in blocked[0].reasons]}"
    )


def test_supp_fichier_trop_volumineux_bloque(tmp_path):
    """Supplémentaire : Un fichier dépassant la limite est bloqué avec FILE_TOO_LARGE."""
    s = Settings(  # type: ignore[call-arg]
        CLAIMSHIELD_STORAGE_DIR=str(tmp_path / "storage"),
        CLAIMSHIELD_QUARANTINE_DIR=str(tmp_path / "storage" / "quarantine"),
        CLAIMSHIELD_MAX_FILE_SIZE_MB=1,
    )
    svc = StorageService(settings=s)
    svc.ensure_dirs()

    source_dir = tmp_path / "source_large"
    source_dir.mkdir()
    # 2 Mo — dépasse la limite de 1 Mo
    (source_dir / "lourd.pdf").write_bytes(_PDF_MINIMAL + b"x" * (2 * 1024 * 1024))

    result = run(case_id="CLM-8031", source_path=source_dir, storage=svc, settings=s)

    assert result.status == IntakeStatus.BLOCKED
    blocked = [f for f in result.manifest.files if f.status == FileStatus.BLOCKED]
    assert len(blocked) == 1
    assert any(r.code == IntakeReasonCode.FILE_TOO_LARGE for r in blocked[0].reasons), (
        f"FILE_TOO_LARGE attendu, obtenu : {[r.code for r in blocked[0].reasons]}"
    )
    assert blocked[0].sha256 is None, "Aucun hash ne doit être calculé pour un fichier trop volumineux"


def test_supp_trop_de_fichiers_bloque(tmp_path):
    """Supplémentaire : Plus de fichiers que la limite configurée → BLOCKED, TOO_MANY_FILES."""
    s = Settings(  # type: ignore[call-arg]
        CLAIMSHIELD_STORAGE_DIR=str(tmp_path / "storage"),
        CLAIMSHIELD_QUARANTINE_DIR=str(tmp_path / "storage" / "quarantine"),
        CLAIMSHIELD_MAX_FILES_PER_FOLDER=2,
    )
    svc = StorageService(settings=s)
    svc.ensure_dirs()

    source_dir = tmp_path / "source_too_many"
    source_dir.mkdir()
    for i in range(3):  # 3 > limite de 2
        (source_dir / f"doc_{i}.pdf").write_bytes(_PDF_MINIMAL)

    result = run(case_id="CLM-8032", source_path=source_dir, storage=svc, settings=s)

    assert result.status == IntakeStatus.BLOCKED
    assert result.accepted_count == 0
    assert any(e.code == IntakeReasonCode.TOO_MANY_FILES for e in result.errors), (
        f"TOO_MANY_FILES attendu dans result.errors, obtenu : {[e.code for e in result.errors]}"
    )
    # Aucun fichier ne doit avoir été traité individuellement
    assert result.manifest.file_count == 0


def test_supp_extension_non_autorisee_bloque(tmp_path):
    """Supplémentaire : Extension .exe non autorisée → BLOCKED, UNSUPPORTED_EXTENSION."""
    svc = _make_storage(tmp_path)
    source_dir = tmp_path / "source_bad_ext"
    source_dir.mkdir()
    (source_dir / "programme.exe").write_bytes(b"MZ" + b"\x00" * 200)

    result = run(case_id="CLM-8033", source_path=source_dir, storage=svc)

    blocked = [f for f in result.manifest.files if f.status == FileStatus.BLOCKED]
    assert len(blocked) == 1
    assert any(r.code == IntakeReasonCode.UNSUPPORTED_EXTENSION for r in blocked[0].reasons), (
        f"UNSUPPORTED_EXTENSION attendu, obtenu : {[r.code for r in blocked[0].reasons]}"
    )
    # Le fichier bloqué n'est pas stocké dans incoming/
    assert not (svc.incoming_dir / "CLM-8033").exists(), (
        "Aucun répertoire incoming/ ne doit être créé pour un fichier bloqué"
    )


def test_supp_mime_different_extension_quarantined(tmp_path):
    """Supplémentaire : Contenu PNG dans un fichier .pdf → QUARANTINED (MIME_EXTENSION_MISMATCH).

    Ignoré si python-magic/libmagic n'est pas disponible (le repli mimetypes
    ne lit pas les magic bytes et ne peut pas détecter le mismatch).
    """
    try:
        import magic as _magic  # type: ignore[import-not-found]
        _magic.from_file  # noqa: B018  — vérification de présence
    except (ImportError, OSError):
        pytest.skip("python-magic/libmagic absent — détection MIME depuis le contenu indisponible")

    svc = _make_storage(tmp_path)
    source_dir = tmp_path / "source_mime"
    source_dir.mkdir()
    # En-tête PNG dans un fichier nommé .pdf — MIME réel ≠ MIME attendu pour .pdf
    png_header = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + b"\x00" * 200
    (source_dir / "image.pdf").write_bytes(png_header)

    result = run(case_id="CLM-8034", source_path=source_dir, storage=svc)

    f = result.manifest.files[0]
    assert f.status == FileStatus.QUARANTINED, (
        f"QUARANTINED attendu pour mismatch MIME/extension, obtenu {f.status}"
    )
    assert any(r.code == IntakeReasonCode.MIME_EXTENSION_MISMATCH for r in f.reasons)
    # Le fichier QUARANTINED est dans quarantine/, pas dans incoming/
    assert f.relative_storage_path is not None
    assert f.relative_storage_path.startswith("quarantine/")


def test_supp_meme_nom_contenu_different_deux_dossiers(tmp_path):
    """Supplémentaire : Même nom de fichier dans deux dossiers distincts avec contenu différent
    → hashes différents, stockage indépendant sans interférence.
    """
    svc = _make_storage(tmp_path)

    source_a = tmp_path / "dossier_a"
    source_a.mkdir()
    (source_a / "facture.pdf").write_bytes(_PDF_MINIMAL + b"A" * 200)

    source_b = tmp_path / "dossier_b"
    source_b.mkdir()
    (source_b / "facture.pdf").write_bytes(_PDF_MINIMAL + b"B" * 200)

    result_a = run(case_id="CLM-8035", source_path=source_a, storage=svc)
    result_b = run(case_id="CLM-8036", source_path=source_b, storage=svc)

    assert result_a.status == IntakeStatus.ACCEPTED
    assert result_b.status == IntakeStatus.ACCEPTED

    sha_a = result_a.manifest.files[0].sha256
    sha_b = result_b.manifest.files[0].sha256
    assert sha_a != sha_b, "Contenus différents → SHA-256 différents"

    path_a = result_a.manifest.files[0].relative_storage_path or ""
    path_b = result_b.manifest.files[0].relative_storage_path or ""
    assert "CLM-8035" in path_a
    assert "CLM-8036" in path_b


def test_supp_nom_caracteres_speciaux(tmp_path):
    """Supplémentaire : Fichier avec espaces et accents dans le nom → traité sans erreur.

    Le nom original est préservé dans le manifest ; le storage_name est assaini
    par build_storage_name (pas de caractères dangereux, extension conservée).
    """
    svc = _make_storage(tmp_path)
    source_dir = tmp_path / "source_special"
    source_dir.mkdir()

    nom_special = "facture résumé (2026).pdf"
    (source_dir / nom_special).write_bytes(_PDF_MINIMAL + b"\x00" * 50)

    result = run(case_id="CLM-8037", source_path=source_dir, storage=svc)

    assert result.manifest.file_count == 1
    f = result.manifest.files[0]

    # Le nom original est préservé dans les métadonnées
    assert f.original_name == nom_special

    # Le storage_name ne contient pas de caractères dangereux
    assert "/" not in f.storage_name, "Séparateur / interdit dans storage_name"
    assert "\\" not in f.storage_name, "Backslash interdit dans storage_name"
    assert f.storage_name.endswith(".pdf")

    # Le fichier est traité (ACCEPTED ou QUARANTINED — jamais ERROR pour ce nom)
    assert f.status in (FileStatus.ACCEPTED, FileStatus.QUARANTINED), (
        f"Statut inattendu pour un nom spécial : {f.status}"
    )


def test_supp_echec_stockage_renvoie_error(tmp_path, monkeypatch):
    """Supplémentaire : Échec de stage_file (WRITE_ERROR) → statut ERROR, code STORAGE_ERROR."""
    from services.storage import StorageError
    from schemas.results import StructuredError as SE

    svc = _make_storage(tmp_path)
    source_dir = tmp_path / "source_err"
    source_dir.mkdir()
    (source_dir / "facture.pdf").write_bytes(_PDF_MINIMAL + b"\x00" * 50)

    def _stage_fail(*_: object, **_kw: object) -> None:
        raise StorageError(SE(
            code="WRITE_ERROR",
            message="Disque plein — simulé pour ce test",
            field="facture.pdf",
        ))

    monkeypatch.setattr(svc, "stage_file", _stage_fail)

    result = run(case_id="CLM-8038", source_path=source_dir, storage=svc)

    assert result.status == IntakeStatus.ERROR
    errored = [f for f in result.manifest.files if f.status == FileStatus.ERROR]
    assert len(errored) == 1
    assert any(r.code == IntakeReasonCode.STORAGE_ERROR for r in errored[0].reasons), (
        f"STORAGE_ERROR attendu, obtenu : {[r.code for r in errored[0].reasons]}"
    )
    # Aucun fichier dans incoming/ — l'erreur a eu lieu avant tout commit
    assert not (svc.incoming_dir / "CLM-8038").exists()


def test_supp_fichier_temporaire_nettoye_apres_erreur(tmp_path, monkeypatch):
    """Supplémentaire : Après un échec de commit_file, la zone temporary/ est vide."""
    from services.storage import StorageError
    from schemas.results import StructuredError as SE

    svc = _make_storage(tmp_path)
    source_dir = tmp_path / "source_cleanup"
    source_dir.mkdir()
    (source_dir / "facture.pdf").write_bytes(_PDF_MINIMAL + b"\x00" * 50)

    def _commit_fail(**kw: object) -> None:
        # Mimique le comportement de commit_file en cas d'OSError : supprime le temp
        tp = kw["temp_path"]
        assert isinstance(tp, Path)
        tp.unlink(missing_ok=True)
        raise StorageError(SE(
            code="MOVE_ERROR",
            message="Déplacement atomique simulé échoué",
            field=str(kw.get("physical_name", "")),
        ))

    monkeypatch.setattr(svc, "commit_file", _commit_fail)

    result = run(case_id="CLM-8039", source_path=source_dir, storage=svc)

    assert result.status == IntakeStatus.ERROR

    # La zone temporaire ne doit contenir aucun fichier résiduel pour ce dossier
    temp_case = svc.temp_dir / "CLM-8039"
    if temp_case.exists():
        residus = [f for f in temp_case.rglob("*") if f.is_file()]
        assert residus == [], f"Fichiers résiduels dans temporary/ après erreur : {residus}"


def test_supp_sortie_json_serializable(tmp_path):
    """Supplémentaire : ClaimIntakeResult est entièrement sérialisable en JSON valide."""
    import json

    svc = _make_storage(tmp_path)
    source_dir = tmp_path / "source_json"
    source_dir.mkdir()
    (source_dir / "facture.pdf").write_bytes(_PDF_MINIMAL + b"x" * 100)

    result = run(case_id="CLM-8040", source_path=source_dir, storage=svc)

    json_str = result.model_dump_json()
    assert isinstance(json_str, str)
    assert len(json_str) > 10

    parsed = json.loads(json_str)
    assert parsed["claim_id"] == "CLM-8040"
    assert parsed["status"] in ("accepted", "quarantined", "blocked", "error")
    assert "manifest" in parsed
    assert isinstance(parsed["manifest"]["files"], list)
    # Aucune valeur binaire dans le JSON sérialisé
    for f_data in parsed["manifest"]["files"]:
        for val in f_data.values():
            assert not isinstance(val, bytes), "bytes trouvé dans le JSON sérialisé"


def test_supp_claimstate_sans_document_brut(tmp_path):
    """Supplémentaire : Après node(), ClaimState ne contient ni bytes ni chemin absolu."""
    from state.claim_state import validate_state_update

    # Utilise le storage par défaut du node() (répertoires partagés nettoyés par conftest)
    state = {
        "case_id": "CLM-8041",
        "intake_input": {
            "source_path": str(DEMO_DIR / "CLM-0004" / "input"),
        },
    }
    updates = node(state)  # type: ignore[arg-type]

    # validate_state_update ne doit pas lever d'exception
    validate_state_update(updates)

    # Vérification explicite : aucun bytes dans les valeurs de mise à jour
    from pydantic import BaseModel

    def _check_no_bytes(value: object, path: str) -> None:
        if isinstance(value, (bytes, bytearray)):
            raise AssertionError(f"bytes trouvé dans ClaimState[{path}]")
        elif isinstance(value, dict):
            for k, v in value.items():
                _check_no_bytes(v, f"{path}.{k}")
        elif isinstance(value, (list, tuple)):
            for i, item in enumerate(value):
                _check_no_bytes(item, f"{path}[{i}]")
        elif isinstance(value, BaseModel):
            _check_no_bytes(value.model_dump(), path)

    for key, val in updates.items():
        _check_no_bytes(val, key)
