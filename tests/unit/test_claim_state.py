"""Tests unitaires de ClaimState et validate_state_update.

Vérifie le contrat de sécurité du state LangGraph :
  - contenu autorisé (métadonnées, hashes, chemins relatifs, statuts)
  - contenu interdit (octets bruts, chemins absolus, objets fichier ouverts)
"""
from __future__ import annotations

import io
from pathlib import Path
from datetime import datetime

import pytest

from state.claim_state import validate_state_update


# ── Contenu autorisé ───────────────────────────────────────────────────────────


def test_metadata_propre_acceptee():
    """Un dict de métadonnées propres passe sans exception."""
    validate_state_update(
        {
            "case_id": "CLM-0001",
            "current_step": "claim_intake",
            "completed_steps": ["claim_intake"],
            "intake_status": "accepted",
        }
    )


def test_chemin_relatif_accepte():
    """Un chemin relatif (incoming/CLM-0001/doc.pdf) est autorisé."""
    validate_state_update(
        {
            "relative_path": "incoming/CLM-0001/CLM-0001_doc00.pdf",
        }
    )


def test_chemin_relatif_quarantine_accepte():
    """Un chemin relatif quarantine/ est autorisé."""
    validate_state_update({"path": "quarantine/CLM-0002/CLM-0002_doc01.pdf"})


def test_sha256_hexdigest_accepte():
    """Un hash SHA-256 (64 chars hexadécimaux) est autorisé."""
    validate_state_update({"sha256": "a" * 64})


def test_liste_de_strings_acceptee():
    """Une liste de chaînes non-absolues est autorisée."""
    validate_state_update({"alerts": ["Document manquant", "Hash incohérent"]})


def test_none_accepte():
    """Les valeurs None sont autorisées."""
    validate_state_update({"intake_result": None, "current_step": None})


def test_nombres_acceptes():
    """Les entiers et flottants sont autorisés."""
    validate_state_update({"file_count": 3, "total_size_bytes": 1_024_000})


def test_datetime_acceptee():
    """Un objet datetime est autorisé (non-binaire, non-chemin)."""
    validate_state_update({"received_at": datetime(2026, 6, 24, 10, 0)})


def test_intake_input_none_passe():
    """intake_input=None est le seul contenu autorisé pour ce champ après ingestion.

    Le chemin absolu source_path a déjà été consommé — None signifie 'vidé'.
    """
    validate_state_update({"intake_input": None})


def test_dict_imbrique_propre():
    """Un dict imbriqué avec des métadonnées propres est autorisé."""
    validate_state_update(
        {
            "metadata": {
                "original_name": "facture.pdf",
                "storage_name": "CLM-0001_doc00.pdf",
                "relative_path": "incoming/CLM-0001/CLM-0001_doc00.pdf",
                "size_bytes": 50_000,
            }
        }
    )


# ── Octets bruts interdits ─────────────────────────────────────────────────────


def test_bytes_bruts_rejetes():
    """Des octets bruts (bytes) sont refusés dans le state."""
    with pytest.raises(ValueError, match="contenu binaire"):
        validate_state_update({"pdf_content": b"%PDF-1.4 raw bytes"})


def test_bytearray_rejete():
    """Un bytearray est refusé."""
    with pytest.raises(ValueError, match="contenu binaire"):
        validate_state_update({"data": bytearray(b"raw binary")})


def test_bytes_dans_liste_rejetes():
    """Des octets dans une liste imbriquée sont refusés."""
    with pytest.raises(ValueError, match="contenu binaire"):
        validate_state_update({"files": [b"PDF data", "normal string"]})


def test_bytes_dans_dict_imbrique_rejetes():
    """Des octets dans un dict imbriqué sont refusés."""
    with pytest.raises(ValueError, match="contenu binaire"):
        validate_state_update({"doc": {"content": b"\x25PDF-1.4"}})


# ── Chemins absolus interdits ──────────────────────────────────────────────────


def test_chemin_absolu_posix_rejete():
    """Un chemin absolu POSIX (/home/…) est refusé."""
    with pytest.raises(ValueError, match="chemin absolu"):
        validate_state_update({"source_path": "/home/user/documents/facture.pdf"})


def test_chemin_absolu_racine_rejete():
    """Un chemin POSIX racine (/) est refusé."""
    with pytest.raises(ValueError, match="chemin absolu"):
        validate_state_update({"path": "/etc/passwd"})


def test_chemin_absolu_windows_rejete():
    """Un chemin absolu Windows (C:\\) est refusé."""
    with pytest.raises(ValueError, match="chemin absolu"):
        validate_state_update({"source": "C:\\Users\\user\\Documents\\facture.pdf"})


def test_chemin_absolu_windows_slash_rejete():
    """Un chemin absolu Windows avec slash (C:/) est refusé."""
    with pytest.raises(ValueError, match="chemin absolu"):
        validate_state_update({"source": "C:/Users/user/Documents/facture.pdf"})


def test_chemin_unc_rejete():
    """Un chemin UNC (\\\\server\\share) est refusé."""
    with pytest.raises(ValueError, match="chemin absolu"):
        validate_state_update({"source": "\\\\server\\share\\doc.pdf"})


def test_chemin_absolu_dans_liste_rejete():
    """Un chemin absolu dans une liste est refusé."""
    with pytest.raises(ValueError, match="chemin absolu"):
        validate_state_update({"paths": ["incoming/ok.pdf", "/absolute/bad.pdf"]})


def test_chemin_absolu_dans_dict_imbrique_rejete():
    """Un chemin absolu dans un dict imbriqué est refusé."""
    with pytest.raises(ValueError, match="chemin absolu"):
        validate_state_update({"metadata": {"source_path": "/tmp/upload/facture.pdf"}})


# ── Objets fichier ouverts interdits ──────────────────────────────────────────


def test_objet_fichier_ouvert_rejete(tmp_path: Path):
    """Un objet fichier ouvert est refusé."""
    fpath = tmp_path / "test.pdf"
    fpath.write_bytes(b"%PDF-1.4")
    with fpath.open("rb") as fh:
        with pytest.raises(ValueError, match="objet fichier"):
            validate_state_update({"file_handle": fh})


def test_stringio_rejete():
    """Un StringIO est refusé (implémente io.IOBase)."""
    buf = io.StringIO("contenu texte")
    with pytest.raises(ValueError, match="objet fichier"):
        validate_state_update({"stream": buf})


def test_bytesio_rejete():
    """Un BytesIO est refusé."""
    buf = io.BytesIO(b"raw bytes")
    with pytest.raises(ValueError, match="objet fichier"):
        validate_state_update({"stream": buf})


# ── Message d'erreur lisible ───────────────────────────────────────────────────


def test_message_erreur_cite_le_chemin_du_champ():
    """Le message de ValueError cite le nom du champ fautif."""
    with pytest.raises(ValueError) as exc_info:
        validate_state_update({"mon_champ": b"donnees binaires"})
    assert "mon_champ" in str(exc_info.value)


def test_plusieurs_violations_toutes_rapportees():
    """Toutes les violations sont rapportées dans une seule ValueError."""
    with pytest.raises(ValueError) as exc_info:
        validate_state_update(
            {
                "champ_a": b"bytes",
                "champ_b": "/chemin/absolu",
            }
        )
    msg = str(exc_info.value)
    assert "champ_a" in msg
    assert "champ_b" in msg
