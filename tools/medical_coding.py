"""Codification médicale déterministe — correspondance actes/médicaments → codes.

Fonctions pures — aucun appel LLM, aucun effet de bord.
Charge la table config/rules/medical_codes.yaml (mise en cache après le premier appel).

Logique de recherche par priorité :
  1. Correspondance exacte (description normalisée casefold + strip)
  2. Correspondance partielle par mots-clés (section keywords du YAML)
  3. Non trouvé → proposed_code=None, status=NEEDS_REVIEW

rule_applied : "exact_match" | "keyword_match" | "not_found"
"""
from __future__ import annotations

from schemas.domain import VerificationStatus
from schemas.results import ProcedureCoding
from tools.rule_loader import load_rules

_RULES_FILENAME = "medical_codes.yaml"


def load_code_table() -> dict:
    """Charge la table medical_codes.yaml. Retourne le dict complet (mis en cache).

    Clés retournées : 'version', 'procedures', 'medications', 'keywords'.
    Lève FileNotFoundError si le fichier est absent.
    Lève ValueError si le contenu n'est pas un dictionnaire valide.
    """
    return load_rules(_RULES_FILENAME)


def lookup_code(
    description: str,
    section: str,
    table: dict | None = None,
) -> ProcedureCoding:
    """Cherche un code pour une description dans la table de règles.

    Args:
        description : description brute de l'acte ou du médicament.
        section     : "procedures" ou "medications".
        table       : table YAML déjà chargée (chargée automatiquement si None).

    Returns:
        ProcedureCoding avec :
          - status=PASS et proposed_code renseigné si correspondance exacte trouvée.
          - status=NEEDS_REVIEW et proposed_code=None si correspondance partielle (mots-clés).
          - status=NEEDS_REVIEW et proposed_code=None si aucune correspondance.

    Raises:
        ValueError : si section n'est pas "procedures" ou "medications".
    """
    if section not in ("procedures", "medications"):
        raise ValueError(f"Section invalide : {section!r} — attendu 'procedures' ou 'medications'")

    if table is None:
        table = load_code_table()

    normalized = description.strip().casefold()
    section_data: dict = table.get(section, {})

    # ── Étape 1 : correspondance exacte ──────────────────────────────────────
    for entry_key, entry_value in section_data.items():
        if entry_key.strip().casefold() == normalized:
            return ProcedureCoding(
                original_description=description,
                proposed_code=str(entry_value["code"]),
                rule_applied="exact_match",
                status=VerificationStatus.PASS,
            )

    # ── Étape 2 : correspondance partielle par mots-clés ─────────────────────
    keywords_section: dict = table.get("keywords", {})
    for _category, keyword_list in keywords_section.items():
        for kw in keyword_list:
            if kw.casefold() in normalized:
                return ProcedureCoding(
                    original_description=description,
                    proposed_code=None,
                    rule_applied="keyword_match",
                    status=VerificationStatus.NEEDS_REVIEW,
                )

    # ── Étape 3 : aucune correspondance ──────────────────────────────────────
    return ProcedureCoding(
        original_description=description,
        proposed_code=None,
        rule_applied="not_found",
        status=VerificationStatus.NEEDS_REVIEW,
    )


def code_procedures(
    descriptions: list[str],
    table: dict | None = None,
) -> list[ProcedureCoding]:
    """Code une liste de descriptions de procédures médicales.

    Args:
        descriptions : liste des descriptions brutes de procédures.
        table        : table YAML déjà chargée (chargée automatiquement si None).

    Returns:
        Liste de ProcedureCoding (dans le même ordre que descriptions).
        Retourne une liste vide si descriptions est vide.
    """
    if not descriptions:
        return []
    if table is None:
        table = load_code_table()
    return [lookup_code(desc, "procedures", table) for desc in descriptions]


def code_medications(
    descriptions: list[str],
    table: dict | None = None,
) -> list[ProcedureCoding]:
    """Code une liste de descriptions de médicaments.

    Args:
        descriptions : liste des descriptions brutes de médicaments.
        table        : table YAML déjà chargée (chargée automatiquement si None).

    Returns:
        Liste de ProcedureCoding (dans le même ordre que descriptions).
        Retourne une liste vide si descriptions est vide.
    """
    if not descriptions:
        return []
    if table is None:
        table = load_code_table()
    return [lookup_code(desc, "medications", table) for desc in descriptions]


def compute_global_status(codings: list[ProcedureCoding]) -> VerificationStatus:
    """Calcule le statut global à partir d'une liste de codifications.

    Règles :
      - PASS si toutes les codings ont status=PASS et la liste est non vide.
      - NEEDS_REVIEW si au moins une NEEDS_REVIEW et aucune FAIL.
      - NEEDS_REVIEW si la liste est vide (aucun acte à coder).
      - FAIL si au moins une coding a status=FAIL.

    Args:
        codings : liste de ProcedureCoding à évaluer.

    Returns:
        VerificationStatus global.
    """
    if not codings:
        return VerificationStatus.NEEDS_REVIEW

    statuses = {c.status for c in codings}

    if VerificationStatus.FAIL in statuses:
        return VerificationStatus.FAIL
    if VerificationStatus.NEEDS_REVIEW in statuses:
        return VerificationStatus.NEEDS_REVIEW
    return VerificationStatus.PASS
