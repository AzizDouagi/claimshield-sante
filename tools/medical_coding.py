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


def _system_for_section(section: str) -> str:
    if section == "procedures":
        return "SNOMED-CT"
    if section == "medications":
        return "RxNorm"
    raise ValueError(f"Section invalide : {section!r} — attendu 'procedures' ou 'medications'")


def code_exists_in_reference(code: str | None, section: str, table: dict | None = None) -> bool:
    """Retourne True uniquement si le code actif existe dans le référentiel local."""
    if not code:
        return False
    if table is None:
        table = load_code_table()
    target_system = _system_for_section(section)
    return any(
        str(entry.get("code")) == str(code)
        and entry.get("active", True)
        and entry.get("system") == target_system
        for entry in table.get("codes", [])
    )


def find_code_alternatives(
    description: str,
    section: str,
    table: dict | None = None,
    *,
    limit: int = 3,
) -> list[str]:
    """Propose des alternatives du référentiel local sans sélectionner un code final."""
    if table is None:
        table = load_code_table()
    target_system = _system_for_section(section)
    normalized = description.strip().casefold()
    alternatives: list[str] = []
    for entry in table.get("codes", []):
        if not entry.get("active", True) or entry.get("system") != target_system:
            continue
        candidates = [entry.get("label", ""), *entry.get("synonyms", [])]
        if any(token and token.strip().casefold() in normalized for token in candidates):
            alternatives.append(str(entry["code"]))
        if len(alternatives) >= limit:
            break
    return alternatives


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

    # Nouveau format : liste plate `codes` filtrée par system
    target_system = _system_for_section(section)
    all_codes: list = table.get("codes", [])

    # ── Étape 1 : correspondance exacte (label normalisé ou synonyme) ─────────
    for entry in all_codes:
        if not entry.get("active", True):
            continue
        if entry.get("system") != target_system:
            continue
        if entry.get("label", "").strip().casefold() == normalized:
            return ProcedureCoding(
                original_description=description,
                proposed_code=str(entry["code"]),
                rule_applied="exact_match",
                status=VerificationStatus.PASS,
                evidence=[
                    f"Référentiel local {_RULES_FILENAME}",
                    f"{entry.get('system')}:{entry.get('code')}",
                    f"label={entry.get('label')}",
                ],
            )
        for syn in entry.get("synonyms", []):
            if syn.strip().casefold() == normalized:
                return ProcedureCoding(
                    original_description=description,
                    proposed_code=str(entry["code"]),
                    rule_applied="exact_match",
                    status=VerificationStatus.PASS,
                    evidence=[
                        f"Référentiel local {_RULES_FILENAME}",
                        f"{entry.get('system')}:{entry.get('code')}",
                        f"synonyme={syn}",
                    ],
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
                    alternatives=find_code_alternatives(description, section, table),
                    evidence=[
                        f"Mot-clé local détecté : {kw}",
                        "Aucun code final déterminé sans correspondance exacte.",
                    ],
                )

    # ── Étape 3 : aucune correspondance ──────────────────────────────────────
    return ProcedureCoding(
        original_description=description,
        proposed_code=None,
        rule_applied="not_determined",
        status=VerificationStatus.NEEDS_REVIEW,
        alternatives=[],
        evidence=["Aucune correspondance dans le référentiel local versionné."],
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
