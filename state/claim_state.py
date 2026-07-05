"""État partagé du workflow LangGraph — ClaimShield Santé.

ClaimState est le seul objet qui traverse tous les nœuds du graphe.
Règles :
- Minimal : pas de texte OCR brut, pas de contenu PDF, pas de secrets.
- Append-only pour les listes (reducers operator.add) — LangGraph ne permet
  pas d'écraser une liste accumulée sans reducer explicite.
- Sérialisable JSON (tous les types sont des primitives ou des dict Pydantic).
- Versionné : schema_version permet de détecter un state produit par une
  version antérieure du workflow lors d'une reprise sur checkpoint.

Contenu autorisé après ingestion / OCR :
  case_id · statut ingestion · manifest structuré · métadonnées documents
  chemins relatifs · hashes SHA-256 · alertes · erreurs
  résultats structurés d'agent (Pydantic) · métadonnées de confiance
  références artefacts (artifact_id, artifact_path) · codes raison

Contenu interdit (garanti par validate_state_update) :
  octets bruts · PDF/images en base64 · texte OCR complet (full_text)
  pages OCR brutes (pages[]) · chemins absolus · objets fichier ouverts

Règles spécifiques à l'agent Document/OCR :
  - ocr_result doit être le résultat MINIMISÉ produit par _minimize_for_state().
    full_text et pages doivent être vides avant d'entrer dans le state.
  - ocr_input est consommé et remis à None par le nœud OCR.
  - Le texte OCR complet est écrit dans un artefact externe (artifact_id / artifact_path).
"""

from __future__ import annotations

import io
import operator
import re
from datetime import datetime
from typing import Annotated, Any, Mapping, TypedDict

from schemas.domain import IntakeStatus, Recommendation
from schemas.results import (
    AuditEvent,
    AuditResult,
    CaseReviewerResult,
    ClaimIntakeResult,
    ClinicalConsistencyResult,
    DocumentOcrResult,
    FhirValidatorResult,
    FraudDetectionResult,
    IdentityCoverageResult,
    MedicalCodingResult,
    PrivacyResult,
    SecurityGateResult,
)

# Détecte les chaînes qui ressemblent à des chemins absolus POSIX, Windows ou UNC.
_ABSOLUTE_PATH_RE = re.compile(r"^(?:/|[A-Za-z]:[/\\]|\\\\)")
_SECRET_HINT_RE = re.compile(
    r"(?:api[_-]?key|secret\s*[:=]|password\s*[:=]|token\s*[:=]|bearer\s+[a-z0-9._~+/=-]+)",
    re.IGNORECASE,
)
_SECRET_KEY_RE = re.compile(r"(?:api[_-]?key|secret|password|token|credential)", re.IGNORECASE)
_RAW_DOCUMENT_KEYS: frozenset[str] = frozenset({
    "full_text",
    "text_ocr",
    "raw_text",
    "ocr_text",
    "raw_ocr_text",
    "document_bytes",
    "document_content",
    "image_content",
    "pdf_content",
    "base64_image",
    "base64_pdf",
})
_FORBIDDEN_LLM_PAYLOAD_KEYS: frozenset[str] = frozenset({
    "prompt",
    "system_prompt",
    "developer_prompt",
    "user_prompt",
    "messages",
    "raw_response",
    "raw_model_response",
    "model_response",
    "completion",
})


# ── Décision humaine (Human-in-the-Loop) ─────────────────────────────────────


class HumanDecision(TypedDict, total=False):
    """Décision saisie par le gestionnaire après interruption LangGraph."""

    actor: str
    decision: str          # "APPROVE" | "REJECT" | "NEEDS_MORE_INFO"
    comment: str
    decided_at: datetime
    target_node: str       # obligatoire si decision == "NEEDS_MORE_INFO" — nœud à relancer


# ── État principal ────────────────────────────────────────────────────────────


class ClaimState(TypedDict, total=False):
    """État partagé passé à travers tous les nœuds du StateGraph.

    Groupes de champs
    -----------------
    **Routage** (case_id, schema_version, intake_status, current_step,
    completed_steps) :
        Champs lus par LangGraph pour décider du prochain nœud et reprendre
        l'exécution depuis un checkpoint.  ``intake_status`` est promu depuis
        ``intake_result`` par le nœud d'ingestion afin d'alimenter les arêtes
        conditionnelles sans exiger la lecture d'un objet Pydantic entier.

    **Entrées consommées** (``*_input``) :
        Chaque nœud lit son entrée dans un champ dédié, puis le remet à
        ``None``.  Cela garantit qu'un chemin absolu temporaire (p. ex.
        ``source_path``) ne persiste pas dans les checkpoints et ne fuie pas
        vers les nœuds suivants.

    **Résultats d'agents** (``*_result``) :
        Un champ par agent, écrasable à chaque invocation du nœud.  LangGraph
        écrase le champ entier — les nœuds ne concaténent jamais un résultat.

    **Historiques append-only** (completed_steps, errors, alerts, audit_trail,
    final_justification) :
        Annotés ``Annotated[list, operator.add]``.  LangGraph fusionne les
        mises à jour par concaténation ; chaque nœud peut ajouter des éléments
        sans connaître les éléments déjà présents.  Ne jamais retourner la
        liste complète — retourner uniquement les nouveaux éléments.

    **HITL** (human_decision) :
        Rempli exclusivement par ``graph.technical_nodes.node_await_human_review``
        via ``langgraph.types.interrupt()`` et une reprise ``Command(resume=...)``.
        Aucun agent ne doit écrire ce champ.

    **Décision finale** (final_recommendation, final_justification) :
        Remplis par ``case_reviewer_agent`` en fin de workflow.

    Reducers utilisés
    -----------------
    ``operator.add`` est le seul reducer défini.  Il concatène deux listes :

        completed_steps + ["security_gate"] → ["claim_intake", "security_gate"]

    LangGraph appelle le reducer pour chaque nœud qui retourne le champ.
    Les champs sans reducer sont écrasés par la dernière valeur reçue.
    Les reducers sont déclarés via ``Annotated[list[T], operator.add]`` dans
    les annotations du TypedDict ; ils sont invisibles à l'exécution Python
    classique mais interprétés par le moteur LangGraph au moment de la fusion.

    Pourquoi les documents bruts sont exclus
    ----------------------------------------
    Le state est sérialisé et persisté dans chaque checkpoint LangGraph.
    Stocker des PDF, des octets d'image ou le texte OCR complet (qui peut
    peser plusieurs Mo) rendrait les checkpoints trop lourds, exposerait des
    données médicales en clair dans la base de données et rendrait la
    désérialisation fragile.  Le texte complet est écrit dans un artefact
    externe (``artifact_id`` / ``artifact_path``) ; le state ne conserve que
    les champs structurés minimaux : codes, scores, métadonnées, hashes SHA-256.

    Différence entre errors et alerts
    ----------------------------------
    ``errors`` (append-only) — conditions **bloquantes** qui empêchent la
    progression du dossier.  Exemples : injection détectée, fichier corrompu,
    rôle inconnu.  Format recommandé : ``"[nom_agent] description concise"``.
    Un dossier avec au moins une entrée dans ``errors`` ne doit pas recevoir
    de recommandation ``APPROVE``.

    ``alerts`` (append-only) — observations **non bloquantes** qui demandent
    une attention humaine ou signalent une anomalie mineure.  Exemples :
    document optionnel absent, confiance OCR limite, préautorisation
    recommandée.  Le workflow continue malgré une alerte.

    Comment un agent retourne une mise à jour partielle
    ----------------------------------------------------
    Un nœud LangGraph retourne **uniquement les clés qu'il modifie**.
    LangGraph fusionne le dict retourné avec le state existant ::

        def node(state: ClaimState) -> dict:
            # Traitement...
            validate_state_update(updates)   # obligatoire avant return
            return {
                "security_result": result,
                "security_input": None,            # champ consommé → None
                "current_step": "privacy",
                "completed_steps": ["security_gate"],  # reducer → append
                "errors": [],                      # liste vide = rien ajouté
            }

    Invariants :
    - Appeler ``validate_state_update(updates)`` avant de retourner.
    - Remettre à ``None`` le champ ``*_input`` consommé.
    - N'inclure dans les listes append-only que les **nouveaux éléments**.
    - Ne jamais inclure bytes, chemins absolus, texte OCR complet ou secrets.
    """

    # ── Informations générales — nécessaires au routage et à la reprise ─────
    case_id: str
    schema_version: str                      # version de ce schéma d'état
    intake_status: IntakeStatus | None       # statut promu pour le routage du graphe
    current_step: str                        # nom du nœud actif
    completed_steps: Annotated[list[str], operator.add]

    # ── Entrée d'ingestion — consommée et vidée par le nœud intake ───────────
    # Ce champ peut contenir source_path (chemin absolu temporaire).
    # Le nœud intake le remet à None après traitement ; aucun autre nœud
    # ne doit le lire.
    intake_input: dict | None

    # ── Entrée sécurité — consommée et vidée par le nœud security_gate ───────
    # Contient text_fields et deterministic_injection_flag.
    # Le nœud security_gate le remet à None après traitement.
    security_input: dict | None

    # ── Entrée privacy — consommée et vidée par le nœud privacy ──────────────
    # Contient role, data_classification, contains_real_personal_data, etc.
    # Le nœud privacy le remet à None après traitement.
    privacy_input: dict | None

    # ── Entrée OCR — consommée et vidée par le nœud document_ocr_agent ───────
    # Contient claim_id, document_id, filename, mime_type, sha256, sanitized_path,
    # security_decision, schema_version, file_index.
    # Le nœud OCR le remet à None après traitement.
    # Invariant : jamais de source_path absolu ni de contenu binaire dans ce champ.
    ocr_input: dict | None

    # ── Entrée identité/couverture — consommée par le nœud identity_coverage ──
    # Contient case_id + fhir_bundle_path (chemin relatif sous incoming/).
    # Les données d'extraction proviennent de ocr_result dans le state.
    identity_coverage_input: dict | None

    # ── Entrée FHIR — consommée par le nœud fhir_validator ───────────────────
    # Contient case_id, fhir_bundle_path (relatif), bundle_expected.
    fhir_input: dict | None

    # ── Entrée codification — consommée par le nœud medical_coding ───────────
    # Contient case_id, procedures (list[str]), medications (list[str]).
    coding_input: dict | None

    # ── Résultats des agents (un par agent, écrasable) ────────────────────────
    intake_result: ClaimIntakeResult | None
    security_result: SecurityGateResult | None
    privacy_result: PrivacyResult | None
    identity_coverage_result: IdentityCoverageResult | None
    fhir_result: FhirValidatorResult | None
    ocr_result: DocumentOcrResult | None
    coding_result: MedicalCodingResult | None
    clinical_result: ClinicalConsistencyResult | None
    fraud_result: FraudDetectionResult | None
    review_result: CaseReviewerResult | None
    audit_result: AuditResult | None

    # ── Erreurs bloquantes (append-only, séparées des alertes) ───────────────
    errors: Annotated[list[str], operator.add]

    # ── Alertes non bloquantes / revue humaine (append-only) ────────────────
    alerts: Annotated[list[str], operator.add]

    # ── Audit (append-only) ───────────────────────────────────────────────────
    audit_trail: Annotated[list[AuditEvent], operator.add]

    # ── Décision humaine (HITL) ───────────────────────────────────────────────
    human_decision: HumanDecision | None

    # ── Compteur de corrections (HITL — route de relance) ────────────────────
    # Incrémenté par node_await_human_review à chaque décision NEEDS_MORE_INFO.
    # Écrasé (pas de reducer) : reflète le nombre total de relances demandées
    # pour ce dossier. Comparé à une limite configurable par route_human_review
    # pour empêcher toute boucle infinie de corrections.
    correction_attempts: int

    # ── Recommandation finale ─────────────────────────────────────────────────
    final_recommendation: Recommendation | None
    final_justification: Annotated[list[str], operator.add]


# ── Garde-fou : validation du contenu du state ───────────────────────────────


def _scan_for_forbidden(value: object, breadcrumb: str) -> list[str]:
    """Retourne la liste des violations trouvées dans value (récursif).

    Interdit : octets bruts, objets fichier ouverts, chemins absolus,
    secrets et documents bruts.
    """
    violations: list[str] = []

    if isinstance(value, (bytes, bytearray)):
        violations.append(
            f"{breadcrumb} : contenu binaire interdit (bytes/{type(value).__name__})"
        )
    elif isinstance(value, io.IOBase):
        violations.append(
            f"{breadcrumb} : objet fichier ouvert interdit ({type(value).__name__})"
        )
    elif isinstance(value, str):
        if _ABSOLUTE_PATH_RE.match(value):
            violations.append(
                f"{breadcrumb} : chemin absolu interdit — {value!r}"
            )
        if _SECRET_HINT_RE.search(value):
            violations.append(
                f"{breadcrumb} : secret potentiel interdit dans le ClaimState"
            )
    elif isinstance(value, dict):
        for k, v in value.items():
            key = str(k)
            child_breadcrumb = f"{breadcrumb}.{key}"
            if _SECRET_KEY_RE.search(key):
                violations.append(
                    f"{child_breadcrumb} : clé secrète interdite dans le ClaimState"
                )
            if key in _RAW_DOCUMENT_KEYS and v not in (None, "", [], {}):
                violations.append(
                    f"{child_breadcrumb} : document brut ou texte OCR complet interdit"
                )
            if key.lower() in _FORBIDDEN_LLM_PAYLOAD_KEYS and v not in (None, "", [], {}):
                violations.append(
                    f"{child_breadcrumb} : prompt, messages ou réponse brute LLM interdits"
                )
            violations.extend(_scan_for_forbidden(v, child_breadcrumb))
    elif isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            violations.extend(_scan_for_forbidden(item, f"{breadcrumb}[{i}]"))
    elif hasattr(value, "model_dump"):
        # Modèle Pydantic — inspecter sa représentation dict
        violations.extend(_scan_for_forbidden(value.model_dump(), breadcrumb))
    # datetime, int, float, bool, None, Enum → autorisés, non inspectés

    return violations


def _check_ocr_result_minimized(result: DocumentOcrResult, breadcrumb: str) -> list[str]:
    """Vérifie que DocumentOcrResult a été minimisé avant d'entrer dans le state.

    Le texte OCR complet et les pages brutes doivent avoir été retirés par
    _minimize_for_state() dans le nœud OCR. Seuls les résultats structurés
    (métadonnées, champs extraits, scores, codes, audit_entry) sont autorisés.

    Violations détectées :
      - full_text non vide → document brut interdit dans le state
      - pages non vide → pages OCR brutes interdites dans le state
      - extraction.full_text non vide → même règle dans la vue détaillée
      - extraction.pages non vide → même règle dans la vue détaillée
    """
    violations: list[str] = []
    if result.full_text:
        violations.append(
            f"{breadcrumb}.full_text : texte OCR complet interdit dans le state "
            f"({len(result.full_text)} caractères) — utiliser _minimize_for_state()"
        )
    if result.pages:
        violations.append(
            f"{breadcrumb}.pages : pages OCR brutes interdites dans le state "
            f"({len(result.pages)} page(s)) — utiliser _minimize_for_state()"
        )
    if result.extraction is not None:
        ext = result.extraction
        if ext.full_text:
            violations.append(
                f"{breadcrumb}.extraction.full_text : texte OCR complet interdit "
                f"({len(ext.full_text)} caractères)"
            )
        if ext.pages:
            violations.append(
                f"{breadcrumb}.extraction.pages : pages OCR brutes interdites "
                f"({len(ext.pages)} page(s))"
            )
    return violations


# Clés consommées (vidées à None) qui sont ignorées dans validate_state_update
_CONSUMED_INPUT_KEYS: frozenset[str] = frozenset({
    "intake_input",
    "security_input",
    "privacy_input",
    "ocr_input",
    "identity_coverage_input",
    "fhir_input",
    "coding_input",
})

_REQUIRED_STATE_KEYS: frozenset[str] = frozenset({
    "case_id",
    "schema_version",
    "current_step",
    "completed_steps",
})

_RESULT_MODELS = {
    "intake_result": ClaimIntakeResult,
    "security_result": SecurityGateResult,
    "privacy_result": PrivacyResult,
    "identity_coverage_result": IdentityCoverageResult,
    "fhir_result": FhirValidatorResult,
    "ocr_result": DocumentOcrResult,
    "coding_result": MedicalCodingResult,
    "clinical_result": ClinicalConsistencyResult,
    "fraud_result": FraudDetectionResult,
    "review_result": CaseReviewerResult,
    "audit_result": AuditResult,
}


def _is_list_of(value: object, item_type: type, *, allow_dict_items: bool = False) -> bool:
    if not isinstance(value, list):
        return False
    return all(
        isinstance(item, item_type)
        or (allow_dict_items and isinstance(item, dict))
        for item in value
    )


def _validate_model_value(key: str, value: object, errors: list[str]) -> None:
    if value is None:
        return
    model = _RESULT_MODELS[key]
    if isinstance(value, model):
        return
    if isinstance(value, dict):
        try:
            model.model_validate(value)
        except Exception as exc:  # noqa: BLE001 - convertit en message de contrat stable.
            errors.append(f"{key} : résultat d'agent invalide ({exc})")
        return
    errors.append(f"{key} : type invalide, attendu {model.__name__} | dict | None")


def validate_claim_state(state: Mapping[str, Any]) -> None:
    """Valide le contrat complet du ClaimState avant checkpoint ou reprise.

    Cette validation complète `validate_state_update` :
    - clés inconnues refusées ;
    - champs minimaux de reprise obligatoires ;
    - types des champs de routage/listes append-only vérifiés ;
    - résultats d'agents validés via leurs modèles Pydantic.
    """
    errors: list[str] = []
    allowed_keys = set(ClaimState.__annotations__)
    unknown = sorted(set(state) - allowed_keys)
    if unknown:
        errors.append(f"champs inconnus interdits : {', '.join(unknown)}")

    missing = sorted(key for key in _REQUIRED_STATE_KEYS if key not in state)
    if missing:
        errors.append(f"champs obligatoires manquants : {', '.join(missing)}")

    if "case_id" in state and not isinstance(state["case_id"], str):
        errors.append("case_id : type invalide, attendu str")
    if "schema_version" in state and not isinstance(state["schema_version"], str):
        errors.append("schema_version : type invalide, attendu str")
    if "current_step" in state and not isinstance(state["current_step"], str):
        errors.append("current_step : type invalide, attendu str")
    if "correction_attempts" in state and not isinstance(state["correction_attempts"], int):
        errors.append("correction_attempts : type invalide, attendu int")
    if "completed_steps" in state and not _is_list_of(state["completed_steps"], str):
        errors.append("completed_steps : type invalide, attendu list[str]")
    if "errors" in state and not _is_list_of(state["errors"], str):
        errors.append("errors : type invalide, attendu list[str]")
    if "alerts" in state and not _is_list_of(state["alerts"], str):
        errors.append("alerts : type invalide, attendu list[str]")
    if "audit_trail" in state and not _is_list_of(
        state["audit_trail"], AuditEvent, allow_dict_items=True
    ):
        errors.append("audit_trail : type invalide, attendu list[AuditEvent]")
    if "final_justification" in state and not _is_list_of(state["final_justification"], str):
        errors.append("final_justification : type invalide, attendu list[str]")

    if "intake_status" in state and state["intake_status"] is not None:
        try:
            IntakeStatus(state["intake_status"])
        except ValueError:
            errors.append("intake_status : valeur invalide")
    if "final_recommendation" in state and state["final_recommendation"] is not None:
        try:
            Recommendation(state["final_recommendation"])
        except ValueError:
            errors.append("final_recommendation : valeur invalide")

    for key in _RESULT_MODELS:
        if key in state:
            _validate_model_value(key, state[key], errors)

    try:
        validate_state_update(dict(state))
    except ValueError as exc:
        errors.append(str(exc))

    if errors:
        raise ValueError(
            "ClaimState invalide :\n"
            + "\n".join(f"  • {error}" for error in errors)
        )


def validate_state_update(updates: dict) -> None:
    """Vérifie qu'une mise à jour du ClaimState ne contient pas de contenu interdit.

    Règles appliquées :
      1. Contenu binaire (bytes, bytearray) → interdit.
      2. Objets fichier ouverts (io.IOBase) → interdit.
      3. Chemins absolus dans les chaînes → interdit.
      4. DocumentOcrResult non minimisé (full_text ou pages non vides) → interdit.

    Les clés d'entrée consommées (intake_input, security_input, privacy_input,
    ocr_input) vidées à None sont ignorées sans inspection.

    Lève ValueError avec la liste des violations détectées.
    À appeler par chaque nœud LangGraph avant de retourner ses mises à jour.
    """
    violations: list[str] = []
    for key, value in updates.items():
        if key in _CONSUMED_INPUT_KEYS and value is None:
            continue
        if key == "ocr_result" and isinstance(value, DocumentOcrResult):
            violations.extend(_check_ocr_result_minimized(value, key))
        violations.extend(_scan_for_forbidden(value, key))

    if violations:
        raise ValueError(
            "Mise à jour du ClaimState refusée — contenu interdit détecté :\n"
            + "\n".join(f"  • {v}" for v in violations)
        )
