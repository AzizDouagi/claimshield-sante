"""Modèles Pydantic du domaine ClaimShield Santé.

Chaque modèle interdit les champs inconnus (extra='forbid') pour détecter
immédiatement toute divergence entre agents.
"""

from __future__ import annotations

from datetime import date, datetime  # noqa: TCH003
from decimal import Decimal
from enum import Enum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ── Enums ────────────────────────────────────────────────────────────────────


class Recommendation(str, Enum):
    APPROVE = "APPROVE"
    REJECT = "REJECT"
    PENDING = "PENDING"


class VerificationStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    PENDING = "PENDING"
    NOT_EVALUATED = "NOT_EVALUATED"


class SecurityDecision(str, Enum):
    ALLOW = "ALLOW"
    BLOCK = "BLOCK"
    QUARANTINE = "QUARANTINE"


class PrivacyDecision(str, Enum):
    """Décision d'accès du Privacy Agent — binaire par design.

    ALLOW — accès accordé (VerificationStatus.PASS ou NEEDS_REVIEW).
            La vue minimisée est produite ; une revue humaine peut être requise.
    BLOCK — accès refusé (VerificationStatus.FAIL).
            Aucune vue n'est produite ; l'agent a bloqué l'accès.
    """

    ALLOW = "ALLOW"
    BLOCK = "BLOCK"


class InputType(str, Enum):
    """Type de l'entrée soumise à l'évaluation de sécurité."""

    FILE = "file"
    TEXT = "text"
    URL = "url"
    TOOL = "tool"
    AGENT_OUTPUT = "agent_output"


class SeverityLevel(str, Enum):
    """Niveau de sévérité d'une anomalie de sécurité."""

    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


class FindingCode(str, Enum):
    """Codes stables identifiant le type d'anomalie de sécurité.

    Ces codes ne changent pas entre versions — seul le message peut évoluer.
    Ils ne contiennent aucune donnée métier ou médicale.
    """

    UNSUPPORTED_EXTENSION = "UNSUPPORTED_EXTENSION"
    UNSUPPORTED_MIME = "UNSUPPORTED_MIME"
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    EMPTY_FILE = "EMPTY_FILE"
    FILE_METADATA_INCOMPLETE = "FILE_METADATA_INCOMPLETE"
    MIME_EXTENSION_MISMATCH = "MIME_EXTENSION_MISMATCH"
    PATH_TRAVERSAL = "PATH_TRAVERSAL"
    ABSOLUTE_PATH_FORBIDDEN = "ABSOLUTE_PATH_FORBIDDEN"
    PATH_NULL_BYTE = "PATH_NULL_BYTE"
    PATH_OUTSIDE_STORAGE = "PATH_OUTSIDE_STORAGE"
    STORAGE_ZONE_FORBIDDEN = "STORAGE_ZONE_FORBIDDEN"
    EXTERNAL_URL_FORBIDDEN = "EXTERNAL_URL_FORBIDDEN"
    PRIVATE_NETWORK_URL = "PRIVATE_NETWORK_URL"
    DANGEROUS_URL_SCHEME = "DANGEROUS_URL_SCHEME"
    MALFORMED_URL = "MALFORMED_URL"
    URL_CREDENTIALS_FORBIDDEN = "URL_CREDENTIALS_FORBIDDEN"
    PROMPT_INJECTION_DETECTED = "PROMPT_INJECTION_DETECTED"
    SECRET_ACCESS_ATTEMPT = "SECRET_ACCESS_ATTEMPT"
    SHELL_ACCESS_ATTEMPT = "SHELL_ACCESS_ATTEMPT"
    UNAUTHORIZED_TOOL = "UNAUTHORIZED_TOOL"
    WRITE_PATH_FORBIDDEN = "WRITE_PATH_FORBIDDEN"
    INVALID_AGENT_OUTPUT = "INVALID_AGENT_OUTPUT"
    SUSPICIOUS_DOCUMENT_CONTENT = "SUSPICIOUS_DOCUMENT_CONTENT"
    SUSPICIOUS_CONTENT = "SUSPICIOUS_CONTENT"
    POLICY_VIOLATION = "POLICY_VIOLATION"

    # Alias conservés pour compatibilité avec les étapes précédentes.
    PROMPT_INJECTION = "PROMPT_INJECTION_DETECTED"
    XSS_ATTEMPT = "SUSPICIOUS_DOCUMENT_CONTENT"
    FORBIDDEN_EXTENSION = "UNSUPPORTED_EXTENSION"
    FORBIDDEN_MIME = "UNSUPPORTED_MIME"


SECURITY_CODE_DESCRIPTIONS: dict[FindingCode, str] = {
    FindingCode.UNSUPPORTED_EXTENSION: "Extension de fichier non autorisée.",
    FindingCode.UNSUPPORTED_MIME: "Type MIME détecté non autorisé.",
    FindingCode.FILE_TOO_LARGE: "Taille réelle du fichier supérieure à la limite.",
    FindingCode.EMPTY_FILE: "Fichier vide refusé.",
    FindingCode.FILE_METADATA_INCOMPLETE: "Métadonnées fichier insuffisantes.",
    FindingCode.MIME_EXTENSION_MISMATCH: "Incohérence entre extension et MIME détecté.",
    FindingCode.PATH_TRAVERSAL: "Tentative de traversée de répertoire.",
    FindingCode.ABSOLUTE_PATH_FORBIDDEN: "Chemin absolu interdit.",
    FindingCode.PATH_NULL_BYTE: "Caractère nul interdit dans un chemin.",
    FindingCode.PATH_OUTSIDE_STORAGE: "Chemin résolu hors de la racine storage.",
    FindingCode.STORAGE_ZONE_FORBIDDEN: "Zone de stockage non autorisée.",
    FindingCode.EXTERNAL_URL_FORBIDDEN: "URL externe refusée par défaut.",
    FindingCode.PRIVATE_NETWORK_URL: "URL vers localhost, loopback ou réseau privé.",
    FindingCode.DANGEROUS_URL_SCHEME: "Schéma d'URL dangereux ou non autorisé.",
    FindingCode.MALFORMED_URL: "URL absente ou malformée.",
    FindingCode.URL_CREDENTIALS_FORBIDDEN: "Identifiants présents dans l'URL.",
    FindingCode.PROMPT_INJECTION_DETECTED: "Tentative d'injection de prompt détectée.",
    FindingCode.SECRET_ACCESS_ATTEMPT: "Tentative d'accès à un secret.",
    FindingCode.SHELL_ACCESS_ATTEMPT: "Tentative d'accès shell, terminal ou commande.",
    FindingCode.UNAUTHORIZED_TOOL: "Outil ou agent demandeur non autorisé.",
    FindingCode.WRITE_PATH_FORBIDDEN: "Écriture demandée hors des zones autorisées.",
    FindingCode.INVALID_AGENT_OUTPUT: "Sortie d'agent invalide ou dangereuse.",
    FindingCode.SUSPICIOUS_DOCUMENT_CONTENT: "Contenu documentaire suspect.",
    FindingCode.SUSPICIOUS_CONTENT: "Contenu suspect non classé plus précisément.",
    FindingCode.POLICY_VIOLATION: "Violation générique de politique de sécurité.",
}

SECURITY_CODE_SEVERITIES: dict[FindingCode, SeverityLevel] = {
    FindingCode.UNSUPPORTED_EXTENSION: SeverityLevel.HIGH,
    FindingCode.UNSUPPORTED_MIME: SeverityLevel.HIGH,
    FindingCode.FILE_TOO_LARGE: SeverityLevel.HIGH,
    FindingCode.EMPTY_FILE: SeverityLevel.HIGH,
    FindingCode.FILE_METADATA_INCOMPLETE: SeverityLevel.HIGH,
    FindingCode.MIME_EXTENSION_MISMATCH: SeverityLevel.MEDIUM,
    FindingCode.PATH_TRAVERSAL: SeverityLevel.CRITICAL,
    FindingCode.ABSOLUTE_PATH_FORBIDDEN: SeverityLevel.CRITICAL,
    FindingCode.PATH_NULL_BYTE: SeverityLevel.CRITICAL,
    FindingCode.PATH_OUTSIDE_STORAGE: SeverityLevel.CRITICAL,
    FindingCode.STORAGE_ZONE_FORBIDDEN: SeverityLevel.HIGH,
    FindingCode.EXTERNAL_URL_FORBIDDEN: SeverityLevel.HIGH,
    FindingCode.PRIVATE_NETWORK_URL: SeverityLevel.CRITICAL,
    FindingCode.DANGEROUS_URL_SCHEME: SeverityLevel.HIGH,
    FindingCode.MALFORMED_URL: SeverityLevel.HIGH,
    FindingCode.URL_CREDENTIALS_FORBIDDEN: SeverityLevel.HIGH,
    FindingCode.PROMPT_INJECTION_DETECTED: SeverityLevel.CRITICAL,
    FindingCode.SECRET_ACCESS_ATTEMPT: SeverityLevel.CRITICAL,
    FindingCode.SHELL_ACCESS_ATTEMPT: SeverityLevel.CRITICAL,
    FindingCode.UNAUTHORIZED_TOOL: SeverityLevel.CRITICAL,
    FindingCode.WRITE_PATH_FORBIDDEN: SeverityLevel.HIGH,
    FindingCode.INVALID_AGENT_OUTPUT: SeverityLevel.HIGH,
    FindingCode.SUSPICIOUS_DOCUMENT_CONTENT: SeverityLevel.HIGH,
    FindingCode.SUSPICIOUS_CONTENT: SeverityLevel.MEDIUM,
    FindingCode.POLICY_VIOLATION: SeverityLevel.MEDIUM,
}


class PrivacyCode(str, Enum):
    """Codes stables du Privacy Agent — identifient la cause d'une décision ou d'une erreur.

    Ces codes ne changent pas entre versions — seul le message associé peut évoluer.
    Ils ne contiennent aucune donnée personnelle, médicale ni confidentielle.
    Utilisés dans PrivacyResult.reason_codes et PrivacyAuditEntry.reason_codes.
    """

    MISSING_ROLE = "MISSING_ROLE"
    UNKNOWN_ROLE = "UNKNOWN_ROLE"
    UNKNOWN_POLICY = "UNKNOWN_POLICY"
    MISSING_PSEUDONYMIZATION_KEY = "MISSING_PSEUDONYMIZATION_KEY"
    UNMASKED_IDENTIFIER = "UNMASKED_IDENTIFIER"
    FORBIDDEN_FIELD_EXPOSED = "FORBIDDEN_FIELD_EXPOSED"
    INVALID_PRIVACY_INPUT = "INVALID_PRIVACY_INPUT"
    INVALID_PRIVACY_OUTPUT = "INVALID_PRIVACY_OUTPUT"
    PSEUDONYMIZATION_ERROR = "PSEUDONYMIZATION_ERROR"


PRIVACY_CODE_DESCRIPTIONS: dict[PrivacyCode, str] = {
    PrivacyCode.MISSING_ROLE: (
        "Rôle absent — champ obligatoire pour accéder à une vue minimisée."
    ),
    PrivacyCode.UNKNOWN_ROLE: (
        "Rôle inconnu — valeur hors de l'énumération ReaderRole."
    ),
    PrivacyCode.UNKNOWN_POLICY: (
        "Politique d'accès introuvable pour ce rôle — incohérence interne."
    ),
    PrivacyCode.MISSING_PSEUDONYMIZATION_KEY: (
        "Clé HMAC de pseudonymisation inaccessible ou vide — blocage préventif."
    ),
    PrivacyCode.UNMASKED_IDENTIFIER: (
        "Identifiant sans le préfixe de pseudonymisation attendu (PAT-…, PRV-…)."
    ),
    PrivacyCode.FORBIDDEN_FIELD_EXPOSED: (
        "Champ secret ou identifiant personnel brut exposé dans la vue minimisée."
    ),
    PrivacyCode.INVALID_PRIVACY_INPUT: (
        "Entrée PrivacyInput invalide selon le schéma Pydantic — validation échouée."
    ),
    PrivacyCode.INVALID_PRIVACY_OUTPUT: (
        "Vue minimisée invalide — ne respecte pas le schéma Pydantic du rôle."
    ),
    PrivacyCode.PSEUDONYMIZATION_ERROR: (
        "Erreur technique lors de la pseudonymisation — traitement bloqué."
    ),
}


class DataClassification(str, Enum):
    SYNTHETIC_TEST_DATA = "SYNTHETIC_TEST_DATA"
    ANONYMIZED = "ANONYMIZED"
    CONFIDENTIAL = "CONFIDENTIAL"


class ReaderRole(str, Enum):
    """Rôle du lecteur déterminant la vue minimisée autorisée.

    Quatre rôles stables — politique par défaut DENY.
    Toute valeur absente de cette énumération est automatiquement rejetée.
    Aucun rôle n'a accès à l'intégralité des champs connus du domaine.
    """

    ADMINISTRATIVE_MANAGER = "ADMINISTRATIVE_MANAGER"
    MEDICAL_REVIEWER = "MEDICAL_REVIEWER"
    FRAUD_ANALYST = "FRAUD_ANALYST"
    AUDITOR = "AUDITOR"


class AuthorizationStatus(str, Enum):
    APPROVED = "approved"
    PENDING = "pending"
    NOT_REQUIRED = "not_required"
    REJECTED = "rejected"


class IntakeStatus(str, Enum):
    """Statut global du dossier d'ingestion."""

    ACCEPTED = "accepted"
    QUARANTINED = "quarantined"
    BLOCKED = "blocked"
    ERROR = "error"


class FileStatus(str, Enum):
    """Statut d'un fichier individuel après inspection.

    DUPLICATE et ERROR n'existent qu'au niveau fichier ;
    ils remontent respectivement en QUARANTINED et ERROR au niveau dossier.
    """

    ACCEPTED = "accepted"
    QUARANTINED = "quarantined"
    BLOCKED = "blocked"
    DUPLICATE = "duplicate"
    ERROR = "error"


class IntakeReasonCode(str, Enum):
    """Codes stables identifiant la cause d'un rejet ou d'une alerte d'ingestion.

    Chaque valeur correspond à une entrée dans REASON_DESCRIPTIONS.
    Les codes ne changent pas entre versions — seul le message peut évoluer.
    """

    EMPTY_CLAIM = "EMPTY_CLAIM"
    EMPTY_FILE = "EMPTY_FILE"
    UNSUPPORTED_EXTENSION = "UNSUPPORTED_EXTENSION"
    UNSUPPORTED_MIME_TYPE = "UNSUPPORTED_MIME_TYPE"
    MIME_EXTENSION_MISMATCH = "MIME_EXTENSION_MISMATCH"
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    CLAIM_TOO_LARGE = "CLAIM_TOO_LARGE"
    PATH_TRAVERSAL_ATTEMPT = "PATH_TRAVERSAL_ATTEMPT"
    DUPLICATE_FILE = "DUPLICATE_FILE"
    STORAGE_ERROR = "STORAGE_ERROR"
    TOO_MANY_FILES = "TOO_MANY_FILES"
    FOLDER_QUOTA_EXCEEDED = "FOLDER_QUOTA_EXCEEDED"
    INVALID_FILENAME = "INVALID_FILENAME"


REASON_DESCRIPTIONS: dict[str, str] = {
    IntakeReasonCode.EMPTY_CLAIM: (
        "Dossier vide — aucun fichier soumis"
    ),
    IntakeReasonCode.EMPTY_FILE: (
        "Fichier vide (0 octet) — le contenu est absent"
    ),
    IntakeReasonCode.UNSUPPORTED_EXTENSION: (
        "Extension non autorisée — seuls PDF, PNG, JPEG et JSON sont acceptés"
    ),
    IntakeReasonCode.UNSUPPORTED_MIME_TYPE: (
        "Type MIME non autorisé — le contenu réel du fichier n'est pas reconnu"
    ),
    IntakeReasonCode.MIME_EXTENSION_MISMATCH: (
        "Incohérence MIME/extension — le contenu détecté ne correspond pas à l'extension déclarée"
    ),
    IntakeReasonCode.FILE_TOO_LARGE: (
        "Fichier trop volumineux — dépasse la limite de taille individuelle configurée"
    ),
    IntakeReasonCode.CLAIM_TOO_LARGE: (
        "Dossier trop volumineux — le quota cumulé est dépassé"
    ),
    IntakeReasonCode.PATH_TRAVERSAL_ATTEMPT: (
        "Tentative de traversée de répertoire — nom de fichier dangereux refusé"
    ),
    IntakeReasonCode.DUPLICATE_FILE: (
        "Fichier en double — SHA-256 identique à un fichier déjà reçu dans ce dossier"
    ),
    IntakeReasonCode.STORAGE_ERROR: (
        "Échec technique de stockage — écriture ou déplacement impossible"
    ),
    IntakeReasonCode.TOO_MANY_FILES: (
        "Trop de fichiers — le nombre maximum de fichiers par dossier est atteint"
    ),
    IntakeReasonCode.FOLDER_QUOTA_EXCEEDED: (
        "Quota dépassé — la taille cumulée du dossier dépasse la limite configurée"
    ),
    IntakeReasonCode.INVALID_FILENAME: (
        "Nom de fichier invalide — caractères ou structure non autorisés"
    ),
}


# ── Montants ─────────────────────────────────────────────────────────────────

PositiveDecimal = Annotated[Decimal, Field(gt=Decimal("0"))]
NonNegativeDecimal = Annotated[Decimal, Field(ge=Decimal("0"))]


# ── Enums Document/OCR Agent ─────────────────────────────────────────────────


class DocumentType(str, Enum):
    """Type de document médical identifié par l'agent OCR.

    Les valeurs sont stables entre versions — seul le message peut évoluer.
    La classification combine : nom de fichier > type MIME > mots-clés.
    """

    INVOICE = "INVOICE"               # facture médicale
    PRESCRIPTION = "PRESCRIPTION"     # ordonnance médicale
    CLAIM_REQUEST = "CLAIM_REQUEST"   # demande de remboursement
    FHIR_BUNDLE = "FHIR_BUNDLE"       # bundle FHIR R4 (JSON)
    UNKNOWN = "UNKNOWN"               # type non identifié
    UNSUPPORTED = "UNSUPPORTED"       # type MIME non géré par l'agent


class OcrSource(str, Enum):
    """Origine de l'extraction textuelle pour la traçabilité de provenance."""

    PDF_TEXT = "PDF_TEXT"        # texte natif extrait de pypdf
    PDF_OCR = "PDF_OCR"          # PDF scanné — OCR appliqué sur les images de pages
    IMAGE_OCR = "IMAGE_OCR"      # fichier image (PNG/JPEG) — OCR appliqué
    UNSUPPORTED = "UNSUPPORTED"  # type MIME non pris en charge par l'agent


class OcrCode(str, Enum):
    """Codes stables identifiant les cas d'erreur et d'alerte du Document/OCR Agent.

    Ces codes ne changent pas entre versions — seul le message peut évoluer.
    Ils ne contiennent aucune donnée personnelle ni médicale.

    Codes historiques (Étapes 4–17) :
      SECURITY_GATE_NOT_ALLOW, FILE_NOT_IN_INCOMING, SHA256_MISMATCH,
      UNSUPPORTED_MIME_TYPE, PDF_EXTRACTION_ERROR, OCR_ENGINE_UNAVAILABLE,
      OCR_EXTRACTION_ERROR, UNREADABLE_DOCUMENT, INVALID_OCR_INPUT, OCR_TEXT_SUSPICIOUS

    Codes étendus (Étape 18) — granularité fine par étape du pipeline :
      DOCUMENT_NOT_FOUND, DOCUMENT_NOT_ALLOWED, DOCUMENT_HASH_MISMATCH,
      UNSUPPORTED_DOCUMENT_TYPE, PDF_READ_ERROR, PDF_ENCRYPTED, IMAGE_READ_ERROR,
      OCR_UNAVAILABLE, OCR_FAILED, EMPTY_EXTRACTED_TEXT, DOCUMENT_CLASSIFICATION_FAILED,
      PARSER_FAILED, REQUIRED_FIELD_MISSING, LOW_CONFIDENCE, AMBIGUOUS_VALUE,
      INVALID_DATE, INVALID_AMOUNT, HIDDEN_PROMPT_INJECTION, INVALID_OCR_OUTPUT
    """

    # ── Codes historiques (Étapes 4–17) ──────────────────────────────────────
    SECURITY_GATE_NOT_ALLOW = "SECURITY_GATE_NOT_ALLOW"
    FILE_NOT_IN_INCOMING = "FILE_NOT_IN_INCOMING"
    SHA256_MISMATCH = "SHA256_MISMATCH"
    UNSUPPORTED_MIME_TYPE = "UNSUPPORTED_MIME_TYPE"
    PDF_EXTRACTION_ERROR = "PDF_EXTRACTION_ERROR"
    OCR_ENGINE_UNAVAILABLE = "OCR_ENGINE_UNAVAILABLE"
    OCR_EXTRACTION_ERROR = "OCR_EXTRACTION_ERROR"
    UNREADABLE_DOCUMENT = "UNREADABLE_DOCUMENT"
    INVALID_OCR_INPUT = "INVALID_OCR_INPUT"
    OCR_TEXT_SUSPICIOUS = "OCR_TEXT_SUSPICIOUS"

    # ── Codes étendus (Étape 18) — granularité fine par étape du pipeline ────
    DOCUMENT_NOT_FOUND = "DOCUMENT_NOT_FOUND"
    DOCUMENT_NOT_ALLOWED = "DOCUMENT_NOT_ALLOWED"
    DOCUMENT_HASH_MISMATCH = "DOCUMENT_HASH_MISMATCH"
    UNSUPPORTED_DOCUMENT_TYPE = "UNSUPPORTED_DOCUMENT_TYPE"
    PDF_READ_ERROR = "PDF_READ_ERROR"
    PDF_ENCRYPTED = "PDF_ENCRYPTED"
    IMAGE_READ_ERROR = "IMAGE_READ_ERROR"
    OCR_UNAVAILABLE = "OCR_UNAVAILABLE"
    OCR_FAILED = "OCR_FAILED"
    EMPTY_EXTRACTED_TEXT = "EMPTY_EXTRACTED_TEXT"
    DOCUMENT_CLASSIFICATION_FAILED = "DOCUMENT_CLASSIFICATION_FAILED"
    PARSER_FAILED = "PARSER_FAILED"
    REQUIRED_FIELD_MISSING = "REQUIRED_FIELD_MISSING"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    AMBIGUOUS_VALUE = "AMBIGUOUS_VALUE"
    INVALID_DATE = "INVALID_DATE"
    INVALID_AMOUNT = "INVALID_AMOUNT"
    HIDDEN_PROMPT_INJECTION = "HIDDEN_PROMPT_INJECTION"
    INVALID_OCR_OUTPUT = "INVALID_OCR_OUTPUT"


OCR_CODE_DESCRIPTIONS: dict[str, str] = {
    OcrCode.SECURITY_GATE_NOT_ALLOW: "Le Security Gate n'a pas accordé l'autorisation ALLOW.",
    OcrCode.FILE_NOT_IN_INCOMING: "Le fichier n'est pas dans la zone incoming/ assainie.",
    OcrCode.SHA256_MISMATCH: "L'empreinte SHA-256 du fichier ne correspond pas aux métadonnées.",
    OcrCode.UNSUPPORTED_MIME_TYPE: "Le type MIME du fichier n'est pas pris en charge par l'agent OCR.",
    OcrCode.PDF_EXTRACTION_ERROR: "Erreur lors de l'extraction du texte natif du PDF.",
    OcrCode.OCR_ENGINE_UNAVAILABLE: "Le moteur OCR (Tesseract) n'est pas disponible.",
    OcrCode.OCR_EXTRACTION_ERROR: "Erreur lors de l'extraction OCR du document.",
    OcrCode.UNREADABLE_DOCUMENT: "Le document est illisible — score de confiance insuffisant.",
    OcrCode.INVALID_OCR_INPUT: "L'entrée de l'agent OCR est invalide (erreur de validation Pydantic).",
    OcrCode.OCR_TEXT_SUSPICIOUS: "Le texte extrait contient une instruction ou exfiltration suspecte.",
}

OCR_ERROR_CODE_DESCRIPTIONS: dict[OcrCode, str] = {
    # ── Codes historiques ──────────────────────────────────────────────────
    OcrCode.SECURITY_GATE_NOT_ALLOW: (
        "Le Security Gate n'a pas accordé l'autorisation ALLOW — extraction bloquée."
    ),
    OcrCode.FILE_NOT_IN_INCOMING: (
        "Le fichier n'est pas dans la zone incoming/ assainie — accès refusé."
    ),
    OcrCode.SHA256_MISMATCH: (
        "L'empreinte SHA-256 du fichier ne correspond pas aux métadonnées — intégrité compromise."
    ),
    OcrCode.UNSUPPORTED_MIME_TYPE: (
        "Le type MIME du fichier n'est pas pris en charge par l'agent OCR."
    ),
    OcrCode.PDF_EXTRACTION_ERROR: (
        "Erreur lors de l'extraction du texte natif ou des images du PDF."
    ),
    OcrCode.OCR_ENGINE_UNAVAILABLE: (
        "Le moteur OCR (Tesseract) n'est pas disponible sur ce système."
    ),
    OcrCode.OCR_EXTRACTION_ERROR: (
        "Erreur lors de l'extraction OCR du document — résultat partiel ou nul."
    ),
    OcrCode.UNREADABLE_DOCUMENT: (
        "Le document est illisible — score de confiance global insuffisant."
    ),
    OcrCode.INVALID_OCR_INPUT: (
        "L'entrée DocumentOcrInput est invalide — erreur de validation Pydantic."
    ),
    OcrCode.OCR_TEXT_SUSPICIOUS: (
        "Le texte extrait contient une instruction ou tentative d'exfiltration suspecte."
    ),
    # ── Codes étendus (Étape 18) ───────────────────────────────────────────
    OcrCode.DOCUMENT_NOT_FOUND: (
        "Le document demandé est introuvable dans la zone de stockage assainie."
    ),
    OcrCode.DOCUMENT_NOT_ALLOWED: (
        "Le document n'est pas autorisé par la politique de sécurité en vigueur."
    ),
    OcrCode.DOCUMENT_HASH_MISMATCH: (
        "L'empreinte SHA-256 calculée ne correspond pas à l'empreinte déclarée dans le manifeste."
    ),
    OcrCode.UNSUPPORTED_DOCUMENT_TYPE: (
        "Le type de document identifié n'est pas pris en charge par ce pipeline."
    ),
    OcrCode.PDF_READ_ERROR: (
        "Erreur de lecture du fichier PDF — fichier corrompu ou inaccessible."
    ),
    OcrCode.PDF_ENCRYPTED: (
        "Le PDF est protégé par un mot de passe — déchiffrement requis avant extraction."
    ),
    OcrCode.IMAGE_READ_ERROR: (
        "Erreur de lecture du fichier image — fichier corrompu ou format non supporté."
    ),
    OcrCode.OCR_UNAVAILABLE: (
        "Le moteur OCR n'est pas disponible ou n'a pas pu être initialisé."
    ),
    OcrCode.OCR_FAILED: (
        "L'extraction OCR a échoué — moteur disponible mais traitement impossible."
    ),
    OcrCode.EMPTY_EXTRACTED_TEXT: (
        "Le texte extrait est vide — aucune donnée textuelle exploitable dans le document."
    ),
    OcrCode.DOCUMENT_CLASSIFICATION_FAILED: (
        "Impossible de déterminer le type du document à partir du contenu et des métadonnées."
    ),
    OcrCode.PARSER_FAILED: (
        "Le parseur de champs n'a pas pu analyser la structure du document."
    ),
    OcrCode.REQUIRED_FIELD_MISSING: (
        "Un ou plusieurs champs essentiels sont absents du document extrait."
    ),
    OcrCode.LOW_CONFIDENCE: (
        "Le score de confiance de l'extraction est inférieur au seuil minimal acceptable."
    ),
    OcrCode.AMBIGUOUS_VALUE: (
        "Une valeur extraite est ambiguë et ne peut pas être interprétée de façon déterministe."
    ),
    OcrCode.INVALID_DATE: (
        "Une date extraite est invalide ou ne respecte pas le format attendu."
    ),
    OcrCode.INVALID_AMOUNT: (
        "Un montant extrait est invalide, négatif ou non conforme au format monétaire attendu."
    ),
    OcrCode.HIDDEN_PROMPT_INJECTION: (
        "Une tentative d'injection de prompt masquée a été détectée dans le texte extrait."
    ),
    OcrCode.INVALID_OCR_OUTPUT: (
        "La sortie produite par l'agent OCR ne respecte pas le schéma Pydantic attendu."
    ),
}

OCR_ERROR_CODE_SEVERITIES: dict[OcrCode, SeverityLevel] = {
    # ── Codes historiques ──────────────────────────────────────────────────
    OcrCode.SECURITY_GATE_NOT_ALLOW: SeverityLevel.CRITICAL,
    OcrCode.FILE_NOT_IN_INCOMING: SeverityLevel.HIGH,
    OcrCode.SHA256_MISMATCH: SeverityLevel.CRITICAL,
    OcrCode.UNSUPPORTED_MIME_TYPE: SeverityLevel.HIGH,
    OcrCode.PDF_EXTRACTION_ERROR: SeverityLevel.HIGH,
    OcrCode.OCR_ENGINE_UNAVAILABLE: SeverityLevel.HIGH,
    OcrCode.OCR_EXTRACTION_ERROR: SeverityLevel.HIGH,
    OcrCode.UNREADABLE_DOCUMENT: SeverityLevel.MEDIUM,
    OcrCode.INVALID_OCR_INPUT: SeverityLevel.HIGH,
    OcrCode.OCR_TEXT_SUSPICIOUS: SeverityLevel.CRITICAL,
    # ── Codes étendus (Étape 18) ───────────────────────────────────────────
    OcrCode.DOCUMENT_NOT_FOUND: SeverityLevel.HIGH,
    OcrCode.DOCUMENT_NOT_ALLOWED: SeverityLevel.HIGH,
    OcrCode.DOCUMENT_HASH_MISMATCH: SeverityLevel.CRITICAL,
    OcrCode.UNSUPPORTED_DOCUMENT_TYPE: SeverityLevel.MEDIUM,
    OcrCode.PDF_READ_ERROR: SeverityLevel.HIGH,
    OcrCode.PDF_ENCRYPTED: SeverityLevel.HIGH,
    OcrCode.IMAGE_READ_ERROR: SeverityLevel.HIGH,
    OcrCode.OCR_UNAVAILABLE: SeverityLevel.HIGH,
    OcrCode.OCR_FAILED: SeverityLevel.HIGH,
    OcrCode.EMPTY_EXTRACTED_TEXT: SeverityLevel.MEDIUM,
    OcrCode.DOCUMENT_CLASSIFICATION_FAILED: SeverityLevel.MEDIUM,
    OcrCode.PARSER_FAILED: SeverityLevel.HIGH,
    OcrCode.REQUIRED_FIELD_MISSING: SeverityLevel.MEDIUM,
    OcrCode.LOW_CONFIDENCE: SeverityLevel.LOW,
    OcrCode.AMBIGUOUS_VALUE: SeverityLevel.LOW,
    OcrCode.INVALID_DATE: SeverityLevel.LOW,
    OcrCode.INVALID_AMOUNT: SeverityLevel.LOW,
    OcrCode.HIDDEN_PROMPT_INJECTION: SeverityLevel.CRITICAL,
    OcrCode.INVALID_OCR_OUTPUT: SeverityLevel.HIGH,
}

OCR_ERROR_CODE_RETRYABLE: dict[OcrCode, bool] = {
    # ── Codes historiques ──────────────────────────────────────────────────
    OcrCode.SECURITY_GATE_NOT_ALLOW: False,
    OcrCode.FILE_NOT_IN_INCOMING: False,
    OcrCode.SHA256_MISMATCH: False,
    OcrCode.UNSUPPORTED_MIME_TYPE: False,
    OcrCode.PDF_EXTRACTION_ERROR: True,   # peut être une erreur I/O transitoire
    OcrCode.OCR_ENGINE_UNAVAILABLE: True, # le moteur peut redevenir disponible
    OcrCode.OCR_EXTRACTION_ERROR: True,   # peut être une erreur transitoire
    OcrCode.UNREADABLE_DOCUMENT: False,
    OcrCode.INVALID_OCR_INPUT: False,
    OcrCode.OCR_TEXT_SUSPICIOUS: False,
    # ── Codes étendus (Étape 18) ───────────────────────────────────────────
    OcrCode.DOCUMENT_NOT_FOUND: False,    # fichier absent — pas de retry utile
    OcrCode.DOCUMENT_NOT_ALLOWED: False,  # politique — décision humaine requise
    OcrCode.DOCUMENT_HASH_MISMATCH: False, # violation d'intégrité — pas de retry
    OcrCode.UNSUPPORTED_DOCUMENT_TYPE: False,
    OcrCode.PDF_READ_ERROR: True,         # peut être une erreur I/O transitoire
    OcrCode.PDF_ENCRYPTED: False,         # nécessite une clé — décision humaine
    OcrCode.IMAGE_READ_ERROR: True,       # peut être une erreur I/O transitoire
    OcrCode.OCR_UNAVAILABLE: True,        # le moteur peut redevenir disponible
    OcrCode.OCR_FAILED: True,             # peut être une erreur transitoire
    OcrCode.EMPTY_EXTRACTED_TEXT: False,  # document sans contenu textuel
    OcrCode.DOCUMENT_CLASSIFICATION_FAILED: False,
    OcrCode.PARSER_FAILED: False,         # structure du document non conforme
    OcrCode.REQUIRED_FIELD_MISSING: False,
    OcrCode.LOW_CONFIDENCE: False,        # qualité insuffisante — revue humaine
    OcrCode.AMBIGUOUS_VALUE: False,       # revue humaine requise
    OcrCode.INVALID_DATE: False,
    OcrCode.INVALID_AMOUNT: False,
    OcrCode.HIDDEN_PROMPT_INJECTION: False, # menace — pas de retry
    OcrCode.INVALID_OCR_OUTPUT: False,    # schéma invalide — correctif requis
}


class ExtractionStatus(str, Enum):
    """Statut fin de l'étape d'extraction du Document/OCR Agent.

    Complète VerificationStatus (signal LangGraph générique) avec une
    sémantique propre à l'OCR. Les deux coexistent dans DocumentOcrResult.

    SUCCESS      — tous les champs essentiels extraits, confiance ≥ 0.65
    NEEDS_REVIEW — document lisible mais extraction partielle ou incertaine
    FAILED       — document illisible ou erreur moteur OCR
    SKIPPED      — fichier non-OCR (ex : JSON FHIR traité par un autre agent)
    BLOCKED      — menace détectée (gate non-ALLOW, SHA mismatch, zone invalide)
    """

    SUCCESS = "SUCCESS"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    BLOCKED = "BLOCKED"


EXTRACTION_STATUS_DESCRIPTIONS: dict[str, str] = {
    ExtractionStatus.SUCCESS: "Tous les champs essentiels extraits avec une confiance suffisante.",
    ExtractionStatus.NEEDS_REVIEW: "Document lisible, mais extraction partielle ou ambiguë — revue humaine recommandée.",
    ExtractionStatus.FAILED: "Document illisible ou erreur critique lors de l'extraction.",
    ExtractionStatus.SKIPPED: "Extraction OCR non applicable pour ce type de fichier (traité par un autre agent).",
    ExtractionStatus.BLOCKED: "Extraction bloquée : menace détectée ou pré-condition de sécurité non satisfaite.",
}


# ── Modèles de base ──────────────────────────────────────────────────────────


class StrictModel(BaseModel):
    """Classe de base : champs inconnus interdits, assignation validée."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


# ── Patient ───────────────────────────────────────────────────────────────────


class PatientInfo(StrictModel):
    patient_id: str = Field(..., description="UUID Synthea du patient")
    patient_name: str = Field(..., min_length=1)
    birth_date: date | None = None
    gender: str | None = None


# ── Couverture assurance ──────────────────────────────────────────────────────


class CoverageInfo(StrictModel):
    payer_name: str = Field(..., min_length=1)
    coverage_rate: Decimal = Field(..., ge=Decimal("0"), le=Decimal("1"))
    policy_active: bool = True
    policy_start: date | None = None
    policy_end: date | None = None

    @field_validator("coverage_rate", mode="before")
    @classmethod
    def parse_coverage_rate(cls, v: object) -> Decimal:
        return Decimal(str(v))


# ── Documents ─────────────────────────────────────────────────────────────────


class DocumentInfo(StrictModel):
    filename: str
    sha256: str = Field(..., min_length=64, max_length=64)
    size_bytes: int = Field(..., gt=0)
    mime_type: str


# ── Données extraites par l'agent OCR ────────────────────────────────────────


class ExtractedData(StrictModel):
    """Données extraites consolidées d'un dossier de remboursement.

    Garanties Étape 19 :
      - Montants (`total_billed`, `amount_requested`, `patient_share`) en `Decimal`
        — jamais de `float` pour éviter les erreurs d'arrondi monétaire.
      - Dates (`service_date`) en `date` Python — jamais de chaîne brute.
      - Provenance : `dict[str, str]` mappant chaque nom de champ vers
        "filename:page_N" (source de l'extraction).
      - Champs inconnus interdits via `StrictModel` (`extra="forbid"`) —
        toute clé hors schéma lève immédiatement une `ValidationError`.
    """

    patient_name: str | None = None
    patient_id: str | None = None
    payer_name: str | None = None
    service_date: date | None = None
    claim_reference: str | None = None
    invoice_number: str | None = None
    prescription_number: str | None = None
    procedure_count: int | None = Field(default=None, ge=0)
    medication_count: int | None = Field(default=None, ge=0)
    total_billed: Decimal | None = Field(default=None, ge=Decimal("0"))
    amount_requested: Decimal | None = Field(default=None, ge=Decimal("0"))
    patient_share: Decimal | None = Field(default=None, ge=Decimal("0"))
    currency: str = "USD"
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    provenance: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Provenance champ-par-champ : clé = nom du champ, "
            "valeur = 'nom_fichier:page_N' indiquant la source de l'extraction"
        ),
    )

    @field_validator("total_billed", "amount_requested", "patient_share", mode="before")
    @classmethod
    def parse_decimal(cls, v: object) -> Decimal | None:
        if v is None:
            return None
        return Decimal(str(v))


# ── Prestataire de soins ──────────────────────────────────────────────────────


class ProviderInfo(StrictModel):
    provider_id: str
    organization_id: str | None = None
    name: str | None = None
    specialty: str | None = None


# ── Consultation médicale ─────────────────────────────────────────────────────


class EncounterInfo(StrictModel):
    encounter_id: str
    encounter_class: str = Field(..., description="ambulatory, inpatient, emergency…")
    start: datetime
    stop: datetime | None = None
    patient_id: str
    provider: ProviderInfo | None = None
    diagnosis_codes: list[str] = Field(default_factory=list)

    @field_validator("start", "stop", mode="before")
    @classmethod
    def parse_dt(cls, v: object) -> datetime | None:
        if v is None:
            return None
        if isinstance(v, datetime):
            return v
        return datetime.fromisoformat(str(v).replace("Z", "+00:00"))


# ── Acte médical ──────────────────────────────────────────────────────────────


class MedicalProcedure(StrictModel):
    code: str = Field(..., description="Code SNOMED CT ou CIM-10")
    description: str
    unit_cost: NonNegativeDecimal
    quantity: int = Field(default=1, ge=1)
    performed_date: date | None = None

    @field_validator("unit_cost", mode="before")
    @classmethod
    def parse_cost(cls, v: object) -> Decimal:
        return Decimal(str(v))


# ── Prescription ──────────────────────────────────────────────────────────────


class Prescription(StrictModel):
    medication_code: str
    medication_name: str
    dispenses: int = Field(default=1, ge=1)
    unit_cost: NonNegativeDecimal

    @field_validator("unit_cost", mode="before")
    @classmethod
    def parse_cost(cls, v: object) -> Decimal:
        return Decimal(str(v))


# ── Règles déterministes ──────────────────────────────────────────────────────


class DeterministicRules(StrictModel):
    coverage_rate: Decimal = Field(..., ge=Decimal("0"), le=Decimal("1"))
    authorization_required: bool
    authorization_status: AuthorizationStatus
    duplicate_invoice: bool
    prompt_injection_detected: bool

    @field_validator("coverage_rate", mode="before")
    @classmethod
    def parse_rate(cls, v: object) -> Decimal:
        return Decimal(str(v))


# ── Dossier de remboursement ──────────────────────────────────────────────────


class ClaimSubmission(StrictModel):
    case_id: str = Field(..., pattern=r"^CLM-\d{4,}$")
    schema_version: str = "1.0.0"
    data_classification: DataClassification = DataClassification.SYNTHETIC_TEST_DATA
    contains_real_personal_data: bool = False
    submitted_at: datetime | None = None
    patient: PatientInfo | None = None
    coverage: CoverageInfo | None = None
    encounter: EncounterInfo | None = None
    documents: list[DocumentInfo] = Field(default_factory=list)
    procedures: list[MedicalProcedure] = Field(default_factory=list)
    prescriptions: list[Prescription] = Field(default_factory=list)
    extracted: ExtractedData | None = None
    rules: DeterministicRules | None = None
