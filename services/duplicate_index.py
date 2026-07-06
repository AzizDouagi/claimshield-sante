"""Index de doublons injectable — services/duplicate_index.py.

Détecte deux catégories de rapprochements entre dossiers déjà indexés,
jamais un verdict de fraude :
  - **doublon exact** : même SHA-256 de document (``document_hash``) —
    littéralement le même fichier déjà soumis sous un autre dossier.
  - **quasi-doublon** : même patient (pseudonyme), montant et description
    suffisamment proches (``tools.statistics``), sur une fenêtre de dates
    configurable.

Aucune instance globale cachée — même patron que
``orchestrator.model_registry.ModelRegistry`` : à instancier et injecter
explicitement (``DuplicateIndex(policy=...)``). Aucune E/S, aucun appel LLM
: l'index vit en mémoire pour la durée de vie de l'objet ; la persistance
(base de données, voir ``database/`` — toujours un stub) est hors périmètre
de ce module.

``DuplicateCheckResult``/``DuplicateMatch`` ne portent jamais de champ de
décision (pas de statut, pas de recommandation, pas de score de fraude) —
uniquement des scores de similarité structurels et une référence au dossier
rapproché. L'interprétation métier (accusation, revue, seuil de blocage)
reste entièrement la responsabilité de l'appelant (ex. futur câblage dans
``fraud_detection_agent``, non fait ici).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from enum import Enum

from pydantic import Field, field_validator

from schemas.domain import StrictModel
from tools.statistics import amount_similarity, date_proximity, text_similarity, weighted_composite_score

_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_PATIENT_PSEUDONYM_RE = re.compile(r"^PAT-[0-9A-F]{12}$")


# ── Politique de détection — versionnée, configurable ────────────────────────


@dataclass(frozen=True)
class DuplicateDetectionPolicy:
    """Seuils de détection de doublons — versionnés et configurables.

    Aucune valeur codée en dur ailleurs dans ce module : toute évolution des
    seuils passe par une nouvelle instance (ou un nouveau ``version``),
    jamais par une modification silencieuse du comportement existant.
    """

    version: str = "1.0.0"
    amount_tolerance_ratio: float = 0.02
    """Écart relatif de montant maximal toléré pour qu'un rapprochement
    reste candidat à un quasi-doublon (0.02 = 2 %)."""
    date_window_days: int = 3
    """Fenêtre de jours au-delà de laquelle la proximité de date tombe à 0."""
    near_duplicate_score_threshold: float = 0.85
    """Score composite minimal (``tools.statistics.weighted_composite_score``)
    à partir duquel un rapprochement est qualifié de quasi-doublon."""
    weight_amount: float = 0.4
    weight_text: float = 0.4
    weight_date: float = 0.2

    def __post_init__(self) -> None:
        if not self.version.strip():
            raise ValueError("DuplicateDetectionPolicy.version ne peut pas être vide")
        for name in ("amount_tolerance_ratio", "near_duplicate_score_threshold"):
            value = getattr(self, name)
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{name} doit être dans [0.0, 1.0], reçu {value!r}")
        if self.date_window_days <= 0:
            raise ValueError("date_window_days doit être strictement positif")


DEFAULT_DUPLICATE_POLICY = DuplicateDetectionPolicy()
"""Politique par défaut — jamais mutée, jamais un singleton implicite dans
``DuplicateIndex`` (toujours passée explicitement, même si c'est la valeur
par défaut du constructeur)."""


# ── Schémas structurels — jamais un verdict de fraude ────────────────────────


class ClaimFingerprint(StrictModel):
    """Signature minimisée d'un dossier à indexer — jamais de donnée
    personnelle brute (nom, adresse) ni de contenu de document.
    """

    case_id: str = Field(..., min_length=1)
    document_hash: str = Field(..., description="SHA-256 hexadécimal du document facturé")
    patient_pseudonym: str = Field(..., description="Pseudonyme patient (PAT-…), jamais l'identifiant réel")
    provider_pseudonym: str | None = Field(default=None)
    amount: Decimal
    service_date: date | None = None
    description: str = Field(
        default="",
        max_length=200,
        description="Résumé court déjà minimisé (ex. types d'actes) — jamais un texte OCR complet",
    )

    @field_validator("document_hash")
    @classmethod
    def _validate_hash(cls, v: str) -> str:
        if not _SHA256_RE.match(v):
            raise ValueError("document_hash doit être un SHA-256 hexadécimal (64 caractères)")
        return v.lower()

    @field_validator("patient_pseudonym")
    @classmethod
    def _validate_patient_pseudonym(cls, v: str) -> str:
        if not _PATIENT_PSEUDONYM_RE.match(v):
            raise ValueError("patient_pseudonym doit être au format PAT-XXXXXXXXXXXX")
        return v


class DuplicateMatchType(str, Enum):
    """Nature d'un rapprochement — jamais une gravité ni un verdict."""

    EXACT = "EXACT"
    NEAR = "NEAR"


class DuplicateMatch(StrictModel):
    """Un rapprochement structurel avec un dossier déjà indexé — uniquement
    des scores de similarité et une référence, jamais une accusation."""

    match_type: DuplicateMatchType
    matched_case_id: str
    similarity_score: float = Field(ge=0.0, le=1.0)
    amount_similarity: float = Field(ge=0.0, le=1.0)
    text_similarity: float = Field(ge=0.0, le=1.0)
    date_proximity: float = Field(ge=0.0, le=1.0)
    policy_version: str


class DuplicateCheckResult(StrictModel):
    """Résultat structurel d'une vérification de doublon — jamais de champ
    de décision (pas de statut, pas de recommandation, pas de score de
    fraude) : uniquement la liste des rapprochements détectés, potentiellement
    vide (dossier sans historique comparable)."""

    case_id: str
    policy_version: str
    matches: list[DuplicateMatch] = Field(default_factory=list)

    @property
    def has_exact_duplicate(self) -> bool:
        return any(m.match_type is DuplicateMatchType.EXACT for m in self.matches)

    @property
    def has_near_duplicate(self) -> bool:
        return any(m.match_type is DuplicateMatchType.NEAR for m in self.matches)


# ── Index injectable ─────────────────────────────────────────────────────────


class DuplicateIndex:
    """Index en mémoire des dossiers déjà soumis.

    Ne détecte jamais de fraude : ``check()`` retourne uniquement des
    rapprochements structurels (score de similarité, référence de dossier).
    L'absence d'historique (index vide ou dossier jamais vu) produit un
    ``DuplicateCheckResult`` sans rapprochement — jamais une erreur, jamais
    un rapprochement inventé.
    """

    def __init__(self, policy: DuplicateDetectionPolicy = DEFAULT_DUPLICATE_POLICY) -> None:
        self._policy = policy
        self._by_hash: dict[str, str] = {}
        self._fingerprints: list[ClaimFingerprint] = []

    @property
    def policy(self) -> DuplicateDetectionPolicy:
        return self._policy

    def __len__(self) -> int:
        return len(self._fingerprints)

    def register(self, fingerprint: ClaimFingerprint) -> None:
        """Ajoute un dossier à l'index, disponible pour les vérifications
        futures. N'effectue elle-même aucune vérification — appeler
        ``check()`` avant ``register()`` pour comparer un nouveau dossier à
        l'historique existant sans se comparer à lui-même."""
        self._by_hash[fingerprint.document_hash] = fingerprint.case_id
        self._fingerprints.append(fingerprint)

    def check(self, fingerprint: ClaimFingerprint) -> DuplicateCheckResult:
        """Compare ``fingerprint`` aux dossiers déjà indexés (avant tout
        ``register()`` de ce même ``fingerprint``).

        Doublon exact : un autre dossier déjà indexé partage le même
        ``document_hash``. Quasi-doublon : un autre dossier du même patient
        (``patient_pseudonym``) dont le montant, la description et la date
        sont suffisamment proches (``DuplicateDetectionPolicy``) — jamais
        comparé entre patients différents, une similarité de montant entre
        deux patients distincts n'est pas une preuve de doublon.
        """
        matches: list[DuplicateMatch] = []
        policy = self._policy

        existing_case = self._by_hash.get(fingerprint.document_hash)
        if existing_case is not None and existing_case != fingerprint.case_id:
            matches.append(
                DuplicateMatch(
                    match_type=DuplicateMatchType.EXACT,
                    matched_case_id=existing_case,
                    similarity_score=1.0,
                    amount_similarity=1.0,
                    text_similarity=1.0,
                    date_proximity=1.0,
                    policy_version=policy.version,
                )
            )

        for other in self._fingerprints:
            if other.case_id == fingerprint.case_id:
                continue
            if other.document_hash == fingerprint.document_hash:
                continue  # déjà couvert par le doublon exact ci-dessus
            if other.patient_pseudonym != fingerprint.patient_pseudonym:
                continue  # jamais de rapprochement entre patients différents

            amount_score = amount_similarity(fingerprint.amount, other.amount)
            if (1.0 - amount_score) > policy.amount_tolerance_ratio:
                continue  # écart de montant hors tolérance — pas candidat

            text_score = text_similarity(fingerprint.description, other.description)
            date_score = date_proximity(
                fingerprint.service_date, other.service_date, window_days=policy.date_window_days
            )
            composite = weighted_composite_score(
                amount_score=amount_score,
                text_score=text_score,
                date_score=date_score,
                weight_amount=policy.weight_amount,
                weight_text=policy.weight_text,
                weight_date=policy.weight_date,
            )
            if composite < policy.near_duplicate_score_threshold:
                continue

            matches.append(
                DuplicateMatch(
                    match_type=DuplicateMatchType.NEAR,
                    matched_case_id=other.case_id,
                    similarity_score=composite,
                    amount_similarity=amount_score,
                    text_similarity=text_score,
                    date_proximity=date_score,
                    policy_version=policy.version,
                )
            )

        return DuplicateCheckResult(
            case_id=fingerprint.case_id, policy_version=policy.version, matches=matches
        )
