"""Outils de pseudonymisation et de masquage des données personnelles.

Fonctions déterministes — aucun appel LLM, aucun effet de bord.

Algorithmes de pseudonymisation (HMAC-SHA256 + séparateur de domaine) :
  patient  : HMAC(key, "pat:<id>")  → PAT-XXXXXXXXXXXX
  provider : HMAC(key, "prv:<id>")  → PRV-XXXXXXXXXXXX
  générique: HMAC(key, "generic:<id>") → PSE-XXXXXXXXXXXX

  La clé est lue depuis PSEUDONYMIZATION_KEY (variable d'environnement).
  Elle n'est jamais écrite dans les logs ni dans le code source.
  Les séparateurs de domaine "pat:", "prv:" et "generic:" garantissent que
  le même identifiant produit des pseudonymes distincts selon son contexte.

Propriétés garanties :
  - Déterminisme     : même identifiant + même domaine → même pseudonyme.
  - Opacité          : sans la clé, l'inversion est computationnellement infaisable.
  - Séparabilité     : deux identifiants ou deux domaines distincts → pseudonymes distincts.
  - Toutes les fonctions sont pures et testables unitairement.
"""
from __future__ import annotations

import hashlib
import hmac
import re

# ── Récupération de la clé HMAC ───────────────────────────────────────────────


def _get_hmac_key() -> bytes:
    """Retourne la clé HMAC encodée en UTF-8 depuis la configuration.

    La clé est chargée via get_settings() qui utilise @lru_cache —
    l'appel est donc quasi-gratuit après la première résolution.
    La clé n'est jamais affichée ni loggée (SecretStr dans Settings).
    """
    from config.settings import get_settings  # import différé — évite les imports circulaires
    return get_settings().pseudonymization_key.get_secret_value().encode("utf-8")


def pseudonymization_key_is_available() -> bool:
    """Vérifie que la clé HMAC de pseudonymisation est accessible et non vide.

    Retourne False si la configuration est inaccessible ou si la clé est vide.
    Ne jamais afficher la valeur de la clé dans les messages d'erreur ni dans les logs.
    """
    try:
        key = _get_hmac_key()
        return bool(key)
    except Exception:
        return False


def _hmac_digest(domain: str, value: str) -> str:
    """Calcule HMAC-SHA256(key, "<domain>:<value>") et retourne le digest hex."""
    key = _get_hmac_key()
    msg = f"{domain}:{value}".encode("utf-8")
    return hmac.new(key, msg, hashlib.sha256).hexdigest()


# ── Masquage de noms complets ─────────────────────────────────────────────────


def mask_name(name: str) -> str:
    """Masque un nom propre en ne conservant que la première lettre de chaque mot.

    Exemples :
        "Jean Dupont"  → "J*** D*****"
        "Marie"        → "M****"
        ""             → ""
    """
    if not name:
        return name

    def _mask_word(word: str) -> str:
        return word[0] + "*" * (len(word) - 1) if len(word) > 1 else word

    return " ".join(_mask_word(w) for w in name.split())


# ── Masquage d'une adresse e-mail ─────────────────────────────────────────────


def mask_email(email: str) -> str:
    """Masque une adresse e-mail — conserve la première lettre et le domaine.

    Au plus 5 astérisques remplacent la partie locale (après le premier caractère).
    Le domaine (après @) est conservé intact.

    Exemples :
        "patient@example.com" → "p*****@example.com"
        "ab@test.fr"          → "a*@test.fr"
        "invalide"            → "[EMAIL MASQUÉ]"
    """
    if "@" not in email:
        return "[EMAIL MASQUÉ]"
    local, _, domain = email.partition("@")
    if not local:
        return f"@{domain}"
    stars = "*" * min(5, max(1, len(local) - 1))
    return f"{local[0]}{stars}@{domain}"


# ── Masquage d'un numéro de téléphone ────────────────────────────────────────


def mask_phone(phone: str) -> str:
    """Masque un numéro de téléphone — conserve les 4 derniers chiffres.

    Le masque est toujours affiché avec 8 astérisques, puis les 4 derniers chiffres.
    Les séparateurs (espaces, tirets, points, parenthèses) sont ignorés lors
    de l'extraction des chiffres.

    Exemples :
        "+216 22 123 456" → "********3456"
        "06 12 34 56 78"  → "********5678"
    """
    digits = re.sub(r"\D", "", phone)
    suffix = digits[-4:] if len(digits) >= 4 else digits
    return "*" * 8 + suffix


# ── Masquage d'un numéro de contrat ou de référence ──────────────────────────


def mask_contract_number(contract: str) -> str:
    """Masque un numéro de contrat — conserve les 4 derniers caractères alphanumériques.

    Le masque est toujours affiché avec 8 astérisques, puis les 4 derniers
    caractères alphanumériques (hors tirets, points, espaces).

    Exemples :
        "POL-2026-874512" → "********4512"
        "CTR/2025/98765"  → "********8765"
    """
    alphanum = re.sub(r"[^A-Za-z0-9]", "", contract)
    suffix = alphanum[-4:] if len(alphanum) >= 4 else alphanum
    return "*" * 8 + suffix


# ── Masquage d'une adresse postale ───────────────────────────────────────────


def mask_postal_address(address: str) -> str:
    """Masque une adresse postale — conserve la ville ou le pays si identifiable.

    Lorsque l'adresse contient plusieurs parties séparées par une virgule, la
    dernière partie (généralement ville + code postal ou pays) est conservée.
    Les éléments précédents (numéro, rue) sont masqués.

    Exemples :
        "1 Rue de la Paix, 75001 Paris" → "***, 75001 Paris"
        "12 avenue des Fleurs"          → "***"
        ""                              → ""
    """
    if not address:
        return address
    parts = [p.strip() for p in address.split(",")]
    if len(parts) > 1:
        return "***" + ", " + parts[-1]
    return "***"


# ── Pseudonymisation d'identifiants (HMAC-SHA256 typé par domaine) ────────────


def pseudonymize_patient_id(patient_id: str) -> str:
    """Pseudonymise un identifiant patient avec le domaine "pat:".

    Utilise HMAC-SHA256(key, "pat:<patient_id>").
    Le préfixe "PAT-" + 12 hex majuscules identifie le pseudonyme comme
    appartenant au domaine patient.

    Propriétés :
      - Déterministe pour une clé et un identifiant donnés.
      - Opaque sans la clé (propriété de HMAC).
      - Distinct de pseudonymize_provider_id() pour le même identifiant (séparateur de domaine).

    Exemple :
        "PATIENT-0007" → "PAT-A8F3C921B72E"
    """
    digest = _hmac_digest("pat", patient_id)
    return "PAT-" + digest[:12].upper()


def pseudonymize_provider_id(provider_id: str) -> str:
    """Pseudonymise un identifiant prestataire/organisation avec le domaine "prv:".

    Utilise HMAC-SHA256(key, "prv:<provider_id>").
    Le préfixe "PRV-" + 12 hex majuscules identifie le pseudonyme comme
    appartenant au domaine prestataire.

    Propriétés :
      - Distinct de pseudonymize_patient_id() pour le même identifiant.
      - Déterministe et opaque (mêmes propriétés que pseudonymize_patient_id).
    """
    digest = _hmac_digest("prv", provider_id)
    return "PRV-" + digest[:12].upper()


def pseudonymize_id(value: str) -> str:
    """Pseudonymise un identifiant générique avec le domaine "generic:".

    Conservé pour la compatibilité descendante — préfère `pseudonymize_patient_id`
    ou `pseudonymize_provider_id` lorsque le contexte est connu.

    Retourne "PSE-" + 12 hex majuscules.
    """
    digest = _hmac_digest("generic", value)
    return "PSE-" + digest[:12].upper()


# ── Masquage générique par type de champ ──────────────────────────────────────


def mask_field_value(field_name: str, value: str) -> str:
    """Sélectionne et applique le masquage adapté au type de champ.

    Dispatch basé sur le nom du champ (insensible à la casse) :
      patient + id       → pseudonymize_patient_id (PAT-)
      provider/org + id  → pseudonymize_provider_id (PRV-)
      autre *id*         → pseudonymize_id (PSE-)
      *name* / *nom*     → mask_name
      *email* / *mail*   → mask_email
      *phone* / *tel*    → mask_phone
      *address*/adresse  → mask_postal_address
      *contract*/contrat → mask_contract_number
      *number*/*invoice*/*prescription* → 4 derniers alphanum
      autres             → "[MASQUÉ]"
    """
    f = field_name.lower()

    # Identifiants typés — contexte spécifique en priorité
    if "patient" in f and "id" in f:
        return pseudonymize_patient_id(value)
    if ("provider" in f or "organization" in f) and "id" in f:
        return pseudonymize_provider_id(value)
    if "id" in f:
        return pseudonymize_id(value)

    if "name" in f or "nom" in f:
        return mask_name(value)

    if "email" in f or "mail" in f:
        return mask_email(value)

    if "phone" in f or "tel" in f:
        return mask_phone(value)

    if "address" in f or "adresse" in f:
        return mask_postal_address(value)

    if "contract" in f or "contrat" in f:
        return mask_contract_number(value)

    if any(kw in f for kw in ("number", "invoice", "prescription")):
        alphanum = re.sub(r"[^A-Za-z0-9]", "", value)
        suffix = alphanum[-4:] if len(alphanum) >= 4 else alphanum
        return "***" + suffix

    return "[MASQUÉ]"


# ── Suppression récursive des champs sensibles ────────────────────────────────


def sanitize_recursive(
    data: dict | list | object,
    sensitive_keys: set[str],
) -> dict | list | object:
    """Supprime récursivement les clés sensibles dans une structure dict/list imbriquée.

    Garantit qu'aucun champ sensible ne survive dans une vue interdite, même
    si les données sont imbriquées sur plusieurs niveaux.

    Args:
        data           : structure à assainir (dict, list, ou valeur scalaire).
        sensitive_keys : ensemble des clés à supprimer à tous les niveaux.

    Returns:
        Copie de la structure sans les clés sensibles.
        Les valeurs scalaires sont retournées telles quelles.
    """
    if isinstance(data, dict):
        return {
            key: sanitize_recursive(value, sensitive_keys)
            for key, value in data.items()
            if key not in sensitive_keys
        }
    if isinstance(data, list):
        return [sanitize_recursive(item, sensitive_keys) for item in data]
    return data


# ── Application sur un dictionnaire de champs ─────────────────────────────────


def pseudonymize_fields(
    fields: dict[str, str | None],
    fields_to_mask: list[str],
) -> dict[str, str | None]:
    """Applique le masquage adapté sur les champs indiqués.

    Les champs absents de `fields_to_mask` sont retournés sans modification.
    Les valeurs None sont retournées telles quelles même pour les champs masqués.

    Args:
        fields         : dictionnaire {nom_champ: valeur}.
        fields_to_mask : noms des champs à pseudonymiser.

    Returns:
        Nouveau dictionnaire avec les champs masqués remplacés.
    """
    mask_set = set(fields_to_mask)
    return {
        key: (mask_field_value(key, str(value)) if key in mask_set and value is not None else value)
        for key, value in fields.items()
    }
