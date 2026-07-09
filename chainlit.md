# ClaimShield Santé

Interface conversationnelle minimale pour le traitement des demandes de
remboursement santé — soumission de dossier, consultation de statut, revue
humaine (HITL).

Cette interface est un client HTTP de l'API ClaimShield Santé
(`api/main.py`) : elle ne traite jamais un dossier elle-même, elle ne fait
qu'afficher et transmettre vos décisions à l'API, qui reste seule à
exécuter le pipeline multi-agents.
