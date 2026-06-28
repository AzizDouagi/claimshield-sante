"""
Génération de fixtures images synthétiques pour les tests du Document/OCR Agent.

Crée 4 images réparties sur CLM-0001, CLM-0002, CLM-0003 :
  - CLM-0001 : facture médicale scannée (PNG lisible)
  - CLM-0002 : ordonnance médicale scannée (JPEG lisible)
  - CLM-0003 : facture médicale (PNG qualité réduite)
  - CLM-0003 : document illisible — bruit intense (PNG)

Met à jour le manifest.json de chaque cas après génération.
"""

from __future__ import annotations

import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).parent.parent
FIXTURES = ROOT / "datasets" / "fixtures" / "valid"


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _font(size: int) -> ImageFont.ImageFont:
    """Retourne une police système disponible, avec repli sur la police par défaut."""
    candidates = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def _bold_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Arial Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def _blank_page(width: int = 1240, height: int = 1754) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    """Page A4 à 150 DPI sur fond légèrement ivoire."""
    img = Image.new("RGB", (width, height), color=(252, 250, 245))
    draw = ImageDraw.Draw(img)
    return img, draw


def _add_scan_noise(img: Image.Image, sigma: float = 0.5) -> Image.Image:
    """Simule un léger grain de scanner sans dégrader la lisibilité."""
    import random as rnd
    pixels = img.load()
    w, h = img.size
    for _ in range(int(w * h * 0.003)):
        x, y = rnd.randint(0, w - 1), rnd.randint(0, h - 1)
        v = rnd.randint(180, 230)
        pixels[x, y] = (v, v, v)
    return img


def _separator(draw: ImageDraw.ImageDraw, y: int, width: int = 1240, margin: int = 60) -> None:
    draw.line([(margin, y), (width - margin, y)], fill=(180, 180, 180), width=1)


def _update_manifest(case_dir: Path, new_files: list[dict]) -> None:
    manifest_path = case_dir / "audit" / "manifest.json"
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    existing_names = {entry["filename"] for entry in manifest["files"]}
    for entry in new_files:
        if entry["filename"] not in existing_names:
            manifest["files"].append(entry)

    manifest["generated_at"] = datetime.now(timezone.utc).isoformat()
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"  ✓ manifest mis à jour : {manifest_path.relative_to(ROOT)}")


# ─── Image 1 : Facture médicale — CLM-0001 (PNG lisible) ──────────────────────

def generate_facture_clm0001() -> Path:
    img, draw = _blank_page()
    W = 1240

    fn_title = _bold_font(36)
    fn_body = _font(24)
    fn_small = _font(20)
    fn_label = _bold_font(22)
    gray = (80, 80, 80)
    dark = (30, 30, 30)
    blue = (25, 80, 160)

    # En-tête clinique
    draw.rectangle([(0, 0), (W, 120)], fill=(25, 80, 160))
    draw.text((60, 28), "CLINIQUE SAINT-JOSEPH — SERVICE FACTURATION", font=_bold_font(30), fill=(255, 255, 255))
    draw.text((60, 72), "42 rue des Tilleuls, 75014 Paris  |  Tél : 01 40 XX XX XX  |  SIRET : 123 456 789 00010", font=_font(20), fill=(210, 225, 255))

    y = 150
    draw.text((60, y), "FACTURE MÉDICALE", font=_bold_font(40), fill=blue)
    y += 60
    _separator(draw, y, W)
    y += 20

    # Bloc patient / facture
    draw.text((60, y), "N° Facture :", font=fn_label, fill=gray)
    draw.text((230, y), "INV-CLM-0001", font=fn_body, fill=dark)
    draw.text((700, y), "Date de service :", font=fn_label, fill=gray)
    draw.text((900, y), "03/06/2026", font=fn_body, fill=dark)
    y += 40
    draw.text((60, y), "Référence dossier :", font=fn_label, fill=gray)
    draw.text((280, y), "CLM-0001", font=fn_body, fill=dark)
    draw.text((700, y), "Assureur :", font=fn_label, fill=gray)
    draw.text((830, y), "Cigna Health", font=fn_body, fill=dark)
    y += 40
    draw.text((60, y), "Patient :", font=fn_label, fill=gray)
    draw.text((180, y), "Mrs. Tresa661 Taunya970 Sawayn19", font=fn_body, fill=dark)
    y += 40
    draw.text((60, y), "N° Patient :", font=fn_label, fill=gray)
    draw.text((210, y), "c2bdae5f-4f81-abc3-c9ae-3f60d73d3cfc", font=fn_small, fill=gray)
    y += 50
    _separator(draw, y, W)
    y += 20

    # Tableau des actes
    draw.rectangle([(60, y), (W - 60, y + 36)], fill=(220, 230, 245))
    draw.text((70, y + 6), "Code", font=fn_label, fill=dark)
    draw.text((200, y + 6), "Description de l'acte", font=fn_label, fill=dark)
    draw.text((900, y + 6), "Montant (USD)", font=fn_label, fill=dark)
    y += 36

    actes = [
        ("99213", "Consultation médecin généraliste (30 min)", "250.00"),
        ("71046", "Radiographie thoracique 2 incidences", "420.00"),
        ("93000", "Électrocardiogramme 12 dérivations", "185.00"),
        ("85025", "Numération formule sanguine complète", "95.00"),
        ("80053", "Panel métabolique complet", "120.00"),
        ("36415", "Prélèvement sanguin veineux", "45.00"),
        ("99232", "Visite hospitalière quotidienne", "1 800.00"),
        ("99238", "Sortie hospitalière (≤ 30 min)", "751.69"),
    ]
    row_colors = [(255, 255, 255), (245, 248, 255)]
    for i, (code, desc, montant) in enumerate(actes):
        bg = row_colors[i % 2]
        draw.rectangle([(60, y), (W - 60, y + 32)], fill=bg)
        draw.text((70, y + 4), code, font=fn_small, fill=dark)
        draw.text((200, y + 4), desc, font=fn_small, fill=dark)
        draw.text((920, y + 4), montant, font=fn_small, fill=dark)
        y += 32

    y += 10
    _separator(draw, y, W)
    y += 20

    # Totaux
    draw.text((700, y), "Total facturé :", font=fn_label, fill=gray)
    draw.text((920, y), "3 666,69 USD", font=_bold_font(24), fill=dark)
    y += 36
    draw.text((700, y), "Taux de couverture :", font=fn_label, fill=gray)
    draw.text((920, y), "80 %", font=fn_body, fill=dark)
    y += 36
    draw.text((700, y), "Part assureur :", font=fn_label, fill=gray)
    draw.text((920, y), "2 933,35 USD", font=_bold_font(26), fill=blue)
    y += 36
    draw.text((700, y), "Part patient :", font=fn_label, fill=gray)
    draw.text((920, y), "733,34 USD", font=fn_body, fill=dark)
    y += 60

    # Ordonnance associée
    draw.text((60, y), "Prescription associée :", font=fn_label, fill=gray)
    draw.text((290, y), "RX-CLM-0001  (autorisation préalable approuvée)", font=fn_body, fill=dark)
    y += 50
    _separator(draw, y, W)
    y += 20

    draw.text((60, y), "Document généré à des fins de test — données entièrement synthétiques (Synthea).", font=fn_small, fill=(160, 160, 160))
    draw.text((60, y + 26), "data_classification: SYNTHETIC_TEST_DATA  |  contains_real_personal_data: false", font=fn_small, fill=(180, 180, 180))

    img = _add_scan_noise(img)
    out = FIXTURES / "CLM-0001" / "input" / "facture_image_CLM-0001.png"
    img.save(out, format="PNG", dpi=(150, 150))
    print(f"  ✓ {out.relative_to(ROOT)}  ({out.stat().st_size:,} octets)")
    return out


# ─── Image 2 : Ordonnance médicale — CLM-0002 (JPEG lisible) ──────────────────

def generate_ordonnance_clm0002() -> Path:
    img, draw = _blank_page()
    W = 1240

    fn_title = _bold_font(34)
    fn_body = _font(24)
    fn_small = _font(20)
    fn_label = _bold_font(22)
    gray = (80, 80, 80)
    dark = (30, 30, 30)
    green = (20, 110, 60)

    # En-tête
    draw.rectangle([(0, 0), (W, 120)], fill=(20, 110, 60))
    draw.text((60, 25), "DR MARTIN DUPONT — MÉDECIN GÉNÉRALISTE", font=_bold_font(30), fill=(255, 255, 255))
    draw.text((60, 70), "Cabinet libéral  |  12 av. Gambetta, 69003 Lyon  |  RPPS : 10 003 456 789", font=_font(20), fill=(180, 240, 200))

    y = 150
    draw.text((60, y), "ORDONNANCE MÉDICALE", font=_bold_font(40), fill=green)
    y += 60
    _separator(draw, y, W, margin=60)
    y += 20

    draw.text((60, y), "N° Ordonnance :", font=fn_label, fill=gray)
    draw.text((270, y), "RX-CLM-0002", font=fn_body, fill=dark)
    draw.text((700, y), "Date :", font=fn_label, fill=gray)
    draw.text((790, y), "05/06/2026", font=fn_body, fill=dark)
    y += 40
    draw.text((60, y), "Référence dossier :", font=fn_label, fill=gray)
    draw.text((280, y), "CLM-0002", font=fn_body, fill=dark)
    y += 40
    draw.text((60, y), "Patient :", font=fn_label, fill=gray)
    draw.text((180, y), "Mr. Marcel580 Darnell564 Terry864", font=fn_body, fill=dark)
    y += 40
    draw.text((60, y), "N° Patient :", font=fn_label, fill=gray)
    draw.text((210, y), "d47a3b12-8c2e-4f9a-b1d0-7e5f6c3a2b8d", font=_font(20), fill=gray)
    y += 50
    _separator(draw, y, W)
    y += 30

    draw.text((60, y), "MÉDICAMENTS PRESCRITS", font=_bold_font(28), fill=green)
    y += 50

    medicaments = [
        ("Amoxicilline 500 mg", "gélules", "3 × / jour pendant 7 jours", "Boîte de 21 gélules"),
        ("Ibuprofène 400 mg", "comprimés", "1 × / repas (max 3/j) — 5 jours", "Boîte de 14 comprimés"),
        ("Pantoprazole 20 mg", "comprimés gastro-résistants", "1 × / jour à jeun", "Boîte de 28 comprimés"),
    ]
    for i, (nom, forme, posologie, conditionnement) in enumerate(medicaments, 1):
        draw.text((60, y), f"{i}.", font=_bold_font(28), fill=dark)
        draw.text((100, y), nom, font=_bold_font(28), fill=dark)
        draw.text((100, y + 34), f"{forme} — {posologie}", font=fn_body, fill=gray)
        draw.text((100, y + 64), conditionnement, font=fn_small, fill=(120, 120, 120))
        y += 110

    y += 10
    _separator(draw, y, W)
    y += 30

    draw.text((60, y), "Prescripteur :", font=fn_label, fill=gray)
    draw.text((230, y), "Dr Martin Dupont", font=fn_body, fill=dark)
    y += 40
    draw.text((60, y), "Spécialité :", font=fn_label, fill=gray)
    draw.text((205, y), "Médecine générale", font=fn_body, fill=dark)
    y += 40
    draw.text((60, y), "Signature électronique :", font=fn_label, fill=gray)
    draw.text((290, y), "[ SIGNATURE SYNTHÉTIQUE — À VALIDER ]", font=fn_body, fill=(150, 150, 150))
    y += 80

    draw.text((60, y), "Données entièrement synthétiques (Synthea) — SYNTHETIC_TEST_DATA.", font=fn_small, fill=(170, 170, 170))

    img = _add_scan_noise(img, sigma=0.8)
    out = FIXTURES / "CLM-0002" / "input" / "ordonnance_image_CLM-0002.jpg"
    img.save(out, format="JPEG", quality=82, dpi=(150, 150))
    print(f"  ✓ {out.relative_to(ROOT)}  ({out.stat().st_size:,} octets)")
    return out


# ─── Image 3 : Facture dégradée — CLM-0003 (PNG qualité réduite) ──────────────

def generate_facture_clm0003() -> Path:
    img, draw = _blank_page()
    W = 1240

    fn_body = _font(24)
    fn_small = _font(20)
    fn_label = _bold_font(22)
    gray = (80, 80, 80)
    dark = (30, 30, 30)
    orange = (170, 80, 10)

    draw.rectangle([(0, 0), (W, 120)], fill=(170, 80, 10))
    draw.text((60, 28), "HÔPITAL RÉGIONAL DU SUD — DÉPARTEMENT FACTURATION", font=_bold_font(28), fill=(255, 255, 255))
    draw.text((60, 72), "8 rue Pasteur, 13006 Marseille  |  Tél : 04 91 XX XX XX", font=_font(20), fill=(255, 220, 180))

    y = 150
    draw.text((60, y), "FACTURE MÉDICALE", font=_bold_font(40), fill=orange)
    y += 60
    _separator(draw, y, W)
    y += 20

    draw.text((60, y), "N° Facture :", font=fn_label, fill=gray)
    draw.text((230, y), "INV-CLM-0003", font=fn_body, fill=dark)
    draw.text((700, y), "Date :", font=fn_label, fill=gray)
    draw.text((780, y), "12/05/2026", font=fn_body, fill=dark)
    y += 40
    draw.text((60, y), "Dossier :", font=fn_label, fill=gray)
    draw.text((175, y), "CLM-0003", font=fn_body, fill=dark)
    draw.text((700, y), "Assureur :", font=fn_label, fill=gray)
    draw.text((830, y), "Blue Cross", font=fn_body, fill=dark)
    y += 40
    draw.text((60, y), "Patient :", font=fn_label, fill=gray)
    draw.text((180, y), "Mrs. Tammera223 Leila837 Kutch271", font=fn_body, fill=dark)
    y += 50
    _separator(draw, y, W)
    y += 20

    draw.rectangle([(60, y), (W - 60, y + 36)], fill=(240, 220, 200))
    draw.text((70, y + 6), "Code", font=fn_label, fill=dark)
    draw.text((200, y + 6), "Description", font=fn_label, fill=dark)
    draw.text((900, y + 6), "Montant (USD)", font=fn_label, fill=dark)
    y += 36

    actes = [
        ("99213", "Consultation spécialiste", "350.00"),
        ("73721", "IRM genou sans injection", "354.20"),
    ]
    for i, (code, desc, montant) in enumerate(actes):
        bg = (255, 255, 255) if i % 2 == 0 else (250, 242, 235)
        draw.rectangle([(60, y), (W - 60, y + 32)], fill=bg)
        draw.text((70, y + 4), code, font=fn_small, fill=dark)
        draw.text((200, y + 4), desc, font=fn_small, fill=dark)
        draw.text((920, y + 4), montant, font=fn_small, fill=dark)
        y += 32

    y += 10
    _separator(draw, y, W)
    y += 20
    draw.text((700, y), "Total facturé :", font=fn_label, fill=gray)
    draw.text((920, y), "704,20 USD", font=_bold_font(26), fill=orange)
    y += 36
    draw.text((700, y), "Taux de couverture :", font=fn_label, fill=gray)
    draw.text((920, y), "80 %", font=fn_body, fill=dark)
    y += 36
    draw.text((700, y), "Part assureur :", font=fn_label, fill=gray)
    draw.text((920, y), "563,36 USD", font=_bold_font(24), fill=orange)
    y += 50

    draw.text((60, y), "Données entièrement synthétiques — SYNTHETIC_TEST_DATA.", font=fn_small, fill=(170, 170, 170))

    # Dégradation modérée : léger flou + bruit accentué
    img = img.filter(ImageFilter.GaussianBlur(radius=0.6))
    rng = random.Random(42)
    pixels = img.load()
    w, h = img.size
    for _ in range(int(w * h * 0.015)):
        x, y_px = rng.randint(0, w - 1), rng.randint(0, h - 1)
        v = rng.randint(140, 210)
        pixels[x, y_px] = (v, v, v)

    out = FIXTURES / "CLM-0003" / "input" / "facture_image_CLM-0003.png"
    img.save(out, format="PNG", dpi=(150, 150))
    print(f"  ✓ {out.relative_to(ROOT)}  ({out.stat().st_size:,} octets)")
    return out


# ─── Image 4 : Document illisible — CLM-0003 (PNG bruit intense) ──────────────

def generate_illisible_clm0003() -> Path:
    """
    Simule un document très dégradé : texte de base rendu quasi invisible
    par superposition de bruit dense et de flou fort.
    Sert à tester la détection d'illisibilité dans le Document/OCR Agent.
    """
    img, draw = _blank_page()
    W, H = 1240, 1754

    fn = _font(30)
    fn_small = _font(22)

    draw.text((60, 60), "DOCUMENT MEDICAL - CLM-0003", font=fn, fill=(200, 200, 200))
    draw.text((60, 110), "Contenu illisible — qualité de numérisation insuffisante", font=fn_small, fill=(210, 210, 210))
    draw.text((60, 150), "Patient : Mrs. Tammera223 Leila837 Kutch271", font=fn_small, fill=(205, 205, 205))
    draw.text((60, 190), "Ref : CLM-0003  |  Date : 12/05/2026", font=fn_small, fill=(210, 210, 210))

    # Dégradation sévère : bruit dense + flou fort
    rng = random.Random(99)
    pixels = img.load()
    for _ in range(int(W * H * 0.40)):
        x = rng.randint(0, W - 1)
        y = rng.randint(0, H - 1)
        v = rng.randint(100, 255)
        pixels[x, y] = (v, rng.randint(100, 255), rng.randint(100, 255))

    img = img.filter(ImageFilter.GaussianBlur(radius=3.5))

    # Deuxième couche de bruit après le flou
    pixels = img.load()
    for _ in range(int(W * H * 0.15)):
        x = rng.randint(0, W - 1)
        y = rng.randint(0, H - 1)
        pixels[x, y] = (rng.randint(60, 200), rng.randint(60, 200), rng.randint(60, 200))

    out = FIXTURES / "CLM-0003" / "input" / "document_illisible_CLM-0003.png"
    img.save(out, format="PNG", dpi=(150, 150))
    print(f"  ✓ {out.relative_to(ROOT)}  ({out.stat().st_size:,} octets)")
    return out


# ─── Mise à jour des manifests ────────────────────────────────────────────────

def update_manifests(generated: dict[str, list[Path]]) -> None:
    for case_id, paths in generated.items():
        case_dir = FIXTURES / case_id
        entries = [
            {
                "filename": p.name,
                "size_bytes": p.stat().st_size,
                "sha256": _sha256(p),
                "required": False,
                "image_fixture": True,
                "purpose": "OCR agent test fixture — données synthétiques",
            }
            for p in paths
        ]
        _update_manifest(case_dir, entries)


# ─── Point d'entrée ───────────────────────────────────────────────────────────

def main() -> None:
    print("Génération des fixtures images synthétiques…\n")

    clm0001_facture = generate_facture_clm0001()
    clm0002_ordo = generate_ordonnance_clm0002()
    clm0003_facture = generate_facture_clm0003()
    clm0003_illisible = generate_illisible_clm0003()

    print("\nMise à jour des manifests…")
    update_manifests({
        "CLM-0001": [clm0001_facture],
        "CLM-0002": [clm0002_ordo],
        "CLM-0003": [clm0003_facture, clm0003_illisible],
    })

    print("\nRésumé :")
    print(f"  CLM-0001 : facture_image_CLM-0001.png  (PNG, lisible)")
    print(f"  CLM-0002 : ordonnance_image_CLM-0002.jpg  (JPEG, lisible)")
    print(f"  CLM-0003 : facture_image_CLM-0003.png  (PNG, qualité réduite)")
    print(f"  CLM-0003 : document_illisible_CLM-0003.png  (PNG, bruit intense — test illisibilité)")
    print("\nTerminé.")


if __name__ == "__main__":
    main()
