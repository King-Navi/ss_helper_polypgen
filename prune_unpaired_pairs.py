#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Iterable, Set, Tuple

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MASK_SUFFIX = "_mask"

def iter_media_files(folder: Path, exts: Set[str]) -> Iterable[Path]:
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts and not p.name.startswith("."):
            yield p

def compute_bases(img_dir: Path, mask_dir: Path) -> Tuple[Set[str], Set[str], list[Path], list[Path]]:
    """
    leer todos los archivos de imágenes y máscaras
    """
    
    images = list(iter_media_files(img_dir, IMG_EXTS))
    masks  = list(iter_media_files(mask_dir, IMG_EXTS))
    #p.stem da el nombre del archivo sin extensión
    image_bases = {p.stem for p in images}  # <base>
    mask_bases  = set()                      # <base> (without _mask)

    for m in masks:
        stem = m.stem
        if stem.endswith(MASK_SUFFIX):
            mask_bases.add(stem[: -len(MASK_SUFFIX)])
    # image_bases: conjunto (set[str]) con los nombres base de las imágenes (por ejemplo {"C1_100H0001", "C1_100H0002"}).
    # mask_bases: conjunto (set[str]) con los nombres base de las máscaras, pero sin _mask.
    # images: lista de objetos Path con las rutas completas de todas las imágenes.
    # masks: lista de objetos Path con las rutas completas de todas las máscaras.
    return image_bases, mask_bases, images, masks

def main():
    ap = argparse.ArgumentParser(
        description="Remove images/masks that do not have an exact pair: <base>.* <-> <base>_mask.*"
    )
    ap.add_argument("--img-dir", type=Path, required=True, help="Directory with images")
    ap.add_argument("--mask-dir", type=Path, required=True, help="Directory with masks")
    ap.add_argument("--dry-run", action="store_true", help="Show what would be deleted without deleting")
    args = ap.parse_args()

    img_dir: Path = args.img_dir
    mask_dir: Path = args.mask_dir

    if not img_dir.is_dir() or not mask_dir.is_dir():
        raise SystemExit(f"[ERROR] Invalid directories:\n  img: {img_dir}\n  mask: {mask_dir}")

    image_bases, mask_bases, images, masks = compute_bases(img_dir, mask_dir)

    # Unpaired images: base not present in any mask base
    unpaired_images = [p for p in images if p.stem not in mask_bases]

    # Unpaired masks: (base without _mask) not present in any image base
    unpaired_masks = []
    for m in masks:
        stem = m.stem
        if not stem.endswith(MASK_SUFFIX):
            # Not a valid mask filename => treat as unpaired
            unpaired_masks.append(m)
            continue
        base = stem[: -len(MASK_SUFFIX)]
        if base not in image_bases:
            unpaired_masks.append(m)

    print("=== CHECK ===")
    print(f"Images found: {len(images)}")
    print(f"Masks  found: {len(masks)}")
    print(f"Unpaired images: {len(unpaired_images)}")
    print(f"Unpaired masks:  {len(unpaired_masks)}\n")

    if unpaired_images:
        print("Images to delete (no matching mask):")
        for p in unpaired_images:
            print("  -", p)

    if unpaired_masks:
        print("\nMasks to delete (no matching image or bad name):")
        for p in unpaired_masks:
            print("  -", p)

    if args.dry_run:
        print("\n[DRY-RUN] No files were deleted.")
        return

    # Delete
    deleted = 0
    for p in unpaired_images + unpaired_masks:
        try:
            p.unlink()
            deleted += 1
        except Exception as e:
            print(f"[WARN] Could not delete {p}: {e}")

    print(f"\n=== SUMMARY ===")
    print(f"Deleted files: {deleted}")
    print("Done.")



"""
#Revisar antes de borrar (sirve para simular lo que el script haría sin modificar nada realmente.)
python prune_unpaired_pairs.py --img-dir /home/ivan/Downloads/img_resized_512/output/img --mask-dir /home/ivan/Downloads/img_resized_512/output/mask --dry-run

python prune_unpaired_pairs.py --img-dir /home/ivan/Downloads/img_resized_512/output/img --mask-dir /home/ivan/Downloads/img_resized_512/output/mask
"""
if __name__ == "__main__":
    main()
