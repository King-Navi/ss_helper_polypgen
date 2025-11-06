#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import List, Set, Optional

IMG_EXTS: Set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".png"}
MASK_SUFFIXES = ("_segmentation", "_mask")


def is_img(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS and not p.name.startswith(".")


def mask_base_from_stem(stem: str) -> Optional[str]:
    for suf in MASK_SUFFIXES:
        if stem.endswith(suf):
            return stem[: -len(suf)]
    return None


def validate_pair_folder(split_name: str, img_dir: Path, mask_dir: Path) -> None:
    # Avisar si no existen las carpetas
    if not img_dir.exists():
        print(f"[{split_name}] [WARN] no existe carpeta de imágenes: {img_dir}")
    if not mask_dir.exists():
        print(f"[{split_name}] [WARN] no existe carpeta de masks: {mask_dir}")
    if not img_dir.exists() or not mask_dir.exists():
        # no seguimos porque no hay con qué comparar
        return

    imgs: List[Path] = [p for p in img_dir.rglob("*") if is_img(p)]
    masks: List[Path] = [p for p in mask_dir.rglob("*") if is_img(p)]

    img_bases = {p.stem for p in imgs}
    mask_bases = set()

    for m in masks:
        mb = mask_base_from_stem(m.stem)
        if mb is not None:
            mask_bases.add(mb)

    # imágenes que no tienen base en masks
    unpaired_imgs = [p for p in imgs if p.stem not in mask_bases]

    # masks que no tienen imagen
    unpaired_masks: List[Path] = []
    for m in masks:
        mb = mask_base_from_stem(m.stem)
        if mb is None or mb not in img_bases:
            unpaired_masks.append(m)

    print(f"\n=== {split_name} ===")
    print(f"imgs: {len(imgs)} | masks: {len(masks)}")
    print(f"imgs sin mask: {len(unpaired_imgs)}")
    print(f"masks sin img: {len(unpaired_masks)}")

    if unpaired_imgs:
        print("  -> IMÁGENES SIN MASK:")
        for p in unpaired_imgs:
            print("     -", p)

    if unpaired_masks:
        print("  -> MASKS SIN IMAGEN (o nombre raro):")
        for p in unpaired_masks:
            print("     -", p)


def main():
    ap = argparse.ArgumentParser(
        description="Valida que cada imagen tenga su máscara y cada máscara su imagen."
    )
    ap.add_argument(
        "--root",
        type=Path,
        help="Raíz que contiene train/val/test con inputs/targets (modo automático).",
    )
    ap.add_argument(
        "--img-dir",
        type=Path,
        help="Carpeta de imágenes (modo manual).",
    )
    ap.add_argument(
        "--mask-dir",
        type=Path,
        help="Carpeta de máscaras (modo manual).",
    )
    args = ap.parse_args()

    if args.root:
        root = args.root
        if not root.exists():
            print(f"[ERROR] La carpeta root no existe: {root}")
            return

        for split in ("train", "val", "valid", "test"):
            img_dir = root / split / "inputs"
            mask_dir = root / split / "targets"
            # Aquí avisamos siempre, aunque no exista
            validate_pair_folder(split, img_dir, mask_dir)

        print("\nHecho.")
    else:
        # modo manual
        if not args.img_dir or not args.mask_dir:
            raise SystemExit("Si no usas --root, debes especificar --img-dir y --mask-dir")

        if not args.img_dir.exists():
            print(f"[ERROR] --img-dir no existe: {args.img_dir}")
        if not args.mask_dir.exists():
            print(f"[ERROR] --mask-dir no existe: {args.mask_dir}")

        if not args.img_dir.exists() or not args.mask_dir.exists():
            # no seguimos
            return

        validate_pair_folder("manual", args.img_dir, args.mask_dir)
        print("\nHecho.")


if __name__ == "__main__":
    main()

"""
Con tu estructura train/val/test:

python script_nasgp/validar_pares.py \
  --root /home/ivan/Downloads/img_resized_512/output_2


Con carpetas sueltas:

python script_nasgp/validar_pares.py \
  --img-dir /home/ivan/Downloads/img_resized_512/output_2/train/inputs \
  --mask-dir /home/ivan/Downloads/img_resized_512/output_2/train/targets

"""
if __name__ == "__main__":
    main()
