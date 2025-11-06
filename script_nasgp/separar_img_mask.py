#!/usr/bin/env python3
import argparse
import shutil
import random
from pathlib import Path
from typing import List, Optional, Tuple

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MASK_SUFFIX = "_mask"
DEST_MASK_SUFFIX = "_segmentation"


def find_dir_exact_or_prefix(base: Path, exact: str, prefix: str) -> Optional[Path]:
    """Prefer an exact subdir name; if missing, fall back to the first that startswith(prefix)."""
    exact_dir = base / exact
    if exact_dir.is_dir():
        return exact_dir
    for p in base.iterdir():
        if p.is_dir() and p.name.lower().startswith(prefix.lower()):
            return p
    return None


def choose_mask_for(basename: str, mask_dir: Path) -> Optional[Path]:
    """
    Try common patterns to locate the corresponding mask file.
    Priority:
      1) <basename>_mask.<ext>
      2) exact name match with suffix already present in basename
    Falls back to first file that startswith(basename) and contains '_mask'.
    """
    # strict pattern
    for ext in IMG_EXTS:
        cand = mask_dir / f"{basename}{MASK_SUFFIX}{ext}"
        if cand.exists():
            return cand
    # relaxed pattern
    candidates: List[Path] = []
    for p in mask_dir.iterdir():
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            st = p.stem.lower()
            if st.startswith(basename.lower()) and MASK_SUFFIX in st:
                candidates.append(p)
    if len(candidates) == 1:
        return candidates[0]
    return None


def copy_file(src: Path, dst: Path, dry_run: bool = False, hardlink: bool = False) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        print(f"[DRY] {src} -> {dst}")
        return
    if hardlink:
        try:
            if dst.exists():
                dst.unlink()
            dst.hardlink_to(src)
            return
        except Exception:
            pass
    shutil.copy2(src, dst)


def collect_pairs_from_split(data_dir: Path) -> List[Tuple[Path, Path, str, str]]:
    """
    Recorre un data_C* y devuelve una lista de tuplas:
    (img_path, mask_path, split_tag, base)
    Solo devuelve pares que realmente existen.
    """
    split_tag = data_dir.name.split("data_")[-1]

    images_dir = find_dir_exact_or_prefix(
        data_dir, f"images_{split_tag}", "images_"
    )
    masks_dir = find_dir_exact_or_prefix(
        data_dir, f"masks_{split_tag}", "masks_"
    )

    if images_dir is None or masks_dir is None:
        print(f"[WARN] Skipping {data_dir.name}: missing images_/masks_ folder(s).")
        return []

    images = [
        p for p in images_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    ]

    pairs: List[Tuple[Path, Path, str, str]] = []

    for img in images:
        base = img.stem  # sin extensión
        mask_path = choose_mask_for(base, masks_dir)
        if mask_path is not None:
            pairs.append((img, mask_path, split_tag, base))
        else:
            print(f"[WARN] Image without mask -> {data_dir.name}:{base}")

    return pairs


def make_dest_mask_name(split_tag: str, base: str, mask_path: Path) -> str:
    # C1_algo_segmentation.png
    return f"{split_tag}_{base}{DEST_MASK_SUFFIX}{mask_path.suffix.lower()}"


def make_dest_img_name(split_tag: str, base: str, img_path: Path) -> str:
    return f"{split_tag}_{base}{img_path.suffix.lower()}"


def split_dataset(
    pairs: List[Tuple[Path, Path, str, str]],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[List, List, List]:
    random.seed(seed)
    pairs = pairs[:]  # copy
    random.shuffle(pairs)

    n = len(pairs)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:n_train + n_val]
    test_pairs = pairs[n_train + n_val:]

    return train_pairs, val_pairs, test_pairs


def copy_split(
    name: str,
    pairs: List[Tuple[Path, Path, str, str]],
    out_root: Path,
    dry_run: bool,
    hardlink: bool,
):
    """
    Copia un subconjunto (train/valid/test) a:
      out_root/name/inputs
      out_root/name/targets
    """
    inputs_dir = out_root / name / "inputs"
    targets_dir = out_root / name / "targets"

    copied_imgs = 0
    copied_masks = 0

    for img_path, mask_path, split_tag, base in pairs:
        img_dst = inputs_dir / make_dest_img_name(split_tag, base, img_path)
        copy_file(img_path, img_dst, dry_run=dry_run, hardlink=hardlink)
        copied_imgs += 1

        mask_dst = targets_dir / make_dest_mask_name(split_tag, base, mask_path)
        copy_file(mask_path, mask_dst, dry_run=dry_run, hardlink=hardlink)
        copied_masks += 1

    print(f"[{name}] imgs={copied_imgs}, masks={copied_masks}")


def main():
    ap = argparse.ArgumentParser(
        description="Agrupa imágenes/máscaras de data_C* y las divide en train/valid/test con inputs/targets."
    )
    ap.add_argument("--root", type=Path, required=True, help="Carpeta que contiene data_C1 ... data_C6")
    ap.add_argument("--out", type=Path, required=True, help="Carpeta de salida")
    ap.add_argument("--train", type=float, default=0.7, help="Proporción para train (default 0.7)")
    ap.add_argument("--val", type=float, default=0.15, help="Proporción para valid (default 0.15)")
    ap.add_argument("--seed", type=int, default=1337, help="Seed para shuffling (default 1337)")
    ap.add_argument("--dry-run", action="store_true", help="Mostrar acciones sin copiar")
    ap.add_argument("--hardlink", action="store_true", help="Intentar hardlinks en lugar de copiar")
    # NUEVO: limitar número de pares totales
    ap.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Usar como máximo N pares (antes de dividir en train/valid/test)",
    )
    args = ap.parse_args()

    # calculamos test
    test_ratio = 1.0 - args.train - args.val
    if test_ratio < 0:
        raise ValueError("Las proporciones de train y val no pueden sumar más de 1.0")

    splits = sorted([p for p in args.root.iterdir() if p.is_dir() and p.name.lower().startswith("data_c")])
    if not splits:
        print(f"[ERROR] No se encontraron carpetas 'data_C*' en: {args.root}")
        return

    print(f"Found splits: {[p.name for p in splits]}")

    all_pairs: List[Tuple[Path, Path, str, str]] = []
    for d in splits:
        ps = collect_pairs_from_split(d)
        print(f"[{d.name}] valid pairs={len(ps)}")
        all_pairs.extend(ps)

    if not all_pairs:
        print("[ERROR] No hay pares válidos (imagen+mascara). Nada que hacer.")
        return

    # aplicar límite si lo pidieron
    if args.max_samples is not None and args.max_samples > 0:
        random.seed(args.seed)
        random.shuffle(all_pairs)
        before = len(all_pairs)
        all_pairs = all_pairs[: args.max_samples]
        print(f"[INFO] Limiting dataset from {before} to {len(all_pairs)} samples")

    train_pairs, val_pairs, test_pairs = split_dataset(
        all_pairs,
        train_ratio=args.train,
        val_ratio=args.val,
        seed=args.seed,
    )

    print("\n=== SPLIT SIZES ===")
    print(f"train: {len(train_pairs)}")
    print(f"valid: {len(val_pairs)}")
    print(f"test:  {len(test_pairs)}")

    out_root = args.out

    copy_split("train", train_pairs, out_root, args.dry_run, args.hardlink)
    copy_split("valid", val_pairs, out_root, args.dry_run, args.hardlink)
    copy_split("test", test_pairs, out_root, args.dry_run, args.hardlink)

    print("\n=== DONE ===")
    print(f"Salida en: {out_root}")


if __name__ == "__main__":
    main()
"""
Este es el 1°

Por ejemplo, para usar todo:

python script_nasgp/separar_img_mask.py \
  --root /home/ivan/Downloads/img_resized_512/output \
  --out /home/ivan/Downloads/img_resized_512/output_splitted \
  --train 0.7 \
  --val 0.15 \
  --seed 42


Solo 300 pares en total (y de ahí saca train/val/test):

python script_nasgp/separar_img_mask.py \
  --root /home/ivan/.../PolypGen \
  --out /home/ivan/.../output_demo \
  --max-samples 300


Con tu split 80/10/10 pero sobre 500 pares nada más:

python script_nasgp/separar_img_mask.py \
  --root /home/ivan/Documents/SS/PolypGen/PolypGen2021_MultiCenterData_v2 \
  --out /home/ivan/Downloads/nasga_example/struct \
  --train 0.8 --val 0.1 \
  --max-samples 700 \
  --seed 42


(500 -> 400 train, 50 val, 50 test aprox.)

Con hardlinks y límite:

python script_nasgp/separar_img_mask.py \
  --root /home/ivan/... \
  --out /home/ivan/.../output_demo \
  --max-samples 1000 \
  --hardlink

"""

if __name__ == "__main__":
    main()
