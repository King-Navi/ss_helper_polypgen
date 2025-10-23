#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Set
import numpy as np
from PIL import Image

IMG_EXTS: Set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MASK_SUFFIX = "_mask"

def binarize_array(arr: np.ndarray, thresh: int = 127) -> np.ndarray:
    """
    Accepts grayscale uint8 array. Returns binary uint8 {0,255}.
    Strategy:
      - If there are >2 unique values, threshold.
      - If unique set contains values beyond {0,255}, map non-zero -> 255.
      - If values already {0,1} -> scale to {0,255}.
    """
    u = np.unique(arr)
    if set(u.tolist()) <= {0, 255}:
        # already 0/255
        bin255 = arr
    elif set(u.tolist()) <= {0, 1}:
        bin255 = (arr * 255).astype(np.uint8)
    else:
        # generic: anything > 0 becomes 255, or use threshold if needed
        # prefer a robust map-to-nonzero to collapse 2/3 classes
        # (change to arr>thresh if you strictly want thresholding)
        bin255 = np.where(arr > 0, 255, 0).astype(np.uint8)

        # If you prefer strict thresholding, uncomment next line and comment the line above:
        # bin255 = np.where(arr > thresh, 255, 0).astype(np.uint8)

    return bin255

def process_mask_file(p: Path, out_png: bool, dry_run: bool) -> None:
    # Load and convert to single channel
    with Image.open(p) as im:
        # Convert any mode (RGB/RGBA/P/CMYK/LA/…) to single channel grayscale
        g = im.convert("L")
        arr = np.asarray(g, dtype=np.uint8)

    before = np.unique(arr)
    bin255 = binarize_array(arr)

    after = np.unique(bin255)
    print(f"[OK] {p.name}: before uniques={before.tolist()} -> after uniques={after.tolist()}")

    if dry_run:
        return

    # Decide output path/format
    if out_png:
        dst = p.with_suffix(".png")
        # Ensure suffix contains _mask
        if not dst.stem.endswith(MASK_SUFFIX):
            dst = dst.with_stem(dst.stem + MASK_SUFFIX)
        Image.fromarray(bin255, mode="L").save(dst, optimize=True)
        if dst.resolve() != p.resolve():
            try:
                p.unlink()  # remove original if different path
            except Exception:
                pass
    else:
        # Overwrite as grayscale JPG (note: lossy, no recomendado para máscaras)
        Image.fromarray(bin255, mode="L").save(p, quality=95, subsampling=0)

def main():
    ap = argparse.ArgumentParser(description="Force masks to 1-channel binary {0,255} and save as PNG.")
    ap.add_argument("--mask-dir", type=Path, required=True, help="Folder with mask images")
    ap.add_argument("--out-png", action="store_true",
                    help="Write masks as PNG (recommended). Will rename to *_mask.png")
    ap.add_argument("--dry-run", action="store_true", help="Show uniques and planned writes without saving")
    args = ap.parse_args()

    if not args.mask_dir.is_dir():
        raise SystemExit(f"[ERROR] Not a directory: {args.mask_dir}")

    files = [p for p in args.mask_dir.rglob("*") if p.suffix.lower() in IMG_EXTS and p.is_file()]
    if not files:
        print("[WARN] No image files found.")
        return

    print(f"Found {len(files)} mask files.")
    for p in files:
        process_mask_file(p, out_png=args.out_png, dry_run=args.dry_run)

    print("Done.")




"""

# Solo revisar (no guarda cambios)
python fix_masks_to_binary.py --mask-dir /home/ivan/Downloads/img_resized_512/output/mask --dry-run

# Guardar como PNG binario (recomendado)
python fix_masks_to_binary.py --mask-dir /home/ivan/Downloads/img_resized_512/output/mask --out-png

"""


if __name__ == "__main__":
    main()
