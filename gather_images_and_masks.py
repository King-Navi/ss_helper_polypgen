#!/usr/bin/env python3
import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MASK_SUFFIX = "_mask"

def find_dir_exact_or_prefix(base: Path, exact: str, prefix: str) -> Optional[Path]:
    """Prefer an exact subdir name; if missing, fall back to the first that startswith(prefix)."""
    exact_dir = base / exact
    if exact_dir.is_dir():
        return exact_dir
    for p in base.iterdir():
        if p.is_dir() and p.name.lower().startswith(prefix.lower()):
            return p
    return None

def choose_mask_for(basename: str, mask_dir: Path)  -> Optional[Path]:
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

def process_split(
    data_dir: Path, out_img: Path, out_mask: Path, dry_run: bool, hardlink: bool
) -> Tuple[int, int, List[str], List[str]]:
    """
    Returns: (num_imgs_copied, num_masks_copied, imgs_without_mask, extra_masks)
    """
    # split_tag: e.g., data_C1 -> C1
    split_tag = data_dir.name.split("data_")[-1]

    images_dir = find_dir_exact_or_prefix(
        data_dir, f"images_{split_tag}", "images_"
    )
    masks_dir = find_dir_exact_or_prefix(
        data_dir, f"masks_{split_tag}", "masks_"
    )

    if images_dir is None or masks_dir is None:
        print(f"[WARN] Skipping {data_dir.name}: missing images_/masks_ folder(s).")
        return 0, 0, [], []

    images = [
        p for p in images_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    ]
    masks = [
        p for p in masks_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    ]

    # Index mask stems for later 'extra' detection
    mask_stems: Dict[str, List[Path]] = {}
    for m in masks:
        mask_stems.setdefault(m.stem.lower(), []).append(m)

    copied_imgs = copied_masks = 0
    imgs_without_mask: List[str] = []
    matched_mask_stems: set[str] = set()

    for img in images:
        base = img.stem  # without extension
        # destination file keeps source extension
        img_dst = out_img / f"{split_tag}_{base}{img.suffix.lower()}"
        copy_file(img, img_dst, dry_run=dry_run, hardlink=hardlink)
        copied_imgs += 1

        mask_path = choose_mask_for(base, masks_dir)
        if mask_path is None:
            imgs_without_mask.append(f"{split_tag}:{base}")
        else:
            mask_dst = out_mask / f"{split_tag}_{base}{MASK_SUFFIX}{mask_path.suffix.lower()}"
            copy_file(mask_path, mask_dst, dry_run=dry_run, hardlink=hardlink)
            copied_masks += 1
            matched_mask_stems.add(mask_path.stem.lower())

    # masks that look like masks but didn't match any image
    extra_masks: List[str] = []
    for m in masks:
        if (MASK_SUFFIX in m.stem.lower()) and (m.stem.lower() not in matched_mask_stems):
            extra_masks.append(f"{split_tag}:{m.stem}")

    return copied_imgs, copied_masks, imgs_without_mask, extra_masks

def main():
    ap = argparse.ArgumentParser(
        description="Merge images/masks from data_C* (images_C*, masks_C*) into output/img and output/mask."
    )
    ap.add_argument("--root", type=Path, required=True, help="Folder containing data_C1 ... data_C6")
    ap.add_argument("--out", type=Path, required=True, help="Output root (creates 'img' and 'mask')")
    ap.add_argument("--dry-run", action="store_true", help="List planned actions without copying")
    ap.add_argument("--hardlink", action="store_true", help="Try hardlinks instead of copies when possible")
    args = ap.parse_args()

    out_img = args.out / "img"
    out_mask = args.out / "mask"

    splits = sorted([p for p in args.root.iterdir() if p.is_dir() and p.name.lower().startswith("data_c")])
    if not splits:
        print(f"[ERROR] No 'data_C*' folders under: {args.root}")
        return

    total_i = total_m = 0
    total_missing: List[str] = []
    total_extra: List[str] = []

    print(f"Found splits: {[p.name for p in splits]}")
    for d in splits:
        ni, nm, missing, extra = process_split(d, out_img, out_mask, args.dry_run, args.hardlink)
        print(f"[{d.name}] images={ni}, masks={nm}, missing_masks={len(missing)}, extra_masks={len(extra)}")
        total_i += ni
        total_m += nm
        total_missing.extend(missing)
        total_extra.extend(extra)

    print("\n=== SUMMARY ===")
    print(f"Images copied: {total_i}")
    print(f"Masks  copied: {total_m}")
    print(f"Images without mask: {len(total_missing)}")
    if total_missing:
        print("  - " + "\n  - ".join(total_missing[:50] + (['...'] if len(total_missing) > 50 else [])))
    print(f"Extra masks (no matching image): {len(total_extra)}")
    if total_extra:
        print("  - " + "\n  - ".join(total_extra[:50] + (['...'] if len(total_extra) > 50 else [])))

if __name__ == "__main__":
    main()