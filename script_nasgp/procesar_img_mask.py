#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import List, Set, Tuple, Optional
import numpy as np
from PIL import Image

IMG_EXTS: Set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MASK_SUFFIXES = ("_segmentation", "_mask")


def is_image_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS and not p.name.startswith(".")


def list_imgs_and_masks(inputs_dir: Path, targets_dir: Path) -> Tuple[List[Path], List[Path]]:
    imgs = [p for p in inputs_dir.rglob("*") if is_image_file(p)] if inputs_dir.is_dir() else []
    masks = [p for p in targets_dir.rglob("*") if is_image_file(p)] if targets_dir.is_dir() else []
    return imgs, masks


def mask_base_from_stem(stem: str) -> Optional[str]:
    for suf in MASK_SUFFIXES:
        if stem.endswith(suf):
            return stem[: -len(suf)]
    return None


def check_and_optionally_prune(
    split_name: str,
    imgs: List[Path],
    masks: List[Path],
    prune: bool,
) -> None:
    img_bases = {p.stem for p in imgs}
    mask_bases = {}
    for m in masks:
        mb = mask_base_from_stem(m.stem)
        if mb is not None:
            mask_bases.setdefault(mb, []).append(m)
        else:
            mask_bases.setdefault("__BAD__", []).append(m)

    unpaired_imgs = [p for p in imgs if p.stem not in mask_bases]
    unpaired_masks: List[Path] = []
    for m in masks:
        mb = mask_base_from_stem(m.stem)
        if mb is None or mb not in img_bases:
            unpaired_masks.append(m)

    print(f"[{split_name}] imgs={len(imgs)} masks={len(masks)} unpaired_imgs={len(unpaired_imgs)} unpaired_masks={len(unpaired_masks)}")

    if unpaired_imgs:
        print(f"[{split_name}] Images without mask:")
        for p in unpaired_imgs:
            print("  -", p)

    if unpaired_masks:
        print(f"[{split_name}] Masks without image (or bad name):")
        for p in unpaired_masks:
            print("  -", p)

    if prune:
        deleted = 0
        for p in unpaired_imgs + unpaired_masks:
            try:
                p.unlink()
                deleted += 1
            except Exception as e:
                print(f"[WARN] cannot delete {p}: {e}")
        print(f"[{split_name}] Pruned {deleted} unpaired files.")


def binarize_array(arr: np.ndarray, thresh: int = 0) -> np.ndarray:
    return np.where(arr > thresh, 255, 0).astype(np.uint8)


def binarize_and_png_mask(p: Path, dry_run: bool) -> Optional[Path]:
    with Image.open(p) as im:
        g = im.convert("L")
        arr = np.asarray(g, dtype=np.uint8)

    bin255 = binarize_array(arr)

    dst = p.with_suffix(".png")

    print(f"[BIN] {p} -> {dst} uniques={np.unique(arr).tolist()} -> {np.unique(bin255).tolist()}")

    if dry_run:
        return None

    Image.fromarray(bin255, mode="L").save(dst, optimize=True)

    if dst.resolve() != p.resolve():
        try:
            p.unlink()
        except Exception:
            pass

    return dst


def convert_img_to_png(p: Path, dry_run: bool) -> Path:
    """
    Convierte cualquier imagen (input) a PNG manteniendo el nombre (stem).
    Devuelve la ruta final (que será .png).
    """
    if p.suffix.lower() == ".png":
        return p

    dst = p.with_suffix(".png")
    print(f"[IMG->PNG] {p} -> {dst}")

    if dry_run:
        return dst

    im = Image.open(p)
    if im.mode not in ("RGB", "L"):
        im = im.convert("RGB")

    im.save(dst, optimize=True)

    if dst.resolve() != p.resolve():
        try:
            p.unlink()
        except Exception:
            pass

    return dst


def resize_image(im: Image.Image, size: Tuple[int, int], keep_aspect: bool, is_mask: bool) -> Image.Image:
    target_w, target_h = size
    if not keep_aspect:
        resample = Image.NEAREST if is_mask else Image.BILINEAR
        return im.resize((target_w, target_h), resample=resample)

    w, h = im.size
    scale = min(target_w / w, target_h / h)
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resample = Image.NEAREST if is_mask else Image.BILINEAR
    resized = im.resize((new_w, new_h), resample=resample)

    if is_mask:
        canvas = Image.new("L", (target_w, target_h), 0)
    else:
        mode = "RGB" if im.mode != "L" else "L"
        canvas = Image.new(mode, (target_w, target_h), 0 if mode == "L" else (0, 0, 0))

    left = (target_w - new_w) // 2
    top = (target_h - new_h) // 2
    canvas.paste(resized, (left, top))
    return canvas


def resize_dir(
    files: List[Path],
    size: Tuple[int, int],
    keep_aspect: bool,
    is_mask: bool,
    dry_run: bool,
) -> None:
    for src in files:
        try:
            with Image.open(src) as im:
                if is_mask:
                    im = im.convert("L")
                out_img = resize_image(im, size, keep_aspect, is_mask)
            if dry_run:
                print(f"[DRY-RESIZE] {src} -> {out_img.size} (mask={is_mask})")
            else:
                out_img.save(src)
        except Exception as e:
            print(f"[WARN] resize failed for {src}: {e}")


def process_split(
    root: Path,
    split: str,
    size: int,
    keep_aspect: bool,
    prune_unpaired: bool,
    dry_run: bool,
):
    split_dir = root / split
    if not split_dir.is_dir():
        print(f"[INFO] split '{split}' not found, skipping.")
        return

    inputs_dir = split_dir / "inputs"
    targets_dir = split_dir / "targets"

    imgs, masks = list_imgs_and_masks(inputs_dir, targets_dir)

    # 1. validar pares
    check_and_optionally_prune(split, imgs, masks, prune_unpaired)

    # refrescar por si se borró algo
    imgs, masks = list_imgs_and_masks(inputs_dir, targets_dir)

    # 2. convertir imágenes a PNG
    final_imgs: List[Path] = []
    for img in imgs:
        new_img = convert_img_to_png(img, dry_run=dry_run)
        final_imgs.append(new_img)

    # 3. binarizar + convertir masks a PNG
    final_masks: List[Path] = []
    for m in masks:
        new_m = binarize_and_png_mask(m, dry_run=dry_run)
        if new_m is not None:
            final_masks.append(new_m)
        else:
            final_masks.append(m)

    # 4. resize
    resize_dir(final_imgs, (size, size), keep_aspect, is_mask=False, dry_run=dry_run)
    resize_dir(final_masks, (size, size), keep_aspect, is_mask=True, dry_run=dry_run)


def main():
    ap = argparse.ArgumentParser(
        description="Valida pares en train/val/test, convierte TODO a PNG, binariza masks y hace resize."
    )
    ap.add_argument("--root", type=Path, required=True, help="Carpeta que contiene train/ val/ test/")
    ap.add_argument("--size", type=int, default=256, help="Tamaño destino (cuadrado). Default 256")
    ap.add_argument("--keep-aspect", action="store_true", help="Mantener aspecto con padding")
    ap.add_argument("--prune-unpaired", action="store_true", help="Borrar archivos que no tengan par")
    ap.add_argument("--dry-run", action="store_true", help="Solo mostrar lo que se haría")
    args = ap.parse_args()

    for split in ("train", "val", "test", "valid"):
        process_split(
            root=args.root,
            split=split,
            size=args.size,
            keep_aspect=args.keep_aspect,
            prune_unpaired=args.prune_unpaired,
            dry_run=args.dry_run,
        )

    print("Done.")


if __name__ == "__main__":
    main()


"""
este es el 2°

Solo checar y ver qué haría (recomiendado):

python script_nasgp/procesar_img_mask.py \
  --root /home/ivan/Downloads/img_resized_512/output_2 \
  --dry-run


Checar y borrar los que no tengan par, binarizar, pasar a png y hacer resize 256x256:

python script_nasgp/procesar_img_mask.py \
  --root /home/ivan/Downloads/img_resized_512/output_2 \
  --prune-unpaired


Conservar aspecto (padding):

python script_nasgp/procesar_img_mask.py \
  --root /home/ivan/Downloads/nasga_example/struct \
  --prune-unpaired \
  --keep-aspect \
  --size 256


"""


if __name__ == "__main__":
    main()
