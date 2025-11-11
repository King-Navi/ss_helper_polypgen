#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import List, Set, Tuple, Optional
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from PIL import Image

# ----------------------------- Config -----------------------------
IMG_EXTS: Set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MASK_SUFFIXES = ("_segmentation", "_mask")

# ----------------------------- Utils rápidas -----------------------------
def is_image_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS and not p.name.startswith(".")

def list_imgs_and_masks(inputs_dir: Path, targets_dir: Path) -> Tuple[List[Path], List[Path]]:
    imgs = [p for p in inputs_dir.rglob("*") if is_image_file(p)] if inputs_dir.is_dir() else []
    masks = [p for p in targets_dir.rglob("*") if is_image_file(p)] if targets_dir.is_dir() else []
    imgs.sort(key=lambda p: str(p).lower())
    masks.sort(key=lambda p: str(p).lower())
    return imgs, masks

def mask_base_from_stem(stem: str) -> Optional[str]:
    for suf in MASK_SUFFIXES:
        if stem.endswith(suf):
            return stem[: -len(suf)]
    return None

# ----------------------------- Validación y pruning -----------------------------
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

    print(f"[{split_name}] imgs={len(imgs)} masks={len(masks)} "
            f"unpaired_imgs={len(unpaired_imgs)} unpaired_masks={len(unpaired_masks)}")

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

# ----------------------------- Conversión / Resize -----------------------------
def save_png_fast(im: Image.Image, dst: Path):
    # Compresión moderada, sin optimize -> mucho más rápido
    im.save(dst, format="PNG", compress_level=3)

def binarize_array(arr: np.ndarray, thresh: int = 0) -> np.ndarray:
    return np.where(arr > thresh, 255, 0).astype(np.uint8)

def binarize_and_png_mask(p: Path, dry_run: bool) -> Optional[Path]:
    with Image.open(p) as im:
        g = im.convert("L")
        arr = np.asarray(g, dtype=np.uint8)
    bin255 = binarize_array(arr)
    dst = p.with_suffix(".png")
    if dry_run:
        return None
    save_png_fast(Image.fromarray(bin255, mode="L"), dst)
    if dst.resolve() != p.resolve():
        try:
            p.unlink()
        except Exception:
            pass
    return dst

def convert_img_to_png(p: Path, dry_run: bool) -> Path:
    """Convierte cualquier imagen (input) a PNG manteniendo el stem."""
    if p.suffix.lower() == ".png":
        return p
    dst = p.with_suffix(".png")
    if dry_run:
        return dst
    im = Image.open(p)
    if im.mode not in ("RGB", "L"):
        im = im.convert("RGB")
    save_png_fast(im, dst)
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

# ----------------------------- Skips inteligentes -----------------------------
def is_binary_mask_array(arr: np.ndarray) -> bool:
    u = np.unique(arr)
    return (u.size <= 2) and (u.min() in (0, 255)) and (u.max() in (0, 255))

def should_skip_image(p: Path, size: Tuple[int, int], keep_aspect: bool) -> bool:
    if p.suffix.lower() != ".png":
        return False
    try:
        with Image.open(p) as im:
            return im.size == size  # si ya es PNG y tiene el tamaño objetivo
    except Exception:
        return False

def should_skip_mask(p: Path, size: Tuple[int, int], keep_aspect: bool) -> bool:
    if p.suffix.lower() != ".png":
        return False
    try:
        with Image.open(p) as im:
            im = im.convert("L")
            if im.size != size:
                return False
            arr = np.asarray(im, dtype=np.uint8)
            return is_binary_mask_array(arr)
    except Exception:
        return False

# ----------------------------- Procesamiento paralelo -----------------------------
def process_one_image(src: Path, size: Tuple[int, int], keep_aspect: bool, dry_run: bool) -> None:
    # Skip total si ya es PNG con tamaño correcto
    if should_skip_image(src, size, keep_aspect):
        return
    # Convertir a PNG si hace falta
    target = convert_img_to_png(src, dry_run)
    if dry_run:
        return
    with Image.open(target) as im:
        if im.mode not in ("RGB", "L"):
            im = im.convert("RGB")
        out = resize_image(im, size, keep_aspect, is_mask=False)
    save_png_fast(out, target)

def process_one_mask(src: Path, size: Tuple[int, int], keep_aspect: bool, dry_run: bool) -> None:
    # Binarizar y convertir a PNG si no está ya listo
    if not should_skip_mask(src, size, keep_aspect):
        new_m = binarize_and_png_mask(src, dry_run)
        src = new_m if new_m is not None else src
    if dry_run:
        return
    with Image.open(src) as im:
        im = im.convert("L")
        if im.size != size:
            out = resize_image(im, size, keep_aspect, is_mask=True)
            save_png_fast(out, src)
        # si ya tiene el tamaño, no re-guardamos

def resize_dir_parallel(files: List[Path], size: Tuple[int, int], keep_aspect: bool, is_mask: bool, dry_run: bool, jobs: Optional[int] = None):
    workers = jobs or min(8, (os.cpu_count() or 4))
    fn = process_one_mask if is_mask else process_one_image
    if dry_run:
        # En dry-run, no vale la pena paralelizar
        for p in files:
            fn(p, size, keep_aspect, dry_run=True)
        print(f"[PAR-DRY] {'masks' if is_mask else 'imgs'} scanned: {len(files)}")
        return
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(fn, p, size, keep_aspect, dry_run) for p in files]
        for _ in as_completed(futs):
            done += 1
    kind = "masks" if is_mask else "imgs"
    #print(f"[PAR] {kind} processed: {done} (threads={workers})")

# ----------------------------- Pipeline por split -----------------------------
def process_split(
    root: Path,
    split: str,
    size: int,
    keep_aspect: bool,
    prune_unpaired: bool,
    dry_run: bool,
    jobs: Optional[int],
):
    split_dir = root / split
    if not split_dir.is_dir():
        print(f"[INFO] split '{split}' not found, skipping.")
        return

    inputs_dir = split_dir / "inputs"
    targets_dir = split_dir / "targets"

    imgs, masks = list_imgs_and_masks(inputs_dir, targets_dir)

    # 1) validar pares
    check_and_optionally_prune(split, imgs, masks, prune_unpaired)

    # refrescar por si se borró algo
    imgs, masks = list_imgs_and_masks(inputs_dir, targets_dir)

    # 2) convertir imágenes a PNG (skip si ya están bien)
    final_imgs: List[Path] = []
    for img in imgs:
        if not should_skip_image(img, (size, size), keep_aspect):
            new_img = convert_img_to_png(img, dry_run=dry_run)
            final_imgs.append(new_img)
        else:
            final_imgs.append(img)

    # 3) binarizar + convertir masks a PNG (skip si ya están bien)
    final_masks: List[Path] = []
    for m in masks:
        if not should_skip_mask(m, (size, size), keep_aspect):
            new_m = binarize_and_png_mask(m, dry_run=dry_run)
            final_masks.append(new_m if new_m is not None else m)
        else:
            final_masks.append(m)

    # 4) resize (paralelo)
    resize_dir_parallel(final_imgs, (size, size), keep_aspect, is_mask=False, dry_run=dry_run, jobs=jobs)
    resize_dir_parallel(final_masks, (size, size), keep_aspect, is_mask=True,  dry_run=dry_run, jobs=jobs)

# ----------------------------- CLI -----------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Valida pares en train/val/test, convierte TODO a PNG, binariza masks y hace resize."
    )
    ap.add_argument("--root", type=Path, required=True, help="Carpeta que contiene train/ val/ test/")
    ap.add_argument("--size", type=int, default=256, help="Tamaño destino (cuadrado). Default 256")
    ap.add_argument("--jobs", type=int, default=None, help="Trabajadores paralelos (auto por defecto)")
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
            jobs=args.jobs,
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
  --size 256 \
  --jobs 8

"""


if __name__ == "__main__":
    main()
