#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Tuple, Optional, Set
from PIL import Image

IMG_EXTS: Set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def is_image_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS and not p.name.startswith(".")

def load_image(p: Path) -> Image.Image:
    # Convert images to RGB (3 channels). Adjust if you prefer to preserve modes.
    return Image.open(p)

def resize_stretch(im: Image.Image, size: Tuple[int, int], is_mask: bool) -> Image.Image:
    resample = Image.NEAREST if is_mask else Image.BILINEAR
    return im.resize(size, resample=resample)

def resize_keep_aspect(im: Image.Image, size: Tuple[int, int], is_mask: bool) -> Image.Image:
    target_w, target_h = size
    w, h = im.size
    scale = min(target_w / w, target_h / h)
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))

    resample = Image.NEAREST if is_mask else Image.BILINEAR
    resized = im.resize((new_w, new_h), resample=resample)

    # Create padded canvas
    if is_mask:
        # masks: single channel preferred; if not, still fill with 0
        mode = "L" if resized.mode == "L" else resized.mode
        canvas = Image.new(mode, (target_w, target_h), 0)
    else:
        # images: fill black
        mode = "RGB" if resized.mode == "RGB" else resized.mode
        fill = 0 if mode in ("L", "I;16") else (0, 0, 0)
        canvas = Image.new(mode, (target_w, target_h), fill)

    # center paste
    left = (target_w - new_w) // 2
    top = (target_h - new_h) // 2
    canvas.paste(resized, (left, top))
    return canvas

def ensure_mask_mode(im: Image.Image) -> Image.Image:
    """
    Prefer single-channel for masks. If it's RGB, convert to L (no binarization here).
    """
    if im.mode != "L":
        return im.convert("L")
    return im

def process_dir(
    in_dir: Path,
    out_dir: Optional[Path],
    size: Tuple[int, int],
    keep_aspect: bool,
    is_mask: bool,
    dry_run: bool,
) -> Tuple[int, int]:
    if not in_dir or not in_dir.is_dir():
        return (0, 0)

    files = [p for p in in_dir.rglob("*") if is_image_file(p)]
    if not files:
        print(f"[WARN] No images found in: {in_dir}")
        return (0, 0)

    count = 0
    skipped = 0
    for src in files:
        try:
            im = load_image(src)
            if is_mask:
                im = ensure_mask_mode(im)

            if keep_aspect:
                out_img = resize_keep_aspect(im, size, is_mask)
            else:
                out_img = resize_stretch(im, size, is_mask)

            if out_dir:
                dst = out_dir / src.relative_to(in_dir)
                dst.parent.mkdir(parents=True, exist_ok=True)
            else:
                dst = src

            if dry_run:
                print(f"[DRY] {src} -> {dst} size {im.size} -> {out_img.size} (mask={is_mask})")
            else:
                if is_mask: dst = dst.with_suffix(".png")
                out_img.save(dst)

            count += 1
        except Exception as e:
            print(f"[WARN] Skip {src}: {e}")
            skipped += 1

    return (count, skipped)

def main():
    ap = argparse.ArgumentParser(description="Resize dataset images and masks to 256x256.")
    ap.add_argument("--img-dir", type=Path, required=False, help="Input images directory")
    ap.add_argument("--mask-dir", type=Path, required=False, help="Input masks directory")
    ap.add_argument("--out-img", type=Path, required=False, help="Output dir for images (omit to overwrite)")
    ap.add_argument("--out-mask", type=Path, required=False, help="Output dir for masks (omit to overwrite)")
    ap.add_argument("--size", type=int, default=256, help="Target size (square). Default: 256")
    ap.add_argument("--keep-aspect", action="store_true", help="Keep aspect ratio with padding")
    ap.add_argument("--dry-run", action="store_true", help="Print actions without writing files")
    args = ap.parse_args()

    if not args.img_dir and not args.mask_dir:
        raise SystemExit("[ERROR] Provide at least --img-dir or --mask-dir")

    size = (args.size, args.size)

    total_done = total_skip = 0

    if args.img_dir:
        out_img = args.out_img if args.out_img else None
        if out_img:
            out_img.mkdir(parents=True, exist_ok=True)
        done, skip = process_dir(args.img_dir, out_img, size, args.keep_aspect, False, args.dry_run)
        print(f"[IMAGES] processed={done}, skipped={skip}")
        total_done += done; total_skip += skip

    if args.mask_dir:
        out_mask = args.out_mask if args.out_mask else None
        if out_mask:
            out_mask.mkdir(parents=True, exist_ok=True)
        done, skip = process_dir(args.mask_dir, out_mask, size, args.keep_aspect, True, args.dry_run)
        print(f"[MASKS ] processed={done}, skipped={skip}")
        total_done += done; total_skip += skip

    print(f"\n=== SUMMARY ===\nprocessed={total_done}, skipped={total_skip}")


"""
# Redimensionar estirando (más simple) y sobrescribiendo:
python resize_dataset_256.py \
  --img-dir /ruta/output/img \
  --mask-dir /ruta/output/mask

# Conservar aspecto con padding (mejor si tus imágenes tienen proporciones variadas):
python resize_dataset_256.py \
  --img-dir /ruta/output/img \
  --mask-dir /ruta/output/mask \
  --keep-aspect

# Escribir a nuevas carpetas:
python resize_dataset_256.py \
  --img-dir /home/ivan/Downloads/img_resized_512/output/img \
  --mask-dir /home/ivan/Downloads/img_resized_512/output/mask \
  --out-img /home/ivan/Downloads/img_resized_512/output/img_r \
  --out-mask /home/ivan/Downloads/img_resized_512/output/mask_r \
  --keep-aspect

# Ensayo sin escribir nada:
python resize_dataset_256.py \
  --img-dir /ruta/output/img \
  --mask-dir /ruta/output/mask \
  --keep-aspect --dry-run

"""
if __name__ == "__main__":
    main()
