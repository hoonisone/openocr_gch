import os
import io
import sys
import lmdb
import argparse
from pathlib import Path
from typing import List, Tuple

from PIL import Image, UnidentifiedImageError


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build OpenOCR-compatible LMDB from image directory + label file"
    )
    parser.add_argument(
        "--image-root",
        type=str,
        required=True,
        help="Directory containing image files",
    )
    parser.add_argument(
        "--label-file",
        type=str,
        required=True,
        help="Label file path",
    )
    parser.add_argument(
        "--output-lmdb",
        type=str,
        required=True,
        help="Output LMDB directory to create",
    )
    parser.add_argument(
        "--delimiter",
        type=str,
        default="auto",
        choices=["auto", "tab", "space"],
        help="Delimiter format of label file",
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default="utf-8",
        help="Encoding of label file",
    )
    parser.add_argument(
        "--map-size",
        type=int,
        default=1099511627776,  # 1 TB
        help="LMDB map size in bytes",
    )
    parser.add_argument(
        "--check-image",
        action="store_true",
        help="Fully verify image readability with PIL",
    )
    parser.add_argument(
        "--relative-path",
        action="store_true",
        help="Treat image paths in label file as relative to --image-root",
    )
    parser.add_argument(
        "--absolute-path",
        action="store_true",
        help="Treat image paths in label file as absolute paths",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Allow nested relative paths under image-root",
    )
    return parser.parse_args()


def parse_label_line(line: str, delimiter: str) -> Tuple[str, str]:
    line = line.rstrip("\n\r")
    if not line:
        raise ValueError("Empty line")

    if delimiter == "tab":
        parts = line.split("\t", 1)
        if len(parts) != 2:
            raise ValueError(f"Tab-delimited parse failed: {line}")
        return parts[0].strip(), parts[1]

    if delimiter == "space":
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"Space-delimited parse failed: {line}")
        return parts[0].strip(), parts[1]

    # auto
    if "\t" in line:
        parts = line.split("\t", 1)
        if len(parts) == 2:
            return parts[0].strip(), parts[1]
    parts = line.split(maxsplit=1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1]

    raise ValueError(f"Auto parse failed: {line}")


def load_label_pairs(label_file: str, encoding: str, delimiter: str) -> List[Tuple[str, str]]:
    pairs = []
    with open(label_file, "r", encoding=encoding) as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip("\n\r")
            if not line.strip():
                continue
            try:
                img_name, label = parse_label_line(line, delimiter)
                pairs.append((img_name, label))
            except Exception as e:
                print(f"[WARN] Skip line {line_no}: {e}", file=sys.stderr)
    return pairs


def resolve_image_path(
    img_name: str,
    image_root: Path,
    use_absolute: bool,
    recursive: bool,
) -> Path:
    p = Path(img_name)

    if use_absolute:
        return p

    # default: relative to image_root
    if recursive:
        return image_root / p
    return image_root / p.name


def read_image_bytes_and_wh(img_path: Path, verify: bool = False) -> Tuple[bytes, int, int]:
    with open(img_path, "rb") as f:
        raw = f.read()

    if verify:
        try:
            with Image.open(io.BytesIO(raw)) as im:
                im.load()
                w, h = im.size
        except UnidentifiedImageError as e:
            raise ValueError(f"PIL cannot identify image: {img_path}") from e
    else:
        try:
            with Image.open(io.BytesIO(raw)) as im:
                w, h = im.size
        except Exception as e:
            raise ValueError(f"Failed to read image size: {img_path}") from e

    return raw, w, h


def main():
    args = parse_args()

    image_root = Path(args.image_root).resolve()
    label_file = Path(args.label_file).resolve()
    output_lmdb = Path(args.output_lmdb).resolve()

    if not image_root.exists():
        raise FileNotFoundError(f"image-root not found: {image_root}")
    if not label_file.exists():
        raise FileNotFoundError(f"label-file not found: {label_file}")

    output_lmdb.mkdir(parents=True, exist_ok=True)

    pairs = load_label_pairs(
        label_file=str(label_file),
        encoding=args.encoding,
        delimiter=args.delimiter,
    )

    if not pairs:
        raise RuntimeError("No valid (image, label) pairs found.")

    print(f"[INFO] Loaded {len(pairs)} label lines")

    env = lmdb.open(
        str(output_lmdb),
        map_size=args.map_size,
        subdir=True,
        readonly=False,
        meminit=False,
        map_async=True,
    )

    valid_count = 0
    skipped_count = 0
    cache = {}

    with env.begin(write=True) as txn:
        for i, (img_name, label) in enumerate(pairs, start=1):
            img_path = resolve_image_path(
                img_name=img_name,
                image_root=image_root,
                use_absolute=args.absolute_path,
                recursive=args.recursive,
            )

            if not img_path.exists():
                print(f"[WARN] Missing image: {img_path}", file=sys.stderr)
                skipped_count += 1
                continue

            if img_path.suffix.lower() not in IMG_EXTS:
                print(f"[WARN] Skip non-image extension: {img_path}", file=sys.stderr)
                skipped_count += 1
                continue

            try:
                raw, w, h = read_image_bytes_and_wh(img_path, verify=args.check_image)
            except Exception as e:
                print(f"[WARN] Bad image: {img_path} | {e}", file=sys.stderr)
                skipped_count += 1
                continue

            valid_count += 1
            idx = valid_count

            image_key = f"image-{idx:09d}".encode()
            label_key = f"label-{idx:09d}".encode()
            wh_key = f"wh-{idx:09d}".encode()

            cache[image_key] = raw
            cache[label_key] = label.encode("utf-8")
            cache[wh_key] = f"{w}_{h}".encode("utf-8")

            if idx % 1000 == 0:
                for k, v in cache.items():
                    txn.put(k, v)
                cache.clear()
                print(f"[INFO] Written {idx} samples")

        for k, v in cache.items():
            txn.put(k, v)

        txn.put(b"num-samples", str(valid_count).encode("utf-8"))

    env.sync()
    env.close()

    print(f"[DONE] LMDB created at: {output_lmdb}")
    print(f"[DONE] valid samples   : {valid_count}")
    print(f"[DONE] skipped samples : {skipped_count}")
    print("[DONE] Expected files: data.mdb, lock.mdb")


if __name__ == "__main__":
    main()