import argparse
import json
import multiprocessing as mp
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageOps


def parse_ppocr_line(line: str, delimiter: str = "\t") -> Tuple[str, str]:
    line = line.rstrip("\n\r")
    parts = line.split(delimiter, 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid label line: {line}")
    return parts[0], parts[1]


def is_json_image_list(path_str: str) -> bool:
    s = path_str.strip()
    return s.startswith("[") and s.endswith("]")


def copy_or_rotate_image(
    src_path: Path,
    dst_path: Path,
    threshold: float,
    rotate_equal: bool = False,
) -> bool:
    """
    Returns:
        True  -> rotated
        False -> copied only
    """
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(src_path) as img:
        img = ImageOps.exif_transpose(img)
        w, h = img.size

        if w <= 0 or h <= 0:
            raise ValueError(f"Invalid image size: {src_path}, size={img.size}")

        ratio = w / float(h)
        should_rotate = ratio >= threshold if rotate_equal else ratio > threshold

        if should_rotate:
            # Rotate clockwise 90deg: horizontal -> vertical
            img = img.transpose(Image.Transpose.ROTATE_270)
            img.save(dst_path)
            return True

    shutil.copy2(src_path, dst_path)
    return False


def process_one_image_path(
    rel_path_str: str,
    data_dir: Path,
    out_data_dir: Path,
    threshold: float,
    rotate_equal: bool,
) -> bool:
    rel_path = Path(rel_path_str)
    src_path = data_dir / rel_path
    dst_path = out_data_dir / rel_path

    if not src_path.exists():
        raise FileNotFoundError(f"Image not found: {src_path}")

    return copy_or_rotate_image(
        src_path=src_path,
        dst_path=dst_path,
        threshold=threshold,
        rotate_equal=rotate_equal,
    )


def worker(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Multiprocessing worker.
    한 label line을 처리하고 결과를 반환한다.
    """
    line_no = task["line_no"]
    line = task["line"]
    data_dir = Path(task["data_dir"])
    out_data_dir = Path(task["out_data_dir"])
    threshold = task["threshold"]
    delimiter = task["delimiter"]
    rotate_equal = task["rotate_equal"]
    allow_json_list = task["allow_json_list"]

    try:
        img_field, label = parse_ppocr_line(line, delimiter=delimiter)

        rotated = 0
        copied = 0

        if allow_json_list and is_json_image_list(img_field):
            img_list: List[str] = json.loads(img_field)
            if not isinstance(img_list, list):
                raise ValueError("JSON image field is not a list")

            for rel_path in img_list:
                was_rotated = process_one_image_path(
                    rel_path_str=rel_path,
                    data_dir=data_dir,
                    out_data_dir=out_data_dir,
                    threshold=threshold,
                    rotate_equal=rotate_equal,
                )
                rotated += int(was_rotated)
                copied += int(not was_rotated)

            out_line = f"{img_field}{delimiter}{label}\n"

        else:
            was_rotated = process_one_image_path(
                rel_path_str=img_field,
                data_dir=data_dir,
                out_data_dir=out_data_dir,
                threshold=threshold,
                rotate_equal=rotate_equal,
            )
            rotated += int(was_rotated)
            copied += int(not was_rotated)

            out_line = f"{img_field}{delimiter}{label}\n"

        return {
            "ok": True,
            "line_no": line_no,
            "out_line": out_line,
            "rotated": rotated,
            "copied": copied,
            "error": None,
        }

    except Exception as e:
        return {
            "ok": False,
            "line_no": line_no,
            "out_line": None,
            "rotated": 0,
            "copied": 0,
            "error": str(e),
        }


def make_tasks(
    label_file: Path,
    data_dir: Path,
    out_data_dir: Path,
    threshold: float,
    delimiter: str,
    rotate_equal: bool,
    allow_json_list: bool,
):
    with label_file.open("r", encoding="utf-8") as fin:
        for line_no, line in enumerate(fin, start=1):
            if not line.strip():
                continue

            yield {
                "line_no": line_no,
                "line": line,
                "data_dir": str(data_dir),
                "out_data_dir": str(out_data_dir),
                "threshold": threshold,
                "delimiter": delimiter,
                "rotate_equal": rotate_equal,
                "allow_json_list": allow_json_list,
            }


def process_dataset_mp(
    data_dir: Path,
    label_file: Path,
    out_data_dir: Path,
    out_label_file: Path,
    threshold: float,
    delimiter: str = "\t",
    rotate_equal: bool = False,
    allow_json_list: bool = False,
    num_workers: Optional[int] = None,
    chunksize: int = 128,
) -> None:
    data_dir = data_dir.resolve()
    label_file = label_file.resolve()
    out_data_dir = out_data_dir.resolve()
    out_label_file = out_label_file.resolve()

    if not data_dir.is_dir():
        raise NotADirectoryError(f"data_dir does not exist: {data_dir}")
    if not label_file.is_file():
        raise FileNotFoundError(f"label_file does not exist: {label_file}")

    out_data_dir.mkdir(parents=True, exist_ok=True)
    out_label_file.parent.mkdir(parents=True, exist_ok=True)

    if num_workers is None:
        num_workers = max(1, (mp.cpu_count() or 1) - 1)

    chunksize = max(1, int(chunksize))

    tasks = make_tasks(
        label_file=label_file,
        data_dir=data_dir,
        out_data_dir=out_data_dir,
        threshold=threshold,
        delimiter=delimiter,
        rotate_equal=rotate_equal,
        allow_json_list=allow_json_list,
    )

    results: Dict[int, str] = {}

    total = 0
    rotated = 0
    copied = 0
    skipped = 0

    with mp.Pool(processes=num_workers) as pool:
        for res in pool.imap_unordered(worker, tasks, chunksize=chunksize):
            line_no = res["line_no"]

            if res["ok"]:
                results[line_no] = res["out_line"]
                rotated += res["rotated"]
                copied += res["copied"]
                total += 1
            else:
                skipped += 1
                print(f"[WARN] line {line_no} skipped: {res['error']}")

            if (total + skipped) % 10000 == 0:
                print(
                    f"[INFO] processed={total + skipped}, "
                    f"ok={total}, rotated={rotated}, copied={copied}, skipped={skipped}"
                )

    # imap_unordered를 썼으므로 label file은 line_no 기준으로 다시 정렬해서 저장
    with out_label_file.open("w", encoding="utf-8") as fout:
        for line_no in sorted(results.keys()):
            fout.write(results[line_no])

    print("Done.")
    print(f"Total label lines processed: {total}")
    print(f"Images rotated: {rotated}")
    print(f"Images copied only: {copied}")
    print(f"Skipped lines: {skipped}")
    print(f"Workers: {num_workers}")
    print(f"Output data dir: {out_data_dir}")
    print(f"Output label file: {out_label_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Copy PPOCR dataset and rotate horizontal images clockwise using multiprocessing."
    )
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Original PPOCR data_dir. Image paths in label file are relative to this directory.",
    )
    parser.add_argument(
        "--label_file",
        required=True,
        help="Original PPOCR label file. Format: relative/path.jpg<TAB>label",
    )
    parser.add_argument(
        "--out_data_dir",
        required=True,
        help="New output data_dir. Relative image paths are preserved.",
    )
    parser.add_argument(
        "--out_label_file",
        required=True,
        help="New output label file.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.5,
        help="Rotate if width / height > threshold. Default: 1.5",
    )
    parser.add_argument(
        "--delimiter",
        default="\t",
        help="Delimiter between image path and label. Default: tab.",
    )
    parser.add_argument(
        "--rotate_equal",
        action="store_true",
        help="Use width / height >= threshold instead of > threshold.",
    )
    parser.add_argument(
        "--allow_json_list",
        action="store_true",
        help='Also handle image field like ["a.jpg", "b.jpg"].',
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of worker processes. Default: CPU count - 1.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=128,
        help="Chunksize for multiprocessing imap_unordered. Default: 128.",
    )

    args = parser.parse_args()

    process_dataset_mp(
        data_dir=Path(args.data_dir),
        label_file=Path(args.label_file),
        out_data_dir=Path(args.out_data_dir),
        out_label_file=Path(args.out_label_file),
        threshold=args.threshold,
        delimiter=args.delimiter,
        rotate_equal=args.rotate_equal,
        allow_json_list=args.allow_json_list,
        num_workers=args.num_workers,
        chunksize=args.chunksize,
    )


if __name__ == "__main__":
    main()
