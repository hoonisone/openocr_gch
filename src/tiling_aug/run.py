import argparse
import hashlib
import json
import random
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from PIL import Image

from .aug import tile_rotate_augment

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable


def load_config(config_path: str) -> Dict[str, Any]:
    """
    yaml 또는 json config를 로드합니다.
    yaml 사용 시 pyyaml이 필요합니다.
    """

    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    suffix = path.suffix.lower()

    if suffix in [".yaml", ".yml"]:
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                "YAML config를 사용하려면 pyyaml이 필요합니다. "
                "설치: pip install pyyaml"
            ) from exc

        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

    elif suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            config = json.load(f)

    else:
        raise ValueError("Config file must be .yaml, .yml, or .json")

    if config is None:
        raise ValueError("Config file is empty")

    if not isinstance(config, dict):
        raise ValueError("Config must be a dictionary/object")

    return config


def validate_rotate_range(rotate_range):
    """
    허용 형식:
        rotate_range: 10
        rotate_range: [-15, 15]
        rotate_range:
          - [-20, -10]
          - [10, 20]
    """

    # 숫자 하나: -abs(x) ~ abs(x)
    if isinstance(rotate_range, (int, float)):
        return float(rotate_range)

    if not isinstance(rotate_range, list):
        raise ValueError(
            "rotate_range must be a number, [min, max], or [[min, max], ...]"
        )

    # 단일 구간: [-15, 15]
    if (
        len(rotate_range) == 2
        and all(isinstance(v, (int, float)) for v in rotate_range)
    ):
        low = float(rotate_range[0])
        high = float(rotate_range[1])

        if low > high:
            raise ValueError(f"Invalid rotate_range: [{low}, {high}]")

        return [low, high]

    # 여러 구간: [[-20, -10], [10, 20]]
    if len(rotate_range) == 0:
        raise ValueError("rotate_range cannot be empty")

    normalized = []

    for interval in rotate_range:
        if (
            not isinstance(interval, list)
            or len(interval) != 2
            or not all(isinstance(v, (int, float)) for v in interval)
        ):
            raise ValueError(
                "For multiple intervals, rotate_range must be like "
                "[[-20, -10], [10, 20]]"
            )

        low = float(interval[0])
        high = float(interval[1])

        if low > high:
            raise ValueError(f"Invalid rotate interval: [{low}, {high}]")

        normalized.append([low, high])

    return normalized


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    config 필수 인자와 기본값을 확인합니다.
    """

    required_keys = ["image_dir", "save_dir"]

    for key in required_keys:
        if key not in config:
            raise KeyError(f"Missing required config key: {key}")

    config.setdefault("rotate_range", [-10, 10])
    config.setdefault("rotate_mode", "range_random")
    config.setdefault("rotate_prob", 1.0)
    config.setdefault("use_rotate", True)
    config.setdefault("use_tiling", True)
    config.setdefault("tiling_prob", 1.0)
    config.setdefault("crop", True)
    config.setdefault("fillcolor", None)
    config.setdefault("no_tiling_bg_mode", "fillcolor")
    config.setdefault("extensions", [".jpg", ".jpeg", ".png", ".bmp", ".webp"])
    config.setdefault("overwrite", False)
    config.setdefault("save_format", None)
    config.setdefault("seed", 42)
    config.setdefault("worker_num", 1)
    config.setdefault("thread_num", 1)

    image_dir = Path(config["image_dir"])
    save_dir = Path(config["save_dir"])

    if not image_dir.exists():
        raise FileNotFoundError(f"image_dir does not exist: {image_dir}")

    if not image_dir.is_dir():
        raise NotADirectoryError(f"image_dir is not a directory: {image_dir}")

    rotate_prob = float(config["rotate_prob"])

    if not (0.0 <= rotate_prob <= 1.0):
        raise ValueError("rotate_prob must be between 0.0 and 1.0")

    tiling_prob = float(config["tiling_prob"])
    if not (0.0 <= tiling_prob <= 1.0):
        raise ValueError("tiling_prob must be between 0.0 and 1.0")

    config["rotate_prob"] = rotate_prob
    config["use_rotate"] = bool(config["use_rotate"])
    config["tiling_prob"] = tiling_prob
    config["use_tiling"] = bool(config["use_tiling"])
    config["rotate_range"] = validate_rotate_range(config["rotate_range"])
    rotate_mode = str(config["rotate_mode"]).lower()
    if rotate_mode not in {
        "range_random",
        "aspect_ratio_range_random",
        "aspect_ratio_fixed_random_direction",
    }:
        raise ValueError(
            "rotate_mode must be one of: "
            "range_random, aspect_ratio_range_random, "
            "aspect_ratio_fixed_random_direction"
        )
    config["rotate_mode"] = rotate_mode

    no_tiling_bg_mode = str(config["no_tiling_bg_mode"]).lower()
    if no_tiling_bg_mode not in {"fillcolor", "mean", "inpaint"}:
        raise ValueError(
            "no_tiling_bg_mode must be one of: fillcolor, mean, inpaint"
        )
    if no_tiling_bg_mode == "inpaint":
        try:
            import cv2  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "no_tiling_bg_mode='inpaint' requires opencv-python. "
                "Install with: pip install opencv-python"
            ) from exc
    config["no_tiling_bg_mode"] = no_tiling_bg_mode

    fillcolor = config["fillcolor"]

    if fillcolor is not None:
        if (
            not isinstance(fillcolor, list)
            or len(fillcolor) != 3
            or not all(isinstance(v, int) for v in fillcolor)
        ):
            raise ValueError(
                "fillcolor must be null or a list of three integers, e.g. [0, 0, 0]"
            )

        if not all(0 <= v <= 255 for v in fillcolor):
            raise ValueError("fillcolor values must be between 0 and 255")

        config["fillcolor"] = tuple(fillcolor)

    if not isinstance(config["extensions"], list):
        raise ValueError("extensions must be a list, e.g. ['.jpg', '.png']")

    config["image_dir"] = image_dir
    config["save_dir"] = save_dir
    config["extensions"] = [str(ext).lower() for ext in config["extensions"]]

    save_format = config["save_format"]
    if save_format is not None:
        config["save_format"] = str(save_format).lower().lstrip(".")

    seed = config.get("seed")
    if not isinstance(seed, int):
        raise ValueError("seed must be an integer")

    worker_num = config.get("worker_num")
    thread_num = config.get("thread_num")

    if not isinstance(worker_num, int) or worker_num < 1:
        raise ValueError("worker_num must be an integer >= 1")

    if not isinstance(thread_num, int) or thread_num < 1:
        raise ValueError("thread_num must be an integer >= 1")

    config["crop"] = bool(config["crop"])
    config["overwrite"] = bool(config["overwrite"])
    config["seed"] = seed
    config["worker_num"] = worker_num
    config["thread_num"] = thread_num

    return config


def collect_images(image_dir: Path, extensions: List[str]) -> List[Path]:
    image_paths = []

    for path in image_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in extensions:
            image_paths.append(path)

    return sorted(image_paths)


def make_save_path(
    image_path: Path,
    image_dir: Path,
    save_dir: Path,
    save_format: Optional[str] = None,
) -> Path:
    """
    원본 폴더 구조를 유지해서 저장 경로를 만듭니다.
    """

    relative_path = image_path.relative_to(image_dir)

    if save_format is not None:
        relative_path = relative_path.with_suffix(f".{save_format}")

    return save_dir / relative_path


def split_evenly(items: List[Path], n: int) -> List[List[Path]]:
    if n <= 1:
        return [items]
    return [items[i::n] for i in range(n)]


def build_image_seed(
    base_seed: int,
    image_path: Path,
    worker_idx: int,
    thread_idx: int,
) -> int:
    payload = f"{base_seed}|{worker_idx}|{thread_idx}|{image_path.as_posix()}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return (base_seed + int(digest[:8], 16)) % (2**32)


def process_image_list(
    image_paths: List[Path],
    *,
    worker_idx: int,
    thread_idx: int,
    config: Dict[str, Any],
    show_progress: bool,
) -> Tuple[int, int, int]:
    image_dir: Path = config["image_dir"]
    save_dir: Path = config["save_dir"]

    rotate_range = config["rotate_range"]
    rotate_mode = config["rotate_mode"]
    rotate_prob = config["rotate_prob"]
    use_rotate = config["use_rotate"]
    use_tiling = config["use_tiling"]
    tiling_prob = config["tiling_prob"]
    crop = config["crop"]
    fillcolor = config["fillcolor"]
    no_tiling_bg_mode = config["no_tiling_bg_mode"]
    overwrite = config["overwrite"]
    save_format = config["save_format"]
    base_seed = config["seed"]

    success_count = 0
    fail_count = 0
    skip_count = 0

    iterable = image_paths
    if show_progress:
        iterable = tqdm(
            image_paths,
            total=len(image_paths),
            desc=f"W{worker_idx}-T{thread_idx}",
            dynamic_ncols=True,
        )

    for image_path in iterable:
        save_path = make_save_path(
            image_path=image_path,
            image_dir=image_dir,
            save_dir=save_dir,
            save_format=save_format,
        )

        if save_path.exists() and not overwrite:
            skip_count += 1
            continue

        save_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            per_image_seed = build_image_seed(
                base_seed=base_seed,
                image_path=image_path,
                worker_idx=worker_idx,
                thread_idx=thread_idx,
            )
            rng = random.Random(per_image_seed)

            with Image.open(image_path) as img:
                aug_img = tile_rotate_augment(
                    image=img,
                    use_rotate=use_rotate,
                    rotate_range=rotate_range,
                    rotate_mode=rotate_mode,
                    rotate_prob=rotate_prob,
                    use_tiling=use_tiling,
                    tiling_prob=tiling_prob,
                    crop=crop,
                    fillcolor=fillcolor,
                    no_tiling_bg_mode=no_tiling_bg_mode,
                    rng=rng,
                )
                aug_img.save(save_path)

            success_count += 1
        except Exception as exc:
            fail_count += 1
            print(
                f"FAIL (worker={worker_idx}, thread={thread_idx}): "
                f"{image_path} | {exc}"
            )

    return success_count, skip_count, fail_count


def process_worker(
    worker_idx: int,
    worker_image_paths: List[Path],
    config: Dict[str, Any],
) -> Tuple[int, int, int]:
    thread_num = config["thread_num"]
    thread_chunks = split_evenly(worker_image_paths, thread_num)

    if thread_num == 1:
        return process_image_list(
            worker_image_paths,
            worker_idx=worker_idx,
            thread_idx=0,
            config=config,
            show_progress=(worker_idx == 0),
        )

    success_total = 0
    skip_total = 0
    fail_total = 0

    with ThreadPoolExecutor(max_workers=thread_num) as executor:
        futures = []
        for thread_idx, thread_chunk in enumerate(thread_chunks):
            futures.append(
                executor.submit(
                    process_image_list,
                    thread_chunk,
                    worker_idx=worker_idx,
                    thread_idx=thread_idx,
                    config=config,
                    show_progress=(worker_idx == 0 and thread_idx == 0),
                )
            )

        for future in futures:
            success_count, skip_count, fail_count = future.result()
            success_total += success_count
            skip_total += skip_count
            fail_total += fail_count

    return success_total, skip_total, fail_total


def process_images(config: Dict[str, Any]) -> None:
    image_dir: Path = config["image_dir"]
    save_dir: Path = config["save_dir"]

    rotate_range = config["rotate_range"]
    rotate_mode = config["rotate_mode"]
    rotate_prob = config["rotate_prob"]
    use_rotate = config["use_rotate"]
    use_tiling = config["use_tiling"]
    tiling_prob = config["tiling_prob"]
    crop = config["crop"]
    fillcolor = config["fillcolor"]
    no_tiling_bg_mode = config["no_tiling_bg_mode"]
    extensions = config["extensions"]
    worker_num = config["worker_num"]
    thread_num = config["thread_num"]
    seed = config["seed"]

    image_paths = collect_images(image_dir, extensions)

    if not image_paths:
        print(f"No images found in {image_dir}")
        return

    print(f"Found {len(image_paths)} images.")
    print(f"Saving to: {save_dir}")
    print(f"rotate_range: {rotate_range}")
    print(f"rotate_mode : {rotate_mode}")
    print(f"use_rotate  : {use_rotate}")
    print(f"rotate_prob : {rotate_prob}")
    print(f"use_tiling  : {use_tiling}")
    print(f"tiling_prob : {tiling_prob}")
    print(f"crop        : {crop}")
    print(f"fillcolor   : {fillcolor}")
    print(f"no_tiling_bg_mode: {no_tiling_bg_mode}")
    print(f"seed        : {seed}")
    print(f"worker_num  : {worker_num}")
    print(f"thread_num  : {thread_num}")

    # 단일 모드(멀티프로세스/멀티스레드 미사용)에서는 메인에서 tqdm 표시
    if worker_num == 1 and thread_num == 1:
        success_count, skip_count, fail_count = process_image_list(
            image_paths,
            worker_idx=0,
            thread_idx=0,
            config=config,
            show_progress=True,
        )
    elif worker_num == 1:
        success_count, skip_count, fail_count = process_worker(
            worker_idx=0,
            worker_image_paths=image_paths,
            config=config,
        )
    else:
        worker_chunks = split_evenly(image_paths, worker_num)

        success_count = 0
        skip_count = 0
        fail_count = 0

        with ProcessPoolExecutor(max_workers=worker_num) as executor:
            futures = []
            for worker_idx, worker_chunk in enumerate(worker_chunks):
                futures.append(
                    executor.submit(
                        process_worker,
                        worker_idx,
                        worker_chunk,
                        config,
                    )
                )

            for future in futures:
                s_count, sk_count, f_count = future.result()
                success_count += s_count
                skip_count += sk_count
                fail_count += f_count

    print("\nDone.")
    print(f"Success: {success_count}")
    print(f"Skipped: {skip_count}")
    print(f"Failed : {fail_count}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="3x3 tile rotation augmentation with config file."
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file. Supports .yaml, .yml, .json",
    )

    args = parser.parse_args()

    config = load_config(args.config)
    config = validate_config(config)

    process_images(config)


if __name__ == "__main__":
    main()