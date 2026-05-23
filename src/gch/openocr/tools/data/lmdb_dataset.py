import os
import random
import traceback

import lmdb
import numpy as np
from torch.utils.data import Dataset

from openocr.openrec.preprocess import transform

import openocr.tools.data as data_mod

data_mod.DATASET_MODULES["GCHLMDBDataset"] = "gch.openocr.tools.data.lmdb_dataset"


class GCHLMDBDataset(Dataset):
    """LMDB dataset variant mirroring GCHSimpleDataset style usage."""

    def __init__(self, config, mode, logger, seed=None, epoch=0, task="rec"):
        super().__init__()
        self.logger = logger
        self.mode = mode.lower()
        self.seed = seed

        global_config = config["Global"]
        dataset_config = config[mode]["dataset"]
        loader_config = config[mode]["loader"]

        lmdb_dir_list = dataset_config.get("data_dir_list")
        lmdb_dir = dataset_config.get("data_dir")
        if lmdb_dir_list is None:
            assert lmdb_dir is not None, (
                "Either data_dir or data_dir_list must be provided."
            )
            lmdb_dir_list = [lmdb_dir]
        if isinstance(lmdb_dir_list, str):
            lmdb_dir_list = [lmdb_dir_list]

        missing_data_dirs = [
            path for path in lmdb_dir_list if not os.path.isdir(path)
        ]
        if missing_data_dirs:
            raise FileNotFoundError(
                "Missing LMDB data_dir_list paths:\n"
                + "\n".join(f"  - {path}" for path in missing_data_dirs)
            )

        ratio_list = dataset_config.get("ratio_list", 1.0)
        if isinstance(ratio_list, (float, int)):
            ratio_list = [float(ratio_list)] * len(lmdb_dir_list)
        assert len(ratio_list) == len(lmdb_dir_list), (
            "The length of ratio_list should be the same as data_dir_list."
        )

        self.do_shuffle = loader_config["shuffle"]
        self.lmdb_sets = self._load_lmdb_sets(lmdb_dir_list, ratio_list)
        if not self.lmdb_sets:
            raise RuntimeError(
                "No valid LMDB dataset found under data_dir/data_dir_list."
            )
        logger.info(f"Initialize indexs of LMDB datasets: {lmdb_dir_list}")

        self.data_idx_order_list = self._build_index_order()
        if self.mode == "train" and self.do_shuffle:
            self._shuffle_data_random()

        self._set_epoch_as_seed(self.seed, dataset_config)
        if task == "rec":
            from openocr.openrec.preprocess import create_operators
        elif task == "det":
            from openocr.opendet.preprocess import create_operators
        else:
            from openocr.openrec.preprocess import create_operators
        self.ops = create_operators(dataset_config["transforms"], global_config)
        self.ext_op_transform_idx = dataset_config.get("ext_op_transform_idx", 2)
        self.need_reset = True in [x < 1 for x in ratio_list]

    def _set_epoch_as_seed(self, seed, dataset_config):
        if self.mode == "train":
            try:
                border_map_id = [
                    index
                    for index, dictionary in enumerate(dataset_config["transforms"])
                    if "MakeBorderMap" in dictionary
                ][0]
                shrink_map_id = [
                    index
                    for index, dictionary in enumerate(dataset_config["transforms"])
                    if "MakeShrinkMap" in dictionary
                ][0]
                dataset_config["transforms"][border_map_id]["MakeBorderMap"][
                    "epoch"
                ] = seed if seed is not None else 0
                dataset_config["transforms"][shrink_map_id]["MakeShrinkMap"][
                    "epoch"
                ] = seed if seed is not None else 0
            except Exception:
                return

    @staticmethod
    def _resolve_lmdb_paths(root_dir):
        candidate_paths = []
        if not os.path.isdir(root_dir):
            return candidate_paths

        # Use root directly if it looks like an LMDB dir.
        if os.path.isfile(os.path.join(root_dir, "data.mdb")):
            return [root_dir]

        for dirpath, dirnames, _ in os.walk(root_dir):
            if dirnames:
                continue
            if os.path.isfile(os.path.join(dirpath, "data.mdb")):
                candidate_paths.append(dirpath)
        return candidate_paths

    def _open_lmdb(self, dirpath):
        env = lmdb.open(
            dirpath,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        txn = env.begin(write=False)
        num_samples_bytes = txn.get("num-samples".encode())
        if num_samples_bytes is None:
            env.close()
            raise RuntimeError(f"Invalid LMDB (missing num-samples): {dirpath}")
        num_samples = int(num_samples_bytes)
        return env, txn, num_samples

    def _load_lmdb_sets(self, lmdb_dir_list, ratio_list):
        lmdb_sets = {}
        dataset_idx = 0
        for root_dir, ratio in zip(lmdb_dir_list, ratio_list):
            lmdb_paths = self._resolve_lmdb_paths(root_dir)
            if not lmdb_paths:
                raise RuntimeError(f"No LMDB found under: {root_dir}")
            for lmdb_path in lmdb_paths:
                env, txn, num_samples = self._open_lmdb(lmdb_path)
                ratio_num_samples = round(num_samples * float(ratio))
                ratio_num_samples = max(0, min(num_samples, ratio_num_samples))
                if ratio_num_samples == 0:
                    env.close()
                    continue
                lmdb_sets[dataset_idx] = {
                    "dirpath": lmdb_path,
                    "env": env,
                    "txn": txn,
                    "num_samples": num_samples,
                    "ratio_num_samples": ratio_num_samples,
                }
                dataset_idx += 1
        return lmdb_sets

    def _build_index_order(self):
        total_sample_num = sum(
            self.lmdb_sets[idx]["ratio_num_samples"] for idx in self.lmdb_sets
        )
        data_idx_order_list = np.zeros((total_sample_num, 2), dtype=np.float64)
        beg_idx = 0
        rng = random.Random(self.seed)
        for lno in sorted(self.lmdb_sets):
            cfg = self.lmdb_sets[lno]
            sample_num = cfg["ratio_num_samples"]
            end_idx = beg_idx + sample_num
            data_idx_order_list[beg_idx:end_idx, 0] = lno
            ids = rng.sample(range(1, cfg["num_samples"] + 1), sample_num)
            data_idx_order_list[beg_idx:end_idx, 1] = ids
            beg_idx = end_idx
        return data_idx_order_list

    def _shuffle_data_random(self):
        random.seed(self.seed)
        np.random.shuffle(self.data_idx_order_list)

    def get_lmdb_sample_info(self, txn, index):
        label_key = "label-%09d".encode() % index
        label = txn.get(label_key)
        if label is None:
            return None
        label = label.decode("utf-8")
        img_key = "image-%09d".encode() % index
        imgbuf = txn.get(img_key)
        if not imgbuf:
            return None
        return imgbuf, label

    def get_ext_data(self):
        ext_data_num = 0
        for op in self.ops:
            if hasattr(op, "ext_data_num"):
                ext_data_num = getattr(op, "ext_data_num")
                break
        load_data_ops = self.ops[: self.ext_op_transform_idx]
        ext_data = []

        while len(ext_data) < ext_data_num:
            lmdb_idx, file_idx = self.data_idx_order_list[np.random.randint(self.__len__())]
            lmdb_idx = int(lmdb_idx)
            file_idx = int(file_idx)
            sample_info = self.get_lmdb_sample_info(
                self.lmdb_sets[lmdb_idx]["txn"], file_idx
            )
            if sample_info is None:
                continue
            img, label = sample_info
            data = {
                "image": img,
                "label": label,
                "img_path": f"{self.lmdb_sets[lmdb_idx]['dirpath']}::{file_idx}",
            }
            data = transform(data, load_data_ops)
            if data is None:
                continue
            if "polys" in data.keys():
                if data["polys"].shape[1] != 4:
                    continue
            ext_data.append(data)
        return ext_data

    def __getitem__(self, idx):
        lmdb_idx, file_idx = self.data_idx_order_list[idx]
        lmdb_idx = int(lmdb_idx)
        file_idx = int(file_idx)
        try:
            sample_info = self.get_lmdb_sample_info(
                self.lmdb_sets[lmdb_idx]["txn"], file_idx
            )
            if sample_info is None:
                raise RuntimeError("Failed to fetch sample from LMDB.")
            img, label = sample_info
            data = {
                "image": img,
                "label": label,
                "img_path": f"{self.lmdb_sets[lmdb_idx]['dirpath']}::{file_idx}",
            }
            data["ext_data"] = self.get_ext_data()
            outs = transform(data, self.ops)
        except Exception:
            self.logger.error(
                "When parsing LMDB sample (%s, %s), error happened with msg: %s"
                % (self.lmdb_sets[lmdb_idx]["dirpath"], file_idx, traceback.format_exc())
            )
            outs = None
        if outs is None:
            rnd_idx = (
                np.random.randint(self.__len__())
                if self.mode == "train"
                else (idx + 1) % self.__len__()
            )
            return self.__getitem__(rnd_idx)
        return outs

    def __len__(self):
        return self.data_idx_order_list.shape[0]
