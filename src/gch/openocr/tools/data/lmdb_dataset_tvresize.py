import io
import math
import os
import random

import lmdb
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F

from openocr.openrec.preprocess import create_operators, transform

import openocr.tools.data as data_mod

data_mod.DATASET_MODULES["LMDBDatasetTVResize"] = (
    "gch.openocr.tools.data.lmdb_dataset_tvresize"
)


class LMDBDatasetTVResize(Dataset):
    """LMDB + TVResize variant mirroring SimpleDatasetTVResize behavior."""

    def __init__(self, config, mode, logger, seed=None, epoch=1, task="rec"):
        super().__init__()
        self.ds_width = config[mode]["dataset"].get("ds_width", True)
        global_config = config["Global"]
        dataset_config = config[mode]["dataset"]
        loader_config = config[mode]["loader"]
        max_ratio = loader_config.get("max_ratio", 10)
        min_ratio = loader_config.get("min_ratio", 1)

        self.logger = logger
        self.mode = mode.lower()
        self.padding = dataset_config.get("padding", True)
        self.padding_rand = dataset_config.get("padding_rand", False)
        self.padding_doub = dataset_config.get("padding_doub", False)
        self.do_shuffle = loader_config["shuffle"]
        self.seed = epoch

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

        self.lmdb_sets = self._load_lmdb_sets(lmdb_dir_list, ratio_list)
        if not self.lmdb_sets:
            raise RuntimeError(
                "No valid LMDB dataset found under data_dir/data_dir_list."
            )
        logger.info(f"Initialize indexs of LMDB datasets: {lmdb_dir_list}")
        self.data_idx_order_list = self._build_index_order(seed, epoch)

        if self.mode == "train" and self.do_shuffle:
            self._shuffle_data_random(seed, epoch)
        self._set_epoch_as_seed(seed, dataset_config)

        self.ops = create_operators(dataset_config["transforms"], global_config)

        wh_ratio = np.around(np.array(self.get_wh_ratio()))
        self.wh_ratio = np.clip(wh_ratio, a_min=min_ratio, a_max=max_ratio)
        for i in range(max_ratio + 1):
            logger.info((1 * (self.wh_ratio == i)).sum())
        self.wh_ratio_sort = np.argsort(self.wh_ratio)

        self.need_reset = True in [x < 1 for x in ratio_list]
        self.error = 0
        self.base_shape = dataset_config.get(
            "base_shape", [[64, 64], [96, 48], [112, 40], [128, 32]]
        )
        self.base_h = dataset_config.get("base_h", 32)
        self.interpolation = T.InterpolationMode.BICUBIC
        self.transforms = T.Compose([T.ToTensor(), T.Normalize(0.5, 0.5)])

    def _set_epoch_as_seed(self, seed, dataset_config):
        if self.mode != "train":
            return
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
            ep = seed if seed is not None else 0
            dataset_config["transforms"][border_map_id]["MakeBorderMap"]["epoch"] = ep
            dataset_config["transforms"][shrink_map_id]["MakeShrinkMap"]["epoch"] = ep
        except Exception:
            return

    @staticmethod
    def _resolve_lmdb_paths(root_dir):
        candidate_paths = []
        if not os.path.isdir(root_dir):
            return candidate_paths

        if os.path.isfile(os.path.join(root_dir, "data.mdb")):
            return [root_dir]

        for dirpath, dirnames, _ in os.walk(root_dir):
            if dirnames:
                continue
            if os.path.isfile(os.path.join(dirpath, "data.mdb")):
                candidate_paths.append(dirpath)
        return candidate_paths

    @staticmethod
    def _open_lmdb(dirpath):
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

    def _build_index_order(self, seed, epoch):
        total_sample_num = sum(
            self.lmdb_sets[idx]["ratio_num_samples"] for idx in self.lmdb_sets
        )
        data_idx_order_list = np.zeros((total_sample_num, 2), dtype=np.float64)
        beg_idx = 0
        rnd = seed if seed is not None else epoch
        random.seed(rnd)
        for lno in sorted(self.lmdb_sets):
            cfg = self.lmdb_sets[lno]
            sample_num = cfg["ratio_num_samples"]
            end_idx = beg_idx + sample_num
            data_idx_order_list[beg_idx:end_idx, 0] = lno
            data_idx_order_list[beg_idx:end_idx, 1] = list(
                random.sample(range(1, cfg["num_samples"] + 1), sample_num)
            )
            beg_idx = end_idx
        return data_idx_order_list

    def _shuffle_data_random(self, seed, epoch):
        rnd = seed if seed is not None else epoch
        random.seed(rnd)
        np.random.shuffle(self.data_idx_order_list)

    def get_wh_ratio(self):
        wh_ratio = []
        for idx in range(self.data_idx_order_list.shape[0]):
            lmdb_idx, file_idx = self.data_idx_order_list[idx]
            lmdb_idx = int(lmdb_idx)
            file_idx = int(file_idx)
            wh_key = "wh-%09d".encode() % file_idx
            wh = self.lmdb_sets[lmdb_idx]["txn"].get(wh_key)
            if wh is None:
                img_key = f"image-{file_idx:09d}".encode()
                img = self.lmdb_sets[lmdb_idx]["txn"].get(img_key)
                if not img:
                    wh_ratio.append(1.0)
                    continue
                w, h = Image.open(io.BytesIO(img)).size
            else:
                wh = wh.decode("utf-8")
                w, h = wh.split("_")
            if int(h) == 0:
                wh_ratio.append(1.0)
            else:
                wh_ratio.append(float(w) / float(h))
        return wh_ratio

    def resize_norm_img(self, data, gen_ratio, padding=True):
        img = data["image"]
        w, h = img.size
        if self.padding_rand and random.random() < 0.5:
            padding = not padding
        if gen_ratio <= 4:
            imgW, imgH = self.base_shape[gen_ratio - 1]
        else:
            imgW = self.base_h * gen_ratio
            imgH = self.base_h
        imgW, imgH = int(imgW), int(imgH)

        use_ratio = imgW // imgH
        if use_ratio >= (w // h) + 2:
            self.error += 1
            return None
        if not padding:
            resized_w = imgW
        else:
            ratio = w / float(h)
            if math.ceil(imgH * ratio) > imgW:
                resized_w = imgW
            else:
                resized_w = int(math.ceil(imgH * ratio * (random.random() + 0.5)))
                resized_w = min(imgW, resized_w)
        resized_image = F.resize(
            img, [imgH, int(resized_w)], interpolation=self.interpolation
        )
        img = self.transforms(resized_image)
        if resized_w < imgW and padding:
            if self.padding_doub and random.random() < 0.5:
                img = F.pad(img, [0, 0, imgW - resized_w, 0], fill=0.0)
            else:
                img = F.pad(img, [imgW - resized_w, 0, 0, 0], fill=0.0)
        valid_ratio = min(1.0, float(resized_w / imgW))
        data["image"] = img
        data["valid_ratio"] = valid_ratio
        r = float(w) / float(h)
        data["real_ratio"] = max(1, round(r))
        return data

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

    def __getitem__(self, properties):
        img_width = properties[0]
        img_height = properties[1]
        idx = properties[2]
        ratio = properties[3]
        lmdb_idx, file_idx = self.data_idx_order_list[idx]
        lmdb_idx = int(lmdb_idx)
        file_idx = int(file_idx)
        sample_info = self.get_lmdb_sample_info(
            self.lmdb_sets[lmdb_idx]["txn"], file_idx
        )
        if sample_info is None:
            ratio_ids = np.where(self.wh_ratio == ratio)[0].tolist()
            if not ratio_ids:
                ratio_ids = list(range(len(self)))
            ids = random.sample(ratio_ids, 1)
            return self.__getitem__([img_width, img_height, ids[0], ratio])

        img, label = sample_info
        data = {
            "image": img,
            "label": label,
            "img_path": f"{self.lmdb_sets[lmdb_idx]['dirpath']}::{file_idx}",
        }
        outs = transform(data, self.ops[:-1])
        if outs is not None and hasattr(outs.get("image", None), "size"):
            outs = self.resize_norm_img(outs, ratio, padding=self.padding)
            if outs is None:
                ratio_ids = np.where(self.wh_ratio == ratio)[0].tolist()
                if not ratio_ids:
                    ratio_ids = list(range(len(self)))
                ids = random.sample(ratio_ids, 1)
                return self.__getitem__([img_width, img_height, ids[0], ratio])
            outs = transform(outs, self.ops[-1:])
        elif outs is not None:
            outs = None

        if outs is None:
            ratio_ids = np.where(self.wh_ratio == ratio)[0].tolist()
            if not ratio_ids:
                ratio_ids = list(range(len(self)))
            ids = random.sample(ratio_ids, 1)
            return self.__getitem__([img_width, img_height, ids[0], ratio])
        return outs

    def __len__(self):
        return self.data_idx_order_list.shape[0]
