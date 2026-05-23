import io
import json
import math
import os
import random

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F

from openocr.openrec.preprocess import create_operators, transform

import openocr.tools.data as data_mod
data_mod.DATASET_MODULES['SimpleDatasetTVResize_HV_LogScale'] = (
    'gch.openocr.tools.data.simple_dataset_tvresize_hv_log_scale')


def _signed_log2_ratio_worker(args):
    line, delimiter, data_dir, max_log_ratio, scale_factor = args
    try:
        text = line.decode('utf-8')
        substr = text.rstrip('\r\n').split(delimiter)
        if len(substr) < 2:
            return 0

        file_name = substr[0]
        if len(file_name) > 0 and file_name[0] == '[':
            try:
                info = json.loads(file_name)
                file_name = random.choice(info)
            except Exception:
                pass

        img_path = os.path.join(data_dir, file_name)
        if not os.path.exists(img_path):
            return 0

        with Image.open(img_path) as img:
            w, h = img.size

        if h == 0 or w == 0:
            return 0

        ratio_wh = float(w) / float(h)
        if scale_factor <= 1.0:
            return 0
        log_bin = int(round(math.log(ratio_wh, scale_factor)))
        log_bin = int(np.clip(log_bin, -max_log_ratio, max_log_ratio))
        return log_bin

    except Exception:
        return 0


class SimpleDatasetTVResize_HV_LogScale(Dataset):
    """Simple label_file_list + data_dir layout with RatioDataSetTVResize behavior.

    This version uses signed log-base(scale_factor) aspect-ratio bins.

    signed_log_bin (when scale_factor=2):
        0  -> near square
        1  -> near horizontal 2x
       -1  -> near vertical 2x
        2  -> near horizontal 4x
       -2  -> near vertical 4x

    Shape mapping with base_size=32:
        0  -> [32, 32]
        1  -> [64, 32]
       -1  -> [32, 64]
        2  -> [128, 32]
       -2  -> [32, 128]

    This dataset keeps signed bins directly in self.wh_ratio.
    (e.g., -2, -1, 0, 1, 2)
    """

    def __init__(self, config, mode, logger, seed=None, epoch=1, task='rec'):
        super().__init__()

        self.ds_width = config[mode]['dataset'].get('ds_width', True)
        global_config = config['Global']
        dataset_config = config[mode]['dataset']
        loader_config = config[mode]['loader']

        self.logger = logger
        self.mode = mode.lower()
        self.delimiter = dataset_config.get('delimiter', '\t')

        # Log-ratio bin config.
        # max_log_ratio=2 gives signed bins: -2, -1, 0, 1, 2
        self.max_log_ratio = int(loader_config.get('max_log_ratio', 3))
        self.scale_factor = float(
            dataset_config.get('scale_factor', loader_config.get('scale_factor', 2.0))
        )
        if self.scale_factor <= 1.0:
            raise ValueError("scale_factor must be > 1.0")
        self.base_size = int(dataset_config.get('base_size', 32))
        self.min_shape_size = int(dataset_config.get('min_shape_size', 1))
        base_shape_cfg = dataset_config.get('base_shape', {'0': [64, 64], '1': [96, 48], '-1': [48, 96]})
        self.base_shape = {}
        if base_shape_cfg is not None:
            if not isinstance(base_shape_cfg, dict):
                raise ValueError(
                    "base_shape must be a dict, e.g. "
                    '{"-1": [32, 64], "0": [32, 32], "1": [64, 32]}'
                )
            for log_bin, shape in base_shape_cfg.items():
                if not isinstance(log_bin, (int, float, str)):
                    raise ValueError("base_shape log_bin key must be int-compatible")
                if (
                    not isinstance(shape, (list, tuple))
                    or len(shape) != 2
                    or not all(isinstance(v, (int, float)) for v in shape)
                ):
                    raise ValueError(
                        "base_shape shape must be [width, height], "
                        "e.g. [64, 32]"
                    )
                key = int(log_bin)
                if key < -self.max_log_ratio or key > self.max_log_ratio:
                    raise ValueError(
                        f"base_shape log_bin={key} is out of range "
                        f"[-{self.max_log_ratio}, {self.max_log_ratio}]"
                    )
                w = max(self.min_shape_size, int(round(float(shape[0]))))
                h = max(self.min_shape_size, int(round(float(shape[1]))))
                self.base_shape[key] = [w, h]

        label_file_list = dataset_config['label_file_list']
        if isinstance(label_file_list, str):
            label_file_list = [label_file_list]

        data_source_num = len(label_file_list)

        ratio_list = dataset_config.get('ratio_list', 1.0)
        if isinstance(ratio_list, (float, int)):
            ratio_list = [float(ratio_list)] * int(data_source_num)
        assert len(ratio_list) == data_source_num, (
            'The length of ratio_list should be the same as label_file_list.')

        self.padding = dataset_config.get('padding', True)
        self.padding_rand = dataset_config.get('padding_rand', False)
        self.padding_doub = dataset_config.get('padding_doub', False)
        self.do_shuffle = loader_config['shuffle']
        self.seed = epoch

        self.data_dir = dataset_config.get('data_dir')
        data_dir_list = dataset_config.get('data_dir_list')
        if data_dir_list is None:
            assert self.data_dir is not None, (
                'Either data_dir or data_dir_list must be provided.')
            data_dir_list = [self.data_dir] * data_source_num

        if isinstance(data_dir_list, str):
            data_dir_list = [data_dir_list]

        assert len(data_dir_list) == data_source_num, (
            'The length of data_dir_list should be the same as label_file_list.')

        missing_label_files = [
            path for path in label_file_list if not os.path.isfile(path)
        ]
        if missing_label_files:
            raise FileNotFoundError(
                'Missing label_file_list paths:\n' +
                '\n'.join(f'  - {path}' for path in missing_label_files))

        missing_data_dirs = [
            path for path in data_dir_list if not os.path.isdir(path)
        ]
        if missing_data_dirs:
            raise FileNotFoundError(
                'Missing data_dir_list paths:\n' +
                '\n'.join(f'  - {path}' for path in missing_data_dirs))

        logger.info(f'Initialize indexs of datasets: {label_file_list}')
        self.data_lines = self._get_image_info_list(
            label_file_list,
            data_dir_list,
            ratio_list,
            seed,
            epoch,
        )

        n = len(self.data_lines)
        self.data_idx_order_list = np.zeros((n, 2), dtype=np.float64)
        self.data_idx_order_list[:, 1] = np.arange(n, dtype=np.float64)

        if self.mode == 'train' and self.do_shuffle:
            self._shuffle_data_random(seed, epoch)

        self._set_epoch_as_seed(seed, dataset_config)

        self.ops = create_operators(dataset_config['transforms'],
                                    global_config)

        # signed bins: -max_log_ratio ... +max_log_ratio
        signed_bins = np.array(self.get_wh_ratio(), dtype=np.int32)
        signed_bins = np.clip(
            signed_bins,
            -self.max_log_ratio,
            self.max_log_ratio,
        ).astype(np.int32)

        self.wh_signed_log_ratio = signed_bins

        # Use signed bins directly for sampler grouping.
        self.wh_ratio = signed_bins.astype(np.int32)

        for signed_bin in range(-self.max_log_ratio, self.max_log_ratio + 1):
            count = int((self.wh_signed_log_ratio == signed_bin).sum())
            logger.info(f'signed_log2_ratio_bin={signed_bin}: {count}')

        self.wh_ratio_sort = np.argsort(self.wh_ratio)

        self.need_reset = True in [x < 1 for x in ratio_list]
        self.error = 0

        self.interpolation = T.InterpolationMode.BICUBIC
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(0.5, 0.5),
        ])


        

    def _shape_from_signed_log_bin(self, signed_bin):
        """Return target shape [imgW, imgH] from signed log-base(scale_factor) bin.

        If base_shape defines the bin, use it first.
        Otherwise fallback to base_size policy where base_size is the shorter side.

        Examples with base_size=32:
            0  -> [32, 32]
            1  -> [64, 32]
           -1  -> [32, 64]
            2  -> [128, 32]
           -2  -> [32, 128]
        """
        signed_bin = int(np.clip(
            int(signed_bin),
            -self.max_log_ratio,
            self.max_log_ratio,
        ))

        if signed_bin in self.base_shape:
            shape = self.base_shape[signed_bin]
            return int(shape[0]), int(shape[1])

        if signed_bin == 0:
            imgW = max(self.min_shape_size, self.base_size)
            imgH = max(self.min_shape_size, self.base_size)
            return int(imgW), int(imgH)

        scale = self.scale_factor ** abs(signed_bin)

        if signed_bin > 0:
            imgW = int(round(self.base_size * scale))
            imgH = self.base_size
        else:
            imgW = self.base_size
            imgH = int(round(self.base_size * scale))

        imgW = max(self.min_shape_size, imgW)
        imgH = max(self.min_shape_size, imgH)
        return int(imgW), int(imgH)

    def _set_epoch_as_seed(self, seed, dataset_config):
        if self.mode != 'train':
            return

        try:
            border_map_id = [
                index for index, dictionary in enumerate(
                    dataset_config['transforms']) if 'MakeBorderMap' in dictionary
            ][0]
            shrink_map_id = [
                index for index, dictionary in enumerate(
                    dataset_config['transforms']) if 'MakeShrinkMap' in dictionary
            ][0]

            ep = seed if seed is not None else 0
            dataset_config['transforms'][border_map_id]['MakeBorderMap'][
                'epoch'] = ep
            dataset_config['transforms'][shrink_map_id]['MakeShrinkMap'][
                'epoch'] = ep

        except Exception:
            return

    def _get_image_info_list(self, file_list, data_dir_list, ratio_list, seed,
                             epoch):
        rnd = seed if seed is not None else epoch
        data_lines = []

        for idx, (file, data_dir) in enumerate(zip(file_list, data_dir_list)):
            with open(file, 'rb') as f:
                lines = f.readlines()

            if self.mode == 'train' or ratio_list[idx] < 1.0:
                random.seed(rnd)
                k = max(0, round(len(lines) * ratio_list[idx]))
                if k == 0:
                    continue
                lines = random.sample(lines, min(k, len(lines)))

            data_lines.extend([(line, data_dir) for line in lines])

        return data_lines

    def _shuffle_data_random(self, seed, epoch):
        rnd = seed if seed is not None else epoch
        random.seed(rnd)
        random.shuffle(self.data_lines)

    @staticmethod
    def _try_parse_filename_list(file_name):
        if len(file_name) > 0 and file_name[0] == '[':
            try:
                info = json.loads(file_name)
                file_name = random.choice(info)
            except Exception:
                pass
        return file_name

    def _get_sample_bytes_and_label(self, line_idx):
        line_idx = int(line_idx)
        line, data_dir = self.data_lines[line_idx]

        try:
            text = line.decode('utf-8')
            substr = text.rstrip('\r\n').split(self.delimiter)

            if len(substr) < 2:
                return None

            file_name = substr[0]
            file_name = self._try_parse_filename_list(file_name)
            label = substr[1]

            img_path = os.path.join(data_dir, file_name)
            if not os.path.exists(img_path):
                return None

            with open(img_path, 'rb') as f:
                imgbuf = f.read()

            if not imgbuf:
                return None

            return imgbuf, label, img_path

        except Exception:
            return None

    def _get_signed_log2_ratio(self):
        signed_bins = []

        for idx in range(self.data_idx_order_list.shape[0]):
            line_idx = int(self.data_idx_order_list[idx, 1])
            info = self._get_sample_bytes_and_label(line_idx)

            if info is None:
                signed_bins.append(0)
                continue

            imgbuf, _, _ = info

            try:
                with Image.open(io.BytesIO(imgbuf)) as img:
                    w, h = img.size

                if h == 0 or w == 0:
                    signed_bins.append(0)
                    continue

                ratio_wh = float(w) / float(h)
                log_bin = int(round(math.log(ratio_wh, self.scale_factor)))
                log_bin = int(np.clip(
                    log_bin,
                    -self.max_log_ratio,
                    self.max_log_ratio,
                ))
                signed_bins.append(log_bin)

            except Exception:
                signed_bins.append(0)

        return signed_bins

    def get_wh_ratio(self):
        return self._get_wh_ratio_multiprocessing(num_workers=30)

    def _get_wh_ratio_multiprocessing(self, num_workers=None, chunksize=1024):
        import multiprocessing as mp

        total = self.data_idx_order_list.shape[0]
        if total == 0:
            return []

        if num_workers is None:
            num_workers = max(1, (mp.cpu_count() or 1) - 1)

        num_workers = max(1, int(num_workers / 2))
        chunksize = max(1, int(chunksize))

        tasks = (
            (
                self.data_lines[int(self.data_idx_order_list[i, 1])][0],
                self.delimiter,
                self.data_lines[int(self.data_idx_order_list[i, 1])][1],
                self.max_log_ratio,
                self.scale_factor,
            )
            for i in range(total)
        )

        with mp.Pool(processes=num_workers) as pool:
            signed_bins = list(pool.imap(
                _signed_log2_ratio_worker,
                tasks,
                chunksize=chunksize,
            ))

        return signed_bins

    def resize_norm_img(self, data, gen_ratio, padding=True):
        img = data['image']
        w, h = img.size

        if w == 0 or h == 0:
            return None

        if self.padding_rand and random.random() < 0.5:
            padding = not padding

        signed_bin = int(gen_ratio)
        imgW, imgH = self._shape_from_signed_log_bin(signed_bin)

        ratio_wh = float(w) / float(h)

        if not padding:
            resized_w = imgW
            resized_h = imgH
            resized_image = F.resize(
                img,
                [imgH, imgW],
                interpolation=self.interpolation,
            )
            valid_ratio = 1.0

        else:
            # Preserve original aspect ratio and pad to target shape.
            target_ratio = float(imgW) / float(imgH)

            if ratio_wh >= target_ratio:
                # Fit by width.
                resized_w = imgW
                resized_h = int(math.ceil(imgW / ratio_wh))
                resized_h = min(imgH, max(1, resized_h))
            else:
                # Fit by height.
                resized_h = imgH
                resized_w = int(math.ceil(imgH * ratio_wh))
                resized_w = min(imgW, max(1, resized_w))

            resized_image = F.resize(
                img,
                [resized_h, resized_w],
                interpolation=self.interpolation,
            )

            if imgW >= imgH:
                valid_ratio = min(1.0, float(resized_w / imgW))
            else:
                valid_ratio = min(1.0, float(resized_h / imgH))

        img = self.transforms(resized_image)

        if padding and (resized_w < imgW or resized_h < imgH):
            pad_left = 0
            pad_top = 0
            pad_right = imgW - resized_w
            pad_bottom = imgH - resized_h

            if self.padding_doub and random.random() < 0.5:
                # Sometimes move padding to the opposite side.
                pad_left, pad_right = pad_right, pad_left
                pad_top, pad_bottom = pad_bottom, pad_top

            img = F.pad(
                img,
                [pad_left, pad_top, pad_right, pad_bottom],
                fill=0.,
            )

        real_log_bin = int(round(math.log(ratio_wh, self.scale_factor)))
        real_log_bin = int(np.clip(
            real_log_bin,
            -self.max_log_ratio,
            self.max_log_ratio,
        ))

        data['image'] = img
        data['valid_ratio'] = valid_ratio
        data['real_ratio'] = real_log_bin
        data['target_log_ratio'] = signed_bin
        data['target_shape'] = [imgW, imgH]

        return data

    def __getitem__(self, properties):
        img_width = properties[0]
        img_height = properties[1]
        idx = properties[2]

        signed_bin = int(properties[3])

        def _sample_fallback_id():
            ratio_ids = np.where(self.wh_ratio == signed_bin)[0]
            if ratio_ids.size > 0:
                return random.choice(ratio_ids.tolist())
            return random.randrange(len(self))

        line_idx = int(self.data_idx_order_list[idx, 1])
        sample_info = self._get_sample_bytes_and_label(line_idx)

        if sample_info is None:
            fallback_id = _sample_fallback_id()
            return self.__getitem__(
                [img_width, img_height, fallback_id, signed_bin]
            )

        img, label, img_path = sample_info
        data = {
            'image': img,
            'label': label,
            'img_path': img_path,
        }

        outs = transform(data, self.ops[:-1])

        if outs is not None and hasattr(outs.get('image', None), 'size'):
            outs = self.resize_norm_img(outs, signed_bin, padding=self.padding)

            if outs is None:
                fallback_id = _sample_fallback_id()
                return self.__getitem__(
                    [img_width, img_height, fallback_id, signed_bin])

            outs = transform(outs, self.ops[-1:])

        elif outs is not None:
            outs = None

        if outs is None:
            fallback_id = _sample_fallback_id()
            return self.__getitem__(
                [img_width, img_height, fallback_id, signed_bin]
            )

        return outs

    def __len__(self):
        return self.data_idx_order_list.shape[0]