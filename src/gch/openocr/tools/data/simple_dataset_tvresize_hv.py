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
data_mod.DATASET_MODULES['SimpleDatasetTVResize_HV'] = (
    'gch.openocr.tools.data.simple_dataset_tvresize_hv')


def _wh_ratio_worker(args):
    line, delimiter, data_dir = args
    try:
        text = line.decode('utf-8')
        substr = text.rstrip('\r\n').split(delimiter)
        if len(substr) < 2:
            return 1.0, 1
        file_name = substr[0]
        if len(file_name) > 0 and file_name[0] == '[':
            try:
                info = json.loads(file_name)
                file_name = random.choice(info)
            except Exception:
                pass
        img_path = os.path.join(data_dir, file_name)
        if not os.path.exists(img_path):
            return 1.0, 1
        with Image.open(img_path) as img:
            w, h = img.size
        if h == 0 or w == 0:
            return 1.0, 1
        # Keep ratio buckets >= 1 for RatioSampler compatibility while
        # preserving vertical difficulty via h/w for portrait samples.
        ratio_wh = float(w) / float(h)
        ratio = max(ratio_wh, 1.0 / ratio_wh)
        orientation = 1 if ratio_wh >= 1.0 else 0
        return ratio, orientation
    except Exception:
        return 1.0, 1


class SimpleDatasetTVResize_HV(Dataset):
    """Simple label_file_list + data_dir layout with RatioDataSetTVResize behavior.

    Use with ``RatioSampler`` and the same dataset config keys as
    ``RatioDataSetTVResize`` except ``data_dir_list`` is replaced by
    ``label_file_list`` and ``data_dir`` (same as ``SimpleDataSet``).
    """

    def __init__(self, config, mode, logger, seed=None, epoch=1, task='rec'):
        super().__init__()
        self.ds_width = config[mode]['dataset'].get('ds_width', True)
        global_config = config['Global']
        dataset_config = config[mode]['dataset']
        loader_config = config[mode]['loader']
        max_ratio = loader_config.get('max_ratio', 10)
        min_ratio = loader_config.get('min_ratio', 1)

        self.logger = logger
        self.mode = mode.lower()
        self.delimiter = dataset_config.get('delimiter', '\t')
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
        self.data_lines = self._get_image_info_list(label_file_list,
                                                    data_dir_list, ratio_list,
                                                    seed, epoch)
        n = len(self.data_lines)
        self.data_idx_order_list = np.zeros((n, 2), dtype=np.float64)
        self.data_idx_order_list[:, 1] = np.arange(n, dtype=np.float64)

        if self.mode == 'train' and self.do_shuffle:
            self._shuffle_data_random(seed, epoch)

        self._set_epoch_as_seed(seed, dataset_config)

        self.ops = create_operators(dataset_config['transforms'],
                                    global_config)

        wh_ratio_raw, wh_orientation = self.get_wh_ratio()
        wh_ratio = np.around(np.array(wh_ratio_raw))
        self.wh_ratio = np.clip(wh_ratio, a_min=min_ratio, a_max=max_ratio)
        self.wh_orientation = np.array(wh_orientation, dtype=np.int32)
        for i in range(max_ratio + 1):
            logger.info((1 * (self.wh_ratio == i)).sum())
        self.wh_ratio_sort = np.argsort(self.wh_ratio)

        self.need_reset = True in [x < 1 for x in ratio_list]
        self.error = 0
        self.base_shape = dataset_config.get(
            'base_shape', [[64, 64], [96, 48], [112, 40], [128, 32]])
        self.base_h = dataset_config.get('base_h', 32)
        self.base_shape_v = dataset_config.get(
            'base_shape_v', [[64, 64], [48, 96], [40, 112], [32, 128]])
        self.base_w = dataset_config.get('base_w', 32)
        self.interpolation = T.InterpolationMode.BICUBIC
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(0.5, 0.5),
        ])

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

    def _get_wh_ratio(self):
        wh_ratio = []
        wh_orientation = []
        for idx in range(self.data_idx_order_list.shape[0]):
            line_idx = int(self.data_idx_order_list[idx, 1])
            info = self._get_sample_bytes_and_label(line_idx)
            if info is None:
                wh_ratio.append(1.0)
                wh_orientation.append(1)
            else:
                imgbuf, _, _ = info
                w, h = Image.open(io.BytesIO(imgbuf)).size
                if h == 0 or w == 0:
                    wh_ratio.append(1.0)
                    wh_orientation.append(1)
                else:
                    ratio_wh = float(w) / float(h)
                    wh_ratio.append(max(ratio_wh, 1.0 / ratio_wh))
                    wh_orientation.append(1 if ratio_wh >= 1.0 else 0)
        return wh_ratio, wh_orientation

    def get_wh_ratio(self):
        return self._get_wh_ratio_multiprocessing(num_workers=30)

    def _get_wh_ratio_multiprocessing(self, num_workers=None, chunksize=1024):
        import multiprocessing as mp

        total = self.data_idx_order_list.shape[0]
        if total == 0:
            return []

        if num_workers is None:
            num_workers = max(1, (mp.cpu_count() or 1) - 1)
        num_workers = max(1, int(num_workers/2)) 
        chunksize = max(1, int(chunksize))

        tasks = (
            (self.data_lines[int(self.data_idx_order_list[i, 1])][0],
             self.delimiter,
             self.data_lines[int(self.data_idx_order_list[i, 1])][1])
            for i in range(total)
        )

        # Keep output order aligned with data_idx_order_list.
        with mp.Pool(processes=num_workers) as pool:
            wh_info = list(pool.imap(_wh_ratio_worker, tasks, chunksize=chunksize))
        if not wh_info:
            return [], []
        wh_ratio, wh_orientation = zip(*wh_info)
        return list(wh_ratio), list(wh_orientation)

    def resize_norm_img(self, data, gen_ratio, padding=True):
        img = data['image']
        w, h = img.size
        if w == 0 or h == 0:
            return None
        if self.padding_rand and random.random() < 0.5:
            padding = not padding
        # Orientation-aware branch:
        # - landscape/square: use horizontal base_shape/base_h and ratio w/h
        # - portrait: use vertical base_shape_v/base_w and ratio h/w
        ratio_wh = float(w) / float(h)
        is_horizontal = ratio_wh >= 1.0
        resized_w = imgW = 0
        resized_h = imgH = 0
        if is_horizontal:
            if gen_ratio <= 4:
                imgW, imgH = self.base_shape[gen_ratio - 1]
            else:
                imgW = self.base_h * gen_ratio
                imgH = self.base_h
        else:
            if gen_ratio <= 4:
                imgW, imgH = self.base_shape_v[gen_ratio - 1]
            else:
                imgW = self.base_w
                imgH = self.base_w * gen_ratio
        imgW, imgH = int(imgW), int(imgH)
        if is_horizontal:
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
                    resized_w = int(
                        math.ceil(imgH * ratio * (random.random() + 0.5)))
                    resized_w = min(imgW, resized_w)
            resized_image = F.resize(
                img, [imgH, int(resized_w)], interpolation=self.interpolation)
        else:
            use_ratio = imgH // imgW
            if use_ratio >= (h // w) + 2:
                self.error += 1
                return None
            if not padding:
                resized_h = imgH
            else:
                ratio = h / float(w)
                if math.ceil(imgW * ratio) > imgH:
                    resized_h = imgH
                else:
                    resized_h = int(
                        math.ceil(imgW * ratio * (random.random() + 0.5)))
                    resized_h = min(imgH, resized_h)
            resized_image = F.resize(
                img, [int(resized_h), imgW], interpolation=self.interpolation)
        img = self.transforms(resized_image)
        if is_horizontal and resized_w < imgW and padding:
            if self.padding_doub and random.random() < 0.5:
                img = F.pad(img, [0, 0, imgW - resized_w, 0], fill=0.)
            else:
                img = F.pad(img, [imgW - resized_w, 0, 0, 0], fill=0.)
            valid_ratio = min(1.0, float(resized_w / imgW))
        elif (not is_horizontal) and resized_h < imgH and padding:
            if self.padding_doub and random.random() < 0.5:
                img = F.pad(img, [0, 0, 0, imgH - resized_h], fill=0.)
            else:
                img = F.pad(img, [0, imgH - resized_h, 0, 0], fill=0.)
            valid_ratio = min(1.0, float(resized_h / imgH))
        else:
            valid_ratio = 1.0
        data['image'] = img
        data['valid_ratio'] = valid_ratio
        data['real_ratio'] = max(1, round(max(ratio_wh, 1.0 / ratio_wh)))
        return data

    def __getitem__(self, properties):
        img_width = properties[0]
        img_height = properties[1]
        idx = properties[2]
        ratio = properties[3]
        target_orientation = 1 if img_width >= img_height else 0

        def _sample_fallback_id():
            ratio_ids = np.where(self.wh_ratio == ratio)[0]
            if ratio_ids.size > 0:
                ratio_orient = self.wh_orientation[ratio_ids]
                oriented_ratio_ids = ratio_ids[ratio_orient == target_orientation]
                if oriented_ratio_ids.size > 0:
                    return random.choice(oriented_ratio_ids.tolist())
                return random.choice(ratio_ids.tolist())
            return random.randrange(len(self))

        line_idx = int(self.data_idx_order_list[idx, 1])
        sample_info = self._get_sample_bytes_and_label(line_idx)
        if sample_info is None:
            fallback_id = _sample_fallback_id()
            return self.__getitem__([img_width, img_height, fallback_id, ratio])
        img, label, img_path = sample_info
        data = {'image': img, 'label': label, "img_path": img_path}
        outs = transform(data, self.ops[:-1])
        if outs is not None and hasattr(outs.get('image', None), 'size'):
            outs = self.resize_norm_img(outs, ratio, padding=self.padding)
            if outs is None:
                fallback_id = _sample_fallback_id()
                return self.__getitem__(
                    [img_width, img_height, fallback_id, ratio])
            outs = transform(outs, self.ops[-1:])
        elif outs is not None:
            outs = None
        if outs is None:
            fallback_id = _sample_fallback_id()
            return self.__getitem__([img_width, img_height, fallback_id, ratio])
        return outs

    def __len__(self):
        return self.data_idx_order_list.shape[0]
