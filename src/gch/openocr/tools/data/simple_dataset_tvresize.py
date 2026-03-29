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
data_mod.DATASET_MODULES['SimpleDatasetTVResize'] = (
    'gch.openocr.tools.data.simple_dataset_tvresize')


def _wh_ratio_worker(args):
    line, delimiter, data_dir = args
    try:
        text = line.decode('utf-8')
        substr = text.strip('\n').split(delimiter)
        file_name = substr[0]
        if len(file_name) > 0 and file_name[0] == '[':
            try:
                info = json.loads(file_name)
                file_name = random.choice(info)
            except Exception:
                pass
        img_path = os.path.join(data_dir, file_name)
        if not os.path.exists(img_path):
            return 1.0
        with Image.open(img_path) as img:
            w, h = img.size
        if h == 0:
            return 1.0
        return float(w) / float(h)
    except Exception:
        return 1.0


class SimpleDatasetTVResize(Dataset):
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
        self.data_dir = dataset_config['data_dir']

        logger.info(f'Initialize indexs of datasets: {label_file_list}')
        self.data_lines = self._get_image_info_list(label_file_list, ratio_list,
                                                    seed, epoch)
        n = len(self.data_lines)
        self.data_idx_order_list = np.zeros((n, 2), dtype=np.float64)
        self.data_idx_order_list[:, 1] = np.arange(n, dtype=np.float64)

        if self.mode == 'train' and self.do_shuffle:
            self._shuffle_data_random(seed, epoch)

        self._set_epoch_as_seed(seed, dataset_config)

        self.ops = create_operators(dataset_config['transforms'],
                                    global_config)

        wh_ratio = np.around(np.array(self.get_wh_ratio()))
        self.wh_ratio = np.clip(wh_ratio, a_min=min_ratio, a_max=max_ratio)
        for i in range(max_ratio + 1):
            logger.info((1 * (self.wh_ratio == i)).sum())
        self.wh_ratio_sort = np.argsort(self.wh_ratio)

        self.need_reset = True in [x < 1 for x in ratio_list]
        self.error = 0
        self.base_shape = dataset_config.get(
            'base_shape', [[64, 64], [96, 48], [112, 40], [128, 32]])
        self.base_h = dataset_config.get('base_h', 32)
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

    def _get_image_info_list(self, file_list, ratio_list, seed, epoch):
        rnd = seed if seed is not None else epoch
        data_lines = []
        for idx, file in enumerate(file_list):
            with open(file, 'rb') as f:
                lines = f.readlines()
            if self.mode == 'train' or ratio_list[idx] < 1.0:
                random.seed(rnd)
                k = max(0, round(len(lines) * ratio_list[idx]))
                if k == 0:
                    continue
                lines = random.sample(lines, min(k, len(lines)))
            data_lines.extend(lines)
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
        line = self.data_lines[line_idx]
        try:
            text = line.decode('utf-8')
            substr = text.strip('\n').split(self.delimiter)
            file_name = substr[0]
            file_name = self._try_parse_filename_list(file_name)
            label = substr[1]
            img_path = os.path.join(self.data_dir, file_name)
            if not os.path.exists(img_path):
                return None
            with open(img_path, 'rb') as f:
                imgbuf = f.read()
            if not imgbuf:
                return None
            return imgbuf, label
        except Exception:
            return None

    def _get_wh_ratio(self):
        wh_ratio = []
        for idx in range(self.data_idx_order_list.shape[0]):
            line_idx = int(self.data_idx_order_list[idx, 1])
            info = self._get_sample_bytes_and_label(line_idx)
            if info is None:
                wh_ratio.append(1.0)
            else:
                imgbuf, _ = info
                w, h = Image.open(io.BytesIO(imgbuf)).size
                wh_ratio.append(float(w) / float(h))
        return wh_ratio

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
            (self.data_lines[int(self.data_idx_order_list[i, 1])],
             self.delimiter, self.data_dir) for i in range(total)
        )

        # Keep output order aligned with data_idx_order_list.
        with mp.Pool(processes=num_workers) as pool:
            wh_ratio = list(
                pool.imap(_wh_ratio_worker, tasks, chunksize=chunksize))
        return wh_ratio

    def resize_norm_img(self, data, gen_ratio, padding=True):
        img = data['image']
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
                resized_w = int(
                    math.ceil(imgH * ratio * (random.random() + 0.5)))
                resized_w = min(imgW, resized_w)
        resized_image = F.resize(
            img, [imgH, int(resized_w)], interpolation=self.interpolation)
        img = self.transforms(resized_image)
        if resized_w < imgW and padding:
            if self.padding_doub and random.random() < 0.5:
                img = F.pad(img, [0, 0, imgW - resized_w, 0], fill=0.)
            else:
                img = F.pad(img, [imgW - resized_w, 0, 0, 0], fill=0.)
        valid_ratio = min(1.0, float(resized_w / imgW))
        data['image'] = img
        data['valid_ratio'] = valid_ratio
        r = float(w) / float(h)
        data['real_ratio'] = max(1, round(r))
        return data

    def __getitem__(self, properties):
        img_width = properties[0]
        img_height = properties[1]
        idx = properties[2]
        ratio = properties[3]
        line_idx = int(self.data_idx_order_list[idx, 1])
        sample_info = self._get_sample_bytes_and_label(line_idx)
        if sample_info is None:
            ratio_ids = np.where(self.wh_ratio == ratio)[0].tolist()
            if not ratio_ids:
                ratio_ids = list(range(len(self)))
            ids = random.sample(ratio_ids, 1)
            return self.__getitem__([img_width, img_height, ids[0], ratio])
        img, label = sample_info
        data = {'image': img, 'label': label}
        outs = transform(data, self.ops[:-1])
        if outs is not None:
            outs = self.resize_norm_img(outs, ratio, padding=self.padding)
            if outs is None:
                ratio_ids = np.where(self.wh_ratio == ratio)[0].tolist()
                if not ratio_ids:
                    ratio_ids = list(range(len(self)))
                ids = random.sample(ratio_ids, 1)
                return self.__getitem__(
                    [img_width, img_height, ids[0], ratio])
            outs = transform(outs, self.ops[-1:])
        if outs is None:
            ratio_ids = np.where(self.wh_ratio == ratio)[0].tolist()
            if not ratio_ids:
                ratio_ids = list(range(len(self)))
            ids = random.sample(ratio_ids, 1)
            return self.__getitem__([img_width, img_height, ids[0], ratio])
        return outs

    def __len__(self):
        return self.data_idx_order_list.shape[0]
