import json
import os
import random
import traceback

import numpy as np
from torch.utils.data import Dataset

from openocr.openrec.preprocess import transform

import openocr.tools.data as data_mod

data_mod.DATASET_MODULES['GCHSimpleDataset'] = 'gch.openocr.tools.data.simple_dataset'


class GCHSimpleDataset(Dataset):
    """SimpleDataSet variant that supports paired data_dir_list/label_file_list."""

    def __init__(self, config, mode, logger, seed=None, epoch=0, task='rec'):
        super().__init__()
        self.logger = logger
        self.mode = mode.lower()

        global_config = config['Global']
        dataset_config = config[mode]['dataset']
        loader_config = config[mode]['loader']

        self.delimiter = dataset_config.get('delimiter', '\t')
        label_file_list = dataset_config.pop('label_file_list')
        if isinstance(label_file_list, str):
            label_file_list = [label_file_list]
        data_source_num = len(label_file_list)

        ratio_list = dataset_config.get('ratio_list', 1.0)
        if isinstance(ratio_list, (float, int)):
            ratio_list = [float(ratio_list)] * int(data_source_num)
        assert len(ratio_list) == data_source_num, (
            'The length of ratio_list should be the same as the file_list.')

        self.data_dir = dataset_config.get('data_dir')
        data_dir_list = dataset_config.get('data_dir_list')
        if data_dir_list is None:
            assert self.data_dir is not None, (
                'Either data_dir or data_dir_list must be provided.')
            data_dir_list = [self.data_dir] * data_source_num
        if isinstance(data_dir_list, str):
            data_dir_list = [data_dir_list]
        assert len(data_dir_list) == data_source_num, (
            'The length of data_dir_list should be the same as the file_list.')

        self.do_shuffle = loader_config['shuffle']
        self.seed = seed
        logger.info(f'Initialize indexs of datasets: {label_file_list}')
        self.data_lines = self.get_image_info_list(label_file_list, data_dir_list,
                                                   ratio_list)
        self.data_idx_order_list = list(range(len(self.data_lines)))
        if self.mode == 'train' and self.do_shuffle:
            self.shuffle_data_random()

        self.set_epoch_as_seed(self.seed, dataset_config)
        if task == 'rec':
            from openocr.openrec.preprocess import create_operators
        elif task == 'det':
            from openocr.opendet.preprocess import create_operators
        else:
            from openocr.openrec.preprocess import create_operators
        self.ops = create_operators(dataset_config['transforms'], global_config)
        self.ext_op_transform_idx = dataset_config.get('ext_op_transform_idx', 2)
        self.need_reset = True in [x < 1 for x in ratio_list]

    def set_epoch_as_seed(self, seed, dataset_config):
        if self.mode == 'train':
            try:
                border_map_id = [
                    index for index, dictionary in enumerate(
                        dataset_config['transforms']) if 'MakeBorderMap' in dictionary
                ][0]
                shrink_map_id = [
                    index for index, dictionary in enumerate(
                        dataset_config['transforms']) if 'MakeShrinkMap' in dictionary
                ][0]
                dataset_config['transforms'][border_map_id]['MakeBorderMap'][
                    'epoch'] = seed if seed is not None else 0
                dataset_config['transforms'][shrink_map_id]['MakeShrinkMap'][
                    'epoch'] = seed if seed is not None else 0
            except Exception:
                return

    def get_image_info_list(self, file_list, data_dir_list, ratio_list):
        if isinstance(file_list, str):
            file_list = [file_list]
        data_lines = []
        for idx, (file, data_dir) in enumerate(zip(file_list, data_dir_list)):
            with open(file, 'rb') as f:
                lines = f.readlines()
            if self.mode == 'train' or ratio_list[idx] < 1.0:
                random.seed(self.seed)
                sample_k = round(len(lines) * ratio_list[idx])
                lines = random.sample(lines, min(sample_k, len(lines)))
            data_lines.extend([(line, data_dir) for line in lines])
        return data_lines

    def shuffle_data_random(self):
        random.seed(self.seed)
        random.shuffle(self.data_lines)
        return

    def _try_parse_filename_list(self, file_name):
        # multiple images -> one gt label
        if len(file_name) > 0 and file_name[0] == '[':
            try:
                info = json.loads(file_name)
                file_name = random.choice(info)
            except Exception:
                pass
        return file_name

    def get_ext_data(self):
        ext_data_num = 0
        for op in self.ops:
            if hasattr(op, 'ext_data_num'):
                ext_data_num = getattr(op, 'ext_data_num')
                break
        load_data_ops = self.ops[:self.ext_op_transform_idx]
        ext_data = []

        while len(ext_data) < ext_data_num:
            file_idx = self.data_idx_order_list[np.random.randint(self.__len__())]
            data_line, data_dir = self.data_lines[file_idx]
            data_line = data_line.decode('utf-8')
            substr = data_line.strip('\n').split(self.delimiter)
            file_name = substr[0]
            file_name = self._try_parse_filename_list(file_name)
            label = substr[1]
            img_path = os.path.join(data_dir, file_name)
            data = {'img_path': img_path, 'label': label}
            if not os.path.exists(img_path):
                continue
            with open(data['img_path'], 'rb') as f:
                data['image'] = f.read()
            data = transform(data, load_data_ops)

            if data is None:
                continue
            if 'polys' in data.keys():
                if data['polys'].shape[1] != 4:
                    continue
            ext_data.append(data)
        return ext_data

    def __getitem__(self, idx):
        file_idx = self.data_idx_order_list[idx]
        data_line, data_dir = self.data_lines[file_idx]
        try:
            data_line = data_line.decode('utf-8')
            substr = data_line.strip('\n').split(self.delimiter)
            file_name = substr[0]
            file_name = self._try_parse_filename_list(file_name)
            label = substr[1]
            img_path = os.path.join(data_dir, file_name)
            data = {'img_path': img_path, 'label': label}

            if not os.path.exists(img_path):
                raise Exception(f'{img_path} does not exist!')
            with open(data['img_path'], 'rb') as f:
                data['image'] = f.read()
            data['ext_data'] = self.get_ext_data()
            outs = transform(data, self.ops)
        except Exception:
            self.logger.error(
                'When parsing line {}, error happened with msg: {}'.format(
                    data_line, traceback.format_exc()))
            outs = None
        if outs is None:
            # During evaluation, fix idx for deterministic fallback.
            rnd_idx = np.random.randint(self.__len__(
            )) if self.mode == 'train' else (idx + 1) % self.__len__()
            return self.__getitem__(rnd_idx)
        return outs

    def __len__(self):
        return len(self.data_idx_order_list)
