import math
import os
import random

import numpy as np
import torch
from torch.utils.data import Sampler

import openocr.tools.data as data_mod

data_mod.SAMPLER_MODULES["RatioSampler_HV"] = (
    "gch.openocr.tools.data.ratio_sampler_hv"
)


class RatioSampler_HV(Sampler):
    """RatioSampler variant that never mixes horizontal/vertical in one batch."""

    def __init__(
        self,
        data_source,
        scales,
        first_bs=512,
        fix_bs=True,
        divided_factor=[8, 16],
        is_training=True,
        max_ratio=10,
        max_bs=1024,
        seed=None,
    ):
        self.data_source = data_source
        self.ds_width = data_source.ds_width
        self.seed = data_source.seed
        if self.ds_width:
            self.wh_ratio = data_source.wh_ratio
            self.wh_ratio_sort = data_source.wh_ratio_sort
            # 1: horizontal/square, 0: vertical
            self.wh_orientation = getattr(
                data_source,
                "wh_orientation",
                np.ones_like(self.wh_ratio, dtype=np.int32),
            )
        self.n_data_samples = len(self.data_source)
        self.max_ratio = max_ratio
        self.max_bs = max_bs

        if isinstance(scales[0], list):
            width_dims = [i[0] for i in scales]
            height_dims = [i[1] for i in scales]
        elif isinstance(scales[0], int):
            width_dims = scales
            height_dims = scales
        else:
            raise ValueError("Unsupported scales format for HVAwareRatioSampler")

        base_im_w = width_dims[0]
        base_im_h = height_dims[0]
        base_batch_size = first_bs
        base_elements = base_im_w * base_im_h * base_batch_size
        self.base_elements = base_elements
        self.base_batch_size = base_batch_size
        self.base_im_h = base_im_h
        self.base_im_w = base_im_w
        self.base_im_w_v = int(getattr(data_source, "base_w", self.base_im_h))

        num_replicas = torch.cuda.device_count() if torch.cuda.is_available() else 1
        rank = int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ else 0
        num_samples_per_replica = int(math.ceil(self.n_data_samples * 1.0 / num_replicas))

        img_indices = [idx for idx in range(self.n_data_samples)]
        self.shuffle = False
        if is_training:
            width_dims = [int((w // divided_factor[0]) * divided_factor[0]) for w in width_dims]
            height_dims = [int((h // divided_factor[1]) * divided_factor[1]) for h in height_dims]

            img_batch_pairs = []
            for (h, w) in zip(height_dims, width_dims):
                if fix_bs:
                    batch_size = base_batch_size
                else:
                    batch_size = int(max(1, (base_elements / (h * w))))
                img_batch_pairs.append((w, h, batch_size))
            self.img_batch_pairs = img_batch_pairs
            self.shuffle = True
            np.random.seed(seed)
            random.seed(seed)
        else:
            self.img_batch_pairs = [(base_im_w, base_im_h, base_batch_size)]

        self.img_indices = img_indices
        self.n_samples_per_replica = num_samples_per_replica
        self.epoch = 0
        self.rank = rank
        self.num_replicas = num_replicas
        self.current = 0
        self.is_training = is_training
        if is_training:
            indices_rank_i = self.img_indices[self.rank : len(self.img_indices) : self.num_replicas]
        else:
            indices_rank_i = self.img_indices
        self.indices_rank_i_ori = np.array(self.wh_ratio_sort[indices_rank_i])
        self.indices_rank_i_ratio = self.wh_ratio[self.indices_rank_i_ori]
        self.indices_rank_i_orientation = self.wh_orientation[self.indices_rank_i_ori]

        self.group_keys = sorted(
            set(
                (
                    int(self.indices_rank_i_orientation[i]),
                    int(self.indices_rank_i_ratio[i]),
                )
                for i in range(len(self.indices_rank_i_ori))
            )
        )
        self.batch_list = self.create_batch()
        self.length = len(self.batch_list)
        self.batchs_in_one_epoch_id = [i for i in range(self.length)]

    def _batch_size_for_group(self, ratio, orientation):
        if ratio < 5:
            return self.base_batch_size
        if orientation == 1:
            ref = self.base_im_h
        else:
            ref = self.base_im_w_v
        denom = ref * ratio * ref
        return min(self.max_bs, int(max(1, (self.base_elements / denom))))

    def _shape_for_group(self, ratio, orientation):
        if orientation == 1:
            return int(ratio * self.base_im_h), int(self.base_im_h)
        return int(self.base_im_w_v), int(ratio * self.base_im_w_v)

    def create_batch(self):
        batch_list = []
        for orientation, ratio in self.group_keys:
            ratio_ids = np.where(
                (self.indices_rank_i_ratio == ratio)
                & (self.indices_rank_i_orientation == orientation)
            )[0]
            ratio_ids = self.indices_rank_i_ori[ratio_ids]
            if ratio_ids.shape[0] == 0:
                continue
            if self.shuffle:
                random.shuffle(ratio_ids)

            num_ratio = ratio_ids.shape[0]
            batch_size_ratio = self._batch_size_for_group(ratio, orientation)
            shape_w, shape_h = self._shape_for_group(ratio, orientation)

            if num_ratio > batch_size_ratio:
                batch_num_ratio = num_ratio // batch_size_ratio
                print(
                    self.rank,
                    "H" if orientation == 1 else "V",
                    num_ratio,
                    shape_w,
                    shape_h,
                    batch_num_ratio,
                    batch_size_ratio,
                )
                ratio_ids_full = ratio_ids[: batch_num_ratio * batch_size_ratio].reshape(
                    batch_num_ratio, batch_size_ratio, 1
                )
                w = np.full_like(ratio_ids_full, shape_w)
                h = np.full_like(ratio_ids_full, shape_h)
                ra_wh = np.full_like(ratio_ids_full, ratio)
                ratio_ids_full = np.concatenate([w, h, ratio_ids_full, ra_wh], axis=-1)
                batch_ratio = ratio_ids_full.tolist()

                if batch_num_ratio * batch_size_ratio < num_ratio:
                    drop = ratio_ids[batch_num_ratio * batch_size_ratio :]
                    if self.is_training:
                        drop_full = ratio_ids[
                            : batch_size_ratio
                            - (num_ratio - batch_num_ratio * batch_size_ratio)
                        ]
                        drop = np.append(drop_full, drop)
                    drop = drop.reshape(-1, 1)
                    w = np.full_like(drop, shape_w)
                    h = np.full_like(drop, shape_h)
                    ra_wh = np.full_like(drop, ratio)
                    drop = np.concatenate([w, h, drop, ra_wh], axis=-1)
                    batch_ratio.append(drop.tolist())
                    batch_list += batch_ratio
            else:
                print(
                    self.rank,
                    "H" if orientation == 1 else "V",
                    num_ratio,
                    shape_w,
                    shape_h,
                    batch_size_ratio,
                )
                ratio_ids = ratio_ids.reshape(-1, 1)
                w = np.full_like(ratio_ids, shape_w)
                h = np.full_like(ratio_ids, shape_h)
                ra_wh = np.full_like(ratio_ids, ratio)
                ratio_ids = np.concatenate([w, h, ratio_ids, ra_wh], axis=-1)
                batch_list.append(ratio_ids.tolist())
        return batch_list

    def __iter__(self):
        if self.shuffle or self.is_training:
            random.seed(self.epoch)
            self.epoch += 1
            self.batch_list = self.create_batch()
            random.shuffle(self.batchs_in_one_epoch_id)
        for batch_tuple_id in self.batchs_in_one_epoch_id:
            yield self.batch_list[batch_tuple_id]

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self):
        return self.length
