import monai
import os
import numpy as np
from glob import glob

from monai.data import Dataset, DataLoader, DatasetSummary, CacheDataset
from monai.transforms import (
Compose,
LoadImaged,
RandAffined,
Resized,
ScaleIntensityRanged,
RandAdjustContrastd,
RandCropByPosNegLabeld,
MapTransform

)
from monai.utils import set_determinism, first
from natsort import natsorted
import copy
import random
import torch
import time
import numpy as np
from monai import transforms

class get_mask(MapTransform):
    def __init__(self, keys, new_key) -> None:
        super().__init__(keys)
        self.new_key = new_key

    def __call__(self, data):
        # start_time = time.time()
        for i, key in enumerate(self.keys):
            data[self.new_key[i]] = np.copy(data[key])
            data[self.new_key[i]] = (data[self.new_key[i]] != 0).astype(np.uint8)
        # print(f"Copyd duration: {time.time() - start_time}s")

        return data


class Copypathd(MapTransform):
    def __init__(self, keys, new_key) -> None:
        """
        Initialize the transform with keys for the original file path and the new key for the copied path.

        Args:
            keys (list or str): the keys in the data dictionary where the file paths are stored.
            new_key (list or str): the new key(s) where the copied file paths will be stored.
        """
        super().__init__(keys)
        if isinstance(new_key, list):
            self.new_key = new_key
        else:
            self.new_key = [new_key] * len(keys)

    def __call__(self, data):
        """
        Copy the file paths from the original keys to the new keys.

        Args:
            data (dict): input data dictionary that includes file paths to copy.

        Returns:
            dict: data dictionary with the copied file paths.
        """
        for i, key in enumerate(self.keys):
            if isinstance(data[key], str):
                # If the data associated with the key is a file path (string), copy it to the new key
                data[self.new_key[i]] = data[key]
            else:
                raise TypeError(f"Expected a file path string for key '{key}', but got {type(data[key])}")

        return data


class Copyd(MapTransform):
    def __init__(self, keys, new_key) -> None:
        super().__init__(keys)
        self.new_key = new_key

    def __call__(self, data):
        # start_time = time.time()
        for i, key in enumerate(self.keys):
            data[self.new_key[i]] = np.copy(data[key])
        # print(f"Copyd duration: {time.time() - start_time}s")

        return data


class OverlayCrop(MapTransform):
    def __init__(self,keys, roi_center, roi_length,ccf, crf)->None:
        MapTransform.__init__(self, keys)
        self.keys = keys
        # self.new_key = new_key
        self.roi_center = roi_center
        self.roi_length = roi_length
        self.ccf = ccf
        self.crf = crf
    def __call__(self, data):
        # 创建一个新的随机数生成器实例，用于非确定性随机数
        non_deterministic_random = random.Random()
        c1 = [self.roi_center[0]+non_deterministic_random.randint(-self.ccf, self.ccf),
              self.roi_center[1]+non_deterministic_random.randint(-self.ccf, self.ccf),
              self.roi_center[2]+non_deterministic_random.randint(-self.ccf, self.ccf)]  # 示例中心点1
        c2 = [self.roi_center[0]+non_deterministic_random.randint(-self.ccf, self.ccf),
              self.roi_center[1]+non_deterministic_random.randint(-self.ccf, self.ccf),
              self.roi_center[2]+non_deterministic_random.randint(-self.ccf, self.ccf)]  # 示例中心点1
        lx1, ly1, lz1 = [self.roi_length[0]+non_deterministic_random.randint(-self.crf, self.crf),
                         self.roi_length[1]+non_deterministic_random.randint(-self.crf, self.crf),
                         self.roi_length[2]+non_deterministic_random.randint(-self.crf, self.crf)]  # 示例边长1
        lx2, ly2, lz2 = [self.roi_length[0]+non_deterministic_random.randint(-self.crf, self.crf),
                         self.roi_length[1]+non_deterministic_random.randint(-self.crf, self.crf),
                         self.roi_length[2]+non_deterministic_random.randint(-self.crf, self.crf)]  # 示例边长2

        # print(f"mask shape is:{mask.shape}")
        mask1 = self.create_mask(c1, lx1, ly1, lz1).astype(bool)
        mask2 = self.create_mask(c2, lx2, ly2, lz2).astype(bool)

        # print(f"img1 shape is:{data['img1'].shape}")
        # print(f"seg1 shape is:{data['seg1'].shape}")

        # 4. 计算交集和差集
        mask_u = mask1 | mask2  # 并集
        mask1_d = mask1 & ~mask2  # mask1上与mask2的差集
        mask2_d = mask2 & ~mask1  # mask2上与mask1的差集
        data['p1'] = data['img1'] * mask1
        data['p2'] = data['img2'] * mask2
        # data['seg_p1'] = data['seg1'] * mask1
        # data['seg_p2'] = data['seg2'] * mask2
        #
        data['pu'] = data['img1'] * mask_u
        # data['seg_pu'] = data['seg1'] * mask_u
        #
        # data['pd1'] = data['img1'] * mask1_d
        # data['seg_pd1'] = mask1_d*mask
        # data['seg_pd1_inv'] = 1 - mask1_d * mask
        # data['pd2'] = data['img2'] * mask2_d
        # data['seg_pd2'] = mask2_d*mask
        # data['seg_pd2_inv'] = 1 - mask2_d * mask

        return data

    def create_mask(self, center, lx, ly, lz):
        x, y, z = center
        mask = np.zeros((128, 128, 128), dtype=np.float32)
        mask[x - lx // 2: x + lx // 2 + 1, y - ly // 2: y + ly // 2 + 1, z - lz // 2: z + lz // 2 + 1] = 1
        return mask



class CenterCrop_patch(MapTransform):
    # The function is to crop the patch without saving the boundary
    def __init__(self,keys, new_key, roi_center, roi_length,ccf, crf)->None:
        MapTransform.__init__(self, keys)
        self.keys = keys
        self.new_key = new_key
        self.roi_center = roi_center
        self.roi_length = roi_length
        self.ccf = ccf
        self.crf = crf
    def __call__(self, data):

        # TODO
        # start_time = time.time()
        for i in range(len(self.keys)):
            x, y, z = self.roi_center
            lx, ly, lz = [self.roi_length[0],
                             self.roi_length[1],
                             self.roi_length[2]]
            # print(data[self.keys[i]].shape)
            data[self.new_key[i]] = data[self.keys[i]][:,x - lx // 2: x + lx // 2, y - ly // 2: y + ly // 2, z - lz // 2: z + lz // 2]

        # print(f"centercrop duration: {time.time() - start_time}s")
        return data

# latest version
class CenterCrop(MapTransform):
    def __init__(self,keys, new_key, roi_center, roi_length, ccf, crf, mask_shape)->None:
        MapTransform.__init__(self, keys)
        self.keys = keys
        self.new_key = new_key
        self.roi_center = roi_center
        self.roi_length = roi_length
        self.ccf = ccf
        self.crf = crf
        self.mask_shape = mask_shape
    def __call__(self, data):
        non_deterministic_random = random.Random()
        for i in range(len(self.keys)):
            # non_deterministic_random = random.Random()
            # c1 = [self.roi_center[0] + non_deterministic_random.randint(-self.ccf, self.ccf),
            #       self.roi_center[1] + non_deterministic_random.randint(-self.ccf, self.ccf),
            #       self.roi_center[2] + non_deterministic_random.randint(-self.ccf, self.ccf)]  # 示例中心点1
            c1 = self.roi_center
            # lx1, ly1, lz1 = [self.roi_length[0] + non_deterministic_random.randint(-self.crf, self.crf),
            #                  self.roi_length[1] + non_deterministic_random.randint(-self.crf, self.crf),
            #                  self.roi_length[2] + non_deterministic_random.randint(-self.crf, self.crf)]  # 示例边长1

            lx1, ly1, lz1 = [self.roi_length[0],
                             self.roi_length[1],
                             self.roi_length[2]]  # 示例边长1
            # mask = np.ones_like(data['seg1'])
            # print(f"mask shape is:{mask.shape}")
            # print(f"the lx1 is {lx1}")
            mask1 = self.create_mask(c1, lx1, ly1, lz1, self.mask_shape).astype(bool)
            data[self.new_key[i]] = data[self.keys[i]] * mask1

        return data

    def create_mask(self, center, lx, ly, lz, mask_shape):
        x, y, z = center
        mask = np.zeros((mask_shape, mask_shape, mask_shape), dtype=np.float32)
        mask[x - lx // 2: x + lx // 2 + 1, y - ly // 2: y + ly // 2 + 1, z - lz // 2: z + lz // 2 + 1] = 1
        return mask



class CenterCrop_multi(MapTransform):
    def __init__(self,keys, new_key, roi_center, roi_length,ccf, crf)->None:
        MapTransform.__init__(self, keys)
        self.keys = keys
        self.new_key = new_key
        self.roi_center = roi_center
        self.roi_length = roi_length
        self.ccf = ccf
        self.crf = crf
    def __call__(self, data):
        for i in range(len(self.keys)):
            non_deterministic_random = random.Random()
            c1 = [self.roi_center[0] + non_deterministic_random.randint(-self.ccf, self.ccf),
                  self.roi_center[1] + non_deterministic_random.randint(-self.ccf, self.ccf),
                  self.roi_center[2] + non_deterministic_random.randint(-self.ccf, self.ccf)]  # 示例中心点1
            lx1, ly1, lz1 = [self.roi_length[0] + non_deterministic_random.randint(-self.crf, self.crf),
                             self.roi_length[1] + non_deterministic_random.randint(-self.crf, self.crf),
                             self.roi_length[2] + non_deterministic_random.randint(-self.crf, self.crf)]  # 示例边长1
            # mask = np.ones_like(data['seg1'])
            # print(f"mask shape is:{mask.shape}")
            mask1 = self.create_mask(c1, lx1, ly1, lz1).astype(bool)
            data[self.new_key[i]] = data[self.keys[i]] * mask1

        return data

    def create_mask(self, center, lx, ly, lz):
        x, y, z = center
        mask = np.zeros((128, 128, 128), dtype=np.float32)
        mask[x - lx // 2: x + lx // 2 + 1, y - ly // 2: y + ly // 2 + 1, z - lz // 2: z + lz // 2 + 1] = 1
        return mask


class ObtainUnion(MapTransform):
    def __init__(self, keys, new_key) -> None:
        MapTransform.__init__(self, keys)
        self.keys = keys
        self.new_key = new_key

    def __call__(self, data):
        p1, seg_p1, p2, seg_p2 = data[self.keys]
        data[self.new_key] = copy.deepcopy(data[self.keys])
        return data


