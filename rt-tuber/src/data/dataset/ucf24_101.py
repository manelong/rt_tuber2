"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py

Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torch.utils.data

import torchvision
torchvision.disable_beta_transforms_warning()

from ._dataset import DetDataset
from ...core import register

from .base_dataset import BaseDataset
from .utils import apply_distort, expand_and_crop, resize_img_keep_ratio
import os
import cv2
import numpy as np
import random


__all__ = ['ucf24_101']

class ucf24_101_dataset(BaseDataset):
    num_classes = 1

    def __init__(self, opt, image_filename, pkl_filename):
        assert opt['split'] == 1, "We use only the first split of NewBasketball"
        self.ROOT_DATASET_PATH = os.path.join(opt['root_dir'], image_filename)
        super(ucf24_101_dataset, self).__init__(opt, self.ROOT_DATASET_PATH, pkl_filename)

    def imagefile(self, v, i):
        return os.path.join(self.ROOT_DATASET_PATH,'rgb-images', v, '{:0>5}.jpg'.format(i))

@register()
class ucf24_101(ucf24_101_dataset, DetDataset):
    __inject__ = ['transforms', ]
    
    def __init__(self, opt, img_folder, ann_file, transforms=None, remap_mscoco_category=False):
        super(ucf24_101, self).__init__(opt, img_folder, ann_file)
        self.img_folder = img_folder
        self.ann_file = ann_file
        self._transforms = transforms
        self.remap_mscoco_category = remap_mscoco_category
        self.opt = opt

    def __getitem__(self, id):
        v, frame = self._indices[id]

        images = [cv2.imread(self.imagefile(v, frame + i)).astype(np.float32) for i in range(self.opt['K'])]

        if self.opt['mode'] == 'train':
            do_mirror = random.getrandbits(1) == 1
            # filp the image
            if do_mirror:
                images = [im[:, ::-1, :] for im in images]
                for i in range(self.opt['K']):
                    images[i][:, :, 2] = 255 - images[i][:, :, 2]
            h, w = self._resolution[v]
            gt_bbox = {}
            for ilabel, tubes in self._gttubes[v].items():
                for t in tubes:
                    if frame not in t[:, 0]:
                        continue
                    assert frame + self.opt['K'] - 1 in t[:, 0]
                    # copy otherwise it will change the gt of the dataset also
                    t = t.copy()
                    if do_mirror:
                        # filp the gt bbox
                        xmin = w - t[:, 3]
                        t[:, 3] = w - t[:, 1]
                        t[:, 1] = xmin
                    boxes = t[(t[:, 0] >= frame) * (t[:, 0] < frame + self.opt['K']), 1:5]

                    assert boxes.shape[0] == self.opt['K']
                    if ilabel not in gt_bbox:
                        gt_bbox[ilabel] = []
                    # gt_bbox[ilabel] ---> a list of numpy array, each one is K, x1, x2, y1, y2
                    gt_bbox[ilabel].append(boxes)

            # apply data augmentation
            images = apply_distort(images, self.distort_param)
            images, gt_bbox = expand_and_crop(images, gt_bbox, self._mean_values)  # augmentation
        else:
            # no data augmentation or flip when validation
            h, w = self._resolution[v]
            gt_bbox = {}
            for ilabel, tubes in self._gttubes[v].items():
                for t in tubes:
                    if frame not in t[:, 0]:
                        continue
                    assert frame + self.opt['K'] - 1 in t[:, 0]
                    t = t.copy()
                    boxes = t[(t[:, 0] >= frame) * (t[:, 0] < frame + self.opt['K']), 1:5]
                    assert boxes.shape[0] == self.opt['K']
                    if ilabel not in gt_bbox:
                        gt_bbox[ilabel] = []
                    gt_bbox[ilabel].append(boxes)

        ## 保持比例缩放
        original_h, original_w = h, w
        input_h, input_w = self.opt['input_h'], self.opt['input_w']
        for i in range(len(images)):
            images[i], left, top, ratio = resize_img_keep_ratio(images[i], (input_h, input_w))
        real_w = int(original_w * ratio)
        real_h = int(original_h * ratio)

        # 将gt与padding、resize之后提取到的特征图对应
        for ilabel in gt_bbox:
            for itube in range(len(gt_bbox[ilabel])):
                gt_bbox[ilabel][itube][:, 0] = (gt_bbox[ilabel][itube][:, 0] / original_w * real_w + left) / input_w
                gt_bbox[ilabel][itube][:, 1] = (gt_bbox[ilabel][itube][:, 1] / original_h * real_h + top) / input_h
                gt_bbox[ilabel][itube][:, 2] = (gt_bbox[ilabel][itube][:, 2] / original_w * real_w + left) / input_w
                gt_bbox[ilabel][itube][:, 3] = (gt_bbox[ilabel][itube][:, 3] / original_h * real_h + top) / input_h

        # gt_bbox 由xyxy转为cxcywh
        for ilabel in gt_bbox:
            for itube in range(len(gt_bbox[ilabel])):
                gt_bbox[ilabel][itube][:, 0] = (gt_bbox[ilabel][itube][:, 0] + gt_bbox[ilabel][itube][:, 2]) / 2
                gt_bbox[ilabel][itube][:, 1] = (gt_bbox[ilabel][itube][:, 1] + gt_bbox[ilabel][itube][:, 3]) / 2
                gt_bbox[ilabel][itube][:, 2] = (gt_bbox[ilabel][itube][:, 2] - gt_bbox[ilabel][itube][:, 0]) * 2
                gt_bbox[ilabel][itube][:, 3] = (gt_bbox[ilabel][itube][:, 3] - gt_bbox[ilabel][itube][:, 1]) * 2

        # 转换格式
        tubes = [images[i].transpose((2, 0, 1)) for i in range(self.opt['K'])]# 调整维度的顺序
        tubes = torch.tensor(np.stack(tubes, axis=1).astype(np.float32))
        
        labels = []
        tubes_bbox = []
        for item in gt_bbox:
            labels.append(item)
            tubes_bbox.append(np.stack(gt_bbox[item][0].flatten(), axis=0))
        labels = np.array(labels).astype(np.int64)
        tubes_bbox = np.stack(tubes_bbox, axis=0).astype(np.float32)
        
        if self.opt['mode'] == 'train':
            targets = {
                        'labels': torch.tensor(labels), 
                        'boxes': torch.tensor(tubes_bbox)
                    }
        else:
            targets = {
                        'labels': torch.tensor(labels), 
                        'boxes': torch.tensor(tubes_bbox),
                        'image_path': self.imagefile(v, frame)
                    }
        
        return tubes, targets


