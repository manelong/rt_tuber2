from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle

import torch.utils.data as data

from .utils import tubelet_in_out_tubes, tubelet_has_gt, tubelet_near_key_frame

class BaseDataset(data.Dataset):

    def __init__(self, opt, ROOT_DATASET_PATH, pkl_filename):

        super(BaseDataset, self).__init__()
        pkl_file = os.path.join(ROOT_DATASET_PATH, pkl_filename)

        with open(pkl_file, 'rb') as fid:
            pkl = pickle.load(fid, encoding='iso-8859-1')
        for k in pkl:
            setattr(self, ('_' if k != 'labels' else '') + k, pkl[k])

        self.split = opt['split']
        self.K = opt['K']
        self.opt = opt

        self._mean_values = [104.0136177, 114.0342201, 119.91659325]

        assert len(self._train_videos[self.split - 1]) + len(self._test_videos[self.split - 1]) == len(self._nframes)
        self._indices = []
        if opt['mode'] == 'train':
            # get train video list
            video_list = self._train_videos[self.split - 1]
        else:
            # get test video list
            video_list = self._test_videos[self.split - 1]

        for v in video_list:
            vtubes = sum(self._gttubes[v].values(), [])
            self._indices += [(v, i) for i in range(1, self._nframes[v] + 2 - self.K, 8)
                        if tubelet_in_out_tubes(vtubes, i, self.K) and tubelet_has_gt(vtubes, i, self.K)]

        self.distort_param = {
            'brightness_prob': 0.5,
            'brightness_delta': 32,
            'contrast_prob': 0.5,
            'contrast_lower': 0.5,
            'contrast_upper': 1.5,
            'hue_prob': 0.5,
            'hue_delta': 18,
            'saturation_prob': 0.5,
            'saturation_lower': 0.5,
            'saturation_upper': 1.5,
            'random_order_prob': 0.0,
        }
        self.expand_param = {
            'expand_prob': 0.5,
            'max_expand_ratio': 4.0,
        }
        self.batch_samplers = [{
            'sampler': {},
            'max_trials': 1,
            'max_sample': 1,
        }, {
            'sampler': {'min_scale': 0.3, 'max_scale': 1.0, 'min_aspect_ratio': 0.5, 'max_aspect_ratio': 2.0, },
            'sample_constraint': {'min_jaccard_overlap': 0.1, },
            'max_trials': 50,
            'max_sample': 1,
        }, {
            'sampler': {'min_scale': 0.3, 'max_scale': 1.0, 'min_aspect_ratio': 0.5, 'max_aspect_ratio': 2.0, },
            'sample_constraint': {'min_jaccard_overlap': 0.3, },
            'max_trials': 50,
            'max_sample': 1,
        }, {
            'sampler': {'min_scale': 0.3, 'max_scale': 1.0, 'min_aspect_ratio': 0.5, 'max_aspect_ratio': 2.0, },
            'sample_constraint': {'min_jaccard_overlap': 0.5, },
            'max_trials': 50,
            'max_sample': 1,
        }, {
            'sampler': {'min_scale': 0.3, 'max_scale': 1.0, 'min_aspect_ratio': 0.5, 'max_aspect_ratio': 2.0, },
            'sample_constraint': {'min_jaccard_overlap': 0.7, },
            'max_trials': 50,
            'max_sample': 1,
        }, {
            'sampler': {'min_scale': 0.3, 'max_scale': 1.0, 'min_aspect_ratio': 0.5, 'max_aspect_ratio': 2.0, },
            'sample_constraint': {'min_jaccard_overlap': 0.9, },
            'max_trials': 50,
            'max_sample': 1,
        }, {
            'sampler': {'min_scale': 0.3, 'max_scale': 1.0, 'min_aspect_ratio': 0.5, 'max_aspect_ratio': 2.0, },
            'sample_constraint': {'max_jaccard_overlap': 1.0, },
            'max_trials': 50,
            'max_sample': 1,
        }, ]
        self.max_objs = 128

    def __len__(self):
        return len(self._indices)

    def imagefile(self, v, i):
        raise NotImplementedError


