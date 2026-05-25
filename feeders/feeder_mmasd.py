"""
Feeder for MMASD+ in NW-UCLA format: one JSON per sample, skeletons (T, 20, 3), label 1-based.

Expects:
  - data_path: directory containing all_sqe/ and train_data_dict.json, val_data_dict.json (e.g. data/MMASD+)
  - label_path: 'train' or 'val' to select which data_dict to load

Same pipeline as feeder_ucla (20 joints, 1 person, time_steps=52, bone/motion options).
"""

import os
import os.path as osp
import json
import numpy as np
import random
import math
from torch.utils.data import Dataset


class Feeder(Dataset):
    def __init__(self, data_path, label_path, repeat=1, random_choose=False, random_shift=False, random_move=False,
                 window_size=-1, normalization=False, debug=False, use_mmap=True, time_steps=52):
        self.data_path = data_path
        self.label_path = label_path
        self.time_steps = time_steps
        self.repeat = repeat
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.debug = debug

        if 'val' in label_path:
            self.train_val = 'val'
            dict_name = 'val_data_dict.json'
        else:
            self.train_val = 'train'
            dict_name = 'train_data_dict.json'
        dict_path = osp.join(data_path, dict_name)
        with open(dict_path, 'r') as f:
            self.data_dict = json.load(f)

        self.mmasd_root = osp.join(data_path, 'all_sqe')
        self.bone = [(1, 2), (2, 3), (3, 3), (4, 3), (5, 3), (6, 5), (7, 6), (8, 7), (9, 3), (10, 9), (11, 10),
                     (12, 11), (13, 1), (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18), (20, 19)]
        self.label = [int(info['label']) - 1 for info in self.data_dict]  # 0-based
        self.load_data()
        if normalization:
            self.get_mean_map()

    def load_data(self):
        self.data = []
        for info in self.data_dict:
            file_name = info['file_name']
            with open(osp.join(self.mmasd_root, file_name + '.json'), 'r') as f:
                obj = json.load(f)
            value = np.array(obj['skeletons'], dtype=np.float32)  # (T, 20, 3)
            self.data.append(value)

    def get_mean_map(self):
        data = np.concatenate([x.reshape(-1, 20, 3) for x in self.data], axis=0)
        N, V, C = data.shape
        data = data.transpose(0, 2, 1).reshape(N, C, V, 1)
        self.mean_map = data.mean(axis=0, keepdims=True)
        self.std_map = data.std(axis=0, keepdims=True) + 1e-6

    def __len__(self):
        return len(self.data_dict) * self.repeat

    def rand_view_transform(self, X, agx, agy, s):
        agx = math.radians(agx)
        agy = math.radians(agy)
        Rx = np.asarray([[1, 0, 0], [0, math.cos(agx), math.sin(agx)], [0, -math.sin(agx), math.cos(agx)]])
        Ry = np.asarray([[math.cos(agy), 0, -math.sin(agy)], [0, 1, 0], [math.sin(agy), 0, math.cos(agy)]])
        Ss = np.asarray([[s, 0, 0], [0, s, 0], [0, 0, s]])
        X0 = np.dot(np.reshape(X, (-1, 3)), np.dot(Ry, np.dot(Rx, Ss)))
        return np.reshape(X0, X.shape)

    def __getitem__(self, index):
        label = self.label[index % len(self.data_dict)]
        value = self.data[index % len(self.data_dict)].copy()

        if self.train_val == 'train':
            agx = random.randint(-60, 60)
            agy = random.randint(-60, 60)
            s = random.uniform(0.5, 1.5)
            center = value[0, 1, :]
            value = value - center
            scalerValue = self.rand_view_transform(value, agx, agy, s)
            scalerValue = np.reshape(scalerValue, (-1, 3))
            scalerValue = (scalerValue - np.min(scalerValue, axis=0)) / (np.max(scalerValue, axis=0) - np.min(scalerValue, axis=0) + 1e-6)
            scalerValue = scalerValue * 2 - 1
            scalerValue = np.reshape(scalerValue, (-1, 20, 3))
            data = np.zeros((self.time_steps, 20, 3), dtype=np.float32)
            length = scalerValue.shape[0]
            random_idx = random.sample(list(np.arange(length)) * 100, min(self.time_steps, length * 100))
            random_idx = sorted(random_idx)[:self.time_steps]
            if len(random_idx) < self.time_steps:
                idx = np.linspace(0, length - 1, self.time_steps).astype(int)
                data[:, :, :] = scalerValue[np.clip(idx, 0, length - 1), :, :]
            else:
                data[:, :, :] = scalerValue[random_idx, :, :]
        else:
            center = value[0, 1, :]
            value = value - center
            scalerValue = self.rand_view_transform(value, 0, 0, 1.0)
            scalerValue = np.reshape(scalerValue, (-1, 3))
            scalerValue = (scalerValue - np.min(scalerValue, axis=0)) / (np.max(scalerValue, axis=0) - np.min(scalerValue, axis=0) + 1e-6)
            scalerValue = scalerValue * 2 - 1
            scalerValue = np.reshape(scalerValue, (-1, 20, 3))
            data = np.zeros((self.time_steps, 20, 3), dtype=np.float32)
            length = scalerValue.shape[0]
            idx = np.linspace(0, length - 1, self.time_steps).astype(int)
            data[:, :, :] = scalerValue[np.clip(idx, 0, length - 1), :, :]

        if 'bone' in self.data_path:
            data_bone = np.zeros_like(data)
            for bone_idx in range(20):
                data_bone[:, self.bone[bone_idx][0] - 1, :] = data[:, self.bone[bone_idx][0] - 1, :] - data[:, self.bone[bone_idx][1] - 1, :]
            data = data_bone
        if 'motion' in self.data_path:
            data_motion = np.zeros_like(data)
            data_motion[:-1, :, :] = data[1:, :, :] - data[:-1, :, :]
            data = data_motion

        data = np.transpose(data, (2, 0, 1))  # (C, T, V)
        C, T, V = data.shape
        data = np.reshape(data, (C, T, V, 1))
        return data, label, index

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod
