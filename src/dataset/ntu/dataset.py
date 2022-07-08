##
##
##

import os
import logging
from typing import Tuple
import numpy as np
import torch

from .skeleton import NTUSkeletonGraph
from .config import NTUDatasetConfig
from ..dataset import Dataset

class NTUDataset(Dataset):
    
    def __init__(self, 
                 config: NTUDatasetConfig, 
                 skeleton: NTUSkeletonGraph,
                 logger: logging.Logger, 
                 train: bool) -> None:
        super().__init__()
        self._config = config
        self._logger = logger
        self._skeleton = skeleton
        
        self._num_classes = 60 if '60' in self._config.name else 120
        
        self._data, self._labels = self._load_data(train)
        if self._config.debug: 
            self._data = self._data[:300]
            self._labels = self._labels[:300]
    
    def _load_data(self, train: bool):
        phase = 'train' if train else 'test'
        path = self._config.dataset_path
        data_path = os.path.join(path, f'{phase}_data.npy')
        labels_path = os.path.join(path, f'{phase}_labels.npy')
        
        try:
            data = np.load(data_path, mmap_mode='r')
            labels = np.load(labels_path, mmap_mode='r')
        except Exception as e:
            if self._config.debug:
                self._logger.exception(e)
            else:
                self._logger.error(e)
            raise e

        return data, labels
    
    def __len__(self): 
        return len(self._labels)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, int, str]:
        data = torch.from_numpy(np.array(self._data[index]))
        label = torch.tensor(self._labels[index])
        data = data[:, :self._config.num_frames]
        
        C, T, V, M = data.shape
        joints = torch.zeros((C * 3, T, V, M))
        bones = torch.zeros((C * 3, T, V, M))
        
        joints[:C, :, :, :] = data
        joints[C:C*2, :-1] = joints[:C, 1:] - data[:C, :-1]
        joints[C*2:] = joints[:C] - data[:C, :, 1].unsqueeze(2)
        
        conn = self._skeleton.joints_connections
        for u, v in conn:
            bones[:C, :, u, :] = data[:, :, u, :] - data[:, :, v, :]
        bones[C:C*2, :-1] = bones[:C, 1:] - bones[:C, :-1]
        
        bone_length = 0
        for c in range(C):
            bone_length += bones[c,:,:,:] ** 2
        bone_length = np.sqrt(bone_length) + 0.0001
        for c in range(C):
            bones[C+c] = np.arccos(bones[c] / bone_length)
        
        return joints, bones, label
    
    @property
    def name(self) -> str:
        return self._config.name
    
    @property
    def num_classes(self) -> int:
        return self._num_classes
    
    @property
    def num_frames(self) -> int:
        return self._config.num_frames
    