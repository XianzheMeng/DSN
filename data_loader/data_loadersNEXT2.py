import numpy as np
import pandas as pd
import random
import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, Sampler, DataLoader

from sklearn.model_selection import train_test_split

import librosa
from utils import util

EPS = 1E-8

def standard_normal_variate(data):
    mean = np.mean(data)
    std = np.std(data)

    return (data - mean) / std

class HeartSoundDataSet(Dataset):

    def __init__(self, 
                 fea_path: h5py.File,
                 labels: pd.DataFrame,
                 duration: int,
                 training: bool,
                 delta: bool = False,
                 norm: bool=False):
        self._fea_path = fea_path
        self._h5database = None
        self._labels = labels 
        self._colname = ['filename', 'label']
        self._len = len(self._labels)
        self._duration = duration
        self._delta = delta
        self.train = training 
        self.hop_length = 60
        self.norm = norm

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        filename, target_bin = self._labels.iloc[idx].reindex(self._colname)
        if self._h5database is None:
            self._h5database = h5py.File(self._fea_path, 'r')

        feature = self._h5database[filename][()]
        if self.norm:
            feature = standard_normal_variate(feature)
        if self._delta: 
            delta = librosa.feature.delta(feature)
            delta_2 = librosa.feature.delta(delta)
            feature = np.concatenate((feature, delta, delta_2),axis=0)
        
        
        mel_bins, num_frames = feature.shape
        # print("STARTFEATURESHAPE",feature.shape)

        cycle_len = num_frames #int(self._duration * 1000 / self.hop_length)  #为了保证符合Dataset 0402 原#84

        if num_frames >= cycle_len:
            if self.train:
                start_ind = random.randint(0, num_frames - cycle_len)
            else:
                start_ind = int((num_frames - cycle_len)/2)
            feature_pad = feature[:, start_ind:start_ind + cycle_len]
        elif num_frames < cycle_len:
            feature_pad = np.pad(feature, ((0,0),(0, cycle_len-num_frames)), mode='wrap')
        elif num_frames == cycle_len:
            feature_pad = feature
        else:
            print('Wrong audio length!')
        # 特征处理逻辑不变，最后添加通道维度
        feature_pad = np.expand_dims(feature_pad, axis=0)

        # # ===== 新增调试代码 =====
        # print(f"[{'TRAIN' if self.train else 'TEST'}] 文件名: {filename}, 切片起点: {start_ind}, 特征形状: {feature_pad.shape}")
        # # ======================

    


        return feature_pad, target_bin
    
    

class MinimumOccupancySampler(Sampler):
    """
        samples at least one instance from each class sequentially
    """
    def __init__(self, labels, sampling_mode='over', random_state=None):
        data_samples = labels.shape
        n_labels = len(np.unique(labels))
        label_to_idx_list, label_to_length = [], []
        self.random_state = np.random.RandomState(seed=random_state)
        for lb_idx in range(n_labels):
            label_indexes = np.where(labels == lb_idx)[0]
            self.random_state.shuffle(label_indexes)
            label_to_length.append(len(label_indexes))
            label_to_idx_list.append(label_indexes)

        self.longest_seq = max(label_to_length)
        self.data_source = np.empty((self.longest_seq, len(label_to_length)), dtype=int)
        # Each column represents one "single instance per class" data piece
        for ix, leng in enumerate(label_to_length):
            self.data_source[:leng, ix] = label_to_idx_list[ix]
        self.label_to_length = label_to_length
        self.label_to_idx_list = label_to_idx_list

        if sampling_mode == 'same':
            self.data_length = data_samples
        elif sampling_mode == 'over':  # Sample all items
            self.data_length = np.prod(self.data_source.shape)

    def _resample(self):
        for ix, leng in enumerate(self.label_to_length):
            leftover = self.longest_seq - leng
            random_idxs = np.random.randint(leng, size=leftover)
            self.data_source[leng:, ix] = self.label_to_idx_list[ix][random_idxs]

    def __iter__(self):
        self._resample()
        n_samples = len(self.data_source)
        random_indices = self.random_state.permutation(n_samples)
        data = np.concatenate(
            self.data_source[random_indices])[:self.data_length]
        return iter(data)

    def __len__(self):
        return self.data_length
 
class HeartSoundDataLoader(DataLoader):
    def __init__(self,
                 fea_path, 
                 label_df,
                 duration,
                 batch_size,
                 delta=False,
                 norm = False,
                 shuffle=True, 
                 validation_split=0.0, 
                 num_workers=1, 
                 training=True,
                 collate_fn=None):
        # if training:
        #     # self.train_df, self.val_df = train_test_split(label_df, test_size=validation_split, shuffle=True)
        
        #     self.train_sampler = MinimumOccupancySampler(np.stack(self.train_df['label']), random_state=100)
        #     # self.val_dataset = HeartSoundDataSet(fea_path, self.val_df, duration=duration, training=False, delta=delta, norm=norm)

        # else:
        self.train_df = label_df 
        self.train_sampler = MinimumOccupancySampler(np.stack(self.train_df['label']), random_state=100)
        self.dataset = HeartSoundDataSet(fea_path, self.train_df, duration=duration, 
                                               training=training, delta=delta, norm=norm)
        self.init_kwargs = {
            'batch_size': batch_size,
            'shuffle': shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers
        }
        super().__init__(sampler=self.train_sampler, 
                         dataset = self.dataset,
                         **self.init_kwargs)
    
    def split_validation(self):
        return DataLoader(dataset= self.val_dataset, **self.init_kwargs)

import os
class KDEnhancedDataset(HeartSoundDataSet):
    def __init__(self, soft_label_path, fea_path, labels, duration, training=True, delta=False, norm=True):
        super().__init__(fea_path, labels, duration, training, delta, norm)
        
        # 加载软标签（添加路径检查）
        if not os.path.exists(soft_label_path):
            raise FileNotFoundError(f"软标签文件 {soft_label_path} 不存在")
        soft_df = pd.read_csv(soft_label_path)
        self.soft_label_map = dict(zip(soft_df['filename'], soft_df[['class_0', 'class_1']].values))
        print("原始文件名示例:", self._labels['filename'].head(3).tolist())
        print("软标签文件名示例:", list(self.soft_label_map.keys())[:3])
        # 使用父类的正确属性名 _labels（关键修复点）
        missing = [f for f in self._labels['filename'] if f not in self.soft_label_map]
        if missing:
            raise ValueError(f"缺失{len(missing)}个样本的软标签，示例：{missing[:3]}")
        print(f"数据集验证通过 → 总样本数: {len(self)}")
        print("首样本信息 →", {
            'filename': self._labels.iloc[0]['filename'],
            'hard_label': self._labels.iloc[0]['label'],
            'soft_label': self.soft_label_map[self._labels.iloc[0]['filename']]
        })
    def __getitem__(self, idx):
        # 获取父类原始数据（假设父类返回numpy array）
        feature, label = super().__getitem__(idx)
        
        # 显式转换数据类型
        feature_tensor = torch.from_numpy(feature.copy()).float()  # 添加copy避免内存错误
        # feature_tensor = feature_tensor.unsqueeze(0)  # 添加通道维度（若适用）
        
        # 获取软标签并转换
        filename = self._labels.iloc[idx]['filename']
        soft_label = torch.FloatTensor(self.soft_label_map[filename])
        
        return feature_tensor, label, soft_label

    # def validate_sample(self, idx):
    #     sample = self[idx]
    #     assert len(sample) == 3, "样本必须包含特征、硬标签、软标签"
    #     assert isinstance(sample[0], torch.Tensor), f"特征类型错误：{type(sample[0])}"
    #     assert sample[0].dim() in [2, 3], f"特征维度错误：{sample[0].shape}"
    #     return True

class KDDataLoader(DataLoader):
    def __init__(self,
                 fea_path, 
                 label_df,
                 duration,
                 batch_size,
                 delta=False,
                 norm = False,
                 shuffle=True, 
                 validation_split=0.0, 
                 num_workers=1, 
                 training=True,
                 collate_fn=None,
                 soft_label_path=None):
        # if training:
        #     # self.train_df, self.val_df = train_test_split(label_df, test_size=validation_split, shuffle=True)
        
        #     self.train_sampler = MinimumOccupancySampler(np.stack(self.train_df['label']), random_state=100)
        #     # self.val_dataset = HeartSoundDataSet(fea_path, self.val_df, duration=duration, training=False, delta=delta, norm=norm)

        # else:
        self.train_df = label_df 
        self.train_sampler = MinimumOccupancySampler(np.stack(self.train_df['label']), random_state=100)
        self.dataset = KDEnhancedDataset(soft_label_path,fea_path, self.train_df, duration=duration, 
                                               training=training, delta=delta, norm=norm)
        self.init_kwargs = {
            'batch_size': batch_size,
            'shuffle': shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers
        }
        super().__init__(sampler=self.train_sampler, 
                         dataset = self.dataset,
                         **self.init_kwargs)
    
    def split_validation(self):
        return DataLoader(dataset= self.val_dataset, **self.init_kwargs)
