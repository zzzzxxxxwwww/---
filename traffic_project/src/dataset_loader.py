# -*- coding: utf-8 -*-
"""
Dataset Loader（中文注释）
用于把 /mnt/data/preprocessed_for_st 下的 numpy 时间序列和邻接加载为训练样本
输出的数据形状：
  - X: (num_samples, window, N)
  - Y: (num_samples, horizon, N)
"""
import numpy as np
import os

class TrafficDatasetSimple:
    def __init__(self, data_dir="../data/preprocessed_for_st", city="milano",
                 split=(0.7, 0.1, 0.2), window=12, horizon=6):
        self.data_dir = data_dir
        self.city = city
        self.window = window
        self.horizon = horizon

        # 读取数据（T x N）
        data_path = os.path.join(data_dir, f"{city}_norm.npy")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"找不到 {data_path}")
        self.data = np.load(data_path)  # T x N
        self.adj = np.load(os.path.join(data_dir, f"{city}_combined_adj.npy"))
        T = self.data.shape[0]
        tr = int(T * split[0])
        vr = int(T * (split[0] + split[1]))
        self.train_ts = self.data[:tr]
        self.val_ts = self.data[tr:vr]
        self.test_ts = self.data[vr:]

    def _create_xy(self, arr):
        X, Y = [], []
        T = arr.shape[0]
        for i in range(T - self.window - self.horizon + 1):
            X.append(arr[i:i+self.window])             # window x N
            Y.append(arr[i+self.window:i+self.window+self.horizon])  # horizon x N
        if len(X) == 0:
            return None, None
        return np.stack(X), np.stack(Y)

    def get_train(self):
        return self._create_xy(self.train_ts)

    def get_val(self):
        return self._create_xy(self.val_ts)

    def get_test(self):
        return self._create_xy(self.test_ts)
