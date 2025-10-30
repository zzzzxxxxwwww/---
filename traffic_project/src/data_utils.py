# 读数据、预处理、滑窗、scaler 保存/加载
# src/data_utils.py
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib


def load_and_preprocess(csv_path, scaler_path="../experiments/scaler.pkl", history_len=24, pred_len=1):
    # 1. 读取数据
    df = pd.read_csv(csv_path)

    # 2. 时间列改名、设为索引
    if 'Unnamed: 0' in df.columns:
        df.rename(columns={'Unnamed: 0': 'datetime'}, inplace=True)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)

    # 3. 归一化（0-1）
    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(df.values)
    df_scaled = pd.DataFrame(scaled_values, columns=df.columns, index=df.index)

    # 保存 scaler 方便之后反归一化
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(scaler, scaler_path)

    # 4. 生成滑动窗口
    X, y = [], []
    for i in range(len(df_scaled) - history_len - pred_len + 1):
        X.append(df_scaled.iloc[i:i + history_len].values)
        y.append(df_scaled.iloc[i + history_len:i + history_len + pred_len].values)

    X = np.array(X)
    y = np.array(y)
    print(f"✅ Shape of X: {X.shape}, y: {y.shape}")

    return X, y, scaler

def load_and_preprocess_data(csv_path, window_size=24):
    """
    加载交通流量数据并预处理：
      - 读取 CSV
      - 去掉非数值列
      - 归一化
      - 构造滑动窗口 (X, y)
    返回：
      X, y, scaler, site_names
    """
    # 1️⃣ 读取数据
    df = pd.read_csv(csv_path)
    df = df.select_dtypes(include=[np.number])  # 去掉非数值列
    site_names = df.columns.tolist()
    data = df.values

    # 2️⃣ 归一化
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    # 3️⃣ 构造滑动窗口
    X, y = [], []
    for i in range(len(scaled_data) - window_size):
        X.append(scaled_data[i:i+window_size])
        y.append(scaled_data[i+window_size:i+window_size+1])
    X = np.array(X)
    y = np.array(y)

    return X, y, scaler, site_names
