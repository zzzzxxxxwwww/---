# src/data_utils.py
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import joblib
import traceback


# 🔴 移除：不再需要 detect_and_smooth_outliers 函数
# ...

# 🔴 移除：不再需要 add_traffic_features 函数，让LSTM自己学习
# def add_traffic_features(df):
#    ... (函数体已删除) ...


def add_time_features(df):
    """添加时间相关特征 (基于索引)"""
    df_time = pd.DataFrame(index=df.index)

    if hasattr(df.index, 'hour'):
        df_time['hour'] = df.index.hour
        df_time['dayofweek'] = df.index.dayofweek
        df_time['month'] = df.index.month
        df_time['is_weekend'] = (df.index.dayofweek >= 5).astype(int)

        # 周期性编码
        df_time['hour_sin'] = np.sin(2 * np.pi * df_time['hour'] / 24)
        df_time['hour_cos'] = np.cos(2 * np.pi * df_time['hour'] / 24)
        df_time['day_sin'] = np.sin(2 * np.pi * df_time['dayofweek'] / 7)
        df_time['day_cos'] = np.cos(2 * np.pi * df_time['dayofweek'] / 7)

        # 移除原始的 'hour', 'dayofweek', 'month'，只保留编码后的
        df_time = df_time.drop(columns=['hour', 'dayofweek', 'month'])

    return df_time


def load_and_preprocess(csv_path, traffic_scaler_path="../experiments/traffic_scaler.pkl",
                        time_scaler_path="../experiments/time_scaler.pkl", history_len=24, pred_len=1):
    try:
        print("正在读取数据...")
        df = pd.read_csv(csv_path)

        if 'Unnamed: 0' in df.columns:
            print("检测到时间索引列 'Unnamed: 0'")
            df.rename(columns={'Unnamed: 0': 'datetime'}, inplace=True)

        if 'datetime' in df.columns:
            print("设置时间索引...")
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
            print(f"时间范围: {df.index.min()} 到 {df.index.max()}")
        else:
            raise ValueError("CSV文件中缺少时间列（需包含'datetime'或'Unnamed: 0'列）")

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        print(f"找到 {len(numeric_cols)} 个原始交通流量列")

        if len(numeric_cols) == 0:
            raise ValueError("读取的数据中没有有效数值列")

        df_numeric = df[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

        # 🔴 移除：异常值检测和平滑处理
        df_smoothed = df_numeric

        print("数据基本信息:")
        print(f"数据形状: {df_smoothed.shape}")

        # 🔴 移除：不再调用 add_traffic_features
        # print("添加交通流量特征工程...")
        # df_traffic_features = add_traffic_features(df_smoothed)

        print("添加时间特征...")
        df_time_features = add_time_features(df_smoothed)  # 传入 df_smoothed 以使用其索引

        # 分离特征
        original_traffic_cols = numeric_cols  # 原始交通流量列 (88个)
        # engineered_traffic_cols = df_traffic_features.columns.tolist() # 🔴 已移除
        time_feature_cols = df_time_features.columns.tolist()  # 纯时间特征

        print(f"原始交通特征 (用于缩放和预测): {len(original_traffic_cols)} 个")
        # print(f"工程交通特征 (作为时间特征输入): {len(engineered_traffic_cols)} 个") # 🔴 已移除
        print(f"时间特征 (作为时间特征输入): {len(time_feature_cols)} 个")

        # 对原始交通特征归一化 (Scaler 1)
        print("对原始交通特征归一化...")
        traffic_scaler = RobustScaler(quantile_range=(5, 95))
        scaled_traffic_values = traffic_scaler.fit_transform(df_smoothed[original_traffic_cols].values)
        df_scaled_traffic = pd.DataFrame(scaled_traffic_values, columns=original_traffic_cols, index=df_smoothed.index)

        # 🔴 修改：只合并纯时间特征
        all_time_features_cols = time_feature_cols
        df_all_time_features = df_time_features

        # 对所有“时间类”特征归一化 (Scaler 2)
        print("对所有时间类特征归一化...")
        time_scaler = RobustScaler(quantile_range=(5, 95))

        if not df_all_time_features.empty:
            scaled_time_values = time_scaler.fit_transform(df_all_time_features.values)
            df_scaled_time = pd.DataFrame(scaled_time_values, columns=all_time_features_cols, index=df_smoothed.index)
            time_feat_dim = len(all_time_features_cols)
        else:
            df_scaled_time = pd.DataFrame(index=df_smoothed.index)
            time_feat_dim = 0

            # 保存scaler
        os.makedirs(os.path.dirname(traffic_scaler_path), exist_ok=True)
        joblib.dump(traffic_scaler, traffic_scaler_path)
        joblib.dump(time_scaler, time_scaler_path)
        print(f"✅ 保存交通特征scaler到: {traffic_scaler_path}")
        print(f"✅ 保存时间特征scaler到: {time_scaler_path}")

        # ... (后续的滑动窗口代码保持不变) ...

        X_list, y_list = [], []
        valid_indices = 0
        traffic_feat_dim = len(original_traffic_cols)  # 原始交通特征维度

        print(f"✅ 交通特征数 (X_traffic): {traffic_feat_dim}, 时间特征数 (X_time): {time_feat_dim}")

        traffic_data_np = df_scaled_traffic.values
        if time_feat_dim > 0:
            time_data_np = df_scaled_time.values
        else:
            time_data_np = np.zeros((len(traffic_data_np), 1))
            time_feat_dim = 1
            print("警告: 没有找到时间特征，将使用0填充。")

        for i in range(len(df_scaled_traffic) - history_len - pred_len + 1):
            traffic_window = traffic_data_np[i: i + history_len]
            time_window = time_data_np[i: i + history_len]
            y_window = traffic_data_np[i + history_len: i + history_len + pred_len]

            if not (np.all(np.isfinite(traffic_window)) and
                    np.all(np.isfinite(time_window)) and
                    np.all(np.isfinite(y_window))):
                continue

            X_list.append((traffic_window, time_window))
            y_list.append(y_window)
            valid_indices += 1

        if valid_indices == 0:
            raise ValueError("没有生成有效的训练样本，请检查 history_len 或数据")

        X_traffic = np.array([x[0] for x in X_list])
        X_time = np.array([x[1] for x in X_list])
        y = np.array(y_list)

        print(f"✅ 有效样本数量: {valid_indices}")
        print(f"✅ X_traffic形状: {X_traffic.shape}, X_time形状: {X_time.shape}, y形状: {y.shape}")

        # 🔴 关键：time_feat_dim 必须返回正确的维度
        # 如果 time_feat_dim 为 0，但 X_time 是 (N, 144, 1) 的 0 填充
        # 我们需要返回 1，否则模型 time_emb 层会出错
        final_time_feat_dim = X_time.shape[2]
        print(f"✅ 最终返回的时间特征维度: {final_time_feat_dim}")

        return X_traffic, X_time, y, traffic_scaler, time_scaler, traffic_feat_dim, final_time_feat_dim

    except Exception as e:
        print(f"❌ 数据预处理失败: {e}")
        traceback.print_exc()
        raise