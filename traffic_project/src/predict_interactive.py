# src/predict_interactive.py
import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from data_utils import load_and_preprocess
from model_tcn import TCNForecast
import os
import joblib
from datetime import datetime, timedelta

# 设备配置
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# 模型参数（必须与训练时一致）
HISTORY_LEN = 144
PRED_LEN = 6
TCN_CHANNELS = [128, 128, 128, 128, 128, 128, 128]
KERNEL_SIZE = 3
DROPOUT = 0.3
SPLIT_RATIO = 0.8  # 与训练时保持一致


def load_full_data():
    """加载完整数据集（不分割）"""
    print("正在加载完整数据集...")

    # 读取原始CSV文件获取完整时间索引
    df_raw = pd.read_csv("../data/milano_traffic_nid.csv")
    if 'Unnamed: 0' in df_raw.columns:
        df_raw.rename(columns={'Unnamed: 0': 'datetime'}, inplace=True)
    df_raw['datetime'] = pd.to_datetime(df_raw['datetime'])
    df_raw.set_index('datetime', inplace=True)

    # 使用data_utils预处理（但返回完整数据）
    X_traffic, X_time, y, traffic_scaler, time_scaler, traffic_feat_dim, time_feat_dim = load_and_preprocess(
        "../data/milano_traffic_nid.csv",
        "../experiments/traffic_scaler.pkl",
        "../experiments/time_scaler.pkl",
        history_len=HISTORY_LEN,
        pred_len=PRED_LEN
    )

    return X_traffic, X_time, y, traffic_scaler, time_scaler, traffic_feat_dim, time_feat_dim, df_raw


def create_full_predictions(model, X_traffic, X_time, traffic_scaler, batch_size=32):
    """对整个数据集进行预测"""
    model.eval()
    all_predictions = []

    print("正在进行全数据预测...")
    with torch.no_grad():
        for i in range(0, len(X_traffic), batch_size):
            # 获取当前batch
            end_idx = min(i + batch_size, len(X_traffic))
            batch_traffic = torch.tensor(X_traffic[i:end_idx], dtype=torch.float32).to(DEVICE)
            batch_time = torch.tensor(X_time[i:end_idx], dtype=torch.float32).to(DEVICE)

            # 预测
            pred = model(batch_traffic, batch_time)

            # 逆变换回原始空间
            batch_pred_log = traffic_scaler.inverse_transform(
                pred.cpu().numpy().reshape(-1, pred.shape[2])
            )
            batch_pred_orig = np.expm1(batch_pred_log).reshape(pred.shape)
            batch_pred_orig[batch_pred_orig < 0] = 0

            all_predictions.append(batch_pred_orig)

    # 合并所有预测结果
    full_predictions = np.concatenate(all_predictions, axis=0)
    print(f"预测完成！形状: {full_predictions.shape}")

    return full_predictions


def generate_future_predictions(model, df_raw, traffic_scaler, time_scaler,
                                traffic_feat_dim, time_feat_dim, station_idx,
                                future_days=60):
    """生成未来预测"""
    print(f"正在生成未来 {future_days} 天的预测...")

    # 获取最后HISTORY_LEN个时间点的数据作为预测起点
    last_traffic_data = df_raw.iloc[-HISTORY_LEN:].values

    # 应用与训练时相同的预处理
    last_traffic_log = np.log1p(last_traffic_data)
    last_traffic_scaled = traffic_scaler.transform(last_traffic_log)

    # 生成未来时间序列
    last_time = df_raw.index[-1]
    future_times = [last_time + timedelta(hours=i + 1) for i in range(future_days * 24)]

    # 生成未来时间特征
    future_df = pd.DataFrame(index=future_times)
    future_time_features = add_time_features(future_df)

    # 对时间特征进行归一化
    scaled_future_time = time_scaler.transform(future_time_features.values)

    # 初始化预测结果
    future_predictions = []
    current_traffic = last_traffic_scaled.copy()

    model.eval()
    with torch.no_grad():
        # 我们需要为每个预测步骤准备HISTORY_LEN长度的时间特征
        # 所以我们将未来时间特征扩展为滑动窗口
        for i in range(0, len(future_times), PRED_LEN):
            # 获取当前时间窗口的时间特征 - 需要HISTORY_LEN长度
            # 对于第一个预测，我们使用历史最后的时间特征和未来时间特征
            if i == 0:
                # 获取历史最后的时间特征
                historical_times = df_raw.index[-HISTORY_LEN:]
                historical_df = pd.DataFrame(index=historical_times)
                historical_time_features = add_time_features(historical_df)
                scaled_historical_time = time_scaler.transform(historical_time_features.values)

                # 使用历史时间特征作为初始输入
                current_time_window = scaled_historical_time
            else:
                # 对于后续预测，我们使用未来的时间特征
                # 我们需要HISTORY_LEN长度的时间特征
                start_idx = max(0, i - HISTORY_LEN)
                end_idx = i + PRED_LEN
                if end_idx > len(scaled_future_time):
                    # 如果不够，用最后一个值填充
                    padding_needed = end_idx - len(scaled_future_time)
                    current_time_window = np.vstack([
                        scaled_future_time[start_idx:],
                        np.tile(scaled_future_time[-1:], (padding_needed, 1))
                    ])
                else:
                    current_time_window = scaled_future_time[start_idx:end_idx]

            # 确保时间窗口长度为HISTORY_LEN
            if len(current_time_window) < HISTORY_LEN:
                padding = np.tile(current_time_window[-1:], (HISTORY_LEN - len(current_time_window), 1))
                current_time_window = np.vstack([current_time_window, padding])
            elif len(current_time_window) > HISTORY_LEN:
                current_time_window = current_time_window[:HISTORY_LEN]

            # 准备输入数据
            traffic_input = current_traffic.reshape(1, HISTORY_LEN, traffic_feat_dim)
            time_input = current_time_window.reshape(1, HISTORY_LEN, time_feat_dim)

            # 转换为tensor
            traffic_tensor = torch.tensor(traffic_input, dtype=torch.float32).to(DEVICE)
            time_tensor = torch.tensor(time_input, dtype=torch.float32).to(DEVICE)

            # 预测
            pred = model(traffic_tensor, time_tensor)

            # 逆变换
            pred_log = traffic_scaler.inverse_transform(
                pred.cpu().numpy().reshape(-1, pred.shape[2])
            )
            pred_orig = np.expm1(pred_log).reshape(pred.shape)
            pred_orig[pred_orig < 0] = 0

            # 存储预测结果
            future_predictions.append(pred_orig[0, :, station_idx])

            # 更新当前交通数据（使用预测值）
            # 首先将预测值转换为与输入相同的尺度
            pred_for_update = pred_orig[0, :, :]  # 取第一个batch的所有预测

            # 将预测值转换回归一化对数空间
            pred_log_update = np.log1p(pred_for_update)
            pred_scaled_update = traffic_scaler.transform(pred_log_update)

            # 滑动窗口：移除最旧的数据，添加最新的预测
            current_traffic = np.vstack([current_traffic[PRED_LEN:], pred_scaled_update])

    # 合并所有预测结果
    future_predictions = np.concatenate(future_predictions, axis=0)
    # 只取实际需要的长度
    future_predictions = future_predictions[:len(future_times)]

    print(f"未来预测完成！形状: {future_predictions.shape}")
    return future_times, future_predictions


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

        df_time = df_time.drop(columns=['hour', 'dayofweek', 'month'])

    return df_time


def get_station_names(df_raw):
    """获取站点名称列表"""
    numeric_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
    return numeric_cols


def create_interactive_plot(df_raw, full_predictions, station_idx, station_name, future_times=None,
                            future_predictions=None):
    """创建交互式时间序列图表"""

    # 获取真实值（逆变换）
    traffic_data_log = np.log1p(df_raw.values)
    traffic_scaler = joblib.load("../experiments/traffic_scaler.pkl")
    traffic_data_scaled = traffic_scaler.transform(traffic_data_log)
    traffic_data_restored = traffic_scaler.inverse_transform(traffic_data_scaled)
    true_values_orig = np.expm1(traffic_data_restored)

    # 由于预测是从HISTORY_LEN开始的，我们需要对齐时间索引
    start_idx = HISTORY_LEN
    end_idx = start_idx + len(full_predictions)

    # 获取对应的时间段
    time_index = df_raw.index[start_idx:end_idx]

    # 获取真实值和预测值（只取T+1的预测）
    true_vals = true_values_orig[start_idx:end_idx, station_idx]
    pred_vals = full_predictions[:, 0, station_idx]  # 只使用T+1预测

    # 计算训练集和测试集的分割点
    total_samples = len(full_predictions)
    train_split_idx = int(total_samples * SPLIT_RATIO)
    train_split_time = time_index[train_split_idx]

    # 将时间索引转换为数值时间戳（毫秒）
    time_index_numeric = time_index.astype(np.int64) // 10 ** 6  # 转换为毫秒

    # 创建Plotly图表
    fig = go.Figure()

    # 添加训练集区域背景（使用数值时间戳）
    fig.add_vrect(
        x0=time_index_numeric[0],
        x1=time_index_numeric[train_split_idx],
        fillcolor="lightgreen",
        opacity=0.2,
        layer="below",
        line_width=0,
        annotation_text="训练集",
        annotation_position="top left"
    )

    # 添加测试集区域背景（使用数值时间戳）
    fig.add_vrect(
        x0=time_index_numeric[train_split_idx],
        x1=time_index_numeric[-1],
        fillcolor="lightcoral",
        opacity=0.2,
        layer="below",
        line_width=0,
        annotation_text="测试集",
        annotation_position="top right"
    )

    # 如果有未来预测，添加未来区域背景
    if future_times is not None and future_predictions is not None:
        future_times_numeric = pd.DatetimeIndex(future_times).astype(np.int64) // 10 ** 6
        fig.add_vrect(
            x0=time_index_numeric[-1],
            x1=future_times_numeric[-1],
            fillcolor="lightblue",
            opacity=0.2,
            layer="below",
            line_width=0,
            annotation_text="未来预测",
            annotation_position="top right"
        )

    # 添加分割线（使用数值时间戳）
    fig.add_vline(
        x=time_index_numeric[train_split_idx],
        line_width=2,
        line_dash="dash",
        line_color="red",
        annotation_text="训练/测试分割线",
        annotation_position="top"
    )

    # 如果有未来预测，添加未来分割线
    if future_times is not None and future_predictions is not None:
        fig.add_vline(
            x=time_index_numeric[-1],
            line_width=2,
            line_dash="dash",
            line_color="blue",
            annotation_text="未来预测开始",
            annotation_position="top"
        )

    # 添加真实值曲线（训练集部分）
    train_true_vals = true_vals[:train_split_idx]
    train_times = time_index_numeric[:train_split_idx]
    fig.add_trace(go.Scatter(
        x=train_times,
        y=train_true_vals,
        mode='lines',
        name='真实值 (训练集)',
        line=dict(color='blue', width=1.5),
        opacity=0.8
    ))

    # 添加真实值曲线（测试集部分）
    test_true_vals = true_vals[train_split_idx:]
    test_times = time_index_numeric[train_split_idx:]
    fig.add_trace(go.Scatter(
        x=test_times,
        y=test_true_vals,
        mode='lines',
        name='真实值 (测试集)',
        line=dict(color='green', width=1.5),
        opacity=0.8
    ))

    # 添加预测值曲线（训练集部分）
    train_pred_vals = pred_vals[:train_split_idx]
    fig.add_trace(go.Scatter(
        x=train_times,
        y=train_pred_vals,
        mode='lines',
        name='预测值 (训练集)',
        line=dict(color='red', width=1.5, dash='dot'),
        opacity=0.8
    ))

    # 添加预测值曲线（测试集部分）
    test_pred_vals = pred_vals[train_split_idx:]
    fig.add_trace(go.Scatter(
        x=test_times,
        y=test_pred_vals,
        mode='lines',
        name='预测值 (测试集)',
        line=dict(color='orange', width=1.5, dash='dot'),
        opacity=0.8
    ))

    # 如果有未来预测，添加未来预测曲线
    if future_times is not None and future_predictions is not None:
        future_times_numeric = pd.DatetimeIndex(future_times).astype(np.int64) // 10 ** 6
        fig.add_trace(go.Scatter(
            x=future_times_numeric,
            y=future_predictions,
            mode='lines',
            name='未来预测',
            line=dict(color='purple', width=2),
            opacity=0.8
        ))

    # 更新布局，设置x轴为时间格式
    fig.update_layout(
        title=f'{station_name} - 交通流量预测 (TCN模型) - 训练集/测试集/未来预测对比',
        xaxis_title='时间',
        yaxis_title='交通流量',
        template='plotly_white',
        height=700,
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )

    # 设置x轴为时间格式
    fig.update_xaxes(
        type='date',
        tickformat='%Y-%m-%d %H:%M',
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1天", step="day", stepmode="backward"),
                dict(count=7, label="1周", step="day", stepmode="backward"),
                dict(count=1, label="1月", step="month", stepmode="backward"),
                dict(count=2, label="2月", step="month", stepmode="backward"),
                dict(step="all", label="全部")
            ])
        )
    )

    return fig


def calculate_model_metrics(true_vals, pred_vals, split_idx):
    """计算模型在训练集和测试集上的指标"""
    # 训练集指标
    train_true = true_vals[:split_idx]
    train_pred = pred_vals[:split_idx]

    train_mae = np.mean(np.abs(train_true - train_pred))
    train_rmse = np.sqrt(np.mean((train_true - train_pred) ** 2))
    train_mape = np.mean(np.abs((train_true - train_pred) / (np.abs(train_true) + 1e-8))) * 100

    # 测试集指标
    test_true = true_vals[split_idx:]
    test_pred = pred_vals[split_idx:]

    test_mae = np.mean(np.abs(test_true - test_pred))
    test_rmse = np.sqrt(np.mean((test_true - test_pred) ** 2))
    test_mape = np.mean(np.abs((test_true - test_pred) / (np.abs(test_true) + 1e-8))) * 100

    print("\n=== 模型性能指标 ===")
    print(f"训练集 (样本数: {len(train_true)})")
    print(f"  MAE:  {train_mae:.2f}")
    print(f"  RMSE: {train_rmse:.2f}")
    print(f"  MAPE: {train_mape:.2f}%")

    print(f"\n测试集 (样本数: {len(test_true)})")
    print(f"  MAE:  {test_mae:.2f}")
    print(f"  RMSE: {test_rmse:.2f}")
    print(f"  MAPE: {test_mape:.2f}%")

    return {
        'train': {'mae': train_mae, 'rmse': train_rmse, 'mape': train_mape},
        'test': {'mae': test_mae, 'rmse': test_rmse, 'mape': test_mape}
    }


def main():
    """主函数"""
    print("=== TCN模型交互式预测可视化 ===")

    # 检查模型文件
    model_path = "../models/best_tcn.pth"
    if not os.path.exists(model_path):
        print(f"❌ 未找到模型文件: {model_path}")
        print("请先运行 train_tcn.py 训练模型")
        return

    try:
        # 加载数据
        X_traffic, X_time, y, traffic_scaler, time_scaler, traffic_feat_dim, time_feat_dim, df_raw = load_full_data()
        num_stations = y.shape[2]

        # 加载模型
        model = TCNForecast(
            traffic_feat_dim=traffic_feat_dim,
            time_feat_dim=time_feat_dim,
            num_stations=num_stations,
            num_channels=TCN_CHANNELS,
            kernel_size=KERNEL_SIZE,
            dropout=DROPOUT,
            pred_len=PRED_LEN
        ).to(DEVICE)

        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print("✅ 模型加载成功!")

        # 获取站点名称
        station_names = get_station_names(df_raw)
        print(f"找到 {len(station_names)} 个监测站点")

        # 对整个数据集进行预测
        full_predictions = create_full_predictions(model, X_traffic, X_time, traffic_scaler)

        # 用户选择站点
        while True:
            print("\n可选站点:")
            for i, name in enumerate(station_names[:20]):  # 只显示前20个以免太多
                print(f"  {i}: {name}")
            print("  ... (更多站点)")

            try:
                station_input = input(f"\n请选择站点编号 (0-{len(station_names) - 1}) 或输入站点名称: ").strip()

                if station_input.isdigit():
                    station_idx = int(station_input)
                    if 0 <= station_idx < len(station_names):
                        station_name = station_names[station_idx]
                        break
                    else:
                        print(f"❌ 编号超出范围，请输入 0-{len(station_names) - 1} 之间的数字")
                else:
                    # 尝试按名称查找
                    if station_input in station_names:
                        station_idx = station_names.index(station_input)
                        station_name = station_input
                        break
                    else:
                        print("❌ 未找到该站点名称")

            except (ValueError, IndexError):
                print("❌ 输入无效，请重新输入")

        print(f"正在生成 {station_name} 的交互式图表...")

        # 准备计算指标的数据
        traffic_data_log = np.log1p(df_raw.values)
        traffic_data_scaled = traffic_scaler.transform(traffic_data_log)
        traffic_data_restored = traffic_scaler.inverse_transform(traffic_data_scaled)
        true_values_orig = np.expm1(traffic_data_restored)

        start_idx = HISTORY_LEN
        true_vals = true_values_orig[start_idx:start_idx + len(full_predictions), station_idx]
        pred_vals = full_predictions[:, 0, station_idx]
        split_idx = int(len(full_predictions) * SPLIT_RATIO)

        # 计算并显示指标
        metrics = calculate_model_metrics(true_vals, pred_vals, split_idx)

        # 询问是否生成未来预测
        generate_future = input("\n是否生成未来两个月的预测？(y/n): ").lower().strip()
        future_times = None
        future_predictions = None

        if generate_future == 'y':
            future_times, future_predictions = generate_future_predictions(
                model, df_raw, traffic_scaler, time_scaler,
                traffic_feat_dim, time_feat_dim, station_idx,
                future_days=60
            )
            print("✅ 未来预测生成完成！")

        # 创建交互式图表
        fig = create_interactive_plot(
            df_raw, full_predictions, station_idx, station_name,
            future_times, future_predictions
        )

        # 显示图表
        fig.show()

        # 可选：保存为HTML文件
        save_html = input("\n是否保存为HTML文件？(y/n): ").lower().strip()
        if save_html == 'y':
            html_path = f"../experiments/interactive_plot_{station_name}.html"
            fig.write_html(html_path)
            print(f"✅ 图表已保存至: {html_path}")

        print("\n✅ 可视化完成！")
        print("💡 在图表中你可以:")
        print("   - 使用鼠标滚轮缩放")
        print("   - 拖拽平移")
        print("   - 双击重置视图")
        print("   - 使用左上角的时间范围选择器")
        print("   - 鼠标悬停查看具体数值")
        print("   - 查看训练集(浅绿)、测试集(浅红)和未来预测(浅蓝)区域")

    except Exception as e:
        print(f"❌ 程序执行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()