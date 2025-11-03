# src/predict_tcn.py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from data_utils import load_and_preprocess
from dataset import create_dataloaders
import os
from model_tcn import TCNForecast  # 🔴 关键：导入新模型

# ... (中文字体设置代码 保持不变) ...
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'PingFang SC', 'Source Han Sans SC',
                                       'WenQuanYi Micro Hei']
    plt.rcParams['axes.unicode_minus'] = False
    print("✅ 已设置中文字体。")
except Exception as e:
    print(f"⚠️ 设置中文字体失败: {e}。图表中的中文可能显示为方框。")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# 保持与训练时一致的参数
HISTORY_LEN = 144
PRED_LEN = 6
BATCH_SIZE = 32
# 🔴 关键：TCN 模型参数
KERNEL_SIZE = 3
DROPOUT = 0.3
TCN_CHANNELS = [128, 128, 128, 128]


def check_model_files():
    # 🔴 关键：加载 TCN 模型
    model_path = "../models/best_tcn.pth"
    if os.path.exists(model_path):
        print(f"✅ 找到模型: {model_path}")
        return model_path
    else:
        print(f"❌ 未找到模型: {model_path}")
        return None


# ... (calculate_metrics 和 plot_predictions 函数保持不变) ...
def calculate_metrics(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100

    print(f"  MAE:  {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAPE: {mape:.2f}%")
    return mae, rmse, mape


def plot_predictions(yb_true, yb_pred, pred_len=6):
    plt.figure(figsize=(18, 12))

    plt.subplot(2, 2, 1)
    station_idx = 0
    step_idx = 0
    plt.plot(yb_true[:100, step_idx, station_idx], label="Actual", linewidth=2)
    plt.plot(yb_pred[:100, step_idx, station_idx], label="Predicted", linewidth=2, linestyle="--")
    plt.title(f"Station {station_idx} - 预测 (T+{step_idx + 1})")
    plt.xlabel("Time Step")
    plt.ylabel("Traffic Flow")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 2)
    station_idx = 0
    step_idx = pred_len - 1
    plt.plot(yb_true[:100, step_idx, station_idx], label="Actual", linewidth=2)
    plt.plot(yb_pred[:100, step_idx, station_idx], label="Predicted", linewidth=2, linestyle="--")
    plt.title(f"Station {station_idx} - 预测 (T+{step_idx + 1})")
    plt.xlabel("Time Step")
    plt.ylabel("Traffic Flow")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 3)
    plt.scatter(yb_true.flatten(), yb_pred.flatten(), alpha=0.3, s=1)
    max_val = max(yb_true.max(), yb_pred.max())
    plt.plot([0, max_val], [0, max_val], 'r--', linewidth=2)
    plt.title("Predicted vs Actual (所有 6 个步长)")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 4)
    step_mapes = []
    for i in range(pred_len):
        step_mape = np.mean(np.abs((yb_true[:, i, :] - yb_pred[:, i, :]) / (np.abs(yb_true[:, i, :]) + 1e-8))) * 100
        step_mapes.append(step_mape)

    plt.bar(range(1, pred_len + 1), step_mapes, color='c')
    plt.title("MAPE by Prediction Step")
    plt.xlabel("Prediction Step (T+n)")
    plt.ylabel("MAPE (%)")
    plt.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig("../experiments/prediction_results_tcn.png")  # 🔴 保存为新图片
    print("✅ TCN 预测结果图已保存。")


def predict_tcn():
    print("检查 TCN 模型文件...")
    model_path = check_model_files()

    if not model_path:
        print("❌ 没有找到 TCN 模型文件，请先运行 train_tcn.py")
        return

    print("加载数据 (应用对数变换)...")
    X_traffic, X_time, y, traffic_scaler, time_scaler, traffic_feat_dim, time_feat_dim = load_and_preprocess(
        "../data/milano_traffic_nid.csv",
        "../experiments/traffic_scaler.pkl",
        "../experiments/time_scaler.pkl",
        history_len=HISTORY_LEN,
        pred_len=PRED_LEN
    )

    num_stations = y.shape[2]

    _, val_loader = create_dataloaders(
        X_traffic, X_time, y, batch_size=BATCH_SIZE, split_ratio=0.8
    )
    print(f"评估数据 (验证集) 加载成功: {len(val_loader)} batches")

    # 🔴 关键：初始化 TCNForecast 模型 (参数必须一致)
    model = TCNForecast(
        traffic_feat_dim=traffic_feat_dim,
        time_feat_dim=time_feat_dim,
        num_stations=num_stations,
        num_channels=TCN_CHANNELS,
        kernel_size=KERNEL_SIZE,
        dropout=DROPOUT,
        pred_len=PRED_LEN
    ).to(DEVICE)

    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print("✅ TCN 模型权重加载成功!")
    except Exception as e:
        print(f"❌ 模型权重加载失败: {e}")
        return

    model.eval()
    all_preds = []
    all_trues = []

    print("正在进行 TCN 预测...")
    with torch.no_grad():
        for xb_traffic, xb_time, yb in val_loader:
            xb_traffic, xb_time = xb_traffic.to(DEVICE), xb_time.to(DEVICE)

            pred = model(xb_traffic, xb_time)

            # 🔴 关键：还原 np.log1p
            yb_true_log = traffic_scaler.inverse_transform(yb.numpy().reshape(-1, num_stations))
            yb_pred_log = traffic_scaler.inverse_transform(pred.cpu().numpy().reshape(-1, num_stations))

            yb_true_orig = np.expm1(yb_true_log.reshape(-1, PRED_LEN, num_stations))
            yb_pred_orig = np.expm1(yb_pred_log.reshape(-1, PRED_LEN, num_stations))

            yb_true_orig[yb_true_orig < 0] = 0
            yb_pred_orig[yb_pred_orig < 0] = 0

            all_trues.append(yb_true_orig)
            all_preds.append(yb_pred_orig)

    yb_true_full = np.concatenate(all_trues, axis=0)
    yb_pred_full = np.concatenate(all_preds, axis=0)

    print("\n--- TCN 整体评估 (所有 6 个步长) ---")
    calculate_metrics(yb_true_full, yb_pred_full)

    print("\n--- TCN 按步长评估 ---")
    for i in range(PRED_LEN):
        print(f" [T+{i + 1}]")
        calculate_metrics(yb_true_full[:, i, :], yb_pred_full[:, i, :])

    print("\n正在绘制 TCN 结果图...")
    plot_predictions(yb_true_full, yb_pred_full, PRED_LEN)


if __name__ == "__main__":
    predict_tcn()