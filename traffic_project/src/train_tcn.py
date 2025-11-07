# src/train_tcn.py
import torch
import torch.nn as nn
import torch.optim as optim
from data_utils import load_and_preprocess
from dataset import create_dataloaders
from model_tcn import TCNForecast  # 🔴 关键：导入新模型
import matplotlib.pyplot as plt
import os
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random # [!!! 1. 新增导入 !!!]

# ... (中文字体设置代码 保持不变) ...
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'PingFang SC', 'Source Han Sans SC',
                                       'WenQuanYi Micro Hei']
    plt.rcParams['axes.unicode_minus'] = False
    print("✅ 已设置中文字体。")
except Exception as e:
    print(f"⚠️ 设置中文字体失败: {e}。图表中的中文可能显示为方框。")

# # [!!! 2. 新增：可复现性设置 !!!]
# SEED = 42
#
# def set_seed(seed):
#     """设置所有随机种子以保证可复现性"""
#     random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)
#     # 强制 PyTorch 使用确定性的 cuDNN 算法
#     torch.backends.cudnn.deterministic = True
#     # 禁用 cuDNN 自动调优，它会引入随机性
#     torch.backends.cudnn.benchmark = False
#     print(f"✅ 所有随机种子已设置为: {seed}")
#
# set_seed(SEED)
# # [!!! 结束修改 !!!]


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# [!!! 关键：确保 TCN 训练速度最快 !!!]
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False # 确保关闭确定性
    print("✅ cuDNN Benchmark 已启用 (TCN 训练速度最优化)")

# TCN 的参数
HISTORY_LEN = 144
PRED_LEN = 6
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-3  # 🔴 TCN 对学习率更敏感，我们用 1e-3 开始
WEIGHT_DECAY = 1e-4
KERNEL_SIZE = 5  # TCN 卷积核大小
DROPOUT = 0.3
# TCN 层级和通道数，例如 4 层，每层 128 通道,修改为七层，为了匹配感受野与历史长度
TCN_CHANNELS = [128, 128, 128, 128, 128, 128, 128 ]
# TCN_CHANNELS = [128, 128, 128, 128 ]



def calculate_mape(y_t, y_p):
    return torch.mean(torch.abs((y_t - y_p) / (torch.abs(y_t) + 1e-8))) * 100


def train_tcn():
    try:
        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("正在加载数据 (应用对数变换)...")
        X_traffic, X_time, y, traffic_scaler, time_scaler, traffic_feat_dim, time_feat_dim = load_and_preprocess(
            "../data/milano_traffic_nid.csv",
            "../experiments/traffic_scaler.pkl",
            "../experiments/time_scaler.pkl",
            history_len=HISTORY_LEN,
            pred_len=PRED_LEN
        )
        print(f"数据加载成功: X_traffic.shape={X_traffic.shape}, X_time.shape={X_time.shape}, y.shape={y.shape}")

        num_stations = y.shape[2]
        print(f"交通特征维度: {traffic_feat_dim}, 时间特征维度: {time_feat_dim}, 输出站点数: {num_stations}")

        train_loader, val_loader = create_dataloaders(
            X_traffic, X_time, y, batch_size=BATCH_SIZE, split_ratio=0.8
        )
        print(f"创建Dataloaders: Train batches={len(train_loader)}, Val batches={len(val_loader)}")

        # 🔴 关键：初始化 TCNForecast 模型
        model = TCNForecast(
            traffic_feat_dim=traffic_feat_dim,
            time_feat_dim=time_feat_dim,
            num_stations=num_stations,
            num_channels=TCN_CHANNELS,
            kernel_size=KERNEL_SIZE,
            dropout=DROPOUT,
            pred_len=PRED_LEN
        ).to(DEVICE)

        print("模型结构 (TCN):")
        print(model)

        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        loss_fn = nn.HuberLoss(delta=1.0)  # 仍在对数空间计算 Loss
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, verbose=True)

        history = {'train_loss': [], 'val_loss': [], 'val_mape': []}
        best_mape = float('inf')

        print(f"--- 开始训练 TCN (Epochs: {EPOCHS}, LR: {LEARNING_RATE}, K: {KERNEL_SIZE}) ---")

        for epoch in range(EPOCHS):
            model.train()
            train_losses = []
            for xb_traffic, xb_time, yb in train_loader:
                xb_traffic, xb_time, yb = xb_traffic.to(DEVICE), xb_time.to(DEVICE), yb.to(DEVICE)

                optimizer.zero_grad()
                pred = model(xb_traffic, xb_time)
                loss = loss_fn(pred, yb)

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_losses.append(loss.item())

            epoch_train_loss = np.mean(train_losses)
            history['train_loss'].append(epoch_train_loss)

            model.eval()
            val_losses = []
            val_mapes = []
            with torch.no_grad():
                for xb_traffic, xb_time, yb in val_loader:
                    xb_traffic, xb_time, yb = xb_traffic.to(DEVICE), xb_time.to(DEVICE), yb.to(DEVICE)
                    pred = model(xb_traffic, xb_time)

                    val_loss = loss_fn(pred, yb)
                    val_losses.append(val_loss.item())

                    # 🔴 关键：还原 np.log1p
                    yb_true_log = traffic_scaler.inverse_transform(yb.cpu().numpy().reshape(-1, num_stations))
                    yb_pred_log = traffic_scaler.inverse_transform(pred.cpu().numpy().reshape(-1, num_stations))

                    yb_true_orig = np.expm1(yb_true_log.reshape(-1, PRED_LEN, num_stations))
                    yb_pred_orig = np.expm1(yb_pred_log.reshape(-1, PRED_LEN, num_stations))

                    yb_true_orig[yb_true_orig < 0] = 0
                    yb_pred_orig[yb_pred_orig < 0] = 0

                    val_mape = calculate_mape(torch.tensor(yb_true_orig), torch.tensor(yb_pred_orig))
                    val_mapes.append(val_mape.item())

            epoch_val_loss = np.mean(val_losses)
            epoch_val_mape = np.mean(val_mapes)
            history['val_loss'].append(epoch_val_loss)
            history['val_mape'].append(epoch_val_mape)

            scheduler.step(epoch_val_loss)

            print(f"Epoch {epoch + 1}/{EPOCHS} - "
                  f"Train Loss: {epoch_train_loss:.4f} (Log-Space), "
                  f"Val Loss: {epoch_val_loss:.4f} (Log-Space), "
                  f"Val MAPE: {epoch_val_mape:.2f}% (Original-Space)")

            if epoch_val_mape < best_mape:
                best_mape = epoch_val_mape
                # 🔴 关键：保存为 TCN 模型
                torch.save(model.state_dict(), "../models/best_tcn4.pth")
                print(f"  ✨ 新的最佳 TCN 模型 (MAPE: {best_mape:.2f}%) 已保存!")

        torch.save(model.state_dict(), "../models/tcn_final4.pth")
        print(f"✅ 训练完成！最终 TCN 模型已保存，最佳验证MAPE: {best_mape:.2f}%")

        # ... (绘图代码与 train_lstm_improved.py 完全相同，此处省略以保持简洁) ...
        plt.figure(figsize=(15, 10))
        plt.subplot(2, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss (Log-Space)')
        plt.plot(history['val_loss'], label='Validation Loss (Log-Space)')
        plt.title('Loss 曲线 (对数空间)')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 2, 2)
        plt.plot(history['val_mape'], label='Validation MAPE', color='orange')
        plt.title('Validation MAPE (%) (原始空间)')
        plt.legend()
        plt.grid(True)

        model.eval()
        with torch.no_grad():
            for xb_traffic, xb_time, yb in val_loader:
                xb_traffic, xb_time, yb = xb_traffic.to(DEVICE), xb_time.to(DEVICE), yb.to(DEVICE)
                pred = model(xb_traffic, xb_time)

                yb_true_log = traffic_scaler.inverse_transform(yb.cpu().numpy().reshape(-1, num_stations))
                yb_pred_log = traffic_scaler.inverse_transform(pred.cpu().numpy().reshape(-1, num_stations))

                yb_true = np.expm1(yb_true_log.reshape(-1, PRED_LEN, num_stations))
                yb_pred = np.expm1(yb_pred_log.reshape(-1, PRED_LEN, num_stations))

                yb_true[yb_true < 0] = 0
                yb_pred[yb_pred < 0] = 0
                break

        plt.subplot(2, 2, 3)
        station_idx = 0
        plt.plot(yb_true[:50, 0, station_idx], label="Actual (Step 1)", linewidth=2)
        plt.plot(yb_pred[:50, 0, station_idx], label="Predicted (Step 1)", linewidth=2, linestyle="--")
        plt.title(f"Station {station_idx} - 预测 (T+1) (原始空间)")
        plt.xlabel("Time Step")
        plt.ylabel("Traffic Flow")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 2, 4)
        plt.scatter(yb_true.flatten(), yb_pred.flatten(), alpha=0.5, s=1)
        max_val = max(yb_true.max(), yb_pred.max())
        plt.plot([0, max_val], [0, max_val], 'r--', linewidth=2)
        plt.title("Predicted vs Actual (所有 6 个步长, 原始空间)")
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("../experiments/training_results_tcn4.png")  # 🔴 保存为新图片
        print("✅ TCN 训练结果图已保存。")

    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    os.makedirs("../models", exist_ok=True)
    os.makedirs("../experiments", exist_ok=True)
    train_tcn()