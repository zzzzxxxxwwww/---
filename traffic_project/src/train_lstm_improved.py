# src/train_lstm_improved.py
import torch
import torch.nn as nn
import torch.optim as optim
from data_utils import load_and_preprocess
from dataset import create_dataloaders
from model_lstm import ImprovedLSTMForecast
import matplotlib.pyplot as plt
import os
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau

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

# 保持我们找到的最佳参数
HISTORY_LEN = 144
PRED_LEN = 6
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4
HIDDEN_SIZE = 128


def calculate_mape(y_t, y_p):
    # 🔴 关键：确保在计算MAPE之前数据已被 np.expm1 还原
    return torch.mean(torch.abs((y_t - y_p) / (torch.abs(y_t) + 1e-8))) * 100


def train_improved():
    try:
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

        model = ImprovedLSTMForecast(
            traffic_feat_dim=traffic_feat_dim,
            time_feat_dim=time_feat_dim,
            num_stations=num_stations,
            hidden_size=HIDDEN_SIZE,
            num_layers=2,
            dropout=0.4,
            pred_len=PRED_LEN
        ).to(DEVICE)

        print("模型结构:")
        print(model)

        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        # 🔴 注意：损失函数现在在“对数空间”中计算，这对模型更友好
        loss_fn = nn.HuberLoss(delta=1.0)
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, verbose=True)

        history = {'train_loss': [], 'val_loss': [], 'val_mape': []}
        best_mape = float('inf')

        print(f"--- 开始训练 (Epochs: {EPOCHS}, LR: {LEARNING_RATE}, Hidden: {HIDDEN_SIZE}, Log-Transform: True) ---")

        for epoch in range(EPOCHS):
            model.train()
            train_losses = []
            for xb_traffic, xb_time, yb in train_loader:
                xb_traffic, xb_time, yb = xb_traffic.to(DEVICE), xb_time.to(DEVICE), yb.to(DEVICE)

                optimizer.zero_grad()
                pred = model(xb_traffic, xb_time)
                loss = loss_fn(pred, yb)  # 在对数空间计算Loss

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

                    val_loss = loss_fn(pred, yb)  # 在对数空间计算Loss
                    val_losses.append(val_loss.item())

                    # 🔴 关键修改 1：反归一化 (得到对数值)
                    yb_true_log = traffic_scaler.inverse_transform(yb.cpu().numpy().reshape(-1, num_stations))
                    yb_pred_log = traffic_scaler.inverse_transform(pred.cpu().numpy().reshape(-1, num_stations))

                    # 🔴 关键修改 2：应用 np.expm1 还原到原始空间
                    yb_true_orig = np.expm1(yb_true_log.reshape(-1, PRED_LEN, num_stations))
                    yb_pred_orig = np.expm1(yb_pred_log.reshape(-1, PRED_LEN, num_stations))

                    # 确保非负
                    yb_true_orig[yb_true_orig < 0] = 0
                    yb_pred_orig[yb_pred_orig < 0] = 0

                    # 🔴 关键修改 3：在“原始空间”计算 MAPE
                    val_mape = calculate_mape(torch.tensor(yb_true_orig), torch.tensor(yb_pred_orig))
                    val_mapes.append(val_mape.item())

            epoch_val_loss = np.mean(val_losses)
            epoch_val_mape = np.mean(val_mapes)
            history['val_loss'].append(epoch_val_loss)
            history['val_mape'].append(epoch_val_mape)

            scheduler.step(epoch_val_loss)

            print(f"Epoch {epoch + 1}/{EPOCHS} - "
                  f"Train Loss: {epoch_train_loss:.4f} (Log-Space), "  # 标注 Loss 是在对数空间
                  f"Val Loss: {epoch_val_loss:.4f} (Log-Space), "
                  f"Val MAPE: {epoch_val_mape:.2f}% (Original-Space)")

            if epoch_val_mape < best_mape:
                best_mape = epoch_val_mape
                torch.save(model.state_dict(), "../models/best_lstm_improved.pth")
                print(f"  ✨ 新的最佳模型 (MAPE: {best_mape:.2f}%) 已保存!")

        torch.save(model.state_dict(), "../models/lstm_final_improved.pth")
        print(f"✅ 训练完成！最终模型已保存，最佳验证MAPE: {best_mape:.2f}%")

        # 绘制结果
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

        # 绘制预测 vs 真实
        model.eval()
        with torch.no_grad():
            for xb_traffic, xb_time, yb in val_loader:
                xb_traffic, xb_time, yb = xb_traffic.to(DEVICE), xb_time.to(DEVICE), yb.to(DEVICE)
                pred = model(xb_traffic, xb_time)

                # 🔴 关键修改 4：绘图时也必须还原
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
        plt.savefig("../experiments/training_results_improved3.png")
        print("✅ 训练结果图已保存。")

    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    os.makedirs("../models", exist_ok=True)
    os.makedirs("../experiments", exist_ok=True)
    train_improved()