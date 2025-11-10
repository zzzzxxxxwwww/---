# -*- coding: utf-8 -*-
"""
训练与评估脚本（中文注释）
说明：
 - 先把数据载入（TrafficDatasetSimple）
 - 构造 PyTorch DataLoader（把 X, Y 转为 tensor）
 - 定义模型（STTransformer）并训练若干 epoch
 - 每个 epoch 后在验证集上评估 RMSE / MAE / MAPE
 - 保存最佳 checkpoint（按 val loss）
"""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from dataset_loader import TrafficDatasetSimple
from model_sttransformer import STTransformer
import json
import math

# 可配置超参（你可以修改）
DATA_DIR = "../data/preprocessed_for_st"
CITY = "milano"           # milano 或 trentino
WINDOW = 12               # 历史窗口（12 * 10min = 2小时）
HORIZON = 6               # 预测步数（6 * 10min = 1小时）
BATCH_SIZE = 32
LR = 1e-3
EPOCHS = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR = "./checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

# 评估指标
def mae(pred, true):
    return float(torch.mean(torch.abs(pred - true)).cpu().item())

def rmse(pred, true):
    return float(torch.sqrt(torch.mean((pred - true)**2)).cpu().item())

def mape(pred, true, eps=1e-6):
    return float(torch.mean(torch.abs((true - pred) / (true + eps))).cpu().item() * 100.0)

# 准备数据
print("加载数据...")
ds = TrafficDatasetSimple(data_dir=DATA_DIR, city=CITY, window=WINDOW, horizon=HORIZON)
X_train, Y_train = ds.get_train()
X_val, Y_val = ds.get_val()
X_test, Y_test = ds.get_test()
print("train samples:", None if X_train is None else X_train.shape[0])

# 转成 TensorDataset
def to_loader(X, Y, batch_size, shuffle=True):
    if X is None:
        return None
    Xt = torch.tensor(X, dtype=torch.float32)
    Yt = torch.tensor(Y, dtype=torch.float32)
    ds = TensorDataset(Xt, Yt)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
    return loader

train_loader = to_loader(X_train, Y_train, BATCH_SIZE, shuffle=True)
val_loader = to_loader(X_val, Y_val, BATCH_SIZE, shuffle=False)
test_loader = to_loader(X_test, Y_test, BATCH_SIZE, shuffle=False)

# 读取邻接矩阵（N x N）
adj_path = os.path.join(DATA_DIR, f"{CITY}_combined_adj.npy")
adj = np.load(adj_path)
N = adj.shape[0]
print(f"节点数 N = {N}")

# 构建模型
model = STTransformer(N=N, in_channels=1, d_model=64, nhead=4, num_layers=2, dropout=0.1, horizon=HORIZON)
model = model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
criterion = nn.MSELoss()

best_val_loss = float('inf')
best_ckpt = None

for epoch in range(1, EPOCHS+1):
    model.train()
    total_loss = 0.0
    n_batches = 0
    for Xb, Yb in train_loader:
        Xb = Xb.to(DEVICE)  # (B, window, N)
        Yb = Yb.to(DEVICE)  # (B, horizon, N)
        optimizer.zero_grad()
        preds = model(Xb, adj)  # (B, horizon, N)
        loss = criterion(preds, Yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    train_loss = total_loss / max(1, n_batches)

    # 验证
    model.eval()
    with torch.no_grad():
        total_vloss = 0.0
        n_v = 0
        maes, rmses, mapes = [], [], []
        for Xv, Yv in val_loader:
            Xv = Xv.to(DEVICE); Yv = Yv.to(DEVICE)
            preds = model(Xv, adj)
            vloss = criterion(preds, Yv).item()
            total_vloss += vloss; n_v += 1
            maes.append(mae(preds, Yv))
            rmses.append(rmse(preds, Yv))
            mapes.append(mape(preds, Yv))
        val_loss = total_vloss / max(1, n_v)
        mean_mae = np.mean(maes) if len(maes)>0 else float('nan')
        mean_rmse = np.mean(rmses) if len(rmses)>0 else float('nan')
        mean_mape = np.mean(mapes) if len(mapes)>0 else float('nan')

    print(f"Epoch {epoch}/{EPOCHS} | train_loss: {train_loss:.6f} | val_loss: {val_loss:.6f} | val_mae: {mean_mae:.6f} | val_rmse: {mean_rmse:.6f} | val_mape: {mean_mape:.3f}%")

    # 保存最优模型
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_ckpt = os.path.join(SAVE_DIR, f"best_{CITY}.pth")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optim_state_dict": optimizer.state_dict(),
            "val_loss": val_loss
        }, best_ckpt)
        print("保存最佳模型到:", best_ckpt)

# 测试集评估（载入最佳模型）
if best_ckpt is not None:
    ckpt = torch.load(best_ckpt, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    all_mae, all_rmse, all_mape = [], [], []
    with torch.no_grad():
        for Xt, Yt in test_loader:
            Xt = Xt.to(DEVICE); Yt = Yt.to(DEVICE)
            preds = model(Xt, adj)
            all_mae.append(mae(preds, Yt)); all_rmse.append(rmse(preds, Yt)); all_mape.append(mape(preds, Yt))
    print("测试集结果 -> MAE:", np.mean(all_mae), "RMSE:", np.mean(all_rmse), "MAPE(%):", np.mean(all_mape))
else:
    print("未找到最佳模型 checkpoint，无法进行测试评估。")
