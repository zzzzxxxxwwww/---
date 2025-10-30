import torch
import matplotlib.pyplot as plt
from data_utils import load_and_preprocess
from dataset import create_dataloaders
from model_lstm import LSTMForecast

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# 1️⃣ 加载数据与归一化器
X, y, scaler = load_and_preprocess("../data/milano_traffic_nid.csv")
train_loader, test_loader = create_dataloaders(X, y, batch_size=64)

# 2️⃣ 初始化并加载模型
num_stations = X.shape[2]
model = LSTMForecast(num_stations=num_stations).to(DEVICE)
model.load_state_dict(torch.load("../models/lstm_forecast.pth", map_location=DEVICE))
model.eval()
print("✅ 模型加载完成")

# 3️⃣ 对测试集前几个 batch 预测
xb, yb = next(iter(test_loader))
xb, yb = xb.to(DEVICE), yb.to(DEVICE)
with torch.no_grad():
    pred = model(xb)  # (batch, 1, num_stations)

# 4️⃣ 反归一化
yb_true = scaler.inverse_transform(yb.squeeze(1).cpu().numpy())
yb_pred = scaler.inverse_transform(pred.squeeze(1).cpu().numpy())

# 5️⃣ 可视化（选取一个站点对比）
station_idx = 0  # 比如第 0 个站点
plt.figure(figsize=(10,5))
plt.plot(yb_true[:, station_idx], label="Actual")
plt.plot(yb_pred[:, station_idx], label="Predicted")
plt.title(f"LSTM Prediction - Station {station_idx}")
plt.xlabel("Sample Index")
plt.ylabel("Traffic Flow")
plt.legend()
plt.show()
