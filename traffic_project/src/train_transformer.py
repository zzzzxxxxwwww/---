import torch
import torch.nn as nn
import torch.optim as optim
from data_utils import load_and_preprocess
from dataset import create_dataloaders
from model_transformer import TransformerForecast
import matplotlib.pyplot as plt
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# 1️⃣ 数据加载
X, y, scaler = load_and_preprocess("../data/milano_traffic_nid.csv")
train_loader, test_loader = create_dataloaders(X, y, batch_size=64)

# 2️⃣ 初始化模型
num_stations = X.shape[2]
model = TransformerForecast(num_stations=num_stations).to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 3️⃣ 训练
EPOCHS = 20
train_losses, test_losses = [], []

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        pred = model(xb)
        loss = criterion(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    train_loss = total_loss / len(train_loader)

    # 验证
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred = model(xb)
            test_loss += criterion(pred, yb).item()
    test_loss /= len(test_loader)

    train_losses.append(train_loss)
    test_losses.append(test_loss)

    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.6f} | Test Loss: {test_loss:.6f}")

# 4️⃣ 保存模型
os.makedirs("../models", exist_ok=True)
torch.save(model.state_dict(), "../models/transformer_forecast.pth")
print("✅ Transformer 模型已保存到 models/transformer_forecast.pth")

# 5️⃣ 可视化
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.legend()
plt.title("Transformer Training Curve")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.show()
