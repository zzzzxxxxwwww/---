import torch
import numpy as np
import matplotlib.pyplot as plt
from model_lstm import LSTMForecast
from data_utils import load_and_preprocess_data
from sklearn.preprocessing import MinMaxScaler

# =================== 参数配置 ===================
DATA_PATH = "../data/milano_traffic_nid.csv"
MODEL_PATH = "../models/lstm_forecast.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =================== 数据加载 ===================
data = load_and_preprocess_data(DATA_PATH)
X, y, scaler, site_names = data
num_stations = X.shape[-1]

print(f"X形状: {X.shape}")  # 应该是 (样本数, 时间步长, 站点数)
print(f"y形状: {y.shape}")  # 应该是 (样本数, 站点数) 或 (样本数, 预测步长, 站点数)
print(f"站点数量: {num_stations}")
print(f"站点名称: {site_names[:5]}...")  # 显示前5个站点名称

# =================== 模型加载 ===================
model = LSTMForecast(num_stations=num_stations).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model.eval()

# =================== 可视化单个站点 ===================
site_index = 0
site_name = site_names[site_index]

# 取最后100个样本进行预测
X_sample = torch.tensor(X[-100:], dtype=torch.float32).to(DEVICE)

# 根据y的实际形状获取真实值
if len(y.shape) == 2:  # y形状: (样本数, 站点数)
    y_true = y[-100:, site_index]  # 形状: (100,)
elif len(y.shape) == 3:  # y形状: (样本数, 预测步长, 站点数)
    y_true = y[-100:, 0, site_index]  # 取第一个预测步长，形状: (100,)
else:
    raise ValueError(f"未知的y形状: {y.shape}")

print(f"y_true形状: {y_true.shape}")

with torch.no_grad():
    y_pred_full = model(X_sample).cpu().numpy()  # 获取所有站点的预测

# 根据模型输出的形状获取目标站点的预测
if len(y_pred_full.shape) == 2:  # 形状: (样本数, 站点数)
    y_pred = y_pred_full[:, site_index]  # 形状: (100,)
elif len(y_pred_full.shape) == 3:  # 形状: (样本数, 预测步长, 站点数)
    y_pred = y_pred_full[:, 0, site_index]  # 取第一个预测步长，形状: (100,)
else:
    raise ValueError(f"未知的预测输出形状: {y_pred_full.shape}")

print(f"y_pred形状: {y_pred.shape}")

# 反归一化处理
# 方法1: 如果scaler支持单变量反归一化
try:
    # 尝试直接反归一化单变量数据
    y_true_inv = scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()
    y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
except:
    # 方法2: 创建完整的多变量数组进行反归一化
    y_true_full = np.zeros((len(y_true), num_stations))
    y_pred_full_array = np.zeros((len(y_pred), num_stations))

    y_true_full[:, site_index] = y_true
    y_pred_full_array[:, site_index] = y_pred

    y_true_inv = scaler.inverse_transform(y_true_full)[:, site_index]
    y_pred_inv = scaler.inverse_transform(y_pred_full_array)[:, site_index]

print(f"反归一化后 - y_true_inv形状: {y_true_inv.shape}, y_pred_inv形状: {y_pred_inv.shape}")

# =================== 绘图 ===================
plt.figure(figsize=(12, 6))
plt.plot(y_true_inv, label="真实值", color="blue", linewidth=1.5, alpha=0.8)
plt.plot(y_pred_inv, label="预测值", color="red", linewidth=1.5, linestyle='--', alpha=0.8)
plt.title(f"交通流量预测对比 - {site_name}", fontproperties="SimHei", fontsize=16)
plt.xlabel("时间步", fontproperties="SimHei", fontsize=12)
plt.ylabel("交通流量", fontproperties="SimHei", fontsize=12)
plt.legend(prop={"family": "SimHei"})
plt.grid(True, alpha=0.3)

# 添加一些统计信息
mse = np.mean((y_true_inv - y_pred_inv) ** 2)
mae = np.mean(np.abs(y_true_inv - y_pred_inv))
plt.text(0.02, 0.98, f'MSE: {mse:.2f}\nMAE: {mae:.2f}',
         transform=plt.gca().transAxes, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.show()

# 打印预测效果统计
print(f"\n预测效果统计 - {site_name}:")
print(f"均方误差 (MSE): {mse:.2f}")
print(f"平均绝对误差 (MAE): {mae:.2f}")
print(f"真实值范围: {y_true_inv.min():.2f} - {y_true_inv.max():.2f}")
print(f"预测值范围: {y_pred_inv.min():.2f} - {y_pred_inv.max():.2f}")