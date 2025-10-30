import torch
import matplotlib.pyplot as plt
import numpy as np
from model_transformer import TransformerForecast
from data_utils import load_and_preprocess_data
import matplotlib.pyplot as plt

# ======= 配置 =======
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "../models/transformer_forecast.pth"
DATA_PATH = "../data/milano_traffic_nid.csv"
SITE_NAME = "AFFORI"   # 👉 修改为你想看的站点名称（例如 "ADRIANO"）

# ======= 加载数据 =======
X, y, scaler, site_names = load_and_preprocess_data(DATA_PATH)
site_index = site_names.index(SITE_NAME)

# ======= 加载模型 =======
input_dim = X.shape[2]
model = TransformerForecast(num_stations=input_dim).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ======= 预测 =======
with torch.no_grad():
    preds = model(torch.tensor(X, dtype=torch.float32).to(DEVICE)).cpu().numpy()

# ======= 取目标站点的真实值与预测值 =======
y_true = y[:, 0, site_index]
y_pred = preds[:, 0, site_index]

# ======= 反归一化 =======
# 只反归一化当前选定站点
min_val = scaler.data_min_[site_index]
max_val = scaler.data_max_[site_index]

y_true = y_true.flatten() * (max_val - min_val) + min_val
y_pred = y_pred.flatten() * (max_val - min_val) + min_val


# ======= 可视化 =======
plt.rcParams['font.sans-serif'] = ['SimHei']       # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False          # 解决负号显示问题
plt.figure(figsize=(10, 5))
plt.plot(y_true[:500], label="真实值", color="blue", linewidth=2)   # 蓝色实线
plt.plot(y_pred[:500], label="预测值", color="red", linewidth=2)     # 红色实线
plt.title(f"交通流量预测对比图（{site_names[site_index]} 站点）")
plt.xlabel("时间步")
plt.ylabel("交通流量")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
