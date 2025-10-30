1.项目总体结构：
traffic_project/
├─ data/                    # 放 CSV, geojson（你已经上传的文件）
├─ notebooks/               # 探索性分析（可选）
├─ src/
│  ├─ __init__.py
│  ├─ data_utils.py         # 读数据、预处理、滑窗、scaler 保存/加载
│  ├─ dataset.py            # PyTorch Dataset/ DataLoader 封装
│  ├─ models.py             # LSTM & Transformer 模型类
│  ├─ train.py              # 训练循环、日志、模型保存
│  ├─ eval.py               # 测试/评估脚本、可视化
│  └─ utils.py              # 通用函数（metrics, plot）
├─ experiments/             # 保存训练输出、模型权重、图表
├─ requirements.txt         # 依赖（用于记录）
└─ README.md
