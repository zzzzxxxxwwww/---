from data_utils import load_and_preprocess
from dataset import create_dataloaders

X, y, _ = load_and_preprocess("../data/milano_traffic_nid.csv")
train_loader, test_loader = create_dataloaders(X, y, batch_size=32)

# 查看一个batch的维度
for xb, yb in train_loader:
    print("X batch shape:", xb.shape)
    print("y batch shape:", yb.shape)
    break
