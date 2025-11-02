# PyTorch Dataset/ DataLoader 封装
# src/dataset.py
# dataset.py
import torch
from torch.utils.data import Dataset, DataLoader


class TrafficDataset(Dataset):
    def __init__(self, X_traffic, X_time, y):
        self.X_traffic = torch.tensor(X_traffic, dtype=torch.float32)
        self.X_time = torch.tensor(X_time, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X_traffic)

    def __getitem__(self, idx):
        return self.X_traffic[idx], self.X_time[idx], self.y[idx]


# 🔴 关键修复：参数顺序改为「X_traffic, X_time, y, batch_size, split_ratio」
# 默认参数（batch_size、split_ratio）放在最后，避免位置传参冲突
def create_dataloaders(X_traffic, X_time, y, batch_size=32, split_ratio=0.8):
    split = int(len(X_traffic) * split_ratio)
    train_dataset = TrafficDataset(X_traffic[:split], X_time[:split], y[:split])
    test_dataset = TrafficDataset(X_traffic[split:], X_time[split:], y[split:])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader
