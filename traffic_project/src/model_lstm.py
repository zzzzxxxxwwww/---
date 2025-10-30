import torch
import torch.nn as nn

class LSTMForecast(nn.Module):
    def __init__(self, num_stations, hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=num_stations,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, num_stations)

    def forward(self, x):
        # x: (batch, seq_len, num_stations)
        out, _ = self.lstm(x)   # out: (batch, seq_len, hidden)
        last_out = out[:, -1, :]  # 取最后一个时间步
        pred = self.fc(last_out)  # (batch, num_stations)
        return pred.unsqueeze(1)  # (batch, 1, num_stations)
