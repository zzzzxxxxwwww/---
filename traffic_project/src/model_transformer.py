import torch
import torch.nn as nn


class TransformerForecast(nn.Module):
    def __init__(self, num_stations, d_model=128, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(num_stations, d_model)  # 输入映射到 embedding
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, num_stations)

    def forward(self, x):
        # x: (batch, seq_len, num_stations)
        x = self.input_proj(x)  # (batch, seq_len, d_model)
        enc_out = self.encoder(x)  # (batch, seq_len, d_model)
        last_out = enc_out[:, -1, :]  # 取最后时间步的特征
        pred = self.fc_out(last_out)  # (batch, num_stations)
        return pred.unsqueeze(1)  # (batch, 1, num_stations)
