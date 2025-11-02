# src/model_lstm.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class ImprovedLSTMForecast(nn.Module):
    # 🔴 关键修改 1：
    # - 增加 pred_len=1 作为参数
    # - 保持你设置的良好默认值 (hidden_size=128, num_layers=2, dropout=0.4)
    def __init__(self, traffic_feat_dim, time_feat_dim, num_stations,
                 hidden_size=128, num_layers=2, dropout=0.4, pred_len=1):
        super().__init__()

        self.pred_len = pred_len
        self.num_stations = num_stations

        # 时间特征处理
        self.time_emb = nn.Sequential(
            nn.Linear(time_feat_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32)
        )

        # LSTM层
        self.lstm = nn.LSTM(
            input_size=traffic_feat_dim + 32,  # 交通特征 + 时间嵌入
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False
        )

        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # 🔴 关键修改 2：输出层
        # 必须输出 (pred_len * num_stations) 个值
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, pred_len * num_stations)  # 修改了这里
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0.1)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, traffic_x, time_x):
        # 处理时间特征
        time_emb = self.time_emb(time_x)

        # 拼接特征
        x = torch.cat([traffic_x, time_emb], dim=2)

        # LSTM处理
        lstm_out, (h_n, c_n) = self.lstm(x)  # lstm_out: (batch, seq_len, hidden_size)

        # 注意力机制
        attn_weights = self.attention(lstm_out)  # (batch, seq_len, 1)
        attn_weights = F.softmax(attn_weights, dim=1)

        context_vector = torch.sum(attn_weights * lstm_out, dim=1)  # (batch, hidden_size)

        # 输出层
        pred = self.output_layers(context_vector)  # (batch, pred_len * num_stations)

        # 🔴 关键修改 3：重塑输出形状
        # 将 (batch, pred_len * num_stations) 变为 (batch, pred_len, num_stations)
        # 这对于匹配 y (target) 的形状 (batch, 6, 88) 至关重要
        pred = pred.view(-1, self.pred_len, self.num_stations)

        return pred