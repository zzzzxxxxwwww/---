# src/model_tcn.py
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

"""
TCN (时间卷积网络) 模型
专门用于捕捉局部波动和高频模式，同时通过空洞卷积(dilation)
来覆盖长的历史窗口 (history_len=144)。
"""


class Chomp1d(nn.Module):
    """
    一个简单的辅助模块，用于"裁剪"掉卷积层末尾多余的padding。
    确保因果卷积的输出长度与输入长度一致。
    """

    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        # x.shape: (batch_size, channels, length)
        # 裁剪掉 length 维度的最后 chomp_size 个元素
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """
    TCN 的核心：一个残差块 (Residual Block)
    包含两层空洞因果卷积 (Dilated Causal Conv)。
    """

    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()

        # 第一层卷积
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # 第二层卷积
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # 完整的卷积-激活-裁剪 序列
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)

        # 残差连接 (Residual Connection)
        # 如果输入/输出通道数不同，使用一个 1x1 卷积来匹配维度
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self._init_weights()

    def _init_weights(self):
        # 初始化权重
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        # x.shape: (batch, n_inputs, length)
        out = self.net(x)
        # (x + out) 是残差连接
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNForecast(nn.Module):
    """
    最终的 TCN 预测模型
    """

    def __init__(self, traffic_feat_dim, time_feat_dim, num_stations,
                 num_channels=[128, 128, 128, 128], kernel_size=3, dropout=0.3, pred_len=1):
        super(TCNForecast, self).__init__()
        self.pred_len = pred_len
        self.num_stations = num_stations

        # 1. 时间特征嵌入
        self.time_emb_dim = 32
        self.time_emb = nn.Sequential(
            nn.Linear(time_feat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.time_emb_dim)
        )

        # 2. TCN 网络
        layers = []
        # TCN 的输入通道 = 交通特征数 + 时间嵌入维度
        num_inputs = traffic_feat_dim + self.time_emb_dim

        # 🔴 关键修复：正确管理通道数变化
        for i, num_outputs in enumerate(num_channels):
            # 空洞(dilation)大小按 2 的幂指数增长
            dilation_size = 2 ** i

            # TCN 的 padding 必须是 (kernel_size-1) * dilation_size
            padding_size = (kernel_size - 1) * dilation_size

            layers.append(
                TemporalBlock(
                    num_inputs, num_outputs, kernel_size,
                    stride=1, dilation=dilation_size,
                    padding=padding_size, dropout=dropout
                )
            )
            # 🔴 关键修复：更新下一层的输入通道数为当前层的输出通道数
            num_inputs = num_outputs

        self.tcn_network = nn.Sequential(*layers)

        # 3. 输出层
        # TCN 网络的最终输出通道数是 num_channels[-1]
        self.last_channel_size = num_channels[-1]

        # 我们用一个线性层来处理 TCN 的最后一个时间步的输出
        self.output_layers = nn.Sequential(
            nn.Linear(self.last_channel_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, pred_len * num_stations)
        )

    def forward(self, traffic_x, time_x):
        # traffic_x: (batch, seq_len, traffic_feat_dim)
        # time_x:    (batch, seq_len, time_feat_dim)

        # 1. 处理时间特征
        time_emb_out = self.time_emb(time_x)  # (batch, seq_len, time_emb_dim)

        # 2. 拼接特征
        x = torch.cat([traffic_x, time_emb_out], dim=2)  # (batch, seq_len, channels_in)

        # 3. 🔴 关键：TCN (Conv1d) 需要 (batch, channels, seq_len)
        #    我们必须进行维度转换 (permute)
        x = x.permute(0, 2, 1)  # (batch, channels_in, seq_len)

        # 4. 通过 TCN 网络
        tcn_out = self.tcn_network(x)  # (batch, channels_out, seq_len)

        # 5. 🔴 关键：我们只取最后一个时间步的输出 (:, :, -1)
        #    来代表 TCN 对整个历史的"总结"
        tcn_last_step = tcn_out[:, :, -1]  # (batch, channels_out)

        # 6. 通过输出层
        pred = self.output_layers(tcn_last_step)  # (batch, pred_len * num_stations)

        # 7. 重塑形状以匹配 y (target)
        pred = pred.view(-1, self.pred_len, self.num_stations)  # (batch, pred_len, num_stations)

        return pred