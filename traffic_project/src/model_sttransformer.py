# -*- coding: utf-8 -*-
"""
简化的 ST-Transformer 模型（中文注释）
结构说明（尽量保持简单、易运行）：
 - 时间编码（可选）
 - 每时间步的节点特征通过线性投影变为 d_model 维
 - Transformer Encoder 进行时间维度的自注意力（同时保留节点维）
 - 空间注意力：使用邻接矩阵做简单的图卷积（GNN-like aggregator）
 - 最终解码为 horizon 步的回归输出
此实现为教育/示例用途，训练速度与性能可接受，易于理解与调参。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleGraphConv(nn.Module):
    """基于邻接矩阵的简单图卷积（行归一化权重）"""
    def __init__(self):
        super().__init__()

    def forward(self, x, adj):
        # x: (B, T, N, C)
        # adj: (N, N)
        # 先将 adj 转为 tensor（在训练中建议传入 device）
        # 输出维度同 x
        A = adj
        # row-normalize
        row_sum = A.sum(axis=1, keepdims=True) + 1e-6
        A_norm = A / row_sum
        # 使用矩阵乘法（N x N 与 N x C）
        # 做 batch 运算
        B, T, N, C = x.shape
        x2 = x.reshape(B*T, N, C)  # (B*T, N, C)
        A_t = torch.from_numpy(A_norm).to(x.device).float()
        out = torch.matmul(A_t, x2)  # (B*T, N, C)
        out = out.view(B, T, N, C)
        return out

class STTransformer(nn.Module):
    def __init__(self, N, in_channels=1, d_model=64, nhead=4, num_layers=2, dropout=0.1, horizon=6):
        """
        N: 节点数量
        in_channels: 每个节点每时间步的输入通道（一般为1 流量）
        d_model: Transformer 维度
        nhead: multi-head
        num_layers: encoder 层数
        horizon: 预测步数
        """
        super().__init__()
        self.N = N
        self.d_model = d_model
        self.in_proj = nn.Linear(in_channels, d_model)  # 将节点特征投影到 d_model
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model*4, dropout=dropout, batch_first=True)
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.graph_conv = SimpleGraphConv()
        # 交替使用时间编码与空间聚合若干次：下面简化为 single-layer temporal + graph + temporal
        self.temporal_encoder2 = nn.TransformerEncoder(encoder_layer, num_layers=1)
        # decoder：把每个节点的 d_model 映射到 horizon 个值（回归）
        self.out_proj = nn.Linear(d_model, horizon)  # 对每个节点输出 horizon 步
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        """
        x: (B, window, N)  注意输入为标量流量，没有 channel 维
        adj: numpy array (N, N) 或 tensor
        返回: preds (B, horizon, N)
        """
        B, T, N = x.shape
        device = x.device
        # 先把节点 channel 维补上
        x = x.unsqueeze(-1)  # (B, T, N, 1)
        # 将节点维合并到最后并投影
        # 先把 (B, T, N, C) -> (B*T*N, C) 然后 linear，再 reshape
        x_proj = self.in_proj(x)  # (B, T, N, d_model)
        # 我们想在时间维做 self-attention：需要形状 (B*N, T, d_model)
        x_time = x_proj.permute(0,2,1,3).contiguous()  # (B, N, T, d_model)
        x_time = x_time.view(B*N, T, self.d_model)      # (B*N, T, d_model)
        # temporal encoder
        x_time_enc = self.temporal_encoder(x_time)      # (B*N, T, d_model)
        # 恢复形状 (B, T, N, d_model)
        x_time_enc = x_time_enc.view(B, N, T, self.d_model).permute(0,2,1,3).contiguous()
        # 空间图卷积（在节点维度上）
        x_spatial = self.graph_conv(x_time_enc, adj)   # (B, T, N, d_model)
        # 再次在时间维编码（增强时间依赖）
        x_time2 = x_spatial.permute(0,2,1,3).contiguous().view(B*N, T, self.d_model)
        x_time2 = self.temporal_encoder2(x_time2)
        x_time2 = x_time2.view(B, N, T, self.d_model).permute(0,2,1,3).contiguous()  # (B, T, N, d_model)
        # 对时间维做聚合（例如取最后一步特征，或平均）
        feat = x_time2[:, -1, :, :]  # (B, N, d_model) 取最后时间步
        feat = self.dropout(feat)
        # 输出每个节点的 horizon 个值
        out = self.out_proj(feat)    # (B, N, horizon)
        out = out.permute(0,2,1).contiguous()  # (B, horizon, N)
        return out
