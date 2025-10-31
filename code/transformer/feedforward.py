import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    """
    前馈神经网络 (Position-wise Feed-Forward Network)
    FFN(x) = max(0, xW1 + b1)W2 + b2
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: [seq_len, batch_size, d_model]
        Returns:
            output: [seq_len, batch_size, d_model]
        """
        # 第一个线性层 + ReLU激活
        output = F.relu(self.linear1(x))
        # Dropout
        output = self.dropout(output)
        # 第二个线性层
        output = self.linear2(output)
        return output


class LayerNorm(nn.Module):
    """
    层归一化
    LayerNorm(x) = γ * (x - μ) / sqrt(σ² + ε) + β
    """
    def __init__(self, d_model, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.size = d_model
        # 可学习的缩放和偏移参数
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
        
    def forward(self, x):
        """
        Args:
            x: [seq_len, batch_size, d_model]
        Returns:
            output: [seq_len, batch_size, d_model]
        """
        # 在最后一个维度（特征维度）上计算均值和方差
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        
        # 归一化
        normalized = (x - mean) / (std + self.eps)
        
        # 缩放和偏移
        return self.weight * normalized + self.bias


class SubLayerConnection(nn.Module):
    """
    残差连接 + 层归一化
    output = LayerNorm(x + Sublayer(x))
    """
    def __init__(self, d_model, dropout=0.1):
        super(SubLayerConnection, self).__init__()
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, sublayer):
        """
        Args:
            x: 输入张量
            sublayer: 子层函数（attention或feedforward）
        Returns:
            残差连接后的输出
        """
        # 先应用子层，再dropout，最后残差连接和层归一化
        return self.norm(x + self.dropout(sublayer(x)))


# 可以选择使用PyTorch内置的LayerNorm替代自定义实现
# from torch.nn import LayerNorm