import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    """
    多头注意力机制
    attention(Q,K,V) = softmax(QK^T/sqrt(d_k))V
    MHA(Q,K,V) = Concat(head_1,...,head_h)W^O
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # 线性变换层
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        缩放点积注意力
        Args:
            Q, K, V: [batch_size, n_heads, seq_len, d_k]
            mask: [batch_size, 1, seq_len, seq_len] 或 [batch_size, 1, 1, seq_len]
        """
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 应用掩码
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax归一化
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 计算输出
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights
    
    def forward(self, query, key, value, mask=None):
        """
        Args:
            query, key, value: [seq_len, batch_size, d_model]
            mask: 注意力掩码
        Returns:
            output: [seq_len, batch_size, d_model]
            attention_weights: [batch_size, n_heads, seq_len, seq_len]
        """
        batch_size = query.size(1)
        query_len = query.size(0)
        key_len = key.size(0)
        
        # 转换维度: [seq_len, batch_size, d_model] -> [batch_size, seq_len, d_model]
        query = query.transpose(0, 1)
        key = key.transpose(0, 1)
        value = value.transpose(0, 1)
        
        # 线性变换得到Q, K, V
        Q = self.w_q(query)  # [batch_size, query_len, d_model]
        K = self.w_k(key)    # [batch_size, key_len, d_model]
        V = self.w_v(value)  # [batch_size, key_len, d_model]
        
        # 重塑为多头形式: [batch_size, seq_len, d_model] -> [batch_size, n_heads, seq_len, d_k]
        Q = Q.view(batch_size, query_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, key_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, key_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # 应用缩放点积注意力
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # 合并多头: [batch_size, n_heads, query_len, d_k] -> [batch_size, query_len, d_model]
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, query_len, self.d_model
        )
        
        # 最终线性变换
        output = self.w_o(attention_output)
        
        # 转换回原始维度: [batch_size, query_len, d_model] -> [query_len, batch_size, d_model]
        output = output.transpose(0, 1)
        
        return output, attention_weights


def create_padding_mask(seq, pad_token=0):
    """
    创建padding掩码
    Args:
        seq: [batch_size, seq_len]
        pad_token: padding token的值
    Returns:
        mask: [batch_size, 1, 1, seq_len]
    """
    mask = (seq != pad_token).unsqueeze(1).unsqueeze(2)
    return mask


def create_look_ahead_mask(size, device=None):
    """
    创建前瞻掩码(用于Decoder的掩码自注意力)
    Args:
        size: 序列长度
        device: 设备
    Returns:
        mask: [1, 1, size, size] 下三角矩阵
    """
    mask = torch.tril(torch.ones(size, size, device=device)).unsqueeze(0).unsqueeze(0)
    return mask


def create_masks(src, tgt, pad_token=0):
    """
    创建训练时需要的所有掩码
    Args:
        src: 源序列 [batch_size, src_len]
        tgt: 目标序列 [batch_size, tgt_len]
        pad_token: padding token
    Returns:
        src_mask: encoder掩码
        tgt_mask: decoder self-attention掩码
        src_tgt_mask: decoder cross-attention掩码
    """
    device = src.device
    
    # Encoder掩码 (只需要padding掩码)
    src_mask = create_padding_mask(src, pad_token)
    
    # Decoder self-attention掩码 (padding + look-ahead)
    tgt_padding_mask = create_padding_mask(tgt, pad_token)
    tgt_look_ahead_mask = create_look_ahead_mask(tgt.size(1), device)
    tgt_mask = torch.minimum(tgt_padding_mask, tgt_look_ahead_mask)
    
    # Decoder cross-attention掩码 (只需要源序列的padding掩码)
    src_tgt_mask = create_padding_mask(src, pad_token)
    
    return src_mask, tgt_mask, src_tgt_mask