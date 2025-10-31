import torch
import torch.nn as nn
import math

class TokenEmbedding(nn.Module):
    """
    Token Embedding层
    将词汇索引转换为向量表示
    """
    def __init__(self, vocab_size, d_model):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
    
    def forward(self, x):
        # 根据论文，embedding需要乘以sqrt(d_model)进行缩放
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """
    位置编码层
    使用正弦和余弦函数为序列添加位置信息
    PE(pos,2i) = sin(pos/10000^(2i/d_model))
    PE(pos,2i+1) = cos(pos/10000^(2i/d_model))
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # 计算div_term: 10000^(2i/d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        # 应用正弦和余弦函数
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位置使用sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数位置使用cos
        
        # 添加batch维度并注册为buffer（不参与梯度更新）
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: 输入张量，shape: [seq_len, batch_size, d_model]
        Returns:
            添加位置编码后的张量
        """
        return x + self.pe[:x.size(0), :]


class TransformerEmbedding(nn.Module):
    """
    完整的Transformer Embedding层
    包含Token Embedding + Position Encoding + Dropout
    """
    def __init__(self, vocab_size, d_model, max_len=5000, dropout=0.1):
        super(TransformerEmbedding, self).__init__()
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.position_encoding = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: 输入序列，shape: [batch_size, seq_len]
        Returns:
            嵌入表示，shape: [seq_len, batch_size, d_model]
        """
        # Token embedding
        token_emb = self.token_embedding(x)  # [batch_size, seq_len, d_model]
        
        # 转换维度为 [seq_len, batch_size, d_model]
        token_emb = token_emb.transpose(0, 1)
        
        # 添加位置编码
        output = self.position_encoding(token_emb)
        
        # Dropout
        return self.dropout(output)