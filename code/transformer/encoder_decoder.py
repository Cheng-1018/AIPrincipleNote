import torch
import torch.nn as nn
try:
    from .attention import MultiHeadAttention
    from .feedforward import FeedForward, SubLayerConnection
except ImportError:
    from attention import MultiHeadAttention
    from feedforward import FeedForward, SubLayerConnection

class EncoderLayer(nn.Module):
    """
    单个Encoder层
    包含: Multi-Head Self-Attention + Feed Forward
    每个子层都有残差连接和层归一化
    """
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        # 两个残差连接层
        self.sublayer = nn.ModuleList([
            SubLayerConnection(d_model, dropout) for _ in range(2)
        ])
        
    def forward(self, x, mask):
        """
        Args:
            x: [seq_len, batch_size, d_model]
            mask: attention掩码
        Returns:
            output: [seq_len, batch_size, d_model]
        """
        # Self-attention + 残差连接
        x = self.sublayer[0](x, lambda x: self.self_attention(x, x, x, mask)[0])
        
        # Feed forward + 残差连接
        x = self.sublayer[1](x, self.feed_forward)
        
        return x


class TransformerEncoder(nn.Module):
    """
    Transformer Encoder
    由N个EncoderLayer堆叠而成
    """
    def __init__(self, layer, n_layers):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([layer for _ in range(n_layers)])
        self.n_layers = n_layers
        
    def forward(self, x, mask):
        """
        Args:
            x: [seq_len, batch_size, d_model]
            mask: attention掩码
        Returns:
            output: [seq_len, batch_size, d_model]
        """
        for layer in self.layers:
            x = layer(x, mask)
        return x


class DecoderLayer(nn.Module):
    """
    单个Decoder层
    包含: Masked Multi-Head Self-Attention + Multi-Head Cross-Attention + Feed Forward
    每个子层都有残差连接和层归一化
    """
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        # 三个残差连接层
        self.sublayer = nn.ModuleList([
            SubLayerConnection(d_model, dropout) for _ in range(3)
        ])
        
    def forward(self, x, encoder_output, self_mask, cross_mask):
        """
        Args:
            x: decoder输入 [seq_len, batch_size, d_model]
            encoder_output: encoder输出 [src_len, batch_size, d_model]
            self_mask: decoder self-attention掩码
            cross_mask: decoder cross-attention掩码
        Returns:
            output: [seq_len, batch_size, d_model]
        """
        # Masked self-attention + 残差连接
        x = self.sublayer[0](x, lambda x: self.self_attention(x, x, x, self_mask)[0])
        
        # Cross-attention + 残差连接
        x = self.sublayer[1](x, lambda x: self.cross_attention(x, encoder_output, encoder_output, cross_mask)[0])
        
        # Feed forward + 残差连接
        x = self.sublayer[2](x, self.feed_forward)
        
        return x


class TransformerDecoder(nn.Module):
    """
    Transformer Decoder
    由N个DecoderLayer堆叠而成
    """
    def __init__(self, layer, n_layers):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([layer for _ in range(n_layers)])
        self.n_layers = n_layers
        
    def forward(self, x, encoder_output, self_mask, cross_mask):
        """
        Args:
            x: decoder输入 [seq_len, batch_size, d_model]
            encoder_output: encoder输出 [src_len, batch_size, d_model]
            self_mask: decoder self-attention掩码
            cross_mask: decoder cross-attention掩码
        Returns:
            output: [seq_len, batch_size, d_model]
        """
        for layer in self.layers:
            x = layer(x, encoder_output, self_mask, cross_mask)
        return x


def make_encoder(d_model, n_heads, d_ff, n_layers, dropout=0.1):
    """
    构建Encoder的工厂函数
    """
    attention = MultiHeadAttention(d_model, n_heads, dropout)
    feed_forward = FeedForward(d_model, d_ff, dropout)
    layer = EncoderLayer(d_model, n_heads, d_ff, dropout)
    return TransformerEncoder(layer, n_layers)


def make_decoder(d_model, n_heads, d_ff, n_layers, dropout=0.1):
    """
    构建Decoder的工厂函数
    """
    layer = DecoderLayer(d_model, n_heads, d_ff, dropout)
    return TransformerDecoder(layer, n_layers)