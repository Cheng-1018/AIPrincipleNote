import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from .embedding import TransformerEmbedding
    from .encoder_decoder import make_encoder, make_decoder
    from .attention import create_masks
except ImportError:
    from embedding import TransformerEmbedding
    from encoder_decoder import make_encoder, make_decoder
    from attention import create_masks

class Transformer(nn.Module):
    """
    完整的Transformer模型
    用于序列到序列的任务，如机器翻译
    """
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, n_heads=8, 
                 n_layers=6, d_ff=2048, max_len=5000, dropout=0.1, pad_token=0):
        super(Transformer, self).__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.pad_token = pad_token
        
        # Embedding层
        self.src_embedding = TransformerEmbedding(src_vocab_size, d_model, max_len, dropout)
        self.tgt_embedding = TransformerEmbedding(tgt_vocab_size, d_model, max_len, dropout)
        
        # Encoder和Decoder
        self.encoder = make_encoder(d_model, n_heads, d_ff, n_layers, dropout)
        self.decoder = make_decoder(d_model, n_heads, d_ff, n_layers, dropout)
        
        # 输出投影层
        self.linear = nn.Linear(d_model, tgt_vocab_size)
        
        # 初始化参数
        self._init_parameters()
        
    def _init_parameters(self):
        """
        初始化模型参数
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def encode(self, src, src_mask):
        """
        编码源序列
        Args:
            src: [batch_size, src_len]
            src_mask: [batch_size, 1, 1, src_len]
        Returns:
            encoder_output: [src_len, batch_size, d_model]
        """
        src_emb = self.src_embedding(src)
        return self.encoder(src_emb, src_mask)
    
    def decode(self, tgt, encoder_output, tgt_mask, src_tgt_mask):
        """
        解码目标序列
        Args:
            tgt: [batch_size, tgt_len]
            encoder_output: [src_len, batch_size, d_model]
            tgt_mask: [batch_size, 1, tgt_len, tgt_len]
            src_tgt_mask: [batch_size, 1, 1, src_len]
        Returns:
            decoder_output: [tgt_len, batch_size, d_model]
        """
        tgt_emb = self.tgt_embedding(tgt)
        return self.decoder(tgt_emb, encoder_output, tgt_mask, src_tgt_mask)
    
    def forward(self, src, tgt):
        """
        前向传播
        Args:
            src: [batch_size, src_len]
            tgt: [batch_size, tgt_len]
        Returns:
            output: [batch_size, tgt_len, tgt_vocab_size]
        """
        # 创建掩码
        src_mask, tgt_mask, src_tgt_mask = create_masks(src, tgt, self.pad_token)
        
        # 编码
        encoder_output = self.encode(src, src_mask)
        
        # 解码
        decoder_output = self.decode(tgt, encoder_output, tgt_mask, src_tgt_mask)
        
        # 线性投影到词汇表大小
        # [tgt_len, batch_size, d_model] -> [batch_size, tgt_len, d_model]
        decoder_output = decoder_output.transpose(0, 1)
        
        # [batch_size, tgt_len, d_model] -> [batch_size, tgt_len, tgt_vocab_size]
        output = self.linear(decoder_output)
        
        return output
    
    def greedy_decode(self, src, max_len, start_token, end_token):
        """
        贪心解码策略
        Args:
            src: [batch_size, src_len]
            max_len: 最大生成长度
            start_token: 开始标记
            end_token: 结束标记
        Returns:
            output: [batch_size, generated_len]
        """
        batch_size = src.size(0)
        device = src.device
        
        # 编码源序列
        src_mask, _, _ = create_masks(src, src, self.pad_token)
        encoder_output = self.encode(src, src_mask)
        
        # 初始化目标序列
        tgt = torch.ones(batch_size, 1).fill_(start_token).long().to(device)
        
        for _ in range(max_len - 1):
            # 创建目标掩码
            # 注意：这里tgt的长度在不断增长，而src长度保持不变
            tgt_padding_mask = (tgt != self.pad_token).unsqueeze(1).unsqueeze(2)
            tgt_look_ahead_mask = torch.tril(torch.ones(tgt.size(1), tgt.size(1), device=device)).unsqueeze(0).unsqueeze(0)
            tgt_mask = torch.minimum(tgt_padding_mask, tgt_look_ahead_mask)
            
            src_mask = (src != self.pad_token).unsqueeze(1).unsqueeze(2)
            src_tgt_mask = src_mask  # decoder对encoder的注意力使用src的padding掩码
            
            # 解码
            decoder_output = self.decode(tgt, encoder_output, tgt_mask, src_tgt_mask)
            
            # 获取最后一个时间步的输出
            last_output = decoder_output[-1, :, :]  # [batch_size, d_model]
            
            # 线性投影并获取概率最大的token
            probs = self.linear(last_output)  # [batch_size, tgt_vocab_size]
            next_token = probs.argmax(dim=-1).unsqueeze(1)  # [batch_size, 1]
            
            # 添加到目标序列
            tgt = torch.cat([tgt, next_token], dim=1)
            
            # 如果所有序列都生成了结束标记，则停止
            if (next_token == end_token).all():
                break
        
        return tgt
    
    def beam_search_decode(self, src, max_len, start_token, end_token, beam_size=4):
        """
        束搜索解码策略
        Args:
            src: [1, src_len] (只支持单个样本)
            max_len: 最大生成长度
            start_token: 开始标记
            end_token: 结束标记
            beam_size: 束大小
        Returns:
            best_sequence: [1, generated_len]
        """
        device = src.device
        
        # 编码源序列
        src_mask, _, _ = create_masks(src, src, self.pad_token)
        encoder_output = self.encode(src, src_mask)
        
        # 初始化束
        sequences = [[start_token]]
        scores = [0.0]
        
        for _ in range(max_len - 1):
            candidates = []
            
            for i, seq in enumerate(sequences):
                if seq[-1] == end_token:
                    candidates.append((seq, scores[i]))
                    continue
                
                # 准备当前序列
                tgt = torch.LongTensor([seq]).to(device)
                _, tgt_mask, src_tgt_mask = create_masks(src, tgt, self.pad_token)
                
                # 解码
                decoder_output = self.decode(tgt, encoder_output, tgt_mask, src_tgt_mask)
                last_output = decoder_output[-1, :, :]
                
                # 获取概率分布
                probs = F.log_softmax(self.linear(last_output), dim=-1)
                
                # 获取top-k个候选
                top_probs, top_indices = probs.topk(beam_size)
                
                for j in range(beam_size):
                    new_seq = seq + [top_indices[0, j].item()]
                    new_score = scores[i] + top_probs[0, j].item()
                    candidates.append((new_seq, new_score))
            
            # 选择top-k个候选
            candidates.sort(key=lambda x: x[1], reverse=True)
            sequences = [cand[0] for cand in candidates[:beam_size]]
            scores = [cand[1] for cand in candidates[:beam_size]]
            
            # 如果最佳序列以结束标记结尾，则停止
            if sequences[0][-1] == end_token:
                break
        
        # 返回最佳序列
        best_sequence = torch.LongTensor([sequences[0]]).to(device)
        return best_sequence


def create_transformer_model(src_vocab_size, tgt_vocab_size, d_model=512, n_heads=8,
                           n_layers=6, d_ff=2048, max_len=5000, dropout=0.1, pad_token=0):
    """
    创建Transformer模型的工厂函数
    """
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        max_len=max_len,
        dropout=dropout,
        pad_token=pad_token
    )
    return model