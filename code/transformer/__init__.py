"""
Transformer英中翻译模型

基于"Attention Is All You Need"论文实现的Transformer模型
用于英语到中文的机器翻译任务

主要组件:
- embedding.py: Token Embedding和位置编码
- attention.py: 多头注意力机制和掩码
- feedforward.py: 前馈网络和层归一化
- encoder_decoder.py: Encoder和Decoder层
- transformer.py: 完整的Transformer模型
- data_utils.py: 数据处理和词汇表
- train.py: 训练逻辑和优化器
- predict.py: 预测和解码策略
- main.py: 主程序入口

使用示例:
    # 训练模型
    python main.py --mode train --epochs 20
    
    # 进行翻译
    python main.py --mode predict --model_path ./checkpoints/best_model.pt
    
    # 评估模型
    python main.py --mode eval --model_path ./checkpoints/best_model.pt
"""

from .transformer import Transformer, create_transformer_model
from .train import Trainer, train_transformer
from .predict import Translator
from .data_utils import Vocabulary, prepare_data
from .attention import MultiHeadAttention
from .embedding import TransformerEmbedding
from .encoder_decoder import TransformerEncoder, TransformerDecoder

__version__ = "1.0.0"
__author__ = "AI Assistant"
__description__ = "Transformer英中翻译模型实现"

__all__ = [
    'Transformer',
    'create_transformer_model',
    'Trainer',
    'train_transformer',
    'Translator',
    'Vocabulary',
    'prepare_data',
    'MultiHeadAttention',
    'TransformerEmbedding',
    'TransformerEncoder',
    'TransformerDecoder'
]