# Transformer 英中翻译模型

基于论文 "Attention Is All You Need" 实现的Transformer模型，用于英语到中文的机器翻译任务。

## 项目结构

```
transformer/
├── __init__.py           # Python包初始化
├── main.py              # 主程序入口
├── embedding.py         # Token Embedding和位置编码
├── attention.py         # 多头注意力机制
├── feedforward.py       # 前馈网络和层归一化
├── encoder_decoder.py   # Encoder和Decoder层
├── transformer.py       # 完整的Transformer模型
├── data_utils.py        # 数据处理和词汇表
├── train.py            # 训练逻辑
├── predict.py          # 预测和解码
└── README.md           # 说明文档
```

## 核心特性

### 模型组件
- **Embedding层**: Token Embedding + 正弦位置编码
- **多头注意力**: 自注意力、掩码注意力、交叉注意力
- **前馈网络**: 两层全连接网络 + ReLU激活
- **层归一化**: LayerNorm + 残差连接
- **Encoder-Decoder**: 6层Encoder + 6层Decoder

### 训练特性
- **标签平滑**: 减少过拟合
- **Noam学习率调度**: 预热 + 衰减策略
- **梯度裁剪**: 防止梯度爆炸
- **Teacher Forcing**: 并行训练策略

### 解码策略
- **贪心解码**: 快速生成
- **束搜索**: 更高质量的翻译结果

## 安装依赖

```bash
pip install torch torchvision torchaudio
pip install jieba  # 中文分词
```

## 使用方法

### 1. 训练模型

```bash
# 使用JSONL数据文件训练
python main.py --mode train --data_file your_data.jsonl --epochs 20

# 使用内置示例数据训练
python main.py --mode train --epochs 20

# 使用GPU训练
python main.py --mode train --data_file your_data.jsonl --epochs 20 --device cuda

# 指定更多样本和参数
python main.py --mode train --data_file your_data.jsonl --epochs 20 --max_samples 5000 --max_len 100
```

### 2. 交互式翻译

```bash
# 使用训练好的模型进行翻译
python main.py --mode predict --model_path ./checkpoints/best_model.pt
```

交互式界面示例：
```
Transformer英中翻译器
输入英文句子进行翻译，输入'quit'退出
--------------------------------------------------
英文: Hello world
中文 (贪心): 你好 世界
中文 (束搜索): 你好 世界
翻译时间: 0.123秒
--------------------------------------------------
英文: How are you
中文 (贪心): 你 好 吗
中文 (束搜索): 你 好 吗
翻译时间: 0.089秒
--------------------------------------------------
```

### 3. 模型评估

```bash
# 评估模型性能
python main.py --mode eval --model_path ./checkpoints/best_model.pt
```

### 4. 快速演示

```bash
# 运行快速演示（无参数启动）
python main.py
```

### 5. 词汇表提取

```python
# 提取训练好的模型的词汇表
from train import Trainer
import json

# 加载模型和词汇表
model, src_vocab, tgt_vocab = Trainer.load_model('./checkpoints/best_model.pt')

# 保存词汇表为JSON
vocab_data = {
    'src_vocab': {
        'word2idx': src_vocab.word2idx,
        'idx2word': src_vocab.idx2word,
        'vocab_size': len(src_vocab)
    },
    'tgt_vocab': {
        'word2idx': tgt_vocab.word2idx,
        'idx2word': tgt_vocab.idx2word,
        'vocab_size': len(tgt_vocab)
    }
}

with open('vocabulary.json', 'w', encoding='utf-8') as f:
    json.dump(vocab_data, f, ensure_ascii=False, indent=2)
```

## 数据格式

### 训练数据
模型支持两种数据输入方式：

1. **JSONL文件输入**（推荐用于真实数据）：
   - 每行一个JSON对象
   - 英文字段：`instruction`
   - 中文字段：`output`
   - 示例格式：
     ```json
     {"instruction": "Hello world", "output": "你好世界"}
     {"instruction": "How are you", "output": "你好吗"}
     {"instruction": "Good morning", "output": "早上好"}
     ```

2. **内置示例数据**：
   - 包含20个英中句子对的示例数据
   - 用于快速测试和演示

### 数据预处理
- **英文分词**：简单的空格分词 + 标点清理
- **中文分词**：使用jieba进行分词
- **词汇表构建**：支持最小词频过滤
- **序列处理**：添加SOS/EOS标记，padding到固定长度

## 模型架构

### 超参数配置
```python
d_model = 512        # 模型维度
n_heads = 8          # 注意力头数
n_layers = 6         # 层数
d_ff = 2048         # 前馈网络维度
dropout = 0.1       # Dropout率
max_len = 5000      # 最大序列长度

# 数据参数
data_file = "data.jsonl"    # JSONL数据文件路径
vocab_size = 10000          # 词汇表大小
min_freq = 2               # 最小词频阈值
```

### 数学公式

**多头注意力**：
$$MHA(Q,K,V) = Concat(head_1,...,head_h)W^O$$
$$head_i = attention(QW^Q_i, KW^K_i, VW^V_i)$$

**缩放点积注意力**：
$$attention(Q,K,V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$$

**位置编码**：
$$PE(pos,2i) = sin(\frac{pos}{10000^{2i/d_{model}}})$$
$$PE(pos,2i+1) = cos(\frac{pos}{10000^{2i/d_{model}}})$$

**前馈网络**：
$$FFN(x) = max(0, xW_1 + b_1)W_2 + b_2$$

## 性能优化

### 训练优化
- 使用Adam优化器 + Noam学习率调度
- 梯度累积支持更大的有效批次
- 混合精度训练（可选）
- 检查点保存和恢复

### 推理优化
- 模型并行化支持
- 批量解码
- KV缓存优化（可扩展）

## 扩展功能

### 自定义数据
```python
from transformer import prepare_data, train_transformer

# 使用JSONL格式的自定义数据
train_loader, src_vocab, tgt_vocab = prepare_data(
    data_file="path/to/your_data.jsonl",
    max_samples=10000,
    max_len=100,
    min_freq=2
)

# 训练模型
trainer = train_transformer(
    data_file="path/to/your_data.jsonl",
    num_epochs=50,
    device="cuda"
)
```

### 模型配置
```python
from transformer import create_transformer_model

# 创建自定义模型
model = create_transformer_model(
    src_vocab_size=10000,
    tgt_vocab_size=8000,
    d_model=256,
    n_heads=4,
    n_layers=4
)
```

## 文件说明

| 文件 | 功能 |
|------|------|
| `embedding.py` | Token Embedding和正弦位置编码 |
| `attention.py` | 多头注意力机制和掩码处理 |
| `feedforward.py` | 前馈网络、层归一化、残差连接 |
| `encoder_decoder.py` | Encoder层和Decoder层实现 |
| `transformer.py` | 完整Transformer模型和解码策略 |
| `data_utils.py` | 数据加载、预处理、词汇表管理 |
| `train.py` | 训练循环、优化器、损失函数 |
| `predict.py` | 翻译器、解码策略、评估函数 |
| `main.py` | 统一入口、命令行界面 |

## 注意事项

1. **内存使用**：注意力矩阵的内存复杂度是O(n²)
2. **训练时间**：完整训练需要较长时间，建议使用GPU
3. **数据质量**：翻译质量很大程度上取决于训练数据的质量和数量
4. **超参数调优**：不同的数据集可能需要调整超参数

## 论文参考

- Vaswani, A., et al. "Attention is all you need." Advances in neural information processing systems 30 (2017).

## 许可证

本项目仅用于学习和研究目的。