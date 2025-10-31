# AI原理学习笔记

本仓库包含AI算法的学习笔记和代码实现，涵盖从基础理论到具体实现的完整学习路径。

## 📚 目录

### 理论篇

1. **[Transformer](note/1.transformer.md)**
   - 注意力机制
   - 多头注意力
   - 位置编码
   - 编码器-解码器架构

### 实践篇

- **[Transformer实现](code/transformer/)** - 完整的Transformer英中翻译模型
  - 多头注意力机制
  - 位置编码
  - 编码器-解码器层
  - 训练与预测接口

## 🛠️ 环境配置

```bash
# 克隆仓库
git clone https://github.com/Cheng-1018/AIPrincipleNote.git
cd AIPrincipleNote

# 安装依赖（以Transformer为例）
cd code/transformer
pip install torch jieba
```

## 学习资源

- **[Attention Is All You Need](https://arxiv.org/abs/1706.03762)** - Transformer原论文
- **[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)** - Transformer可视化讲解
