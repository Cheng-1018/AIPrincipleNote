<script type="text/javascript" async
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML">
</script>
<script type="text/x-mathjax-config">
MathJax.Hub.Config({
  tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']]}
});
</script>

# AI原理学习笔记

本仓库包含AI算法的学习笔记和代码实现，涵盖从基础理论到具体实现的完整学习路径。

## 📚 目录

### 理论篇

1. **[Transformer](note/1.transformer.md)**
   - Embedding
   - 位置编码
   - 注意力机制
   - FNN
   - LayerNorm
   - 训练预测
2. **[预训练语言模型](note/2.预训练语言模型.md)**
   - Encoder Only:BERT,RoBERTa,ALBERT 
   - Encoder-Decoder:T5
   - Decoder:GPT,LLaMA,GLM
3. **[大模型训练](note/3.大模型训练.md)**
   - Pretrain
   - SFT
   - RLHF:PPO,DPO

4. **[分词算法](note/4.tokenizer.md)**
   - BPE
   - WordPiece
   - Unigram

### 实践篇

- **[Transformer实现](code/transformer/)** - 完整的Transformer英中翻译模型
  - 多头注意力机制
  - 位置编码
  - 编码器-解码器层
  - 训练与预测接口

- **[分词算法](code/tokenizer/)**
  - [BPE](code/tokenizer/bpetokenizer.py)
  - [wordpiece](code/tokenizer/wordpiecetokenizer.py)
  - [Unigram](code/tokenizer/unigramtokenizer.py)

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
- **[huggingface cource](https://huggingface.co/learn/llm-course/chapter1/1)** This course will teach you about large language models (LLMs) and natural language processing (NLP) using libraries from the Hugging Face ecosystem — Transformers,  Datasets,  Tokenizers, and  Accelerate — as well as the Hugging Face Hub.
- **[Awesome-LLM-Learning](https://github.com/kebijuelun/Awesome-LLM-Learning)** 这里是一个专注于大语言模型学习的仓库，旨在为大语言模型学习入门者和大语言模型研发岗位的面试准备者提供全面的基础知识。
- **[LLM-MCP-RAG 实验项目](https://github.com/StrayDragon/exp-llm-mcp-rag)** 一个基于大语言模型（LLM）、模型上下文协议（MCP）和检索增强生成（RAG）的实验性项目。它展示了如何构建一个能够与外部工具交互并利用检索增强生成技术的 AI 助手系统。
