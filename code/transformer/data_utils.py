import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import re
import jieba
from collections import Counter, defaultdict
import pickle
import os
import json
class Vocabulary:
    """
    词汇表类，用于处理文本到索引的映射
    """
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.word_count = Counter()
        
        # 特殊标记
        self.PAD_TOKEN = '<PAD>'
        self.SOS_TOKEN = '<SOS>'  # Start of Sequence
        self.EOS_TOKEN = '<EOS>'  # End of Sequence
        self.UNK_TOKEN = '<UNK>'  # Unknown token
        
        # 添加特殊标记
        self.add_word(self.PAD_TOKEN)
        self.add_word(self.SOS_TOKEN)
        self.add_word(self.EOS_TOKEN)
        self.add_word(self.UNK_TOKEN)
        
        self.pad_idx = self.word2idx[self.PAD_TOKEN]
        self.sos_idx = self.word2idx[self.SOS_TOKEN]
        self.eos_idx = self.word2idx[self.EOS_TOKEN]
        self.unk_idx = self.word2idx[self.UNK_TOKEN]
    
    def add_word(self, word):
        """添加单词到词汇表"""
        if word not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        self.word_count[word] += 1
    
    def add_sentence(self, sentence):
        """添加句子中的所有单词"""
        for word in sentence:
            self.add_word(word)
    
    def build_vocab(self, sentences, min_freq=2):
        """
        根据句子列表构建词汇表
        Args:
            sentences: 句子列表，每个句子是单词列表
            min_freq: 最小词频，低于此频率的词会被替换为UNK
        """
        # 统计词频
        for sentence in sentences:
            for word in sentence:
                self.word_count[word] += 1
        
        # 只保留高频词
        for word, count in self.word_count.items():
            if count >= min_freq and word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
    
    def __len__(self):
        return len(self.word2idx)
    
    def words_to_indices(self, words):
        """将单词列表转换为索引列表"""
        return [self.word2idx.get(word, self.unk_idx) for word in words]
    
    def indices_to_words(self, indices):
        """将索引列表转换为单词列表"""
        return [self.idx2word.get(idx, self.UNK_TOKEN) for idx in indices]
    
    def save(self, path):
        """保存词汇表"""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path):
        """加载词汇表"""
        with open(path, 'rb') as f:
            return pickle.load(f)


def tokenize_english(text):
    """
    英文分词
    """
    # 简单的英文分词，可以使用更复杂的分词器如spacy
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text.lower())
    return text.split()


def tokenize_chinese(text):
    """
    中文分词，使用jieba
    """
    # 去除标点符号，只保留中文字符和数字
    text = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9\s]", "", text)
    return list(jieba.cut(text))


class TranslationDataset(Dataset):
    """
    翻译数据集类
    """
    def __init__(self, src_sentences, tgt_sentences, src_vocab, tgt_vocab, max_len=100):
        self.src_sentences = src_sentences
        self.tgt_sentences = tgt_sentences
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len
    
    def __len__(self):
        return len(self.src_sentences)
    
    def __getitem__(self, idx):
        src_sentence = self.src_sentences[idx]
        tgt_sentence = self.tgt_sentences[idx]
        
        # 转换为索引
        src_indices = self.src_vocab.words_to_indices(src_sentence)
        tgt_indices = self.tgt_vocab.words_to_indices(tgt_sentence)
        
        # 添加SOS和EOS标记到目标句子
        tgt_input = [self.tgt_vocab.sos_idx] + tgt_indices
        tgt_output = tgt_indices + [self.tgt_vocab.eos_idx]
        
        # 截断或填充到固定长度
        src_indices = self.pad_sequence(src_indices, self.max_len, self.src_vocab.pad_idx)
        tgt_input = self.pad_sequence(tgt_input, self.max_len, self.tgt_vocab.pad_idx)
        tgt_output = self.pad_sequence(tgt_output, self.max_len, self.tgt_vocab.pad_idx)
        
        return {
            'src': torch.LongTensor(src_indices),
            'tgt_input': torch.LongTensor(tgt_input),
            'tgt_output': torch.LongTensor(tgt_output)
        }
    
    def pad_sequence(self, sequence, max_len, pad_idx):
        """填充或截断序列到指定长度"""
        if len(sequence) > max_len:
            return sequence[:max_len]
        else:
            return sequence + [pad_idx] * (max_len - len(sequence))


def load_translation_data(data_file, max_samples=None,max_len=50):
    """
    加载英中翻译数据
    Args:
        data_file: 数据文件路径
        max_samples: 最大样本数量
    Returns:
        en_sentences, zh_sentences: 分词后的句子列表
    """
    en_sentences = []
    zh_sentences = []
    
    # 如果没有提供文件路径，直接使用示例数据
    if data_file is None or not os.path.exists(data_file):
        print("未提供数据文件，使用示例数据...")
        return create_sample_data()
    
    try:
        with open(data_file, 'r', encoding='utf-8') as df:
            for i, line in enumerate(df):
                if max_samples and i >= max_samples:
                    break
                data=json.loads(line.strip())
                en_text=data['instruction']
                zh_text=data['output']
                # 分词
                en_tokens = tokenize_english(en_text)
                zh_tokens = tokenize_chinese(zh_text)

                # 过滤空句子和过长句子
                if len(en_tokens) > 0 and len(zh_tokens) > 0 and \
                   len(en_tokens) <= max_len and len(zh_tokens) <= max_len:
                    en_sentences.append(en_tokens)
                    zh_sentences.append(zh_tokens)
    
    except FileNotFoundError:
        print("数据文件未找到，创建示例数据...")
        # 创建示例数据
        en_sentences, zh_sentences = create_sample_data()
    except Exception as e:
        print(f"加载数据文件时出错: {e}")
        print("使用示例数据...")
        en_sentences, zh_sentences = create_sample_data()
    
    return en_sentences, zh_sentences


def create_sample_data():
    """
    创建示例英中翻译数据
    """
    sample_pairs = [
        ("Hello world", "你好世界"),
        ("How are you", "你好吗"),
        ("Good morning", "早上好"),
        ("Thank you", "谢谢"),
        ("Goodbye", "再见"),
        ("I love you", "我爱你"),
        ("What is your name", "你叫什么名字"),
        ("Nice to meet you", "很高兴见到你"),
        ("How old are you", "你多大了"),
        ("Where are you from", "你来自哪里"),
        ("I am learning Chinese", "我在学中文"),
        ("The weather is nice today", "今天天气很好"),
        ("I like to eat Chinese food", "我喜欢吃中国菜"),
        ("Can you speak English", "你会说英语吗"),
        ("See you tomorrow", "明天见"),
        ("Have a good day", "祝你今天愉快"),
        ("I am hungry", "我饿了"),
        ("What time is it", "现在几点了"),
        ("Where is the bathroom", "洗手间在哪里"),
        ("How much does it cost", "这个多少钱")
    ]
    
    en_sentences = []
    zh_sentences = []
    
    for en, zh in sample_pairs:
        en_tokens = tokenize_english(en)
        zh_tokens = tokenize_chinese(zh)
        en_sentences.append(en_tokens)
        zh_sentences.append(zh_tokens)
    
    return en_sentences, zh_sentences


def prepare_data(data_file=None, max_samples=None, min_freq=1, max_len=50):
    """
    准备训练数据
    Args:
        data_file: 数据文件
        max_samples: 最大样本数
        min_freq: 最小词频
        max_len: 最大序列长度
    Returns:
        train_loader: 训练数据加载器
        src_vocab: 源语言词汇表
        tgt_vocab: 目标语言词汇表
    """
    # 加载数据
    en_sentences, zh_sentences = load_translation_data(data_file,max_samples,max_len)
    
    print(f"加载了 {len(en_sentences)} 个句子对")
    
    # 构建词汇表
    src_vocab = Vocabulary()
    tgt_vocab = Vocabulary()
    
    src_vocab.build_vocab(en_sentences, min_freq)
    tgt_vocab.build_vocab(zh_sentences, min_freq)
    
    print(f"英文词汇表大小: {len(src_vocab)}")
    print(f"中文词汇表大小: {len(tgt_vocab)}")
    
    # 创建数据集
    dataset = TranslationDataset(en_sentences, zh_sentences, src_vocab, tgt_vocab, max_len)
    
    # 创建数据加载器
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
    
    return train_loader, src_vocab, tgt_vocab


if __name__ == "__main__":
    # 测试数据处理
    train_loader, src_vocab, tgt_vocab = prepare_data()
    
    # 打印一个批次的数据
    for batch in train_loader:
        print("源序列形状:", batch['src'].shape)
        print("目标输入形状:", batch['tgt_input'].shape)
        print("目标输出形状:", batch['tgt_output'].shape)
        
        # 打印第一个样本
        src_words = src_vocab.indices_to_words(batch['src'][0].tolist())
        tgt_words = tgt_vocab.indices_to_words(batch['tgt_output'][0].tolist())
        print("源句子:", ' '.join([w for w in src_words if w != '<PAD>']))
        print("目标句子:", ' '.join([w for w in tgt_words if w != '<PAD>']))
        break