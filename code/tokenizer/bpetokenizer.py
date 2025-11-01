import re, collections
from collections import defaultdict
import json
from typing import List, Tuple, Dict, Optional
import os

class BPE:
    def __init__(self, vocab_size: int = 500):
        """
        初始化BPE分词器
        
        Args:
            vocab_size: 目标词汇表大小
        """
        self.vocab_size = vocab_size
        self.vocab: List[str] = []
        self.merges: Dict[Tuple[str, str], str] = {}
        self.word_freqs: Dict[str, int] = defaultdict(int)
        self.alphabet: List[str] = []
        
    def load_corpus(self, file_path: str) -> List[str]:
        """
        从JSONL文件加载语料库
        
        Args:
            file_path: JSONL文件路径
            
        Returns:
            语料库文本列表
        """
        corpus = []
        
        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            return corpus
            
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                text = data['instruction']
                corpus.append(text)
        
        return corpus
    
    def build_word_freqs(self, corpus: List[str]) -> None:
        """
        构建词频统计
        
        Args:
            corpus: 语料库文本列表
        """
        self.word_freqs = defaultdict(int)
        
        for text in corpus:
            for word in text.strip().split():
                self.word_freqs[word] += 1
    
    def build_alphabet(self) -> None:
        """构建字符表"""
        alphabet = []
        
        for word in self.word_freqs.keys():
            for letter in word:
                if letter not in alphabet:
                    alphabet.append(letter)
        
        alphabet.sort()
        self.alphabet = alphabet
        self.vocab = ["<|endoftext|>"] + alphabet.copy()
    
    def compute_pair_freqs(self, splits: Dict[str, List[str]]) -> Dict[Tuple[str, str], int]:
        """
        计算字符对频率
        
        Args:
            splits: 单词分割字典
            
        Returns:
            字符对频率字典
        """
        pair_freqs = defaultdict(int)
        
        for word, freqs in self.word_freqs.items():
            split = splits[word]
            for i in range(len(split) - 1):
                pair_freqs[(split[i], split[i + 1])] += freqs
                
        return pair_freqs
    
    def get_best_pair(self, pair_freqs: Dict[Tuple[str, str], int]) -> Tuple[Tuple[str, str], int]:
        """
        获取频率最高的字符对
        
        Args:
            pair_freqs: 字符对频率字典
            
        Returns:
            最佳字符对和其频率
        """
        best_pair = ("", "")
        max_freq = 0
        
        for pair, freq in pair_freqs.items():
            if freq > max_freq:
                best_pair = pair
                max_freq = freq
                
        return best_pair, max_freq
    
    def merge_pair(self, a: str, b: str, splits: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """
        合并指定的字符对
        
        Args:
            a: 第一个字符
            b: 第二个字符
            splits: 单词分割字典
            
        Returns:
            更新后的分割字典
        """
        for word in self.word_freqs.keys():
            split = splits[word]
            if len(split) == 1:
                continue
                
            i = 0
            while i < len(split) - 1:
                if split[i] == a and split[i + 1] == b:
                    split = split[:i] + [a + b] + split[i + 2:]
                else:
                    i += 1
            splits[word] = split
            
        return splits
    
    def train(self, corpus: List[str]) -> None:
        """
        训练BPE分词器
        
        Args:
            corpus: 训练语料库
        """
        # 构建词频和字符表
        self.build_word_freqs(corpus)
        self.build_alphabet()
        
        # 初始化分割
        splits = {word: [c for c in word] for word in self.word_freqs.keys()}
        
        # 初始化合并规则
        self.merges = {}
        
        # 迭代合并直到达到目标词汇表大小
        while len(self.vocab) < self.vocab_size:
            pair_freqs = self.compute_pair_freqs(splits)
            
            if not pair_freqs:
                break
                
            best_pair, max_freq = self.get_best_pair(pair_freqs)
            
            if max_freq == 0:
                break
                
            splits = self.merge_pair(best_pair[0], best_pair[1], splits)
            self.merges[best_pair] = best_pair[0] + best_pair[1]
            self.vocab.append(best_pair[0] + best_pair[1])
    
    def tokenize(self, text: str) -> List[List[str]]:
        """
        对文本进行分词
        
        Args:
            text: 输入文本
            
        Returns:
            分词结果列表
        """
        pre_tokenized_text = text.split()
        splits = [[c for c in word] for word in pre_tokenized_text]
        
        for pair, merge in self.merges.items():
            for idx, split in enumerate(splits):
                i = 0
                while i < len(split) - 1:
                    if split[i] == pair[0] and split[i + 1] == pair[1]:
                        split = split[:i] + [merge] + split[i + 2:]
                    else:
                        i += 1
                splits[idx] = split
                
        return splits
    
    def save_model(self, save_path: str) -> None:
        """
        保存模型到JSON文件
        
        Args:
            save_path: 保存路径
        """
        model_data = {
            "vocab": self.vocab,
            "merges": {f"{pair[0]}_{pair[1]}": merge for pair, merge in self.merges.items()},
            "vocab_size": self.vocab_size
        }
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(model_data, f, ensure_ascii=False, indent=2)
    
    def load_model(self, model_path: str) -> None:
        """
        从JSON文件加载模型
        
        Args:
            model_path: 模型文件路径
        """
        with open(model_path, 'r', encoding='utf-8') as f:
            model_data = json.load(f)
            
        self.vocab = model_data["vocab"]
        self.vocab_size = model_data["vocab_size"]
        
        # 恢复merges格式
        self.merges = {}
        for key, value in model_data["merges"].items():
            pair = tuple(key.split("_", 1))
            self.merges[pair] = value


# 示例使用
if __name__ == "__main__":
    # 创建BPE分词器
    bpe = BPE(vocab_size=500)
    
    # 加载语料库
    corpus = bpe.load_corpus(r'..\transformer\data\en-zh-cn-39k-without-think-alpaca.jsonl')
    
    if corpus:
        # 训练分词器
        print("开始训练BPE分词器...")
        bpe.train(corpus[:1000])  # 使用前1000个样本训练，可根据需要调整
        
        print(f"词汇表大小: {len(bpe.vocab)}")
        print(f"合并规则数量: {len(bpe.merges)}")
        
        # 测试分词
        test_text = "The Guttenberg discontinuity marks the boundary between Earth's outer core and lower mantle."
        tokens = bpe.tokenize(test_text)
        print(f"原文: {test_text}")
        print(f"分词结果: {tokens}")
        
        # 保存模型
        bpe.save_model("bpe_model.json")
        print("模型已保存到 bpe_model.json")
    