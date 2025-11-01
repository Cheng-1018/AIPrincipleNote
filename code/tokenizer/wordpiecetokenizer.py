import re, collections
from collections import defaultdict
import json
from typing import List, Tuple, Dict, Optional
import os
import math

class WordPiece:
    def __init__(self, vocab_size: int = 70, unk_token: str = "[UNK]"):
        """
        初始化WordPiece分词器
        
        Args:
            vocab_size: 目标词汇表大小
            unk_token: 未知词符号
        """
        self.vocab_size = vocab_size
        self.unk_token = unk_token
        self.vocab: List[str] = []
        self.word_freqs: Dict[str, int] = defaultdict(int)
        self.alphabet: List[str] = []
        
    def load_corpus(self, file_path: str = None, corpus: List[str] = None) -> List[str]:
        """
        加载语料库
        
        Args:
            file_path: JSONL文件路径（可选）
            corpus: 直接提供的语料库（可选）
            
        Returns:
            语料库文本列表
        """
        if corpus is not None:
            return corpus
            
        if file_path is None:
            # 默认示例语料
            return [
                "This is the Hugging Face Course.",
                "This chapter is about tokenization.",
                "This section shows several tokenizer algorithms.",
                "Hopefully, you will be able to understand how they are trained and generate tokens.",
            ]
        
        corpus_list = []
        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            return corpus_list
            
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                text = data['instruction']
                corpus_list.append(text)
        
        return corpus_list
    
    def build_word_freqs(self, corpus: List[str]) -> None:
        """
        构建词频统计
        
        Args:
            corpus: 语料库文本列表
        """
        self.word_freqs = defaultdict(int)
        
        for text in corpus:
            new_words = text.strip().split()
            for word in new_words:
                self.word_freqs[word] += 1
    
    def build_alphabet(self) -> None:
        """构建字符表，第一个字符不加##，其余字符加##前缀"""
        alphabet = []
        
        for word in self.word_freqs.keys():
            if word[0] not in alphabet:
                alphabet.append(word[0])
            for letter in word[1:]:
                if f"##{letter}" not in alphabet:
                    alphabet.append(f"##{letter}")
        
        alphabet.sort()
        self.alphabet = alphabet
        
        # 初始化词汇表，包含特殊符号
        self.vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + alphabet.copy()
    
    def get_initial_splits(self) -> Dict[str, List[str]]:
        """
        获取初始分割，第一个字符不加##，其余字符加##前缀
        
        Returns:
            单词分割字典
        """
        splits = {
            word: [c if i == 0 else f"##{c}" for i, c in enumerate(word)]
            for word in self.word_freqs.keys()
        }
        return splits
    
    def compute_pair_scores(self, splits: Dict[str, List[str]]) -> Dict[Tuple[str, str], float]:
        """
        计算字符对评分，使用WordPiece的评分公式
        
        Args:
            splits: 单词分割字典
            
        Returns:
            字符对评分字典
        """
        letter_freqs = defaultdict(int)
        pair_freqs = defaultdict(int)
        
        for word, freq in self.word_freqs.items():
            split = splits[word]
            if len(split) == 1:
                letter_freqs[split[0]] += freq
                continue
                
            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                letter_freqs[split[i]] += freq
                pair_freqs[pair] += freq
            letter_freqs[split[-1]] += freq
        
        # WordPiece评分公式: P(xy) / (P(x) * P(y))
        scores = {
            pair: freq / (letter_freqs[pair[0]] * letter_freqs[pair[1]])
            for pair, freq in pair_freqs.items()
            if letter_freqs[pair[0]] > 0 and letter_freqs[pair[1]] > 0
        }
        
        return scores
    
    def get_best_pair(self, pair_scores: Dict[Tuple[str, str], float]) -> Tuple[Tuple[str, str], float]:
        """
        获取评分最高的字符对
        
        Args:
            pair_scores: 字符对评分字典
            
        Returns:
            最佳字符对和其评分
        """
        best_pair = ("", "")
        max_score = 0.0
        
        for pair, score in pair_scores.items():
            if score > max_score:
                best_pair = pair
                max_score = score
                
        return best_pair, max_score
    
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
        for word in self.word_freqs:
            split = splits[word]
            if len(split) == 1:
                continue
                
            i = 0
            while i < len(split) - 1:
                if split[i] == a and split[i + 1] == b:
                    # 如果b以##开头，需要去掉##前缀
                    merge = a + b[2:] if b.startswith("##") else a + b
                    split = split[:i] + [merge] + split[i + 2:]
                else:
                    i += 1
            splits[word] = split
            
        return splits
    
    def train(self, corpus: List[str]) -> None:
        """
        训练WordPiece分词器
        
        Args:
            corpus: 训练语料库
        """
        # 构建词频和字符表
        self.build_word_freqs(corpus)
        self.build_alphabet()
        
        # 初始化分割
        splits = self.get_initial_splits()
        
        # 迭代合并直到达到目标词汇表大小
        while len(self.vocab) < self.vocab_size:
            scores = self.compute_pair_scores(splits)
            
            if not scores:
                break
                
            best_pair, max_score = self.get_best_pair(scores)
            
            if max_score == 0:
                break
                
            splits = self.merge_pair(best_pair[0], best_pair[1], splits)
            
            # 生成新token
            new_token = (
                best_pair[0] + best_pair[1][2:]
                if best_pair[1].startswith("##")
                else best_pair[0] + best_pair[1]
            )
            self.vocab.append(new_token)
    
    def encode_word(self, word: str) -> List[str]:
        """
        对单个单词进行编码
        
        Args:
            word: 输入单词
            
        Returns:
            编码后的token列表
        """
        tokens = []
        
        while len(word) > 0:
            i = len(word)
            while i > 0 and word[:i] not in self.vocab:
                i -= 1
            if i == 0:
                return [self.unk_token]
            tokens.append(word[:i])
            word = word[i:]
            if len(word) > 0:
                word = f"##{word}"
                
        return tokens
    
    def tokenize(self, text: str) -> List[str]:
        """
        对文本进行分词
        
        Args:
            text: 输入文本
            
        Returns:
            分词结果列表
        """
        words = text.strip().split()
        tokens = []
        
        for word in words:
            word_tokens = self.encode_word(word)
            tokens.extend(word_tokens)
            
        return tokens
    
    def save_model(self, save_path: str) -> None:
        """
        保存模型到JSON文件
        
        Args:
            save_path: 保存路径
        """
        model_data = {
            "vocab": self.vocab,
            "vocab_size": self.vocab_size,
            "unk_token": self.unk_token,
            "word_freqs": dict(self.word_freqs)
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
        self.unk_token = model_data["unk_token"]
        self.word_freqs = defaultdict(int, model_data["word_freqs"])


# 示例使用
if __name__ == "__main__":
    # 创建WordPiece分词器
    wp = WordPiece(vocab_size=500)
    
    # 使用默认语料库训练
    corpus = wp.load_corpus(file_path=r'..\transformer\data\en-zh-cn-39k-without-think-alpaca.jsonl')
    
    print("开始训练WordPiece分词器...")
    wp.train(corpus)
    
    print(f"词汇表大小: {len(wp.vocab)}")
    print(f"词汇表: {wp.vocab}")
    
    # 测试分词
    test_text = "This is tokenization"
    tokens = wp.tokenize(test_text)
    print(f"原文: {test_text}")
    print(f"分词结果: {tokens}")
    
    # 测试单词编码
    test_word = "tokenization"
    word_tokens = wp.encode_word(test_word)
    print(f"单词: {test_word}")
    print(f"编码结果: {word_tokens}")
    
    # 保存模型
    wp.save_model("wordpiece_model.json")
    print("模型已保存到 wordpiece_model.json")