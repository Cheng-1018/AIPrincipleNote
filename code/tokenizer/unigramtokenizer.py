import re, collections
from collections import defaultdict
import json
from typing import List, Tuple, Dict, Optional
import os
import math
import copy

class Unigram:
    def __init__(self, vocab_size: int = 500, unk_token: str = "<unk>"):
        """
        初始化Unigram分词器
        
        Args:
            vocab_size: 目标词汇表大小
            unk_token: 未知词符号
        """
        self.vocab_size = vocab_size
        self.unk_token = unk_token
        self.model: Dict[str, float] = {}
        self.word_freqs: Dict[str, int] = defaultdict(int)
        self.token_freqs: Dict[str, int] = {}
        
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
    
    def build_subword_freqs(self, max_subwords: int = 300) -> None:
        """
        构建subword频率统计
        
        Args:
            max_subwords: 最大subword数量
        """
        char_freqs = defaultdict(int)
        subwords_freqs = defaultdict(int)
        
        for word, freq in self.word_freqs.items():
            for i in range(len(word)):
                char_freqs[word[i]] += freq
                # 生成长度至少为2的所有subword
                for j in range(i + 2, len(word) + 1):
                    subwords_freqs[word[i:j]] += freq
        
        # 按频率排序subwords
        sorted_subwords = sorted(subwords_freqs.items(), key=lambda x: x[1], reverse=True)
        
        # 组合字符和top subwords
        token_freqs = list(char_freqs.items()) + sorted_subwords[:max_subwords - len(char_freqs)]
        self.token_freqs = {token: freq for token, freq in token_freqs}
    
    def build_initial_model(self) -> None:
        """
        构建初始Unigram语言模型
        """
        total_sum = sum([freq for token, freq in self.token_freqs.items()])
        self.model = {token: -math.log(freq / total_sum) for token, freq in self.token_freqs.items()}
    
    def encode_word(self, word: str, model: Dict[str, float] = None) -> Tuple[List[str], Optional[float]]:
        """
        使用动态规划对单词进行最优分割
        
        Args:
            word: 输入单词
            model: 使用的模型（可选，默认使用self.model）
            
        Returns:
            最优分割结果和总分数
        """
        if model is None:
            model = self.model
            
        best_segmentations = [{"start": 0, "score": 1}] + [
            {"start": None, "score": None} for _ in range(len(word))
        ]
        
        for start_idx in range(len(word)):
            # 获取当前位置的最佳分数
            best_score_at_start = best_segmentations[start_idx]["score"]
            
            for end_idx in range(start_idx + 1, len(word) + 1):
                token = word[start_idx:end_idx]
                if token in model and best_score_at_start is not None:
                    score = model[token] + best_score_at_start
                    # 如果找到更好的分割方案，则更新
                    if (
                        best_segmentations[end_idx]["score"] is None
                        or best_segmentations[end_idx]["score"] > score
                    ):
                        best_segmentations[end_idx] = {"start": start_idx, "score": score}
        
        segmentation = best_segmentations[-1]
        if segmentation["score"] is None:
            # 无法分词，返回未知词
            return [self.unk_token], None
        
        # 回溯构建最优分割序列
        score = segmentation["score"]
        start = segmentation["start"]
        end = len(word)
        tokens = []
        
        while start != 0:
            tokens.insert(0, word[start:end])
            next_start = best_segmentations[start]["start"]
            end = start
            start = next_start
        tokens.insert(0, word[start:end])
        
        return tokens, score
    
    def compute_loss(self, model: Dict[str, float] = None) -> float:
        """
        计算当前模型在语料库上的总损失
        
        Args:
            model: 使用的模型（可选，默认使用self.model）
            
        Returns:
            总损失值
        """
        if model is None:
            model = self.model
            
        loss = 0
        for word, freq in self.word_freqs.items():
            _, word_loss = self.encode_word(word, model)
            if word_loss is not None:
                loss += freq * word_loss
        return loss
    
    def compute_scores(self, model: Dict[str, float] = None) -> Dict[str, float]:
        """
        计算每个token被移除后对损失的影响
        
        Args:
            model: 使用的模型（可选，默认使用self.model）
            
        Returns:
            token影响分数字典
        """
        if model is None:
            model = self.model
            
        scores = {}
        model_loss = self.compute_loss(model)
        
        for token, score in model.items():
            # 保留长度为1的token（字符）
            if len(token) == 1:
                continue
                
            model_without_token = copy.deepcopy(model)
            _ = model_without_token.pop(token)
            scores[token] = self.compute_loss(model_without_token) - model_loss
            
        return scores
    
    def prune_model(self, percent_to_remove: float = 0.1) -> None:
        """
        剪枝模型，移除影响最小的token
        
        Args:
            percent_to_remove: 每次移除的token比例
        """
        while len(self.model) > self.vocab_size:
            scores = self.compute_scores()
            if not scores:
                break
                
            sorted_scores = sorted(scores.items(), key=lambda x: x[1])
            
            # 移除影响最小的token
            tokens_to_remove = int(len(self.model) * percent_to_remove)
            for i in range(min(tokens_to_remove, len(sorted_scores))):
                token_to_remove = sorted_scores[i][0]
                if token_to_remove in self.token_freqs:
                    _ = self.token_freqs.pop(token_to_remove)
            
            # 重新构建模型
            self.build_initial_model()
    
    def train(self, corpus: List[str], max_subwords: int = 300, percent_to_remove: float = 0.1) -> None:
        """
        训练Unigram分词器
        
        Args:
            corpus: 训练语料库
            max_subwords: 最大subword数量
            percent_to_remove: 每次剪枝移除的比例
        """
        print("构建词频统计...")
        self.build_word_freqs(corpus)
        
        print("构建subword频率...")
        self.build_subword_freqs(max_subwords)
        
        print("构建初始模型...")
        self.build_initial_model()
        
        print(f"初始模型大小: {len(self.model)}")
        print("开始剪枝...")
        self.prune_model(percent_to_remove)
        print(f"最终模型大小: {len(self.model)}")
    
    def tokenize(self, text: str) -> List[str]:
        """
        对文本进行分词
        
        Args:
            text: 输入文本
            
        Returns:
            分词结果列表
        """
        pre_tokenized_text = text.strip().split()
        encoded_words = [self.encode_word(word)[0] for word in pre_tokenized_text]
        return sum(encoded_words, [])  # 展平列表
    
    def save_model(self, save_path: str) -> None:
        """
        保存模型到JSON文件
        
        Args:
            save_path: 保存路径
        """
        model_data = {
            "model": self.model,
            "vocab_size": self.vocab_size,
            "unk_token": self.unk_token,
            "word_freqs": dict(self.word_freqs),
            "token_freqs": self.token_freqs
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
            
        self.model = model_data["model"]
        self.vocab_size = model_data["vocab_size"]
        self.unk_token = model_data["unk_token"]
        self.word_freqs = defaultdict(int, model_data["word_freqs"])
        self.token_freqs = model_data["token_freqs"]
    
    def get_vocab(self) -> List[str]:
        """
        获取词汇表
        
        Returns:
            词汇表列表
        """
        return list(self.model.keys())


# 示例使用
if __name__ == "__main__":
    # 创建Unigram分词器
    unigram = Unigram(vocab_size=500)
    
    # 使用默认语料库训练
    corpus = unigram.load_corpus(file_path=r'..\transformer\data\en-zh-cn-39k-without-think-alpaca.jsonl')
    
    print("开始训练Unigram分词器...")
    unigram.train(corpus, max_subwords=1000, percent_to_remove=0.1)
    
    print(f"最终词汇表大小: {len(unigram.model)}")
    print(f"词汇表: {unigram.get_vocab()}")
    
    # 测试单词编码
    test_words = ["Hopefully", "This", "tokenization"]
    for word in test_words:
        tokens, score = unigram.encode_word(word)
        print(f"单词: {word}")
        print(f"编码结果: {tokens}, 分数: {score}")
    
    # 测试文本分词
    test_text = "This is the Hugging Face course."
    tokens = unigram.tokenize(test_text)
    print(f"原文: {test_text}")
    print(f"分词结果: {tokens}")
    
    # 计算模型损失
    loss = unigram.compute_loss()
    print(f"模型损失: {loss}")
    
    # 保存模型
    unigram.save_model("unigram_model.json")
    print("模型已保存到 unigram_model.json")
