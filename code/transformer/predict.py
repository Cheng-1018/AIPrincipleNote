import torch
import torch.nn.functional as F
try:
    from .transformer import create_transformer_model
    from .train import Trainer
    from .data_utils import tokenize_english, tokenize_chinese
except ImportError:
    from transformer import create_transformer_model
    from train import Trainer
    from data_utils import tokenize_english, tokenize_chinese
import time

class Translator:
    """
    Transformer翻译器，用于推理预测
    """
    def __init__(self, model, src_vocab, tgt_vocab, device='cpu'):
        self.model = model
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.device = device
        
        # 将模型移动到设备并设置为评估模式
        self.model.to(device)
        self.model.eval()
    
    @classmethod
    def from_checkpoint(cls, checkpoint_path, device='cpu'):
        """
        从检查点加载翻译器
        """
        model, src_vocab, tgt_vocab = Trainer.load_model(checkpoint_path, device)
        return cls(model, src_vocab, tgt_vocab, device)
    
    def preprocess_sentence(self, sentence, is_source=True):
        """
        预处理句子
        Args:
            sentence: 输入句子字符串
            is_source: 是否为源语言
        Returns:
            tokens: 分词后的token列表
        """
        if is_source:
            tokens = tokenize_english(sentence)
        else:
            tokens = tokenize_chinese(sentence)
        return tokens
    
    def postprocess_sentence(self, tokens):
        """
        后处理句子，移除特殊标记
        """
        # 移除特殊标记
        filtered_tokens = []
        for token in tokens:
            if token in [self.tgt_vocab.PAD_TOKEN, self.tgt_vocab.SOS_TOKEN, self.tgt_vocab.EOS_TOKEN]:
                if token == self.tgt_vocab.EOS_TOKEN:
                    break
                continue
            filtered_tokens.append(token)
        
        result = ' '.join(filtered_tokens)
        
        # 如果结果为空或只有UNK，返回提示信息
        if not result.strip():
            return "[模型没有生成翻译]"
        elif all(token == self.tgt_vocab.UNK_TOKEN for token in filtered_tokens):
            return "[输入词汇不在训练数据中]"
        
        return result
    
    def translate_greedy(self, sentence, max_len=50):
        """
        使用贪心解码进行翻译
        Args:
            sentence: 英文句子字符串
            max_len: 最大生成长度
        Returns:
            translation: 中文翻译结果
        """
        with torch.no_grad():
            # 预处理输入句子
            src_tokens = self.preprocess_sentence(sentence, is_source=True)
            src_indices = self.src_vocab.words_to_indices(src_tokens)
            
            # 添加padding到固定长度（与训练时保持一致）
            if len(src_indices) > max_len:
                src_indices = src_indices[:max_len]
            else:
                src_indices = src_indices + [self.src_vocab.pad_idx] * (max_len - len(src_indices))
            
            # 转换为tensor并添加batch维度
            src = torch.LongTensor([src_indices]).to(self.device)
            
            # 使用模型的贪心解码
            result = self.model.greedy_decode(
                src=src,
                max_len=max_len,
                start_token=self.tgt_vocab.sos_idx,
                end_token=self.tgt_vocab.eos_idx
            )
            
            # 转换回单词
            result_tokens = self.tgt_vocab.indices_to_words(result[0].tolist())
            
            # 后处理
            translation = self.postprocess_sentence(result_tokens)
            
            return translation
    
    def translate_beam_search(self, sentence, max_len=50, beam_size=4):
        """
        使用束搜索进行翻译
        Args:
            sentence: 英文句子字符串
            max_len: 最大生成长度
            beam_size: 束大小
        Returns:
            translation: 中文翻译结果
        """
        with torch.no_grad():
            # 预处理输入句子
            src_tokens = self.preprocess_sentence(sentence, is_source=True)
            src_indices = self.src_vocab.words_to_indices(src_tokens)
            
            # 添加padding到固定长度（与训练时保持一致）
            if len(src_indices) > max_len:
                src_indices = src_indices[:max_len]
            else:
                src_indices = src_indices + [self.src_vocab.pad_idx] * (max_len - len(src_indices))
            
            # 转换为tensor并添加batch维度
            src = torch.LongTensor([src_indices]).to(self.device)
            
            # 使用模型的束搜索解码
            result = self.model.beam_search_decode(
                src=src,
                max_len=max_len,
                start_token=self.tgt_vocab.sos_idx,
                end_token=self.tgt_vocab.eos_idx,
                beam_size=beam_size
            )
            
            # 转换回单词
            result_tokens = self.tgt_vocab.indices_to_words(result[0].tolist())
            
            # 后处理
            translation = self.postprocess_sentence(result_tokens)
            
            return translation
    
    def translate_batch(self, sentences, max_len=50, method='greedy'):
        """
        批量翻译
        Args:
            sentences: 英文句子列表
            max_len: 最大生成长度
            method: 解码方法 ('greedy' 或 'beam')
        Returns:
            translations: 翻译结果列表
        """
        translations = []
        
        for sentence in sentences:
            if method == 'greedy':
                translation = self.translate_greedy(sentence, max_len)
            elif method == 'beam':
                translation = self.translate_beam_search(sentence, max_len)
            else:
                raise ValueError("method must be 'greedy' or 'beam'")
            
            translations.append(translation)
        
        return translations
    
    def interactive_translate(self):
        """
        交互式翻译
        """
        print("Transformer英中翻译器")
        print("输入英文句子进行翻译，输入'quit'退出")
        print("-" * 50)
        
        while True:
            sentence = input("英文: ").strip()
            
            if sentence.lower() == 'quit':
                print("再见！")
                break
            
            if not sentence:
                continue
            
            # 记录翻译时间
            start_time = time.time()
            
            try:
                # 贪心解码
                greedy_translation = self.translate_greedy(sentence)
                
                # 束搜索解码
                beam_translation = self.translate_beam_search(sentence, beam_size=4)
                
                end_time = time.time()
                
                print(f"中文 (贪心): {greedy_translation}")
                print(f"中文 (束搜索): {beam_translation}")
                print(f"翻译时间: {end_time - start_time:.3f}秒")
                print("-" * 50)
                
            except Exception as e:
                print(f"翻译出错: {e}")
                print("-" * 50)


def evaluate_model(model_path, test_sentences=None, device='cpu'):
    """
    评估模型性能
    Args:
        model_path: 模型检查点路径
        test_sentences: 测试句子列表
        device: 设备
    """
    if test_sentences is None:
        test_sentences = [
            "Hello world",
            "How are you",
            "Good morning",
            "Thank you very much",
            "I love you",
            "What is your name",
            "Nice to meet you",
            "The weather is good today",
            "I am learning Chinese",
            "See you tomorrow"
        ]
    
    # 加载翻译器
    translator = Translator.from_checkpoint(model_path, device)
    
    print("模型评估结果:")
    print("=" * 60)
    
    total_time = 0
    
    for sentence in test_sentences:
        start_time = time.time()
        
        # 贪心解码
        greedy_result = translator.translate_greedy(sentence)
        
        # 束搜索解码
        beam_result = translator.translate_beam_search(sentence, beam_size=4)
        
        end_time = time.time()
        total_time += end_time - start_time
        
        print(f"英文: {sentence}")
        print(f"贪心: {greedy_result}")
        print(f"束搜索: {beam_result}")
        print(f"时间: {end_time - start_time:.3f}秒")
        print("-" * 60)
    
    avg_time = total_time / len(test_sentences)
    print(f"平均翻译时间: {avg_time:.3f}秒")


def demo_translation(model_path=None):
    """
    翻译演示
    """
    if model_path is None:
        print("请提供模型检查点路径")
        return
    
    # 检查设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    try:
        # 加载翻译器
        translator = Translator.from_checkpoint(model_path, device)
        
        # 运行交互式翻译
        translator.interactive_translate()
        
    except FileNotFoundError:
        print(f"找不到模型文件: {model_path}")
        print("请先训练模型或检查文件路径")
    except Exception as e:
        print(f"加载模型时出错: {e}")


if __name__ == "__main__":
    # 演示翻译功能
    # 注意：需要先训练模型并保存检查点
    model_path = "./checkpoints/best_model.pt"
    
    print("如果模型已训练，将运行翻译演示")
    print("否则，请先运行训练脚本")
    
    # 尝试运行演示
    demo_translation(model_path)