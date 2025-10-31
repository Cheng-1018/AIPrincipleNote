"""
提取训练好的模型中的词汇表并保存为JSON格式
"""
import torch
import json
import argparse
import os

try:
    from .train import Trainer
except ImportError:
    from train import Trainer

def extract_vocabularies(model_path, output_dir="./vocab"):
    """
    从训练好的模型中提取词汇表
    Args:
        model_path: 模型文件路径
        output_dir: 输出目录
    """
    print(f"加载模型: {model_path}")
    
    # 加载模型和词汇表
    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        src_vocab = checkpoint['src_vocab']
        tgt_vocab = checkpoint['tgt_vocab']
        
        print(f"源语言词汇表大小: {len(src_vocab)}")
        print(f"目标语言词汇表大小: {len(tgt_vocab)}")
        
    except Exception as e:
        print(f"加载模型失败: {e}")
        return
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 提取源语言词汇表
    src_vocab_data = {
        "vocab_size": len(src_vocab),
        "word2idx": src_vocab.word2idx,
        "idx2word": src_vocab.idx2word,
        "special_tokens": {
            "PAD_TOKEN": src_vocab.PAD_TOKEN,
            "SOS_TOKEN": src_vocab.SOS_TOKEN,
            "EOS_TOKEN": src_vocab.EOS_TOKEN,
            "UNK_TOKEN": src_vocab.UNK_TOKEN,
            "pad_idx": src_vocab.pad_idx,
            "sos_idx": src_vocab.sos_idx,
            "eos_idx": src_vocab.eos_idx,
            "unk_idx": src_vocab.unk_idx
        },
        "word_count": dict(src_vocab.word_count)
    }
    
    # 提取目标语言词汇表
    tgt_vocab_data = {
        "vocab_size": len(tgt_vocab),
        "word2idx": tgt_vocab.word2idx,
        "idx2word": tgt_vocab.idx2word,
        "special_tokens": {
            "PAD_TOKEN": tgt_vocab.PAD_TOKEN,
            "SOS_TOKEN": tgt_vocab.SOS_TOKEN,
            "EOS_TOKEN": tgt_vocab.EOS_TOKEN,
            "UNK_TOKEN": tgt_vocab.UNK_TOKEN,
            "pad_idx": tgt_vocab.pad_idx,
            "sos_idx": tgt_vocab.sos_idx,
            "eos_idx": tgt_vocab.eos_idx,
            "unk_idx": tgt_vocab.unk_idx
        },
        "word_count": dict(tgt_vocab.word_count)
    }
    
    # 保存源语言词汇表
    src_vocab_path = os.path.join(output_dir, "src_vocab.json")
    with open(src_vocab_path, 'w', encoding='utf-8') as f:
        json.dump(src_vocab_data, f, ensure_ascii=False, indent=2)
    print(f"源语言词汇表已保存到: {src_vocab_path}")
    
    # 保存目标语言词汇表
    tgt_vocab_path = os.path.join(output_dir, "tgt_vocab.json")
    with open(tgt_vocab_path, 'w', encoding='utf-8') as f:
        json.dump(tgt_vocab_data, f, ensure_ascii=False, indent=2)
    print(f"目标语言词汇表已保存到: {tgt_vocab_path}")
    
    # 创建合并的词汇表信息
    combined_vocab = {
        "model_path": model_path,
        "src_vocab": src_vocab_data,
        "tgt_vocab": tgt_vocab_data,
        "vocab_summary": {
            "src_vocab_size": len(src_vocab),
            "tgt_vocab_size": len(tgt_vocab),
            "src_top_words": list(src_vocab.word_count.most_common(10)),
            "tgt_top_words": list(tgt_vocab.word_count.most_common(10))
        }
    }
    
    # 保存合并的词汇表
    combined_path = os.path.join(output_dir, "vocabularies.json")
    with open(combined_path, 'w', encoding='utf-8') as f:
        json.dump(combined_vocab, f, ensure_ascii=False, indent=2)
    print(f"合并词汇表已保存到: {combined_path}")
    
    # 打印词汇表统计信息
    print("\n=== 词汇表统计信息 ===")
    print(f"源语言词汇表大小: {len(src_vocab)}")
    print(f"目标语言词汇表大小: {len(tgt_vocab)}")
    
    print(f"\n源语言高频词TOP10:")
    for word, count in src_vocab.word_count.most_common(10):
        print(f"  {word}: {count}")
    
    print(f"\n目标语言高频词TOP10:")
    for word, count in tgt_vocab.word_count.most_common(10):
        print(f"  {word}: {count}")

def create_human_readable_vocab(model_path, output_dir="./vocab"):
    """
    创建人类可读的词汇表格式
    """
    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        src_vocab = checkpoint['src_vocab']
        tgt_vocab = checkpoint['tgt_vocab']
        
    except Exception as e:
        print(f"加载模型失败: {e}")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建源语言词汇表文件
    src_txt_path = os.path.join(output_dir, "src_vocab.txt")
    with open(src_txt_path, 'w', encoding='utf-8') as f:
        f.write("源语言（英文）词汇表\n")
        f.write("=" * 50 + "\n")
        f.write(f"总词汇数: {len(src_vocab)}\n\n")
        
        f.write("索引\t词汇\t频次\n")
        f.write("-" * 30 + "\n")
        for idx, word in src_vocab.idx2word.items():
            count = src_vocab.word_count.get(word, 0)
            f.write(f"{idx}\t{word}\t{count}\n")
    
    # 创建目标语言词汇表文件
    tgt_txt_path = os.path.join(output_dir, "tgt_vocab.txt")
    with open(tgt_txt_path, 'w', encoding='utf-8') as f:
        f.write("目标语言（中文）词汇表\n")
        f.write("=" * 50 + "\n")
        f.write(f"总词汇数: {len(tgt_vocab)}\n\n")
        
        f.write("索引\t词汇\t频次\n")
        f.write("-" * 30 + "\n")
        for idx, word in tgt_vocab.idx2word.items():
            count = tgt_vocab.word_count.get(word, 0)
            f.write(f"{idx}\t{word}\t{count}\n")
    
    print(f"人类可读的词汇表已保存到:")
    print(f"  {src_txt_path}")
    print(f"  {tgt_txt_path}")

def main():
    parser = argparse.ArgumentParser(description='提取Transformer模型词汇表')
    parser.add_argument('--model_path', type=str, default='./checkpoints/best_model.pt',
                       help='模型文件路径')
    parser.add_argument('--output_dir', type=str, default='./vocab',
                       help='输出目录')
    parser.add_argument('--format', choices=['json', 'txt', 'both'], default='both',
                       help='输出格式: json, txt, 或 both')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"错误: 找不到模型文件 {args.model_path}")
        return
    
    print("提取Transformer模型词汇表")
    print("=" * 50)
    
    if args.format in ['json', 'both']:
        extract_vocabularies(args.model_path, args.output_dir)
    
    if args.format in ['txt', 'both']:
        create_human_readable_vocab(args.model_path, args.output_dir)
    
    print("\n词汇表提取完成!")

if __name__ == "__main__":
    main()