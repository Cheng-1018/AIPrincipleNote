"""
Transformer 英中翻译模型
主程序入口，整合训练和预测功能

使用方法:
    python main.py --mode train --epochs 20          # 训练模型
    python main.py --mode predict --model_path ./checkpoints/best_model.pt  # 预测翻译
    python main.py --mode eval --model_path ./checkpoints/best_model.pt     # 评估模型
"""

import argparse
import torch
import os
import sys
try:
    from .train import train_transformer, Trainer
    from .predict import Translator, evaluate_model, demo_translation
    from .data_utils import prepare_data
except ImportError:
    from train import train_transformer, Trainer
    from predict import Translator, evaluate_model, demo_translation
    from data_utils import prepare_data

def main():
    parser = argparse.ArgumentParser(description='Transformer英中翻译模型')
    parser.add_argument('--mode', choices=['train', 'predict', 'eval'], required=True,
                       help='运行模式: train(训练), predict(预测), eval(评估)')
    parser.add_argument('--epochs', type=int, default=20,
                       help='训练轮数 (默认: 20)')
    parser.add_argument('--model_path', type=str, default='./checkpoints/best_model.pt',
                       help='模型路径 (默认: ./checkpoints/best_model.pt)')
    parser.add_argument('--data_file', type=str, default='./data/en-zh-cn-39k-without-think-alpaca.jsonl',
                       help='英文数据文件路径 (可选)')
    parser.add_argument('--device', type=str, default=None,
                       help='设备: cpu 或 cuda (默认: 自动检测)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小 (默认: 32)')
    parser.add_argument('--max_samples', type=int, default=1000,
                       help='最大样本数 (默认: 1000)')
    
    args = parser.parse_args()
    
    # 设置设备
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"使用设备: {device}")
    
    if args.mode == 'train':
        print("=" * 60)
        print("训练 Transformer 英中翻译模型")
        print("=" * 60)
        #如果中英文数据文件不存在，提示用户
        if not os.path.exists(args.data_file):
            args.data_file = None
            print("未找到指定的数据文件，使用默认数据集进行训练。")
        try:
            # 训练模型
            trainer = train_transformer(
                data_file=args.data_file,
                num_epochs=args.epochs,
                device=device
            )
            
            print("\n训练完成！")
            print(f"最佳模型已保存到: {os.path.join('./checkpoints', 'best_model.pt')}")
            
        except Exception as e:
            print(f"训练过程中出错: {e}")
            sys.exit(1)
    
    elif args.mode == 'predict':
        print("=" * 60)
        print("Transformer 英中翻译预测")
        print("=" * 60)
        
        if not os.path.exists(args.model_path):
            print(f"错误: 找不到模型文件 {args.model_path}")
            print("请先训练模型或检查模型路径")
            sys.exit(1)
        
        try:
            # 运行交互式翻译
            demo_translation(args.model_path)
            
        except Exception as e:
            print(f"预测过程中出错: {e}")
            sys.exit(1)
    
    elif args.mode == 'eval':
        print("=" * 60)
        print("Transformer 模型评估")
        print("=" * 60)
        
        if not os.path.exists(args.model_path):
            print(f"错误: 找不到模型文件 {args.model_path}")
            print("请先训练模型或检查模型路径")
            sys.exit(1)
        
        try:
            # 评估模型
            evaluate_model(args.model_path, device=device)
            
        except Exception as e:
            print(f"评估过程中出错: {e}")
            sys.exit(1)


def quick_demo():
    """
    快速演示：训练一个小模型并进行翻译测试
    """
    print("=" * 60)
    print("Transformer 快速演示")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    print("\n1. 准备数据...")
    train_loader, src_vocab, tgt_vocab = prepare_data(max_samples=100)
    
    print("\n2. 创建并训练模型...")
    # 使用更小的模型进行快速演示
    try:
        from .transformer import create_transformer_model
        from .train import Trainer
    except ImportError:
        from transformer import create_transformer_model
        from train import Trainer
    
    model = create_transformer_model(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=128,  # 更小的模型
        n_heads=4,
        n_layers=2,
        d_ff=512,
        dropout=0.1,
        pad_token=src_vocab.pad_idx
    )
    
    trainer = Trainer(model, train_loader, src_vocab, tgt_vocab, device)
    
    # 训练几个epoch
    for epoch in range(1, 4):
        loss = trainer.train_epoch(epoch)
        print(f"Epoch {epoch}: Loss = {loss:.4f}")
    
    print("\n3. 测试翻译...")
    translator = Translator(model, src_vocab, tgt_vocab, device)
    
    test_sentences = [
        "hello world",
        "good morning",
        "thank you",
        "how are you"
    ]
    
    for sentence in test_sentences:
        translation = translator.translate_greedy(sentence)
        print(f"英文: {sentence}")
        print(f"中文: {translation}")
        print("-" * 40)
    
    print("演示完成！")


def print_usage():
    """
    打印使用说明
    """
    print("Transformer 英中翻译模型使用说明")
    print("=" * 60)
    print()
    print("1. 训练模型:")
    print("   python main.py --mode train --epochs 20")
    print()
    print("2. 交互式翻译:")
    print("   python main.py --mode predict --model_path ./checkpoints/best_model.pt")
    print()
    print("3. 评估模型:")
    print("   python main.py --mode eval --model_path ./checkpoints/best_model.pt")
    print()
    print("4. 快速演示:")
    print("   python -c \"from main import quick_demo; quick_demo()\"")
    print()
    print("可选参数:")
    print("   --device cpu/cuda     指定设备")
    print("   --epochs N            训练轮数")
    print("   --max_samples N       最大样本数")
    print("   --batch_size N        批次大小")
    print()


if __name__ == "__main__":
    # 如果没有参数，显示使用说明
    if len(sys.argv) == 1:
        print_usage()
        
        # 询问是否运行快速演示
        response = input("是否运行快速演示? (y/n): ").strip().lower()
        if response in ['y', 'yes', '是']:
            quick_demo()
    else:
        main()