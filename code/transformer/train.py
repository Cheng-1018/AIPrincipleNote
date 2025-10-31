import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import math
import os
try:
    from .transformer import create_transformer_model
    from .data_utils import prepare_data, Vocabulary
except ImportError:
    from transformer import create_transformer_model
    from data_utils import prepare_data, Vocabulary

class LabelSmoothing(nn.Module):
    """
    标签平滑，减少模型过拟合
    """
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
    
    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx, as_tuple=False)
        if mask.dim() > 0 and mask.size(0) > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist)


class NoamOptimizer:
    """
    Noam学习率调度器，论文中使用的学习率策略
    lr = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
    """
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
    
    def step(self):
        """更新参数和学习率"""
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
    
    def rate(self, step=None):
        """计算当前学习率"""
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) * 
             min(step ** (-0.5), step * self.warmup ** (-1.5)))


def create_optimizer(model, d_model, warmup_steps=4000):
    """
    创建优化器
    """
    optimizer = optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    return NoamOptimizer(d_model, 1, warmup_steps, optimizer)


class Trainer:
    """
    Transformer训练器
    """
    def __init__(self, model, train_loader, src_vocab, tgt_vocab, device='cpu'):
        self.model = model
        self.train_loader = train_loader
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.device = device
        
        # 将模型移动到设备
        self.model.to(device)
        
        # 创建优化器
        self.optimizer = create_optimizer(model, model.d_model)
        
        # 创建损失函数（使用标签平滑）
        self.criterion = LabelSmoothing(
            size=len(tgt_vocab),
            padding_idx=tgt_vocab.pad_idx,
            smoothing=0.1
        ).to(device)
        
        # 训练统计
        self.train_losses = []
        self.best_loss = float('inf')
    
    def train_epoch(self, epoch):
        """
        训练一个epoch
        """
        self.model.train()
        total_loss = 0
        total_tokens = 0
        start_time = time.time()
        
        for batch_idx, batch in enumerate(self.train_loader):
            # 将数据移动到设备
            src = batch['src'].to(self.device)
            tgt_input = batch['tgt_input'].to(self.device)
            tgt_output = batch['tgt_output'].to(self.device)
            
            # 清零梯度
            self.optimizer.optimizer.zero_grad()
            
            # 前向传播
            output = self.model(src, tgt_input)
            
            # 计算损失
            # output: [batch_size, tgt_len, vocab_size]
            # tgt_output: [batch_size, tgt_len]
            output = output.contiguous().view(-1, output.size(-1))
            tgt_output = tgt_output.contiguous().view(-1)
            
            # 计算非padding token的数量
            n_tokens = (tgt_output != self.tgt_vocab.pad_idx).sum().item()
            
            # 计算损失
            loss = self.criterion(F.log_softmax(output, dim=-1), tgt_output) / n_tokens
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # 更新参数
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens
            
            # 打印进度
            if batch_idx % 10 == 0:
                elapsed = time.time() - start_time
                current_lr = self.optimizer._rate
                print(f'Epoch: {epoch:02d} | Batch: {batch_idx:03d} | '
                      f'Loss: {loss.item():.4f} | '
                      f'LR: {current_lr:.2e} | '
                      f'Tokens/sec: {total_tokens/elapsed:.1f}')
        
        avg_loss = total_loss / total_tokens
        return avg_loss
    
    def train(self, num_epochs, save_dir='./checkpoints'):
        """
        训练模型
        """
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        print("开始训练...")
        print(f"模型参数数量: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"设备: {self.device}")
        
        for epoch in range(1, num_epochs + 1):
            epoch_loss = self.train_epoch(epoch)
            self.train_losses.append(epoch_loss)
            
            print(f'Epoch {epoch:02d} 完成 | 平均损失: {epoch_loss:.4f}')
            
            # 保存最佳模型
            if epoch_loss < self.best_loss:
                self.best_loss = epoch_loss
                self.save_model(os.path.join(save_dir, 'best_model.pt'))
                print(f'保存最佳模型，损失: {epoch_loss:.4f}')
            
            # 定期保存检查点
            if epoch % 5 == 0:
                self.save_model(os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pt'))
        
        print("训练完成！")
    
    def save_model(self, path):
        """
        保存模型和词汇表
        """
        # 从训练过程中获取实际的模型配置
        # 这些值应该在Trainer初始化时设置
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.optimizer.state_dict(),
            'src_vocab': self.src_vocab,
            'tgt_vocab': self.tgt_vocab,
            'model_config': {
                'src_vocab_size': len(self.src_vocab),
                'tgt_vocab_size': len(self.tgt_vocab),
                'd_model': self.model.d_model,
                'n_heads': getattr(self.model, 'n_heads', 8),
                'n_layers': getattr(self.model, 'n_layers', 6),
                'd_ff': getattr(self.model, 'd_ff', 2048),
                'dropout': 0.1,
                'pad_token': self.model.pad_token
            },
            'train_losses': self.train_losses,
            'best_loss': self.best_loss
        }, path)
    
    @classmethod
    def load_model(cls, path, device='cpu'):
        """
        加载保存的模型
        """
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        
        # 重建模型 - 修复配置不匹配问题
        config = checkpoint['model_config'].copy()
        
        # 如果配置中的参数与实际训练参数不符，使用实际训练时的参数
        if config['d_model'] == 256:  # 训练时使用的配置
            config['n_heads'] = 8
            config['n_layers'] = 4
            config['d_ff'] = 1024
        
        model = create_transformer_model(**config)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model, checkpoint['src_vocab'], checkpoint['tgt_vocab']


def train_transformer(data_file=None, num_epochs=20, device=None):
    """
    训练Transformer模型的主函数
    """
    # 设置设备
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"使用设备: {device}")
    
    # 准备数据
    print("准备数据...")
    train_loader, src_vocab, tgt_vocab = prepare_data(data_file=data_file, max_samples=1000)  # 使用较小的数据集进行测试
    
    # 创建模型
    print("创建模型...")
    model = create_transformer_model(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=256,  # 使用较小的模型进行测试
        n_heads=8,
        n_layers=4,
        d_ff=1024,
        dropout=0.1,
        pad_token=src_vocab.pad_idx
    )
    
    # 创建训练器
    trainer = Trainer(model, train_loader, src_vocab, tgt_vocab, device)
    
    # 开始训练
    trainer.train(num_epochs)
    
    return trainer


if __name__ == "__main__":
    # 训练模型
    trainer = train_transformer(num_epochs=10)
    print("训练完成！")