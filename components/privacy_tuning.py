import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM
from dp_transformers import PrivacyEngine
from tqdm import tqdm as progress_bar

from assets.static_vars import device, dtype

class PrivacyTuner:
    """
    使用差分隐私对模型进行微调的类
    """
    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        self.privacy_engine = None
        
    def load_model(self, model_name_or_path):
        """
        加载预训练模型
        
        参数:
        - model_name_or_path: 模型名称或路径
        
        返回:
        - model: 加载的模型
        """
        if self.args.model == 'gpt':
            model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
        
        return model.to(device)
    
    def setup_privacy_engine(self, model, dataloader):
        """
        设置隐私引擎
        
        参数:
        - model: 要微调的模型
        - dataloader: 数据加载器
        
        返回:
        - privacy_engine: 隐私引擎
        """
        privacy_engine = PrivacyEngine(
            model,
            batch_size=self.args.batch_size,
            sample_size=len(dataloader.dataset),
            epochs=self.args.privacy_epochs,
            noise_multiplier=self.args.noise_multiplier,
            max_grad_norm=self.args.max_grad_norm,
            target_epsilon=self.args.target_epsilon,
            target_delta=self.args.target_delta,
            accounting_mode=self.args.accounting_mode
        )
        
        self.privacy_engine = privacy_engine
        return privacy_engine
    
    def train(self, model, dataloader, optimizer, scheduler=None):
        """
        使用差分隐私训练模型
        
        参数:
        - model: 要微调的模型
        - dataloader: 数据加载器
        - optimizer: 优化器
        - scheduler: 学习率调度器
        
        返回:
        - model: 微调后的模型
        """
        # 设置隐私引擎
        privacy_engine = self.setup_privacy_engine(model, dataloader)
        
        # 将优化器包装到隐私引擎中
        optimizer = privacy_engine.make_optimizer(optimizer)
        
        # 训练循环
        model.train()
        for epoch in range(self.args.privacy_epochs):
            total_loss = 0
            for batch in progress_bar(dataloader, desc=f"Epoch {epoch+1}/{self.args.privacy_epochs}"):
                optimizer.zero_grad()
                
                # 根据模型类型处理输入
                if self.args.model == 'gpt':
                    inputs = {
                        'input_ids': batch['input_ids'].to(device),
                        'attention_mask': batch['attention_mask'].to(device),
                        'labels': batch['input_ids'].to(device)
                    }
                else:
                    inputs = {
                        'input_ids': batch['input_ids'].to(device),
                        'attention_mask': batch['attention_mask'].to(device),
                        'decoder_input_ids': batch['decoder_input_ids'].to(device),
                        'labels': batch['labels'].to(device)
                    }
                
                outputs = model(**inputs)
                loss = outputs.loss
                total_loss += loss.item()
                
                loss.backward()
                optimizer.step()
                
                if scheduler is not None:
                    scheduler.step()
            
            # 打印每个epoch的损失
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{self.args.privacy_epochs}, Loss: {avg_loss:.4f}")
            
            # 打印隐私预算
            epsilon = privacy_engine.get_epsilon(self.args.target_delta)
            print(f"Privacy budget: ε = {epsilon:.2f}")
        
        return model
    
    def save_model(self, model, save_path):
        """
        保存微调后的模型
        
        参数:
        - model: 微调后的模型
        - save_path: 保存路径
        """
        # 创建保存目录
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 保存模型
        model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        # 保存隐私参数
        privacy_params = {
            'noise_multiplier': self.args.noise_multiplier,
            'max_grad_norm': self.args.max_grad_norm,
            'target_epsilon': self.args.target_epsilon,
            'target_delta': self.args.target_delta,
            'accounting_mode': self.args.accounting_mode,
            'final_epsilon': self.privacy_engine.get_epsilon(self.args.target_delta) if self.privacy_engine else None
        }
        
        torch.save(privacy_params, os.path.join(os.path.dirname(save_path), 'privacy_params.pt'))
        
        print(f"模型已保存到 {save_path}")
        if self.privacy_engine:
            print(f"最终隐私预算: ε = {privacy_params['final_epsilon']:.2f}") 