import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM
from peft import get_peft_model, LoraConfig, TaskType
from dp_transformers import DPTrainer, DPTrainingArguments
from transformers import TrainingArguments
import math

class PrivacyTuner:
    """
    使用差分隐私和 LoRA 对模型进行微调的类
    """
    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        self.trainer = None
        
    def create_lora_config(self):
        """
        创建 LoRA 配置
        
        返回:
        - lora_config: LoRA 配置对象
        """
        return LoraConfig(
            r=8,  # LoRA 的秩
            lora_alpha=32,  # LoRA 的 alpha 参数
            target_modules=["q_proj", "v_proj"],  # 需要应用 LoRA 的模块
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM if self.args.model == 'gpt' else TaskType.SEQ_2_SEQ_LM
        )

    def load_model(self, model_name_or_path):
        """
        加载预训练模型并应用 LoRA
        
        参数:
        - model_name_or_path: 模型名称或路径
        
        返回:
        - model: 加载并应用 LoRA 的模型
        """
        # 加载基础模型
        if self.args.model == 'gpt':
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                device_map="auto",
                torch_dtype=torch.float16
            )
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name_or_path,
                device_map="auto",
                torch_dtype=torch.float16
            )
        
        # 应用 LoRA
        lora_config = self.create_lora_config()
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        return model

    def train(self, model, train_dataset, eval_dataset=None):
        """
        使用差分隐私和 LoRA 训练模型
        
        参数:
        - model: 要微调的模型
        - train_dataset: 训练数据集
        - eval_dataset: 评估数据集（可选）
        
        返回:
        - model: 微调后的模型
        """
        # 计算差分隐私参数
        target_epsilon = self.args.target_epsilon
        sampling_probability = self.args.batch_size / len(train_dataset)
        
        # 创建训练参数
        training_args = DPTrainingArguments(
            output_dir=self.args.output_dir,
            num_train_epochs=self.args.privacy_epochs,
            per_device_train_batch_size=self.args.batch_size,
            target_epsilon=target_epsilon,
            target_delta=self.args.target_delta,
            noise_multiplier=self.args.noise_multiplier,
            max_grad_norm=self.args.max_grad_norm,
            gradient_accumulation_steps=1,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=10,
            save_strategy="epoch",
            evaluation_strategy="epoch" if eval_dataset else "no",
            remove_unused_columns=False,
            push_to_hub=False,
        )

        # 创建 DP Trainer
        trainer = DPTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
        )
        
        self.trainer = trainer
        
        # 开始训练
        train_result = trainer.train()
        
        # 打印训练结果和隐私预算
        print(f"Training completed. Final privacy budget: ε = {trainer.get_epsilon():.2f}")
        
        return model

    def save_model(self, model, save_path):
        """
        保存微调后的模型
        
        参数:
        - model: 微调后的模型
        - save_path: 保存路径
        """
        # 创建保存目录
        os.makedirs(save_path, exist_ok=True)
        
        # 保存模型和分词器
        model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        # 保存隐私参数
        privacy_params = {
            'noise_multiplier': self.args.noise_multiplier,
            'max_grad_norm': self.args.max_grad_norm,
            'target_epsilon': self.args.target_epsilon,
            'target_delta': self.args.target_delta,
            'final_epsilon': self.trainer.get_epsilon() if self.trainer else None
        }
        
        torch.save(privacy_params, os.path.join(save_path, 'privacy_params.pt'))
        
        print(f"模型已保存到 {save_path}")
        if self.trainer:
            print(f"最终隐私预算: ε = {privacy_params['final_epsilon']:.2f}") 