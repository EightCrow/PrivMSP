import os
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, TrainingArguments, AutoModelForSequenceClassification
from peft import get_peft_model, LoraConfig, TaskType
from dp_transformers import PrivacyArguments, dp_utils
import math
import json

class PrivacyDataset(Dataset):
    """用于隐私微调的数据集类"""
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 构建输入文本
        text = item["text"]
        
        # 构建标签文本
        label_text = ""
        if "intents" in item:
            label_text += "Intents: " + ", ".join(item["intents"]) + "\n"
        if "slots" in item and item["slots"]:
            slots_text = []
            for slot_name, slot_info in item["slots"].items():
                if isinstance(slot_info, dict):
                    slot_text = f"{slot_name}: {slot_info.get('text', '')}"
                    if "value" in slot_info:
                        slot_text += f" ({slot_info['value']})"
                    slots_text.append(slot_text)
            if slots_text:
                label_text += "Slots: " + "; ".join(slots_text)
        
        # 编码输入文本
        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # 编码标签文本
        labels = self.tokenizer(
            label_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": labels["input_ids"].squeeze()
        }

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
        """
        # 根据模型类型选择目标模块
        if self.args.model == 'gpt':
            target_modules = ["c_attn", "c_proj"]  # GPT 模型的注意力模块
        elif self.args.model == 't5' or self.args.model == 'godel':
            target_modules = ["q_proj", "v_proj"]  # T5/GODEL 模型的注意力模块
        elif self.args.model == 'opt':
            target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]  # OPT 模型的注意力模块
        elif self.args.model == 'bert':
            target_modules = ["query", "value"]  # BERT/RoBERTa/DeBERTa 模型的注意力模块
        else:
            raise ValueError(f"不支持的模型类型: {self.args.model}")
            
        return LoraConfig(
            r=4,  # 减小 LoRA 的秩
            lora_alpha=16,  # 减小 alpha 参数
            target_modules=target_modules,
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM if self.args.model in ['gpt', 'opt'] else TaskType.SEQ_2_SEQ_LM,
            inference_mode=False,
            fan_in_fan_out=False
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
        if self.args.model in ['gpt', 'opt']:
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                device_map="auto"
            )
        elif self.args.model in ['t5', 'godel']:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name_or_path,
                device_map="auto"
            )
        elif self.args.model == 'bert':
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name_or_path,
                device_map="auto"
            )
        else:
            raise ValueError(f"不支持的模型类型: {self.args.model}")
        
        # 在应用LoRA之前启用梯度检查点
        model.gradient_checkpointing_enable()
        
        # 应用 LoRA
        lora_config = self.create_lora_config()
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        return model

    def prepare_dataset(self, dataset, split="train"):
        """准备数据集"""
        # 加载数据
        data = []
        if split == "train":
            with open(f"assets/{self.args.dataset}/train.json", "r", encoding="utf-8") as f:
                data = json.load(f)
        elif split == "dev":
            with open(f"assets/{self.args.dataset}/dev.json", "r", encoding="utf-8") as f:
                data = json.load(f)
        elif split == "test":
            with open(f"assets/{self.args.dataset}/test.json", "r", encoding="utf-8") as f:
                data = json.load(f)
            
        # 过滤指定域的数据
        if self.args.domain:
            data = [item for item in data if item.get("domain") == self.args.domain]
        
        # 创建数据集
        dataset = PrivacyDataset(data, self.tokenizer)
        return dataset

    def train(self, model, train_dataset, eval_dataset=None):
        """
        使用差分隐私和 LoRA 训练模型
        """
        # 准备数据集
        train_dataset = self.prepare_dataset(train_dataset)
        if eval_dataset:
            eval_dataset = self.prepare_dataset(eval_dataset)
        
        # 计算差分隐私参数
        sampling_probability = min(0.1, self.args.batch_size / len(train_dataset))  # 限制采样概率的上限
        print(f"Sampling probability: {sampling_probability:.6f}")  # 打印采样概率以便调试
        
        # 创建训练参数
        training_args = TrainingArguments(
            output_dir=self.args.output_dir,
            num_train_epochs=self.args.privacy_epochs,
            per_device_train_batch_size=self.args.batch_size,
            gradient_accumulation_steps=32,  # 增加梯度累积步数
            learning_rate=2e-4,
            logging_steps=10,
            save_strategy="epoch",
            evaluation_strategy="epoch" if eval_dataset else "no",
            remove_unused_columns=False,
            push_to_hub=False,
            max_grad_norm=self.args.max_grad_norm,  # 使用args中的max_grad_norm
            optim="adamw_torch",
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            group_by_length=True,
        )

        # 创建隐私参数
        privacy_args = PrivacyArguments(
            target_epsilon=self.args.target_epsilon,
            target_delta=self.args.target_delta,
            per_sample_max_grad_norm=self.args.max_grad_norm,  # 添加每个样本的最大梯度范数
            noise_multiplier=self.args.noise_multiplier  # 添加噪声乘数
        )

        # 创建 DP Trainer
        trainer = dp_utils.OpacusDPTrainer(
            model=model,
            args=training_args,
            privacy_args=privacy_args,
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
            'target_delta': self.args.target_delta,
            'final_epsilon': self.trainer.get_epsilon() if self.trainer else None
        }
        
        torch.save(privacy_params, os.path.join(save_path, 'privacy_params.pt'))
        
        print(f"模型已保存到 {save_path}")
        if self.trainer:
            print(f"最终隐私预算: ε = {privacy_params['final_epsilon']:.2f}") 