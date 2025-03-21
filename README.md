# 隐私增强的软提示可控数据生成

## 新增功能

本项目在原有MSP基础上增加了以下新功能：

1. **隐私微调**：使用dp-transformer库对模型进行差分隐私微调，保护训练数据隐私。
2. **改进的去噪过滤算法**：实现了参数化差分隐私去噪过滤算法，支持多种分布调整策略。
3. **MAUVE评分**：使用MAUVE评分衡量指令之间的差异，提供更全面的评估指标。s
4. **可视化支持**：提供数据分布和评估结果的可视化功能。

## 项目结构

```
mixture_soft_prompts/
├── assets/                # 静态资源和变量
├── components/            # 核心组件
│   ├── additional.py      # 额外组件和辅助功能
│   ├── augmenter.py       # 数据增强器实现
│   ├── ct_generator.py    # 可控文本生成器实现
│   ├── datasets.py        # 数据集处理和加载
│   ├── denoise.py         # 去噪机制实现
│   ├── engineer.py        # 提示工程实现
│   ├── logger.py          # 日志记录器实现
│   ├── models.py          # 模型定义和实现
│   ├── privacy_tuning.py  # 隐私微调实现
│   └── soft_embedder.py   # 软提示嵌入实现
├── data/                  # 数据目录
├── models/                # 模型保存目录
│   └── privacy_tuned/     # 隐私微调后的模型
├── scripts/               # 数据处理脚本
├── utils/                 # 工具函数
│   ├── arguments.py       # 命令行参数处理
│   ├── evaluate.py        # 评估功能实现
│   ├── evaluate_mauve.py  # MAUVE评分实现
│   ├── help.py            # 辅助函数
│   ├── load.py            # 数据和模型加载功能
│   ├── process.py         # 数据处理功能
│   └── synthesize.py      # 数据合成功能
├── visualizations/        # 可视化结果保存目录
├── .gitignore             # Git忽略文件
├── interact.py            # 交互式界面实现
├── LICENSE                # 许可证文件
├── main.py                # 主程序入口
├── requirements.txt       # 依赖项列表
└── run.sh                 # 运行脚本
```

## 安装

1. 克隆仓库：

   ```
   git clone https://github.com/yourusername/mixture_soft_prompts.git
   cd mixture_soft_prompts
   ```

2. 安装依赖：

   ```
   pip install -r requirements.txt
   ```

3. 下载数据集：
   - [ActDial](https://github.com/Jianqiao-Zhao/FlowEval/tree/main/ActDial)
   - [NLU++](https://github.com/PolyAI-LDN/task-specific-datasets/tree/master/nlupp)
   - [CrossNER](https://github.com/zliucr/CrossNER)
   - [TopV2](https://fb.me/TOPv2Dataset)

## 使用方法

### 软提示训练

```bash
python main.py --dataset nlu++ --task soft_prompt --do-train --n-tokens 100 --domain hotels \
      --model godel --size large --source-max-len 128 --quantify --qualify --log-interval 100 \
      --n-epochs 14 --learning-rate 0.1 --batch-size 8 --setting cross --verbose
```

### 隐私微调

```bash
python main.py --dataset nlu++ --task privacy_tune --do-train --domain hotels \
      --model godel --size medium --source-max-len 128 --log-interval 100 \
      --privacy_tune --privacy_epochs 3 --noise_multiplier 1.0 --max_grad_norm 1.0 \
      --target_epsilon 8.0 --target_delta 1e-5 --accounting_mode rdp \
      --privacy_model_save_dir models/privacy_tuned --verbose
```

### 使用改进的去噪过滤算法生成数据

```bash
python main.py --dataset crossner --task synthesize --source-max-len 64 --setting full \
      --model godel --size medium --method msp --mixture concat --num-shot 2 --domain music \
      --advanced_filter --do-save --temperature 2.0 --threshold 2.0 \
      --num_clusters 10 --dp_epsilon 1.0 --dp_delta 1e-5 --target_samples 1000 \
      --distribution_alpha 0.5 --rarity_beta 2.0 --similarity_threshold 0.7 \
      --distribution_type balanced --visualize_distribution --visualization_dir visualizations
```

### 使用MAUVE评分评估生成数据

```bash
python main.py --dataset crossner --task synthesize --source-max-len 64 --setting full \
      --model godel --size medium --method msp --mixture concat --num-shot 2 --domain music \
      --advanced_filter --do-save --temperature 2.0 --threshold 2.0 \
      --use_mauve --mauve_model_name gpt2-large --mauve_batch_size 32 --mauve_device cuda
```

### 使用隐私微调后的模型生成数据

```bash
python main.py --dataset nlu++ --task synthesize --source-max-len 64 --setting full \
      --model godel --size medium --method msp --mixture concat --num-shot 2 --domain hotels \
      --advanced_filter --do-save --temperature 2.0 --threshold 2.0 \
      --pretrained_model_name models/privacy_tuned/nlu++_hotels --use_mauve
```

### 使用生成的数据进行下游任务训练

```bash
python main.py --dataset crossner --task end_to_end --log-interval 500 --do-train --do-save \
      --n-epochs 7 --model godel --size medium --learning-rate 3e-5 --quantify --domain literature \
      --source-max-len 256 --threshold 1.2 --method msp --mixture pooling --setting full --verbose
```

## 参数说明

### 隐私微调参数

- `--privacy_tune`: 使用dp-transformer库对模型进行隐私微调
- `--privacy_epochs`: 隐私微调的轮数
- `--noise_multiplier`: 噪声乘数，控制添加到梯度的噪声量
- `--max_grad_norm`: 梯度裁剪的最大范数
- `--target_epsilon`: 目标隐私预算ε
- `--target_delta`: 目标隐私预算δ
- `--accounting_mode`: 隐私预算计算方式
- `--pretrained_model_name`: 预训练模型名称或路径，用于隐私微调
- `--privacy_model_save_dir`: 隐私微调后模型的保存目录

### 改进的去噪过滤算法参数

- `--advanced_filter`: 使用改进的参数化差分隐私去噪过滤算法
- `--embedding_model`: 用于计算嵌入的模型名称
- `--num_clusters`: 聚类数量K
- `--dp_epsilon`: 差分隐私预算ε₂
- `--dp_delta`: 差分隐私预算δ₂
- `--target_samples`: 目标合成样本数T
- `--distribution_alpha`: 分布控制参数α (0-1之间)
- `--rarity_beta`: 稀有属性增强因子β (≥1)
- `--similarity_threshold`: 语义相似度阈值γ (0-1之间)
- `--distribution_type`: 分布类型 (original/uniform/balanced/rare_enhanced)
- `--visualize_distribution`: 可视化过滤前后的数据分布
- `--visualization_dir`: 保存可视化图表的目录

### MAUVE评分参数

- `--use_mauve`: 使用MAUVE评分衡量指令之间的差异
- `--mauve_model_name`: 用于MAUVE评分的模型名称
- `--mauve_batch_size`: MAUVE评分的批处理大小
- `--mauve_device`: MAUVE评分使用的设备
- `--mauve_verbose`: MAUVE评分是否显示详细信息

## 引用

如果您使用了本项目的代码或方法，请引用以下论文：

```
@inproceedings{msp2023,
  title={Mixture of Soft Prompts for Controllable Data Generation},
  author={Authors},
  booktitle={Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year={2023}
}
```