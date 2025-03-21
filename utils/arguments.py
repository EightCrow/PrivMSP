import argparse
import os

def solicit_params():
  parser = argparse.ArgumentParser()

  # Required parameters
  parser.add_argument("--input-dir", default='assets', type=str,
        help="The input training data file (a text file).")
  parser.add_argument("--output-dir", default='results', type=str,
        help="Output directory where the model predictions and checkpoints are written.")
  parser.add_argument("--dataset", default='actdial', type=str,
        choices=['actdial', 'banking', 'crossner', 'nlu++', 'topv2'],
        help="which dataset to choose from out of all possible options")
  parser.add_argument("--domain", default=None, type=str,
        choices=['ai', 'literature', 'music', 'politics', 'science', 'banking', \
        'hotels', 'reminder', 'weather'], help="Target domain to be tested on.")
  parser.add_argument("--task", default='fine_tune', type=str,
        choices=['synthesize', 'end_to_end', 'fine_tune', 'soft_prompt', 'in_context', 'classify', 'privacy_tune'],
        help="synthesize is for DataAug and CTG, e2e trains with generated data, \
        fine_tune updates all gradients, soft_prompt trains prompts only, \
        in_context performs no backprop at all, just inference, \
        classify trains an attribute classifier for correctness automatic evaluation, \
        privacy_tune uses dp-transformer to fine-tune model with privacy guarantees")
  parser.add_argument("--model", default='gpt', type=str, 
        choices=['t5', 'gpt', 'godel', 'aug', 'bert', 'api'],
        help="The model architecture to be trained or fine-tuned.")
  parser.add_argument("--size", default='small', type=str, choices=['small', 'medium', 'large', 'giant'],
        help="Size of the model, use small for debugging, but report results on giant")
  parser.add_argument("--openai-key", default='', type=str,
        help="The API key for OpenAI's GPT-3 API")
  parser.add_argument("--checkpoint", default='', type=str,
        help="Enter the filename of a checkpoint for manual override")
  parser.add_argument("--icl-type", default='base', type=str, 
        choices=['base', 'cot'],
        help="Prompt engineering for ICL prediction: base, chain of thought.")
  parser.add_argument("--seed", default=42, type=int)

  # Custom paper parameters
  parser.add_argument("--method", default='none', type=str,
        choices=['eda', 'para', 'fill', 'rtt', 'cvae', 'dexpert', 'clm', 'msp', 'none'],
        help="Method of dataset generation, includes both DataAug and CTG")
  parser.add_argument("--mixture", default='concat', type=str,
        choices=['concat', 'attention', 'bottleneck', 'cnn', 'pooling', 'injection'],
        help="How to mix the soft prompts together")
  parser.add_argument("--num-shot", default=2, type=int,
        help="How many exemplars or K-shot to include when performing few shot synthesis")
  parser.add_argument("--num-generations", default=4, type=int,
        help="The multiplier on the number of generations compared to size of the seed set")
  parser.add_argument("--threshold", default=1.4, type=float,
        help="Used as the repetition penalty during inference of generation")
  parser.add_argument("--temperature", default=1.0, type=float,
        help="Temperature for increasing diversity when decoding, mainly for paraphrase")
  parser.add_argument("--source-max-len", default=256, type=int,
        help="Default input length for a model")
  parser.add_argument("--target-max-len", default=128, type=int,
        help="Default output length for a model")
  parser.add_argument("--n-tokens", default=100, type=int,
        help="Number of soft prompt tokens to tune")
  parser.add_argument("--do-guide", action="store_true",
        help="Use additional guidance such as domain during generation")
  parser.add_argument("--filter", action="store_true",
        help="Run additional denoising to clean up the dataset")
  parser.add_argument("--metric", default="f1_score", type=str,
        choices=["accuracy", "f1_score", "intents_acc", "slots_acc", "bleu"],
        help="type of metric to optimize")
  parser.add_argument("--setting", default='few_shot', type=str,
        choices=['few_shot', 'full', 'additional', 'kfold', 'cross'],
        help="Method of dataset preparation, details still unclear")

  # 改进的去噪过滤算法参数
  parser.add_argument("--advanced_filter", action="store_true",
        help="使用改进的参数化差分隐私去噪过滤算法")
  parser.add_argument("--embedding_model", default="all-MiniLM-L6-v2", type=str,
        help="用于计算嵌入的模型名称")
  parser.add_argument("--num_clusters", default=10, type=int,
        help="聚类数量K")
  parser.add_argument("--dp_epsilon", default=1.0, type=float,
        help="差分隐私预算ε₂")
  parser.add_argument("--dp_delta", default=1e-5, type=float,
        help="差分隐私预算δ₂")
  parser.add_argument("--target_samples", default=1000, type=int,
        help="目标合成样本数T")
  parser.add_argument("--distribution_alpha", default=0.5, type=float,
        help="分布控制参数α (0-1之间，控制均衡度，0表示完全拟合原始分布，1表示完全均衡分布)")
  parser.add_argument("--rarity_beta", default=2.0, type=float,
        help="稀有属性增强因子β (≥1，控制稀有属性的采样权重)")
  parser.add_argument("--similarity_threshold", default=0.7, type=float,
        help="语义相似度阈值γ (0-1之间，控制语义过滤严格程度)")
  parser.add_argument("--distribution_type", default="balanced", type=str,
        choices=["original", "uniform", "balanced", "rare_enhanced"],
        help="分布类型 (original: 拟合原始分布, uniform: 均匀分布, balanced: 平衡分布, rare_enhanced: 稀有属性增强)")
  parser.add_argument("--visualize_distribution", action="store_true",
        help="可视化过滤前后的数据分布")
  parser.add_argument("--visualization_dir", default="visualizations", type=str,
        help="保存可视化图表的目录")

  # 隐私微调相关参数
  parser.add_argument("--privacy_tune", action="store_true",
        help="使用dp-transformer库对模型进行隐私微调")
  parser.add_argument("--privacy_epochs", default=3, type=int,
        help="隐私微调的轮数")
  parser.add_argument("--noise_multiplier", default=1.0, type=float,
        help="噪声乘数，控制添加到梯度的噪声量")
  parser.add_argument("--max_grad_norm", default=1.0, type=float,
        help="梯度裁剪的最大范数")
  parser.add_argument("--target_epsilon", default=8.0, type=float,
        help="目标隐私预算ε")
  parser.add_argument("--target_delta", default=1e-5, type=float,
        help="目标隐私预算δ")
  parser.add_argument("--accounting_mode", default="rdp", type=str,
        choices=["rdp", "glw", "all"],
        help="隐私预算计算方式")
  parser.add_argument("--pretrained_model_name", default="", type=str,
        help="预训练模型名称或路径，用于隐私微调")
  parser.add_argument("--privacy_model_save_dir", default="models/privacy_tuned", type=str,
        help="隐私微调后模型的保存目录")

  # MAUVE评分相关参数
  parser.add_argument("--use_mauve", action="store_true",
        help="使用MAUVE评分衡量指令之间的差异")
  parser.add_argument("--mauve_model_name", default="gpt2-large", type=str,
        help="用于MAUVE评分的模型名称")
  parser.add_argument("--mauve_batch_size", default=32, type=int,
        help="MAUVE评分的批处理大小")
  parser.add_argument("--mauve_device", default="cuda", type=str,
        help="MAUVE评分使用的设备")
  parser.add_argument("--mauve_verbose", action="store_true",
        help="MAUVE评分是否显示详细信息")

  # Key settings
  parser.add_argument("--accelerate", action="store_true",
        help="Whether to use accelerate during training for multiple machines.")
  parser.add_argument("--ignore-cache", action="store_true",
        help="Whether to ignore cache and create a new input data")
  parser.add_argument("--debug", action="store_true",
        help="Whether to run in debug mode which is exponentially faster")
  parser.add_argument("--verbose", action="store_true",
        help="Whether to run with extra prints to help debug")
  parser.add_argument("--do-train", action="store_true",
        help="Whether to run training.")
  parser.add_argument("--do-eval", action="store_true",
        help="Whether to run eval on the test set.")
  parser.add_argument("--do-save", action="store_true",
        help="Whether to save models, which override previous checkpoints")
  parser.add_argument("--log-interval", type=int, default=500,
        help="Log every X updates steps.")
  parser.add_argument("--qualify", action='store_true',
        help="Whether to include joint accuracy scores during evaluation")
  parser.add_argument("--quantify", action='store_true',
        help="Whether to include inform/success/BLEU scores during evaluation")
  parser.add_argument("--prune-keep", default=-1, type=int,
        help="Number of models to keep around after pruning, by default does not prune")
  parser.add_argument("--parallel", action="store_true",
        help="Whether to run in parallel")
  parser.add_argument("--patience", default=4, type=int,
        help="patience for early stop, applies to both chunks and epochs")
  # temporary flag for experiments in 0084
  parser.add_argument("--pool-size", default=10, type=int,
        help="Number of exemplars to randomly sample from to put in the prompt")

  # Hyper-parameters for tuning
  parser.add_argument("--batch-size", default=12, type=int,
        help="Batch size per GPU/CPU for training and evaluation.")
  parser.add_argument('--grad-accum-steps', default=1, type=int,
        help='Number of steps for gradient accumulation')
  parser.add_argument("--learning-rate", default=3e-4, type=float,
        help="Model learning rate starting point.")
  parser.add_argument("--drop-rate", default=0.1, type=float,
        help="Dropout rate with default of 10%")
  parser.add_argument("--hidden-size", default=300, type=int,
        help="Hidden dimension for intermediate projection of ranking model")
  parser.add_argument("--embed-dim", default=768, type=int,
        help="Embed dimension for intermediate projection of ranking model")
  parser.add_argument("--weight-decay", default=0.0, type=float,
        help="Weight decay if we apply some.")
  parser.add_argument("--n-epochs", default=3, type=int,
        help="Total number of training epochs to perform.")
  parser.add_argument("--warmup-steps", default=0.1, type=float,
        help="Linear warmup over warmup-steps ratio of total steps")
  parser.add_argument("--sig-threshold", default=0.5, type=float,
        help="sigmoid threshold for multilabel classifier evaluation")

  args = parser.parse_args()
  return args
