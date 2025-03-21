import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from mauve import compute_mauve

class MAUVEEvaluator:
    """
    使用MAUVE评分衡量指令之间的差异的评估器
    """
    def __init__(self, args):
        self.args = args
        self.device = args.mauve_device if torch.cuda.is_available() else "cpu"
        
    def compute_mauve_score(self, human_texts, generated_texts, attribute_name=None):
        """
        计算MAUVE评分
        
        参数:
        - human_texts: 人类编写的文本列表
        - generated_texts: 生成的文本列表
        - attribute_name: 属性名称，用于保存结果
        
        返回:
        - mauve_score: MAUVE评分
        """
        print(f"计算MAUVE评分 (人类文本: {len(human_texts)}, 生成文本: {len(generated_texts)})")
        
        # 确保文本列表不为空
        if not human_texts or not generated_texts:
            print("警告: 文本列表为空，无法计算MAUVE评分")
            return 0.0
        
        try:
            # 计算MAUVE评分
            out = compute_mauve(
                p_text=human_texts,
                q_text=generated_texts,
                device_id=0 if self.device == "cuda" else -1,
                max_text_length=self.args.source_max_len,
                verbose=self.args.mauve_verbose,
                batch_size=self.args.mauve_batch_size,
                featurize_model_name=self.args.mauve_model_name
            )
            
            mauve_score = out.mauve
            
            # 如果指定了属性名称，保存结果
            if attribute_name and self.args.visualize_distribution:
                self._save_mauve_results(out, attribute_name)
            
            return mauve_score
        
        except Exception as e:
            print(f"计算MAUVE评分时出错: {e}")
            return 0.0
    
    def evaluate_by_attributes(self, human_data, generated_data):
        """
        按属性评估MAUVE评分
        
        参数:
        - human_data: 人类编写的数据列表，每个元素包含'text'和'attributes'字段
        - generated_data: 生成的数据列表，每个元素包含'text'和'attributes'字段
        
        返回:
        - results: 包含每个属性MAUVE评分的字典
        """
        # 按属性分组文本
        attribute_texts = {}
        
        # 处理人类文本
        for item in human_data:
            for attr in item['attributes']:
                if attr not in attribute_texts:
                    attribute_texts[attr] = {'human': [], 'generated': []}
                attribute_texts[attr]['human'].append(item['text'])
        
        # 处理生成文本
        for item in generated_data:
            for attr in item['attributes']:
                if attr in attribute_texts:  # 只考虑人类数据中存在的属性
                    attribute_texts[attr]['generated'].append(item['text'])
        
        # 计算每个属性的MAUVE评分
        results = {}
        for attr, texts in tqdm(attribute_texts.items(), desc="计算属性MAUVE评分"):
            if len(texts['human']) >= 10 and len(texts['generated']) >= 10:  # 确保有足够的样本
                score = self.compute_mauve_score(texts['human'], texts['generated'], attr)
                results[attr] = score
            else:
                print(f"属性 '{attr}' 的样本数量不足 (人类: {len(texts['human'])}, 生成: {len(texts['generated'])})")
        
        # 计算总体MAUVE评分
        all_human_texts = [item['text'] for item in human_data]
        all_generated_texts = [item['text'] for item in generated_data]
        overall_score = self.compute_mauve_score(all_human_texts, all_generated_texts, "overall")
        results['overall'] = overall_score
        
        # 可视化结果
        if self.args.visualize_distribution:
            self._visualize_mauve_scores(results)
        
        return results
    
    def _save_mauve_results(self, mauve_out, attribute_name):
        """
        保存MAUVE结果
        
        参数:
        - mauve_out: MAUVE计算结果
        - attribute_name: 属性名称
        """
        # 创建保存目录
        save_dir = os.path.join(self.args.visualization_dir, "mauve")
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存MAUVE分布图
        plt.figure(figsize=(10, 6))
        sns.kdeplot(mauve_out.p_features, label="Human")
        sns.kdeplot(mauve_out.q_features, label="Generated")
        plt.title(f"MAUVE Distribution - {attribute_name}")
        plt.xlabel("Feature Value")
        plt.ylabel("Density")
        plt.legend()
        plt.savefig(os.path.join(save_dir, f"mauve_distribution_{attribute_name}.png"))
        plt.close()
    
    def _visualize_mauve_scores(self, results):
        """
        可视化MAUVE评分
        
        参数:
        - results: 包含每个属性MAUVE评分的字典
        """
        # 创建保存目录
        save_dir = os.path.join(self.args.visualization_dir, "mauve")
        os.makedirs(save_dir, exist_ok=True)
        
        # 排除overall，按评分排序
        attrs = {k: v for k, v in results.items() if k != 'overall'}
        sorted_attrs = sorted(attrs.items(), key=lambda x: x[1], reverse=True)
        
        # 绘制条形图
        plt.figure(figsize=(12, 8))
        x = [item[0] for item in sorted_attrs]
        y = [item[1] for item in sorted_attrs]
        
        bars = plt.bar(x, y, color='skyblue')
        plt.axhline(y=results['overall'], color='red', linestyle='--', label=f'Overall: {results["overall"]:.4f}')
        
        plt.title("MAUVE Scores by Attribute")
        plt.xlabel("Attribute")
        plt.ylabel("MAUVE Score")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.legend()
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.4f}', ha='center', va='bottom', rotation=0)
        
        plt.savefig(os.path.join(save_dir, "mauve_scores_by_attribute.png"))
        plt.close()
        
        # 保存结果到CSV
        import pandas as pd
        df = pd.DataFrame(list(results.items()), columns=['Attribute', 'MAUVE Score'])
        df.to_csv(os.path.join(save_dir, "mauve_scores.csv"), index=False)
        
        print(f"MAUVE评分可视化结果已保存到 {save_dir}") 