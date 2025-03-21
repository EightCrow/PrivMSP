import os, sys
import numpy as np
import random
from collections import Counter
from sklearn.cluster import KMeans
import torch
from torch import nn
from sentence_transformers import SentenceTransformer
from assets.static_vars import device

def attribute_frequency(data):
    """计算数据集中各属性的频率分布"""
    attr_freq = Counter()
    for example in data:
        for attribute in example['attributes']:
            attr_freq[attribute] += 1

    # 转换为归一化比例
    frequency_ratio = {}
    total_size = sum(attr_freq.values())
    for attribute, count in attr_freq.items():
        frequency_ratio[attribute] = float(count) / total_size
    return frequency_ratio

def attribute_overlap_score(example, frequency_ratio):
    """基于原始数据中某些属性类别过度表示的观察，
    我们根据示例的属性与原始数据的重叠程度对每个示例进行评分。
    因此，更高的分数表示重叠太多，这是"不好的"。如果我们要采样，这将
    具有重新平衡数据的效果，使所有属性都有平等的出现机会。"""
    attribute_ratios = []
    for attribute in example['attributes']:
        attr_ratio = frequency_ratio[attribute]
        attribute_ratios.append(attr_ratio)
    # 取平均值，这样具有更多属性的示例不会受到不公平的惩罚
    overlap_rate = np.mean(attribute_ratios)
    return overlap_rate

def calculate_sample_rarity(example, frequency_ratio):
    """计算样本的稀有度（基于其属性的稀有程度）"""
    attribute_rarities = []
    for attribute in example['attributes']:
        attr_freq = frequency_ratio[attribute]
        # 稀有度是频率的倒数
        attribute_rarities.append(1.0 / max(attr_freq, 1e-5))
    # 取平均值
    rarity_score = np.mean(attribute_rarities)
    return rarity_score

def embedding_distance_score(new_exp, old_exp, embedder_model):
    """我们希望保留语义上更接近种子示例的生成示例。
    直观上，这增加了新示例被正确标记的可能性。这是通过
    计算合成示例与其原始示例之间的余弦距离来实现的。再次，较大的分数
    是"不好的"，因为较大的距离意味着新示例在语义上离原始示例更远。"""
    sentences = [new_exp['text'], old_exp['text']]
    embeddings = embedder_model.encode(sentences)
    new_embed, old_embed = torch.tensor(embeddings[0]), torch.tensor(embeddings[1])
    similarity = nn.functional.cosine_similarity(new_embed.unsqueeze(0), old_embed.unsqueeze(0), dim=1)
    embed_dist = 1 - similarity.item()
    return embed_dist

def semantic_similarity(new_exp, old_exp, embedder_model):
    """计算两个样本之间的语义相似度"""
    sentences = [new_exp['text'], old_exp['text']]
    embeddings = embedder_model.encode(sentences)
    new_embed, old_embed = torch.tensor(embeddings[0]), torch.tensor(embeddings[1])
    similarity = nn.functional.cosine_similarity(new_embed.unsqueeze(0), old_embed.unsqueeze(0), dim=1)
    return similarity.item()

def assign_noise_scores(args, synthetic_data, seed_data):
    """为每个合成示例分配噪声分数，并展平分组。"""
    frequency_ratio = attribute_frequency(seed_data)
    embedder_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

    noise_scores, flat_data = [], []
    for synth_group, old_example in zip(synthetic_data, seed_data):
        for new_example in synth_group:
            attr_over = attribute_overlap_score(new_example, frequency_ratio)
            embed_dist = embedding_distance_score(new_example, old_example, embedder_model)
            lambda_ao, lambda_ed = 0.5, 0.5  # 调整帮助很小，只需设置为一半以保持简单
            final_score = (lambda_ao * attr_over) + (lambda_ed * embed_dist)

            noise_scores.append(final_score)
            flat_data.append(new_example)
    return noise_scores, flat_data

def parameterized_filter(args, synthetic_data, seed_data):
    """
    参数化去噪过滤算法
    
    参数:
    - args: 包含以下字段的参数对象:
        - embedding_model: 嵌入模型名称
        - num_clusters: 聚类数量K
        - target_samples: 目标合成样本数T
        - distribution_alpha: 分布控制参数α
        - rarity_beta: 稀有属性增强因子β
        - similarity_threshold: 语义相似度阈值γ
        - distribution_type: 分布类型('original', 'uniform', 'balanced', 'rare_enhanced')
    - synthetic_data: 初始合成数据列表（分组形式）
    - seed_data: 真实数据列表
    
    返回:
    - filtered_data: 过滤后的合成数据列表
    - distribution_info: 分布信息字典，用于可视化
    """
    # 0. 展平合成数据
    flat_synthetic_data = []
    for synth_group in synthetic_data:
        flat_synthetic_data.extend(synth_group)
    
    # 1. 加载嵌入模型
    embedder_model = SentenceTransformer(args.embedding_model, device=device)
    
    # 2. 计算嵌入表示
    synthetic_texts = [example['text'] for example in flat_synthetic_data]
    real_texts = [example['text'] for example in seed_data]
    
    synthetic_embeddings = embedder_model.encode(synthetic_texts)
    real_embeddings = embedder_model.encode(real_texts)
    
    # 3. 运行k-means聚类
    kmeans = KMeans(n_clusters=args.num_clusters, random_state=42)
    kmeans.fit(synthetic_embeddings)
    
    synthetic_clusters = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_
    
    # 4. 将真实数据分配到最近的聚类
    real_clusters = []
    for emb in real_embeddings:
        distances = np.linalg.norm(cluster_centers - emb, axis=1)
        nearest_cluster = np.argmin(distances)
        real_clusters.append(nearest_cluster)
    
    # 5. 初始化直方图并计算原始分布
    histogram = np.zeros(args.num_clusters)
    for cluster in real_clusters:
        histogram[cluster] += 1
    
    # 6. 计算目标分布
    original_distribution = histogram / np.sum(histogram)
    uniform_distribution = np.ones(args.num_clusters) / args.num_clusters
    
    if args.distribution_type == 'original':
        target_distribution = original_distribution
    elif args.distribution_type == 'uniform':
        target_distribution = uniform_distribution
    elif args.distribution_type == 'balanced':
        target_distribution = (1 - args.distribution_alpha) * original_distribution + args.distribution_alpha * uniform_distribution
    else:  # 默认为balanced
        target_distribution = (1 - args.distribution_alpha) * original_distribution + args.distribution_alpha * uniform_distribution
    
    # 7. 计算属性频率
    frequency_ratio = attribute_frequency(seed_data)
    
    # 8. 应用稀有属性增强（如果需要）
    if args.distribution_type == 'rare_enhanced':
        # 计算每个聚类的平均稀有度
        cluster_rarities = np.zeros(args.num_clusters)
        cluster_counts = np.zeros(args.num_clusters)
        
        for i, example in enumerate(flat_synthetic_data):
            cluster = synthetic_clusters[i]
            rarity = calculate_sample_rarity(example, frequency_ratio)
            cluster_rarities[cluster] += rarity
            cluster_counts[cluster] += 1
        
        # 避免除以零
        cluster_counts = np.maximum(cluster_counts, 1)
        avg_cluster_rarities = cluster_rarities / cluster_counts
        
        # 应用稀有度增强
        enhancement_factor = 1 + args.rarity_beta * (avg_cluster_rarities - 1)
        target_distribution = target_distribution * enhancement_factor
        
        # 重新归一化
        target_distribution = target_distribution / np.sum(target_distribution)
    
    # 9. 计算每个聚类需要采样的样本数
    target_counts = np.round(args.target_samples * target_distribution).astype(int)
    
    # 10. 重采样合成数据
    filtered_data = []
    distribution_info = {
        'original_distribution': original_distribution.tolist(),
        'target_distribution': target_distribution.tolist(),
        'actual_distribution': [],
        'cluster_sizes': [],
        'cluster_attributes': []
    }
    
    for cluster_id in range(args.num_clusters):
        # 获取该聚类的所有样本
        cluster_indices = np.where(synthetic_clusters == cluster_id)[0]
        cluster_samples = [flat_synthetic_data[i] for i in cluster_indices]
        
        distribution_info['cluster_sizes'].append(len(cluster_samples))
        
        # 收集该聚类的属性信息
        cluster_attrs = Counter()
        for sample in cluster_samples:
            for attr in sample['attributes']:
                cluster_attrs[attr] += 1
        distribution_info['cluster_attributes'].append(dict(cluster_attrs))
        
        # 检查是否有足够的样本
        if len(cluster_samples) < target_counts[cluster_id]:
            print(f"警告: 聚类 {cluster_id} 没有足够的样本 ({len(cluster_samples)} < {target_counts[cluster_id]})")
            # 如果样本不足，采样所有可用样本
            sampled_indices = list(range(len(cluster_samples)))
        else:
            # 随机采样所需数量的样本
            sampled_indices = random.sample(range(len(cluster_samples)), target_counts[cluster_id])
        
        # 应用语义相似度过滤
        for idx in sampled_indices:
            sample = cluster_samples[idx]
            # 找到最相似的真实样本
            max_similarity = 0
            for real_sample in seed_data:
                similarity = semantic_similarity(sample, real_sample, embedder_model)
                max_similarity = max(max_similarity, similarity)
            
            # 如果相似度高于阈值，保留该样本
            if max_similarity >= args.similarity_threshold:
                filtered_data.append(sample)
    
    # 11. 计算实际分布
    actual_counts = np.zeros(args.num_clusters)
    for sample in filtered_data:
        sample_text = sample['text']
        sample_idx = synthetic_texts.index(sample_text)
        cluster = synthetic_clusters[sample_idx]
        actual_counts[cluster] += 1
    
    actual_distribution = actual_counts / np.sum(actual_counts) if np.sum(actual_counts) > 0 else np.zeros(args.num_clusters)
    distribution_info['actual_distribution'] = actual_distribution.tolist()
    
    return filtered_data, distribution_info

def visualize_distributions(distribution_info, save_path=None):
    """
    可视化原始分布、目标分布和实际分布
    
    参数:
    - distribution_info: 包含分布信息的字典
    - save_path: 保存图表的路径，如果为None则显示图表
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd
        
        # 准备数据
        clusters = list(range(len(distribution_info['original_distribution'])))
        df = pd.DataFrame({
            'Cluster': np.repeat(clusters, 3),
            'Distribution': ['Original'] * len(clusters) + ['Target'] * len(clusters) + ['Actual'] * len(clusters),
            'Probability': distribution_info['original_distribution'] + 
                        distribution_info['target_distribution'] + 
                        distribution_info['actual_distribution']
        })
        
        # 创建图表
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Cluster', y='Probability', hue='Distribution', data=df)
        plt.title('Cluster Distribution Comparison')
        plt.xlabel('Cluster ID')
        plt.ylabel('Probability')
        plt.legend(title='Distribution Type')
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    except ImportError:
        print("警告: 缺少可视化所需的库 (matplotlib, seaborn, pandas)。跳过可视化。")

def visualize_attribute_distribution(distribution_info, top_n=10, save_path=None):
    """
    可视化每个聚类中的主要属性分布
    
    参数:
    - distribution_info: 包含分布信息的字典
    - top_n: 每个聚类显示的顶部属性数量
    - save_path: 保存图表的路径，如果为None则显示图表
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        num_clusters = len(distribution_info['cluster_attributes'])
        fig, axes = plt.subplots(num_clusters, 1, figsize=(12, 5 * num_clusters))
        
        if num_clusters == 1:
            axes = [axes]
        
        for i, (cluster_attrs, ax) in enumerate(zip(distribution_info['cluster_attributes'], axes)):
            # 获取前N个最常见的属性
            top_attrs = sorted(cluster_attrs.items(), key=lambda x: x[1], reverse=True)[:top_n]
            attrs = [x[0] for x in top_attrs]
            counts = [x[1] for x in top_attrs]
            
            # 创建条形图
            sns.barplot(x=counts, y=attrs, ax=ax)
            ax.set_title(f'Cluster {i} - Top {top_n} Attributes')
            ax.set_xlabel('Count')
            ax.set_ylabel('Attribute')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    except ImportError:
        print("警告: 缺少可视化所需的库 (matplotlib, seaborn)。跳过可视化。")

