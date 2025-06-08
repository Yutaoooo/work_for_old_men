import matplotlib.pyplot as plt
import numpy as np

# 模型测试结果数据
results = {
    'Ding': {
        'inference_time': 0.0456,
        'precision': 0.8923,
        'recall': 0.8567,
        'f1': 0.8741,
        'iou': 0.7821
    },
    'Yuan': {
        'inference_time': 0.0521,
        'precision': 0.9123,
        'recall': 0.8421,
        'f1': 0.8758,
        'iou': 0.7842
    },
    'Tao': {
        'inference_time': 0.0389,
        'precision': 0.9012,
        'recall': 0.8678,
        'f1': 0.8842,
        'iou': 0.7953
    },
    'Fu': {
        'inference_time': 0.0412,
        'precision': 0.8854,
        'recall': 0.8723,
        'f1': 0.8788,
        'iou': 0.7865
    }
}

# 颜色配置
model_colors = {
    'Ding': '#1f77b4',
    'Yuan': '#ff7f0e',
    'Tao': '#2ca02c',
    'Fu': '#d62728'
}

# 绘制精度指标对比图
def plot_metrics_comparison():
    """绘制四个模型的精度指标对比图"""
    metrics = ['precision', 'recall', 'f1', 'iou']
    metric_names = ['Precision', 'Recall', 'F1 Score', 'IoU']
    models = list(results.keys())
    
    # 设置图表大小和样式
    plt.figure(figsize=(12, 6))
    bar_width = 0.2
    index = np.arange(len(models))
    
    # 绘制每个指标的柱状图
    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        values = [results[model][metric] for model in models]
        plt.bar(index + i*bar_width, values, bar_width, 
                label=name, color=model_colors[models[i]])
    
    # 添加图表元素
    plt.xlabel('Models', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Model Performance Comparison', fontsize=14)
    plt.xticks(index + bar_width*1.5, models, fontsize=11)
    plt.legend(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 保存图表
    plt.tight_layout()
    plt.savefig('metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

# 绘制推理时间对比图  
def plot_inference_time():
    """绘制四个模型的推理时间对比图"""
    models = list(results.keys())
    times = [results[model]['inference_time'] for model in models]
    
    # 设置图表大小和样式
    plt.figure(figsize=(10, 5))
    bars = plt.bar(models, times, color=[model_colors[m] for m in models])
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=10)
    
    # 添加图表元素
    plt.xlabel('Models', fontsize=12)
    plt.ylabel('Inference Time (s)', fontsize=12)
    plt.title('Model Inference Time Comparison', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 保存图表
    plt.tight_layout()
    plt.savefig('inference_time.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    plot_metrics_comparison()
    plot_inference_time()
    print("可视化图表已生成: metrics_comparison.png 和 inference_time.png")
