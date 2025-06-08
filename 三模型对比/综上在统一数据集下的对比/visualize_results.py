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
    }
}

# 绘制精度指标对比图
def plot_metrics():
    models = list(results.keys())
    metrics = ['precision', 'recall', 'f1', 'iou']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(models))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        values = [results[model][metric] for model in models]
        ax.bar(x + i*width, values, width, label=metric)
    
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x + width*1.5)
    ax.set_xticklabels(models)
    ax.legend()
    plt.tight_layout()
    plt.savefig('metrics_comparison.png')
    plt.close()

# 绘制推理时间对比图
def plot_inference_time():
    models = list(results.keys())
    times = [results[model]['inference_time'] for model in models]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(models, times, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    
    ax.set_ylabel('Inference Time (s)')
    ax.set_title('Model Inference Time Comparison')
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('inference_time.png')
    plt.close()

if __name__ == '__main__':
    plot_metrics()
    plot_inference_time()
    print("可视化图表已生成: metrics_comparison.png 和 inference_time.png")
