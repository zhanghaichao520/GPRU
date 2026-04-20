import matplotlib.pyplot as plt
import numpy as np
import seaborn

# 数据
models = ['WMF', 'BPR', 'LightGCN']
methods = ['Retrain', 'SISA', 'RecEraser', 'PRU']
datasets = ['Amazon-Book', 'Gowalla', 'Yelp2018']

# 运行时间数据
data = {
    'Amazon-Book': {
        'WMF': [3977.36, 2194.32, 1968.47, 1539.50],
        'BPR': [4255.03, 4990.47, 3568.43, 2229.38],
        'LightGCN': [20293.41, 7946.94, 4315.26, 1979.17]
    },
    'Gowalla': {
        'WMF': [1480.67, 1744.22, 1096.58, 684.23],
        'BPR': [2038.46, 1569.48, 1487.88, 794.25],
        'LightGCN': [7952.20, 2963.17, 1749.28, 724.32]
    },
    'Yelp2018': {
        'WMF': [7955.08, 3964.82, 1864.93, 734.81],
        'BPR': [8069.47, 3519.42, 2340.18, 1952.74],
        'LightGCN': [65400.00, 4475.86, 2040.00, 1564.31]
    }
}

# 设置图表样式
# plt.style.use('seaborn')
plt.figure(figsize=(15, 10))

# 设置颜色和透明度
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
alpha = 0.7

# 绘制分组柱状图
bar_width = 0.2
index = np.arange(len(methods))

for i, dataset in enumerate(datasets):
    plt.subplot(1, 3, i+1)
    plt.title(f'{dataset}', fontsize=12)
    
    for j, model in enumerate(models):
        plt.bar(index + j*bar_width, data[dataset][model],
                width=bar_width,
                label=model, 
                color=colors[j], 
                alpha=alpha)
    
    plt.xlabel('Methods', fontsize=10)
    plt.ylabel('Runtime (seconds)', fontsize=10)
    plt.xticks(index + bar_width, methods, rotation=45)
    plt.legend()
    plt.tight_layout()

# plt.suptitle('Model Runtime Comparison Across Datasets', fontsize=16)
plt.show()