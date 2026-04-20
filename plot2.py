import numpy as np
import matplotlib.pyplot as plt

# 数据
models = ['WMF', 'BPR', 'LightGCN']
methods = ['SISA', 'Retrain', 'PRU', 'RecEraser']
datasets = ['Amazon-Book', 'Gowalla', 'Yelp2018']

# 运行时间数据
data = {
    'Amazon-Book': {
        'WMF': [2194.32, 3977.36, 1539.50, 1968.47],
        'BPR': [4990.47, 4255.03, 2229.38, 3568.43],
        'LightGCN': [7946.94, 20293.41, 1979.17, 4315.26]
    },
    'Gowalla': {
        'WMF': [1744.22, 1480.67, 684.23, 1096.58],
        'BPR': [1569.48, 2038.46, 794.25, 1487.88],
        'LightGCN': [2963.17, 7952.20, 724.32, 1749.28]
    },
    'Yelp2018': {
        'WMF': [3964.82, 7955.08, 734.81, 1864.93],
        'BPR': [3519.42, 8069.47, 1952.74, 2340.18],
        'LightGCN': [4475.86, 65400.00, 1564.31, 2040.00]
    }
}

# 设置图表样式
# plt.style.use('seaborn')
plt.figure(figsize=(15, 5))

# 设置颜色
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

# # 绘制五边形图
for i, dataset in enumerate(datasets):
    plt.subplot(1, 3, i+1, polar=True)
    plt.title(f'{dataset}', fontsize=12)
    
    for j, model in enumerate(models):
        # 对数据进行对数变换以减少差异
        values = np.log10(data[dataset][model])
        
        # 角度
        angles = np.linspace(0, 2*np.pi, len(methods), endpoint=False)
        
        # 闭合图形
        values = np.concatenate((values, [values[0]]))
        angles = np.concatenate((angles, [angles[0]]))
        
        plt.polar(angles, values,'o-', linewidth=2, label=model, color=colors[j], alpha=0.7)
        plt.fill(angles, values, colors[j], alpha=0.1)
    
    plt.thetagrids(angles[:-1] *180/np.pi, methods)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

# plt.suptitle('Model Runtime Comparison Across Datasets (Log Scale)', fontsize=16)
plt.tight_layout()
plt.show()

