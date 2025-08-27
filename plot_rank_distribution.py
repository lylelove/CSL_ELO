import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# 设置matplotlib后端以获得更好的渲染效果
matplotlib.use('Agg')  # 使用Agg后端以获得更好的图片质量

# 设置中文字体支持和图像质量
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.rcParams['figure.dpi'] = 400  # 设置图像DPI
plt.rcParams['savefig.dpi'] = 400  # 设置保存图像的DPI
plt.rcParams['font.size'] = 12  # 设置默认字体大小

# 加载JSON数据
with open('rank_distribution.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 转换为DataFrame
df = pd.DataFrame(data)

# 计算每支球队的期望排名
def calculate_expected_rank(row):
    total = 0
    for i in range(1, 17):
        total += row[f'rank_{i}'] * i
    return total

df['expected_rank'] = df.apply(calculate_expected_rank, axis=1)

# 根据期望排名排序球队（从小到大）
df_sorted = df.sort_values('expected_rank').reset_index(drop=True)

# 提取排序后的球队名称
teams = df_sorted['team'].tolist()

# 提取排名概率列（rank_1到rank_16）
rank_cols = [f'rank_{i}' for i in range(1, 17)]
prob_matrix = df_sorted[rank_cols].values

# 创建自定义的绿色渐变颜色映射
from matplotlib.colors import LinearSegmentedColormap
colors = ['white', 'lightgreen', 'green', 'darkgreen']
cmap = LinearSegmentedColormap.from_list('green_gradient', colors, N=256)

# 创建热力图
fig, ax = plt.subplots(figsize=(16, 12))
im = ax.imshow(prob_matrix, cmap=cmap, aspect='auto', vmin=0, vmax=1)

# 添加颜色条
cbar = fig.colorbar(im, ax=ax)
cbar.set_label('概率', fontsize=14)

# 设置坐标轴标签
ax.set_title('2025赛季中超排名概率分布热力图(基于前22轮)', fontsize=18, fontweight='bold', pad=20)
ax.set_xlabel('排名', fontsize=14)
ax.set_ylabel('球队', fontsize=14)

# 设置x轴刻度（排名1-16）
ax.set_xticks(np.arange(0, 16, 1))
ax.set_xticklabels(range(1, 17), fontsize=12)

# 设置y轴刻度（球队名称）
ax.set_yticks(np.arange(0, len(teams), 1))
ax.set_yticklabels(teams, fontsize=12)

# 在每个格子上添加文本（概率不为0时显示百分比）
for i in range(len(teams)):
    for j in range(16):
        prob = prob_matrix[i, j]
        if prob > 0:
            # 转换为百分比字符串，保留1位小数
            text = f'{prob*100:.1f}%'
            ax.text(j, i, text, ha='center', va='center', fontsize=9,
                    color='white' if prob > 0.5 else 'black', fontweight='bold')

# 调整布局
plt.tight_layout()

# 显示图表
plt.show()

# 保存图片
plt.savefig('rank_distribution_heatmap.png', dpi=400, bbox_inches='tight', 
            facecolor='white', edgecolor='none')


print("热力图已生成并保存为 rank_distribution_heatmap.png")
