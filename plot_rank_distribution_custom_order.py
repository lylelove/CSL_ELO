import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import argparse

# 设置matplotlib后端以获得更好的渲染效果
matplotlib.use('Agg')  # 使用Agg后端以获得更好的图片质量

# 设置中文字体支持和图像质量
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.rcParams['figure.dpi'] = 400  # 设置图像DPI
plt.rcParams['savefig.dpi'] = 400  # 设置保存图像的DPI
plt.rcParams['font.size'] = 12  # 设置默认字体大小

# 解析命令行参数
parser = argparse.ArgumentParser(description='生成排名概率分布热力图')
parser.add_argument('--round', type=int, default=22, help='当前轮次')
parser.add_argument('--output', type=str, default='rank_distribution_heatmap_custom_order.png', help='输出文件名')
args = parser.parse_args()

# 加载JSON数据
with open('rank_distribution.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 转换为DataFrame
df = pd.DataFrame(data)

# 定义指定的球队顺序
custom_team_order = [
    '上海申花', '北京国安', '上海海港', '成都蓉城', 
    '山东泰山', '浙江', '天津津门虎', '云南玉昆', 
    '大连英博', '青岛西海岸', '河南', '武汉三镇', 
    '梅州客家', '深圳新鹏城', '青岛海牛', '长春亚泰'
]

# 按照指定顺序重新排列DataFrame
df_sorted = df.set_index('team').reindex(custom_team_order).reset_index()

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
ax.set_title(f'2025赛季中超排名概率分布热力图(基于前{args.round}轮)', fontsize=18, fontweight='bold', pad=20)
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


# 保存图片
plt.savefig(args.output, dpi=400, bbox_inches='tight',
            facecolor='white', edgecolor='none')


print(f"热力图已生成并保存为 {args.output}")