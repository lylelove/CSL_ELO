import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

with open('rank_distribution.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

df = pd.DataFrame(data)


def calculate_expected_rank(row):
    total = 0
    for i in range(1, 17):
        total += row[f'rank_{i}'] * i
    return total

df['expected_rank'] = df.apply(calculate_expected_rank, axis=1)

df_sorted = df.sort_values('expected_rank').reset_index(drop=True)

teams = df_sorted['team'].tolist()

rank_cols = [f'rank_{i}' for i in range(1, 17)]
prob_matrix = df_sorted[rank_cols].values

from matplotlib.colors import LinearSegmentedColormap
colors = ['white', 'lightgreen', 'green', 'darkgreen']
cmap = LinearSegmentedColormap.from_list('green_gradient', colors, N=256)

plt.figure(figsize=(16, 12))
im = plt.imshow(prob_matrix, cmap=cmap, aspect='auto', vmin=0, vmax=1)

cbar = plt.colorbar(im)
cbar.set_label('概率', fontsize=12)

plt.title('2025赛季中超排名概率分布热力图(基于前22轮)', fontsize=16, fontweight='bold')
plt.xlabel('排名', fontsize=12)
plt.ylabel('球队', fontsize=12)

plt.xticks(np.arange(0, 16, 1), range(1, 17))

plt.yticks(np.arange(0, len(teams), 1), teams)

for i in range(len(teams)):
    for j in range(16):
        prob = prob_matrix[i, j]
        if prob > 0:
            text = f'{prob*100:.1f}%'
            plt.text(j, i, text, ha='center', va='center', fontsize=8,
                    color='white' if prob > 0.5 else 'black', fontweight='bold')

plt.tight_layout()

plt.savefig('rank_distribution_heatmap.png', dpi=300, bbox_inches='tight')

plt.show()

print("热力图已生成并保存为 rank_distribution_heatmap.png")
