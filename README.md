# CSL ELO 项目

> 本README由AI自动分析生成

## 项目概述

本项目是一个用于分析中国足球协会超级联赛（CSL）的工具集，主要功能包括：

1. 获取比赛数据
2. 计算球队ELO评分
3. 分析球队排名概率分布
4. 识别关键比赛及其重要性

## 文件说明

### 主要脚本

1. `fetch_matches.py` - 从网络获取比赛数据并保存为JSON格式
2. `match_importance_json_v2.py` - 核心分析脚本，用于计算球队排名概率、ELO评分和关键比赛分析
3. `plot_rank_distribution.py` - 生成排名概率分布热力图

### 数据文件

1. `matches.json` - 存储从网络获取的比赛数据，按轮次分组
2. `elo_scores.json` - 存储各球队的ELO评分
3. `rank_distribution.json` - 存储各球队在不同排名位置的概率
4. `match_importance_<球队名>.json` - 存储特定球队的关键比赛分析结果

### 输出文件

1. `rank_distribution_heatmap.png` - 排名概率分布热力图
2. 各球队的关键比赛分析结果CSV/JSON文件

## 使用方法

1. 运行 `fetch_matches.py` 获取比赛数据：
   ```bash
   python fetch_matches.py
   ```

2. 运行 `match_importance_json_v2.py` 进行分析：
   ```bash
   python match_importance_json_v2.py --json matches.json --season 2024-25 --current_round 22 --team "青岛海牛" --relegation_slots 2 --n_sims 200 --form_coeff 40 --progress_every 2 --verbose 1
   ```

3. 运行 `plot_rank_distribution.py` 生成热力图：
   ```bash
   python plot_rank_distribution.py
   ```

## 分析结果说明

### ELO评分
ELO评分用于衡量球队实力，评分越高表示球队实力越强。

### 排名概率分布
通过模拟计算各球队在不同排名位置的概率，帮助预测赛季最终排名。

### 关键比赛分析
对于特定球队，分析剩余比赛中哪些比赛对球队的保级/争冠形势最为关键，包括：
- 对手信息
- 主客场信息
- 比赛结果对球队排名概率的影响

## 依赖库

- requests
- json
- math
- typing
- numpy
- pandas
- matplotlib

## 注意事项

1. 网络数据获取可能受网络状况和目标网站结构变化影响
2. 分析结果基于统计模拟，仅供参考
3. 请确保安装了所有依赖库后再运行脚本