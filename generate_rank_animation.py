#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_rank_animation.py
==========================
自动生成中超排名概率分布热力图的动画GIF
循环运行predict_rank_distribution.py和plot_rank_distribution_custom_order.py
生成r1.png到r22.png，最后合成GIF动画
"""

import os
import subprocess
import sys
from PIL import Image
import glob

def run_command(cmd):
    """运行命令行命令"""
    print(f"执行命令: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"命令执行失败: {cmd}")
        print(f"错误输出: {result.stderr}")
        return False
    print(f"命令执行成功: {cmd}")
    if result.stdout.strip():
        print(f"输出: {result.stdout}")
    return True

def generate_rank_images():
    """生成所有轮次的排名图片"""
    for i in range(1, 23):
        print(f"\n=== 处理第 {i} 轮 ===")
        
        # 运行预测脚本
        predict_cmd = f"python predict_rank_distribution.py --json matches.json --season 2024-25 --current_round {i} --n_sims 2000"
        if not run_command(predict_cmd):
            print(f"第 {i} 轮预测失败，跳过")
            continue
        
        # 运行绘图脚本
        output_file = f"r{i}.png"
        plot_cmd = f"python plot_rank_distribution_custom_order.py --round {i} --output {output_file}"
        if not run_command(plot_cmd):
            print(f"第 {i} 轮绘图失败，跳过")
            continue
        
        print(f"第 {i} 轮处理完成，图片保存为 {output_file}")

def create_gif_animation():
    """创建GIF动画"""
    print("\n=== 创建GIF动画 ===")
    
    # 获取所有r*.png文件并按数字顺序排序
    image_files = []
    for i in range(1, 23):
        filename = f"r{i}.png"
        if os.path.exists(filename):
            image_files.append(filename)
    
    if not image_files:
        print("未找到任何图片文件")
        return False
    
    print(f"找到 {len(image_files)} 张图片: {image_files}")
    
    # 读取所有图片
    frames = []
    for image_file in image_files:
        try:
            frame = Image.open(image_file)
            frames.append(frame)
            print(f"已加载: {image_file}")
        except Exception as e:
            print(f"加载图片失败 {image_file}: {e}")
    
    if not frames:
        print("没有可用的图片帧")
        return False
    
    # 保存为GIF
    output_gif = "rank_animation.gif"
    
    # 确保所有图片尺寸一致
    first_frame = frames[0]
    resized_frames = []
    for frame in frames:
        if frame.size != first_frame.size:
            resized_frame = frame.resize(first_frame.size, Image.Resampling.LANCZOS)
            resized_frames.append(resized_frame)
        else:
            resized_frames.append(frame)
    
    # 保存GIF，每帧间隔1秒
    resized_frames[0].save(
        output_gif,
        format='GIF',
        append_images=resized_frames[1:],
        save_all=True,
        duration=1000,  # 每帧1000毫秒 (1秒)
        loop=0  # 无限循环
    )
    
    print(f"GIF动画已保存为 {output_gif}")
    return True

def main():
    """主函数"""
    print("开始生成中超排名概率分布动画...")
    
    # 检查必要文件
    if not os.path.exists("matches.json"):
        print("错误: 未找到 matches.json 文件")
        return
    
    if not os.path.exists("predict_rank_distribution.py"):
        print("错误: 未找到 predict_rank_distribution.py 文件")
        return
    
    if not os.path.exists("plot_rank_distribution_custom_order.py"):
        print("错误: 未找到 plot_rank_distribution_custom_order.py 文件")
        return
    
    # 生成所有图片
    generate_rank_images()
    
    # 创建GIF动画
    create_gif_animation()
    
    print("\n=== 处理完成 ===")
    print("所有图片已生成并合成为GIF动画")

if __name__ == "__main__":
    main()