#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

def parse_args():
    parser = argparse.ArgumentParser(description="比较基础模型和LoRA模型的评估结果")
    parser.add_argument("--base_result", type=str, default="./results/base_model_evaluation_results_20250315-113245.json", help="基础模型评估结果文件")
    parser.add_argument("--lora_result", type=str, default="./results/lora_evaluation_results_20250314-151308.json", help="LoRA模型评估结果文件")
    parser.add_argument("--output_dir", type=str, default="./results", help="输出目录")
    
    return parser.parse_args()

def load_results(result_file):
    """加载评估结果"""
    with open(result_file, "r", encoding="utf-8") as f:
        results = json.load(f)
    return results

def compare_results(base_results, lora_results):
    """比较基础模型和LoRA模型的评估结果"""
    # 提取基本指标
    base_accuracy = base_results.get("accuracy", 0.0)
    lora_accuracy = lora_results.get("accuracy", 0.0)
    
    # 提取详细结果
    base_details = base_results.get("details", [])
    lora_details = lora_results.get("details", [])
    
    # 创建结果字典
    comparison = {
        "base_model": base_results.get("model_path", "Unknown"),
        "lora_model": lora_results.get("model_path", "Unknown"),
        "base_accuracy": base_accuracy,
        "lora_accuracy": lora_accuracy,
        "accuracy_diff": base_accuracy - lora_accuracy,
        "base_sample_count": base_results.get("total_count", 0),
        "lora_sample_count": lora_results.get("total_count", 0),
    }
    
    # 如果两个模型评估的样本ID相同，可以进行更详细的比较
    base_ids = [item.get("id") for item in base_details]
    lora_ids = [item.get("id") for item in lora_details]
    
    # 找出共同的样本ID
    common_ids = set(base_ids).intersection(set(lora_ids))
    comparison["common_sample_count"] = len(common_ids)
    
    if common_ids:
        # 创建ID到索引的映射
        base_id_to_idx = {item.get("id"): idx for idx, item in enumerate(base_details)}
        lora_id_to_idx = {item.get("id"): idx for idx, item in enumerate(lora_details)}
        
        # 统计共同样本上的准确率
        base_correct_common = 0
        lora_correct_common = 0
        
        for sample_id in common_ids:
            base_idx = base_id_to_idx.get(sample_id)
            lora_idx = lora_id_to_idx.get(sample_id)
            
            if base_idx is not None and lora_idx is not None:
                if base_details[base_idx].get("is_correct", False):
                    base_correct_common += 1
                if lora_details[lora_idx].get("is_correct", False):
                    lora_correct_common += 1
        
        # 计算共同样本上的准确率
        comparison["base_accuracy_common"] = base_correct_common / len(common_ids)
        comparison["lora_accuracy_common"] = lora_correct_common / len(common_ids)
        comparison["accuracy_diff_common"] = comparison["base_accuracy_common"] - comparison["lora_accuracy_common"]
        
        # 统计两个模型都正确、都错误、只有一个正确的样本数量
        both_correct = 0
        both_wrong = 0
        only_base_correct = 0
        only_lora_correct = 0
        
        for sample_id in common_ids:
            base_idx = base_id_to_idx.get(sample_id)
            lora_idx = lora_id_to_idx.get(sample_id)
            
            if base_idx is not None and lora_idx is not None:
                base_correct = base_details[base_idx].get("is_correct", False)
                lora_correct = lora_details[lora_idx].get("is_correct", False)
                
                if base_correct and lora_correct:
                    both_correct += 1
                elif not base_correct and not lora_correct:
                    both_wrong += 1
                elif base_correct and not lora_correct:
                    only_base_correct += 1
                elif not base_correct and lora_correct:
                    only_lora_correct += 1
        
        comparison["both_correct"] = both_correct
        comparison["both_wrong"] = both_wrong
        comparison["only_base_correct"] = only_base_correct
        comparison["only_lora_correct"] = only_lora_correct
    
    return comparison

def plot_comparison(comparison, output_dir):
    """绘制比较结果的图表"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 绘制准确率对比图
    plt.figure(figsize=(10, 6))
    models = ["基础模型", "LoRA模型"]
    accuracies = [comparison["base_accuracy"], comparison["lora_accuracy"]]
    
    plt.bar(models, accuracies, color=["blue", "orange"])
    plt.ylim(0.0, 1.0)
    plt.title("模型准确率对比")
    plt.ylabel("准确率")
    
    # 在柱状图上添加准确率数值
    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 0.01, f"{acc:.4f}", ha="center")
    
    plt.savefig(os.path.join(output_dir, "accuracy_comparison.png"))
    
    # 如果有共同样本，绘制更详细的对比图
    if "common_sample_count" in comparison and comparison["common_sample_count"] > 0:
        # 绘制共同样本上的准确率对比图
        plt.figure(figsize=(10, 6))
        models = ["基础模型", "LoRA模型"]
        common_accuracies = [comparison["base_accuracy_common"], comparison["lora_accuracy_common"]]
        
        plt.bar(models, common_accuracies, color=["blue", "orange"])
        plt.ylim(0.0, 1.0)
        plt.title("共同样本上的模型准确率对比")
        plt.ylabel("准确率")
        
        # 在柱状图上添加准确率数值
        for i, acc in enumerate(common_accuracies):
            plt.text(i, acc + 0.01, f"{acc:.4f}", ha="center")
        
        plt.savefig(os.path.join(output_dir, "common_accuracy_comparison.png"))
        
        # 绘制两个模型预测情况的饼图
        plt.figure(figsize=(10, 6))
        labels = ["两个模型都正确", "两个模型都错误", "只有基础模型正确", "只有LoRA模型正确"]
        sizes = [
            comparison["both_correct"],
            comparison["both_wrong"],
            comparison["only_base_correct"],
            comparison["only_lora_correct"]
        ]
        colors = ["green", "red", "blue", "orange"]
        explode = (0.1, 0, 0, 0)  # 突出显示"两个模型都正确"
        
        plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct="%1.1f%%", shadow=True, startangle=90)
        plt.axis("equal")  # 保证饼图是圆的
        plt.title("两个模型预测情况对比")
        
        plt.savefig(os.path.join(output_dir, "prediction_comparison.png"))

def main():
    args = parse_args()
    
    # 加载评估结果
    base_results = load_results(args.base_result)
    lora_results = load_results(args.lora_result)
    
    # 比较结果
    comparison = compare_results(base_results, lora_results)
    
    # 打印比较结果
    print("模型评估结果比较:")
    print(f"基础模型: {comparison['base_model']}")
    print(f"LoRA模型: {comparison['lora_model']}")
    print(f"基础模型准确率: {comparison['base_accuracy']:.4f}")
    print(f"LoRA模型准确率: {comparison['lora_accuracy']:.4f}")
    print(f"准确率差异 (基础 - LoRA): {comparison['accuracy_diff']:.4f}")
    
    if "common_sample_count" in comparison and comparison["common_sample_count"] > 0:
        print(f"\n共同样本数量: {comparison['common_sample_count']}")
        print(f"基础模型在共同样本上的准确率: {comparison['base_accuracy_common']:.4f}")
        print(f"LoRA模型在共同样本上的准确率: {comparison['lora_accuracy_common']:.4f}")
        print(f"共同样本上的准确率差异 (基础 - LoRA): {comparison['accuracy_diff_common']:.4f}")
        
        print(f"\n两个模型都正确的样本数量: {comparison['both_correct']} ({comparison['both_correct']/comparison['common_sample_count']:.2%})")
        print(f"两个模型都错误的样本数量: {comparison['both_wrong']} ({comparison['both_wrong']/comparison['common_sample_count']:.2%})")
        print(f"只有基础模型正确的样本数量: {comparison['only_base_correct']} ({comparison['only_base_correct']/comparison['common_sample_count']:.2%})")
        print(f"只有LoRA模型正确的样本数量: {comparison['only_lora_correct']} ({comparison['only_lora_correct']/comparison['common_sample_count']:.2%})")
    
    # 保存比较结果
    with open(os.path.join(args.output_dir, "model_comparison.json"), "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    
    # 绘制比较图表
    try:
        plot_comparison(comparison, args.output_dir)
        print(f"\n比较图表已保存到 {args.output_dir} 目录")
    except Exception as e:
        print(f"绘制图表时出错: {e}")
    
    print(f"\n比较结果已保存到 {os.path.join(args.output_dir, 'model_comparison.json')}")

if __name__ == "__main__":
    main() 