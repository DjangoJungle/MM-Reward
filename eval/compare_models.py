#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(description="比较不同奖励模型的评估结果")
    
    parser.add_argument("--results_dir", type=str, required=True, help="评估结果目录")
    parser.add_argument("--output_dir", type=str, default=None, help="输出目录，默认与结果目录相同")
    parser.add_argument("--model_types", type=str, nargs="+", default=None, 
                        help="要比较的模型类型，不指定则比较所有模型")
    parser.add_argument("--plot", action="store_true", help="是否生成可视化图表")
    
    return parser.parse_args()

def load_results(results_dir, model_types=None):
    """加载评估结果"""
    results = {}
    
    for filename in os.listdir(results_dir):
        if not filename.endswith(".json"):
            continue
        
        filepath = os.path.join(results_dir, filename)
        with open(filepath, "r") as f:
            data = json.load(f)
        
        model_type = data.get("model_type", "unknown")
        
        # 如果指定了模型类型，则只加载指定类型的结果
        if model_types and model_type not in model_types:
            continue
        
        # 使用模型类型作为键，可能有多个同类型模型的结果
        if model_type not in results:
            results[model_type] = []
        
        results[model_type].append({
            "filename": filename,
            "accuracy": data.get("accuracy", 0),
            "model_path": data.get("model_path", ""),
            "predictions": data.get("predictions", []),
            "timestamp": filename.split("_")[-1].split(".")[0] if "_" in filename else ""
        })
    
    # 对每种模型类型的结果按时间戳排序
    for model_type in results:
        results[model_type].sort(key=lambda x: x["timestamp"], reverse=True)
    
    return results

def analyze_results(results):
    """分析评估结果"""
    analysis = {}
    
    for model_type, model_results in results.items():
        if not model_results:
            continue
        
        # 获取最新的结果
        latest_result = model_results[0]
        
        # 计算准确率统计
        accuracies = [r["accuracy"] for r in model_results]
        
        analysis[model_type] = {
            "latest_accuracy": latest_result["accuracy"],
            "mean_accuracy": np.mean(accuracies),
            "std_accuracy": np.std(accuracies),
            "min_accuracy": np.min(accuracies),
            "max_accuracy": np.max(accuracies),
            "num_results": len(model_results),
            "latest_model_path": latest_result["model_path"]
        }
        
        # 分析预测分数分布
        if latest_result["predictions"]:
            chosen_scores = [p["chosen_score"] for p in latest_result["predictions"]]
            rejected_scores = [p["rejected_score"] for p in latest_result["predictions"]]
            
            analysis[model_type].update({
                "mean_chosen_score": np.mean(chosen_scores),
                "mean_rejected_score": np.mean(rejected_scores),
                "mean_score_diff": np.mean(np.array(chosen_scores) - np.array(rejected_scores)),
                "std_score_diff": np.std(np.array(chosen_scores) - np.array(rejected_scores)),
                "min_score_diff": np.min(np.array(chosen_scores) - np.array(rejected_scores)),
                "max_score_diff": np.max(np.array(chosen_scores) - np.array(rejected_scores))
            })
    
    return analysis

def compare_predictions(results):
    """比较不同模型的预测结果"""
    comparison = {}
    
    # 获取所有模型类型的最新结果
    latest_results = {model_type: model_results[0] for model_type, model_results in results.items() if model_results}
    
    # 如果只有一个模型，无法比较
    if len(latest_results) <= 1:
        return comparison
    
    # 创建样本ID到预测的映射
    predictions_by_sample = defaultdict(dict)
    
    for model_type, result in latest_results.items():
        for pred in result["predictions"]:
            sample_id = pred.get("sample_id", "unknown")
            predictions_by_sample[sample_id][model_type] = pred
    
    # 计算模型间的一致性和差异
    model_types = list(latest_results.keys())
    agreement_matrix = np.zeros((len(model_types), len(model_types)))
    
    for i, model_i in enumerate(model_types):
        for j, model_j in enumerate(model_types):
            if i == j:
                agreement_matrix[i][j] = 1.0
                continue
            
            # 计算两个模型之间的预测一致性
            agreements = 0
            total = 0
            
            for sample_id, preds in predictions_by_sample.items():
                if model_i in preds and model_j in preds:
                    pred_i = preds[model_i]["prediction"]
                    pred_j = preds[model_j]["prediction"]
                    
                    if pred_i == pred_j:
                        agreements += 1
                    
                    total += 1
            
            agreement_matrix[i][j] = agreements / total if total > 0 else 0
    
    comparison["agreement_matrix"] = agreement_matrix.tolist()
    comparison["model_types"] = model_types
    
    # 找出模型间预测不一致的样本
    disagreements = []
    
    for sample_id, preds in predictions_by_sample.items():
        if len(preds) < 2:
            continue
        
        # 检查是否有不一致的预测
        predictions = [preds[model_type]["prediction"] for model_type in preds]
        if not all(p == predictions[0] for p in predictions):
            disagreements.append({
                "sample_id": sample_id,
                "predictions": {model_type: preds[model_type] for model_type in preds}
            })
    
    comparison["disagreements"] = disagreements
    comparison["num_disagreements"] = len(disagreements)
    comparison["total_samples"] = len(predictions_by_sample)
    
    return comparison

def generate_plots(analysis, output_dir):
    """生成可视化图表"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 准确率比较图
    plt.figure(figsize=(10, 6))
    model_types = list(analysis.keys())
    accuracies = [analysis[model_type]["latest_accuracy"] for model_type in model_types]
    
    plt.bar(model_types, accuracies, color='skyblue')
    plt.xlabel('模型类型')
    plt.ylabel('准确率')
    plt.title('不同模型类型的准确率比较')
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 0.01, f'{acc:.4f}', ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_comparison.png'))
    plt.close()
    
    # 分数差异分布图
    plt.figure(figsize=(10, 6))
    
    for model_type in model_types:
        if "mean_score_diff" in analysis[model_type]:
            plt.bar(model_type, analysis[model_type]["mean_score_diff"], 
                    yerr=analysis[model_type]["std_score_diff"], 
                    capsize=5, alpha=0.7)
    
    plt.xlabel('模型类型')
    plt.ylabel('平均分数差异 (chosen - rejected)')
    plt.title('不同模型类型的分数差异比较')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'score_diff_comparison.png'))
    plt.close()

def main():
    args = parse_args()
    
    # 设置输出目录
    output_dir = args.output_dir or args.results_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载评估结果
    results = load_results(args.results_dir, args.model_types)
    
    if not results:
        print(f"在目录 {args.results_dir} 中未找到评估结果文件")
        return
    
    # 分析结果
    analysis = analyze_results(results)
    
    # 比较预测
    comparison = compare_predictions(results)
    
    # 输出结果表格
    table_data = []
    headers = ["模型类型", "最新准确率", "平均准确率", "标准差", "模型路径"]
    
    for model_type, stats in analysis.items():
        table_data.append([
            model_type,
            f"{stats['latest_accuracy']:.4f}",
            f"{stats['mean_accuracy']:.4f}",
            f"{stats['std_accuracy']:.4f}",
            stats['latest_model_path']
        ])
    
    print("\n模型性能比较:")
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # 输出模型间一致性
    if "agreement_matrix" in comparison:
        print("\n模型预测一致性矩阵:")
        agreement_table = []
        model_types = comparison["model_types"]
        
        for i, model_i in enumerate(model_types):
            row = [model_i]
            for j, model_j in enumerate(model_types):
                row.append(f"{comparison['agreement_matrix'][i][j]:.4f}")
            agreement_table.append(row)
        
        print(tabulate(agreement_table, headers=[""] + model_types, tablefmt="grid"))
        
        print(f"\n不一致预测样本数: {comparison['num_disagreements']} / {comparison['total_samples']}")
    
    # 保存分析结果
    with open(os.path.join(output_dir, "model_comparison.json"), "w") as f:
        json.dump({
            "analysis": analysis,
            "comparison": comparison
        }, f, indent=2)
    
    # 生成可视化图表
    if args.plot:
        generate_plots(analysis, output_dir)
        print(f"\n可视化图表已保存到: {output_dir}")
    
    print(f"\n比较结果已保存到: {os.path.join(output_dir, 'model_comparison.json')}")

if __name__ == "__main__":
    main() 