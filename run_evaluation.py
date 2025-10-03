# run_evaluation.py

import os
import json
import sys
import datetime
from custom_models import CustomOllamaLLM
import config
import rag_agent
from evaluate_metric import run_evaluation

def get_rating(score):
    if score is None: return "[Error]"
    if score >= config.PASS_THRESHOLD: return "[Good]"
    elif score >= config.IMPROVEMENT_THRESHOLD: return "[Needs Improvement]"
    else: return "[Failure]"

def summarize_failures(results):
    print("\n--- Generating Improvement Suggestions ---")
    failures = []
    for metric, data in results.items():
        if metric == "eval_type": continue
        if data['rating'] != "[Good]":
            report_file = f"{metric}_{results['eval_type']}_report.json"
            if os.path.exists(report_file):
                with open(report_file, 'r', encoding='utf-8') as f:
                    report_data = json.load(f)
                metric_failures = [item for item in report_data if item.get('score') is not None and item['score'] < config.PASS_THRESHOLD]
                if metric_failures:
                    failures.append({"metric": metric, "failed_cases": metric_failures})
    if not failures:
        summary = "No specific failures found to summarize. Great job!"
        print(summary)
        return summary
        
    prompt = "You are an expert AI evaluation analyst. Based on the following evaluation results, identify patterns of failure and provide specific, actionable suggestions for improvement. Focus on the reasons provided for the low scores.\n\n"
    for failure in failures:
        prompt += f"Metric: {failure['metric']}\n"
        for case in failure['failed_cases'][:2]:
            prompt += f"  - Input: {case.get('input', 'N/A')}\n"
            prompt += f"  - Score: {case.get('score', 'N/A'):.2f}\n"
            prompt += f"  - Reason: {case.get('reason', 'N/A')}\n"
        prompt += "\n"
    prompt += "Summary of patterns and suggestions for improvement:"
    summarizer_llm = CustomOllamaLLM(model=config.DEEPEVAL_GENERATION_MODEL)
    summary = summarizer_llm.generate(prompt)
    
    return summary

def run_evaluation_main(eval_type):
    if eval_type == 'quick':
        goldens_file, synthesizer_script, goldens_with_output_file = (config.QUICK_GOLDENS_FILE, "synthesizer_quick.py", config.QUICK_GOLDENS_WITH_OUTPUT_FILE)
    else:
        goldens_file, synthesizer_script, goldens_with_output_file = (config.DEEP_GOLDENS_FILE, "synthesizer_deep.py", config.DEEP_GOLDENS_WITH_OUTPUT_FILE)
    
    if not os.path.exists(goldens_file):
        print(f"'{goldens_file}' not found. Running synthesizer...")
        if eval_type == 'quick':
            from synthesizer_quick import generate_quick_goldens
            generate_quick_goldens()
        else:
            from synthesizer_deep import generate_deep_goldens
            generate_deep_goldens()
    else:
        print(f"Found existing '{goldens_file}'. Skipping generation.")
    
    if not os.path.exists(goldens_with_output_file):
        print(f"'{goldens_with_output_file}' not found. Running RAG agent...")
        rag_agent.generate_rag_responses(goldens_file, goldens_with_output_file)
    else:
        print(f"Found existing '{goldens_with_output_file}'. Skipping RAG agent execution.")

    all_results = {"eval_type": eval_type}
    print("\n--- Starting Metric Evaluations ---")
    for metric in config.METRICS_TO_RUN:
        report_file = f"{metric}_{eval_type}_report.json"
        run_evaluation(metric, goldens_with_output_file, report_file)
        with open(report_file, 'r') as f:
            results = json.load(f)
        total_score = sum(item['score'] for item in results if item['score'] is not None)
        average_score = total_score / len(results) if results else 0
        all_results[metric] = {"score": average_score, "rating": get_rating(average_score)}

    report_content = []
    report_content.append("--- Final Evaluation Report ---")
    report_content.append(f"Mode: {eval_type.title()} Eval")
    report_content.append("-" * 35)
    for metric, data in all_results.items():
        if metric == "eval_type": continue
        score_str = f"{data['score']:.2f}" if data['score'] is not None else "Error"
        report_content.append(f"{metric:<22} | Score: {score_str:<5} | {data['rating']}")
    report_content.append("-" * 35)

    summary = summarize_failures(all_results)
    report_content.append("\n--- Improvement Summary ---")
    report_content.append(summary)

    final_report_str = "\n".join(report_content)

    print("\n\n" + final_report_str)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"evaluation_report_{eval_type}_{timestamp}.txt"
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write(final_report_str)
    
    print(f"\n[+] Full report saved to '{report_filename}'")
    return final_report_str, report_filename

if __name__ == "__main__":
    choice = ""
    while choice not in ['1', '2']:
        choice = input("Select evaluation type:\n1. Quick Eval\n2. Deep Eval\nEnter choice (1 or 2): ")
    if choice == '1':
        run_evaluation_main('quick')
    else:
        run_evaluation_main('deep')