# run_evaluation.py

import os
import subprocess
import json
import sys
import datetime
from concurrent.futures import ProcessPoolExecutor
from custom_models import CustomOllamaLLM
import config
import rag_agent

def run_metric_evaluation(metric_name, eval_type, goldens_file):
    # ... (This helper function is unchanged)
    print(f"---> Starting evaluation for: {metric_name}")
    report_file = f"{metric_name}_{eval_type}_report.json"
    command = [sys.executable, "evaluate_metric.py", metric_name, goldens_file, report_file]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True, encoding='utf-8')
        output_lines = result.stdout.strip().split('\n')
        score_line = [line for line in output_lines if "Average Score:" in line]
        if score_line:
            avg_score = float(score_line[0].split(':')[-1].strip())
            print(f"<--- Finished evaluation for: {metric_name} (Score: {avg_score:.2f})")
            return metric_name, avg_score
        else:
            print(f"<--- Finished evaluation for: {metric_name} (Error: Could not parse score)")
            return metric_name, None
    except subprocess.CalledProcessError as e:
        print(f"\n[Error] Subprocess for '{metric_name}' failed.")
        print("--- Subprocess Error Output (stderr) ---")
        print(e.stderr)
        print("----------------------------------------")
        return metric_name, None

def get_rating(score):
    # ... (This function is unchanged)
    if score is None: return "[Error]"
    if score >= config.PASS_THRESHOLD: return "[Good]"
    elif score >= config.IMPROVEMENT_THRESHOLD: return "[Needs Improvement]"
    else: return "[Failure]"

# MODIFIED: This function now returns the summary text
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
    
    # Return the summary instead of just printing it
    return summary


def main():
    # ... (User choice, golden generation, and RAG agent logic is unchanged)
    choice = ""
    while choice not in ['1', '2']:
        choice = input("Select evaluation type:\n1. Quick Eval\n2. Deep Eval\nEnter choice (1 or 2): ")
    if choice == '1':
        eval_type, goldens_file, synthesizer_script, goldens_with_output_file = ("quick", config.QUICK_GOLDENS_FILE, "synthesizer_quick.py", config.QUICK_GOLDENS_WITH_OUTPUT_FILE)
    else:
        eval_type, goldens_file, synthesizer_script, goldens_with_output_file = ("deep", config.DEEP_GOLDENS_FILE, "synthesizer_deep.py", config.DEEP_GOLDENS_WITH_OUTPUT_FILE)
    if not os.path.exists(goldens_file):
        print(f"'{goldens_file}' not found. Running synthesizer...")
        subprocess.run([sys.executable, synthesizer_script], check=True, encoding='utf-8')
    else:
        print(f"Found existing '{goldens_file}'. Skipping generation.")
    if not os.path.exists(goldens_with_output_file):
        print(f"'{goldens_with_output_file}' not found. Running RAG agent...")
        rag_agent.generate_rag_responses(goldens_file, goldens_with_output_file)
    else:
        print(f"Found existing '{goldens_with_output_file}'. Skipping RAG agent execution.")

    all_results = {"eval_type": eval_type}
    print("\n--- Starting Parallel Metric Evaluations ---")
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_metric_evaluation, metric, eval_type, goldens_with_output_file) for metric in config.METRICS_TO_RUN]
        for future in futures:
            metric_name, avg_score = future.result()
            if metric_name:
                all_results[metric_name] = {"score": avg_score, "rating": get_rating(avg_score)}

    # NEW: Build the report as a list of strings
    report_content = []
    report_content.append("--- Final Evaluation Report ---")
    report_content.append(f"Mode: {eval_type.title()} Eval")
    report_content.append("-" * 35)
    for metric, data in all_results.items():
        if metric == "eval_type": continue
        score_str = f"{data['score']:.2f}" if data['score'] is not None else "Error"
        report_content.append(f"{metric:<22} | Score: {score_str:<5} | {data['rating']}")
    report_content.append("-" * 35)

    # Get the AI-powered summary
    summary = summarize_failures(all_results)
    report_content.append("\n--- Improvement Summary ---")
    report_content.append(summary)

    # Combine the report into a single string
    final_report_str = "\n".join(report_content)

    # Print the final report to the console
    print("\n\n" + final_report_str)

    # Save the final report to a timestamped text file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"evaluation_report_{eval_type}_{timestamp}.txt"
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write(final_report_str)
    
    print(f"\n[+] Full report saved to '{report_filename}'")


if __name__ == "__main__":
    main()