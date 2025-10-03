# evaluate_metric.py

import json
import argparse
from deepeval.metrics import (
    AnswerRelevancyMetric, FaithfulnessMetric, ContextualPrecisionMetric,
    ContextualRecallMetric, ContextualRelevancyMetric
)
from deepeval.test_case import LLMTestCase
from custom_models import CustomOllamaLLM
import config

METRIC_MAP = {
    "answer_relevancy": AnswerRelevancyMetric, "faithfulness": FaithfulnessMetric,
    "contextual_precision": ContextualPrecisionMetric, "contextual_recall": ContextualRecallMetric,
    "contextual_relevancy": ContextualRelevancyMetric,
}

def run_evaluation(metric_name, goldens_file, report_file):
    if metric_name not in METRIC_MAP:
        raise ValueError(f"Unknown metric: {metric_name}")
    print(f"\n--- Evaluating metric: {metric_name} ---")
    ollama_judge = CustomOllamaLLM(model=config.DEEPEVAL_GENERATION_MODEL)
    metric = METRIC_MAP[metric_name](threshold=0.5, model=ollama_judge, include_reason=True)
    with open(goldens_file, 'r') as f:
        goldens = json.load(f)
    results_data = []
    total_score = 0
    for i, golden in enumerate(goldens):
        test_case = LLMTestCase(
            input=golden.get("input"),
            actual_output=golden.get("actual_output"),
            expected_output=golden.get("expected_output"),
            retrieval_context=golden.get("retrieval_context")
        )
        try:
            metric.measure(test_case)
            print(f"  - Scored test case {i+1}/{len(goldens)}: {metric.score:.2f}")
            results_data.append({"input": golden.get("input"), "score": metric.score, "reason": metric.reason})
            if metric.score is not None:
                total_score += metric.score
        except Exception as e:
            print(f"  - [Error] scoring test case {i+1}/{len(goldens)}: {e}")
            results_data.append({"input": golden.get("input"), "score": None, "reason": str(e)})
    with open(report_file, 'w') as f:
        json.dump(results_data, f, indent=4)
    average_score = total_score / len(goldens) if goldens else 0

    # CORRECTED: Replaced emoji with '[+]'
    print(f"[+] Detailed report saved to '{report_file}'")
    
    print(f"Average Score: {average_score}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("metric_name", type=str, help="The name of the metric to run.")
    parser.add_argument("goldens_file", type=str, help="Path to the goldens JSON file.")
    parser.add_argument("report_file", type=str, help="Path to save the output report.")
    args = parser.parse_args()
    run_evaluation(args.metric_name, args.goldens_file, args.report_file)