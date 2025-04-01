# SPDX-License-Identifier: MIT

import json
import argparse
import math
import os
from statistics import mean
from scipy.stats import kendalltau, spearmanr, rankdata
import numpy as np

def compute_r2(y_true, y_pred):
    """
    Compute the coefficient of determination (R²).
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot != 0 else 1.0

def process_file(filename, method):
    # Initialize dictionary for the four datasets.
    datasets = {"HE_plus": {}, "HE_base": {}, "MBPP_plus": {}, "MBPP_base": {}}

    # Read and group entries from the JSONL file.
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Skipping invalid JSON: {e}")
                continue

            ds = entry.get("dataset")
            if ds not in datasets:
                continue  # ignore datasets not in the allowed set
            task_id = entry.get("task_id")
            if task_id is None:
                continue
            datasets[ds].setdefault(task_id, []).append(entry)

    # Verification: in each group, ensure all entries share the same dataset and task_id.
    for ds, groups in datasets.items():
        for tid, entries in groups.items():
            for entry in entries:
                if entry.get("dataset") != ds or entry.get("task_id") != tid:
                    raise ValueError(f"Verification failed in dataset {ds} for task_id {tid}")

    # Prepare to accumulate metrics per dataset.
    results = {}
    for ds, groups in datasets.items():
        total_mae =  []
        total_ground_scores = []
        total_noramalized_reward_scores = []
        group_kendall = []
        group_spearman = []
        top1_correct = 0
        topn_correct = 0
        bottom1_correct = 0
        num_groups = 0

        for tid, entries in groups.items():
            num_groups += 1

            # Sort entries in the group by the chosen score (no random tie-breaker).
            try:
                if method == "reward":
                    sorted_entries = sorted(entries, key=lambda x: x["reward"]["reward_score"], reverse=True)
                else:
                    sorted_entries = sorted(entries, key=lambda x: x["average_test_score"], reverse=True)
            except KeyError as e:
                raise KeyError(f"Missing key {e} in one of the entries in task_id {tid}")

            # Extract scores for computed ranking.
            if method == "reward":
                scores = [entry["reward"]["reward_score"] for entry in sorted_entries]
            else:
                scores = [entry["average_test_score"] for entry in sorted_entries]

            # Compute computed ranks using average ranking.
            # (Negate scores so that higher scores get lower rank numbers.)
            computed_ranks_array = rankdata([-s for s in scores], method='average')
            for i, entry in enumerate(sorted_entries):
                entry["computed_rank"] = computed_ranks_array[i]

            # Process the given ranks with average ranking in case of ties.
            raw_given_ranks = [entry["rank"] for entry in sorted_entries]
            given_ranks_avg = rankdata(raw_given_ranks, method='average')

            # Compute Kendall's Tau and Spearman's rho on the averaged ranks.
            tau, _ = kendalltau(given_ranks_avg, computed_ranks_array)
            if tau is None or np.isnan(tau):
                tau = 0.0
            group_kendall.append(tau)


            spearman, _ = spearmanr(given_ranks_avg, computed_ranks_array)
            if spearman is None or np.isnan(spearman):
                spearman = 0.0
            group_spearman.append(spearman)

            # Top-1 accuracy: use fractional ranking for the top group.
            # Identify all entries tied at the top (by computed score).
            if method == "reward":
                top_score = sorted_entries[0]["reward"]["reward_score"]
                score_func = lambda e: e["reward"]["reward_score"]
            else:
                top_score = sorted_entries[0]["average_test_score"]
                score_func = lambda e: e["average_test_score"]

            top_group = [entry for entry in sorted_entries if score_func(entry) == top_score]
            count_top = len(top_group)
            # If the unique ground truth top (rank==1) is among the tie, add fractional credit.
            if any(entry["rank"] == 1 for entry in top_group):
                top1_correct += 1 / count_top

            # Identify the computed bottom group: all entries with the lowest computed score.
            if method == "reward":
                bottom_score = sorted_entries[-1]["reward"]["reward_score"]
                score_func = lambda e: e["reward"]["reward_score"]
            else:
                bottom_score = sorted_entries[-1]["average_test_score"]
                score_func = lambda e: e["average_test_score"]

            bottom_group = [entry for entry in sorted_entries if score_func(entry) == bottom_score]
            count_bottom = len(bottom_group)

            # The ground truth always has a unique bottom (with the maximum rank).
            ground_truth_bottom = max(entry["rank"] for entry in sorted_entries)
            if any(entry["rank"] == ground_truth_bottom for entry in bottom_group):
                bottom1_correct += 1 / count_bottom


            # For MAE and R² computations.
            if method == "reward":
                high = sorted_entries[0]["reward"]["reward_score"]
                low = sorted_entries[-1]["reward"]["reward_score"]
                denom = high - low if high != low else 1.0  # Avoid division by zero.
                normalized_reward_scores = [
                    (entry["reward"]["reward_score"] - low) / denom for entry in sorted_entries
                ]
            else:
                high = 1.0
                low = 0.0
                denom = 1.0  # Avoid division by zero.
                normalized_reward_scores = [
                    entry["average_test_score"] for entry in sorted_entries
                ]

            ground_scores = [entry["ground_average_test_score"] for entry in sorted_entries]

            # Compute MAE and R² between normalized rewards and ground scores.
            deltas = [abs(nr - gs) for nr, gs in zip(normalized_reward_scores, ground_scores)]
            total_mae += deltas
            total_ground_scores += ground_scores
            total_noramalized_reward_scores += normalized_reward_scores

        # Aggregate metrics for the dataset.
        results[ds] = {
            "total_MAE": mean(total_mae) if total_mae else None,
            "total_R2": compute_r2(total_ground_scores, total_noramalized_reward_scores) if total_ground_scores else None,
            "mean_Kendall_tau": mean(group_kendall) if group_kendall else None,
            "mean_spearman": mean(group_spearman) if group_spearman else None,
            "top1_accuracy": top1_correct / num_groups if num_groups else None,
            "bottom1_accuracy": bottom1_correct / num_groups if num_groups else None,
            "topn_accuracy": topn_correct / num_groups if num_groups else None,

        }
    return results

def print_csv_table(results, title):
    """
    Prints a CSV table where:
      - The first row is a header with "Metric" then dataset names.
      - Each subsequent row is a metric and its values for each dataset.
    """
    # Define the desired order for datasets and metrics.
    datasets_order = ["HE_plus", "HE_base", "MBPP_plus", "MBPP_base"]
    metrics_order = [
        "top1_accuracy",
        "bottom1_accuracy",
        "mean_spearman",
        "mean_Kendall_tau",
        "total_MAE",
        "total_R2",
    ]
    
    # Print title.
    print(f"Title: {title}")
    # Print header row.
    header = ["Metric"] + datasets_order
    print(",".join(header))
    
    # Print each metric row.
    for metric in metrics_order:
        row = [metric]
        for ds in datasets_order:
            value = results.get(ds, {}).get(metric, "")
            # Format float values to 4 decimal places if applicable.
            if isinstance(value, float):
                value = f"{value:.4f}"
            row.append(str(value))
        print(",".join(row))

def main():

    parser = argparse.ArgumentParser(description="Evaluate synthetic code scoring.")
    parser.add_argument("input_file", type=str, help="Path to the input JSONL file.")
    parser.add_argument("--method", type=str, default="utg", help="Method used for processing the file, either reward or utg")
    parser.add_argument("--output_file", type=str, default=None, help="Path to the output file.")
    args = parser.parse_args()

    input_file = args.input_file
    method = args.method
    output_file = args.output_file 

    results = process_file(input_file, method)
    for ds, metrics in results.items():
        print(f"Dataset: {ds}")
        for metric, value in metrics.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")
        print()

    print()
    # Use the provided output path filename as title, or fall back to the input file basename.
    if output_file:
        title = os.path.basename(output_file)
    else:
        title = os.path.basename(input_file)
    print_csv_table(results, title)

if __name__ == "__main__":
    main()
