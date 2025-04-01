# SPDX-License-Identifier: MIT

import argparse
import json
from collections import defaultdict

def pick_spaced_solutions(sorted_solutions, k):
    """
    Given a list of solutions sorted by average_test_score (descending),
    select k solutions that best span the score range.
    Always include the solution with the maximum (top) and minimum (bottom) scores.
    Then, for each target quantile t in (0, 1) split evenly into (k-2) parts,
    choose the solution (not already selected) whose average_test_score is closest to:
         target = max_score - t * (max_score - min_score)
    
    Modification:
      If there is any solution with average_test_score > 0.0 and < 0.1,
      use the lowest such solution as the bottom extreme instead of one with 0.0.
    """
    n = len(sorted_solutions)
    if k >= n:
        return sorted_solutions

    max_score = 1.0
    selected_indices = set()

    # Look for a candidate solution with a low positive score (between 0.0 and 0.1).
    # Since sorted_solutions is in descending order, we iterate in reverse (i.e. ascending by score)
    # so that the first candidate we find is the one with the smallest score among those > 0.0.
    candidate_idx = None
    for i in reversed(range(n)):
        score = sorted_solutions[i]["average_test_score"]
        if 0.0 < score < 0.1:
            candidate_idx = i
            break

    if candidate_idx is not None:
        min_score = sorted_solutions[candidate_idx]["average_test_score"]
        selected_indices.add(candidate_idx)
    else:
        min_score = sorted_solutions[-1]["average_test_score"]
        selected_indices.add(n - 1)

    # Generate target quantiles; the extremes (1.0 and our chosen min_score) are already included.
    targets = [max_score - t * (max_score - min_score)
               for t in [i / (k) for i in range(1, k)]]

    # For each target, find the available solution closest to the target score.
    for target in targets:
        best_idx = None
        best_diff = float("inf")
        for i, sol in enumerate(sorted_solutions):
            if i in selected_indices:
                continue
            diff = abs(sol["average_test_score"] - target)
            if diff < best_diff:
                best_diff = diff
                best_idx = i
        if best_idx is not None:
            selected_indices.add(best_idx)

    # Return the selected solutions in the order of their original indices.
    return [sorted_solutions[i] for i in sorted(selected_indices)]


def process_solutions(solutions):
    """
    Process the solutions as follows:
    
    1. Remove duplicate solutions (duplicates are defined by having the same average_test_score).
       For each group of solutions with the same score:
         - If any solution is rank 1, keep all the rank 1 solutions (and drop any non-rank1 duplicate).
         - Otherwise, keep only the non-rank1 solution with the smallest average_time_taken.
    2. From the deduplicated set, if the total is >= 5, select at most 5 solutions (spread out in score),
       but always include any rank-1 solutions even if that makes more than 5.
    3. Finally, reassign the rank numbers based on descending average_test_score (so rank 1 has score 1.0).
    """
    # Group all solutions by average_test_score.
    groups = defaultdict(list)
    for s in solutions:
        score = s["average_test_score"]
        groups[score].append(s)
    
    deduped = []
    for score, group in groups.items():
        # Check if any solution in this score group is rank 1.
        rank1_group = [s for s in group if s.get("rank") == 1]
        if rank1_group:
            # Keep all rank-1 solutions and ignore the non-rank1 duplicates.
            deduped.extend(rank1_group)
        else:
            # Otherwise, keep only the non-rank1 solution with the smallest average_time_taken.
            best = min(group, key=lambda s: s["average_time_taken"])
            deduped.append(best)
    
    # Separate deduped solutions into rank1 and non-rank1.
    rank1_solutions = [s for s in deduped if s.get("rank") == 1]
    non_rank1_solutions = [s for s in deduped if s.get("rank") != 1]

    if len(rank1_solutions) > 1:
        raise ValueError("More than one rank 1 solution found.")

    # Sort by average_test_score in descending order.
    rank1_sorted = sorted(rank1_solutions, key=lambda s: s["average_test_score"], reverse=True)
    non_rank1_sorted = sorted(non_rank1_solutions, key=lambda s: s["average_test_score"], reverse=True)

    # Always keep rank1 solutions. Then, if the total is less than 5, pick additional non-rank1
    # solutions (spread evenly across the score range) to bring the total up to 5.
    if len(rank1_sorted) >= 5:
        final_solutions = rank1_sorted
    else:
        need_more = 5 - len(rank1_sorted)
        # Only pick as many as are available.
        spaced_non_rank1 = pick_spaced_solutions(non_rank1_sorted, min(need_more, len(non_rank1_sorted)))
        final_solutions = rank1_sorted + spaced_non_rank1

    # Re-rank the final solutions by descending average_test_score.
    final_solutions_sorted = sorted(final_solutions, key=lambda s: s["average_test_score"], reverse=True)
    for i, sol in enumerate(final_solutions_sorted, start=1):
        sol["rank"] = i

    return final_solutions_sorted

def filter_jsonl(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        
        for line in fin:
            record = json.loads(line)
            solutions = record.get("all_solutions", [])

            if "base_input" in solutions[0]["solution"]:
                record["base_input"] = solutions[0]["solution"]["base_input"]
                record["plus_input"] = solutions[0]["solution"]["plus_input"]

            updated = process_solutions(solutions)

            if "base_input" in solutions[0]["solution"]:
                for i in range(len(updated)):
                    updated[i]["solution"].pop("base_input")
                    updated[i]["solution"].pop("plus_input")
            else:
                for i in range(len(updated)):
                    updated[i]["solution"].pop("test_list")
                    updated[i]["solution"].pop("challenge_test_list")

            assert updated[0]['rank'] == 1
            updated[0]["solution"]["canonical_solution"] = updated[0]["solution"]["prompt"] + updated[0]["solution"]["canonical_solution"]
            record["all_solutions"] = updated
            
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Filter JSONL files with solutions.")
    parser.add_argument("input_path", type=str, help="Path to the input JSONL file.")
    parser.add_argument("output_path", type=str, help="Path to the output JSONL file.")
    args = parser.parse_args()

    filter_jsonl(args.input_path, args.output_path)
