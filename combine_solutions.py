# SPDX-License-Identifier: MIT

import argparse
import json
import glob
import os
from collections import defaultdict

TIME_RATIO_THRESHOLD = 1.0

###############################################################################
# Helpers
###############################################################################

def get_test_score(sol, dim):
    """
    Retrieve test_score from sol[dim + "_execution_result"]["average_test_score"].
    Returns 0.0 if missing or invalid.
    """
    if not sol:
        return 0.0
    res = sol.get(dim + "_execution_result", {})
    return float(res["average_test_score"])

def get_time_taken(sol, dim):
    """
    Retrieve time_taken from sol[dim + "_execution_result"]["average_time_taken"].
    Returns float('inf') if missing (so that it is considered 'worst').
    """
    if not sol:
        return float('inf')
    res = sol.get(dim + "_execution_result", {})
    if "average_time_taken" in res:
        return res["average_time_taken"]
    times = res.get("time_taken", [])
    if not times:
        return float('inf')
    return sum(times) / len(times)

def rank_dimension(solutions, dim, original_id):
    """
    Rank solutions by `test_score` (descending) for the given dimension ('base' or 'plus'),
    with tie-breaking via *pairwise* time ratio (<2.0 => discard the worse [bigger time]).

    The solution with id == original_id is forced rank=1 and never discarded.
    Returns: dict of solution_id -> rank (int),
             solutions not in the final ranking are omitted.
    """

    # Separate out the forced original solution
    forced_original = None
    others = []
    for s in solutions:
        if s["id"] == original_id:
            forced_original = s
        else:
            others.append(s)
    
    # --- 1) Group 'others' by their test_score for this dimension ---
    score_map = defaultdict(list)
    for sol in others:
        score = get_test_score(sol["solution"], dim)
        if dim == "plus":
            score = (score * len(sol["solution"].get("plus_input", [])) + get_test_score(sol["solution"], "base") * len(sol["solution"]["base_input"])) / (len(sol["solution"].get("plus_input", [])) + len(sol["solution"]["base_input"]))
        score_map[score].append(sol)

    # We will collect survivors here:
    final_survivors = []

    # --- 2) For each test_score group, do pairwise checks ---
    for score, group in score_map.items():
        # If forced_original has this same test_score, treat it as in this group (so it can never be removed)
        in_this_group = group[:]
        if forced_original is not None:
            fo_score = get_test_score(forced_original["solution"], dim)
            if abs(fo_score - score) < 1e-20:
                # forced original is effectively part of this tie group
                in_this_group.append(forced_original)
        # We'll repeatedly compare solutions pairwise until no more can be removed.
        changed = True
        while changed:
            changed = False
            i = 0
            while i < len(in_this_group):
                s1 = in_this_group[i]
                t1 = get_time_taken(s1["solution"], dim)
                j = i + 1
                while j < len(in_this_group):
                    s2 = in_this_group[j]
                    t2 = get_time_taken(s2["solution"], dim)
                    # Check if they have essentially the same test_score:
                    score_s1 = get_test_score(s1["solution"], dim)
                    score_s2 = get_test_score(s2["solution"], dim)
                    if abs(score_s1 - score_s2) < 1e-9:
                        bigger = s1 if t1 > t2 else s2
                        ratio = (max(t1, t2) / min(t1, t2)) if min(t1, t2) > 0 else float('inf')
                        # If ratio < 2 => discard the bigger-time solution unless it is forced_original
                        if ratio < TIME_RATIO_THRESHOLD:
                            if bigger["id"] != original_id:
                                in_this_group.remove(bigger)
                                changed = True
                                # Because we removed bigger, adjust indices accordingly:
                                if bigger is s1:
                                    j -= 1
                                    s1 = None
                                    break
                                else:
                                    j -= 1
                            else:
                                # forced_original is the bigger one => do NOT discard it
                                pass
                        else:
                            # ratio >= 2 => keep both
                            pass
                    j += 1
                i += 1
        
        # After tie-breaking is done for this group, add survivors (except duplicates of forced original)
        for s in in_this_group:
            if s is not forced_original:
                final_survivors.append(s)

    # Ensure forced_original is in final_survivors
    if forced_original is not None and forced_original not in final_survivors:
        final_survivors.append(forced_original)

    # --- 3) Sort final survivors by test_score (descending) ---
    final_survivors.sort(key=lambda x: get_test_score(x["solution"], dim), reverse=True)

    # --- 4) Build the final ranking ---
    result = {}
    result[original_id] = 1
    rank_counter = 2
    for s in final_survivors:
        sid = s["id"]
        if sid != original_id:
            result[sid] = rank_counter
            rank_counter += 1

    return result

def clean_solution(sol):
    """
    Remove unwanted keys from top level (base_input, plus_input),
    and from base_execution_results & plus_execution_results:
      time_taken, unit_test_stderrs, unit_test_stdouts, correct_tests, traceback.
    Modifies 'sol' in place and returns it.
    """
    sol.pop("base_input", None)
    sol.pop("plus_input", None)
    
    # If there's a "solution" key that contains base/plus execution results, remove sub-keys
    if "solution" in sol and sol["solution"]:
        for dim in ["base", "plus"]:
            exec_key = dim + "_execution_result"
            if exec_key in sol["solution"]:
                # Fill average_time_taken if missing
                if "average_time_taken" not in sol["solution"][exec_key]:
                    times = sol["solution"][exec_key].get("time_taken", [])
                    if times:
                        sol["solution"][exec_key]["average_time_taken"] = sum(times) / len(times)
                    else:
                        sol["solution"][exec_key]["average_time_taken"] = float('inf')
                for k in ["time_taken", "unit_test_stderrs", "unit_test_stdouts", "correct_tests", "traceback"]:
                    sol["solution"][exec_key].pop(k, None)
    return sol

def all_stderrs_nonempty(sol):
    """
    Returns True if **all** stderr lines in base_execution_result['unit_test_stderrs']
    + plus_execution_result['unit_test_stderrs'] are non-empty strings.
    Otherwise returns False.
    """
    base_stderrs = sol["solution"]["base_execution_result"]["unit_test_stderrs"]
    if "plus_execution_result" in sol["solution"]:
        plus_stderrs = sol["solution"]["plus_execution_result"]["unit_test_stderrs"]
    else:
        plus_stderrs = []
    
    combined = base_stderrs + plus_stderrs
    if not combined:
        return False
    
    for line in combined:
        if not line.strip() or line.strip() == "AssertionError()":
            return False
    return True

###############################################################################
# Main Script
###############################################################################

def main():

    parser = argparse.ArgumentParser(description="Process and rank solutions.")
    parser.add_argument('--dataset_type', type=str, default="MBPP+",
                        help='Type of dataset to process (e.g., HE, MBPP, MBPP+).')
    parser.add_argument('--input_file', type=str, required=True,
                        help='Path to the input JSONL file containing scored solutions.')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Path to the input directory containing scored solutions.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to the output directory.')
    
    args = parser.parse_args()

    dataset_type = args.dataset_type
    input_file = args.input_file
    input_dir = args.input_dir
    output_dir = args.output_dir

    output_file_unranked = f"{output_dir}/{dataset_type}_unranked.jsonl"
    output_file_ranked = f"{output_dir}/{dataset_type}_ranked.jsonl"

    output_file_base = f"{output_dir}/{dataset_type}_base_ranked.jsonl"
    output_file_plus = f"{output_dir}/{dataset_type}_plus_ranked.jsonl"
    
    # Collect all solution filenames of form exec_*.jsonl
    solution_files = sorted(glob.glob(os.path.join(input_dir, "exec_*.jsonl")))
    
    # Read input lines
    with open(input_file, "r", encoding="utf-8") as fin:
        input_lines = [json.loads(line.strip()) for line in fin]
    
    # Read lines from each solution file
    solutions_per_file = []
    for sf in solution_files:
        with open(sf, "r", encoding="utf-8") as f:
            solutions_per_file.append([json.loads(l.strip()) for l in f])
    
    with open(output_file_unranked, "w", encoding="utf-8") as f_unranked, \
         open(output_file_ranked, "w", encoding="utf-8") as f_ranked, \
         open(output_file_base, "w", encoding="utf-8") as f_base, \
         open(output_file_plus, "w", encoding="utf-8") as f_plus:

        tossed_solutions = 0
        
        for i, original_line in enumerate(input_lines):
            if i % 25 == 0:
                print(f"Processing line {i}...")

            # --------------------------------------------
            #  Build the list of solutions (including original)
            # --------------------------------------------
            # Validate that the original line contains a prompt.
            if dataset_type != "MBPP":
                original_prompt = original_line.get("prompt")
                if original_prompt is None:
                    print(f"Original solution at line {i} does not contain a 'prompt' key.")
            else:
                original_prompt = original_line.get("text")
                if original_prompt is None:
                    print(f"Original solution at line {i} does not contain a 'text' key.")


            all_solutions = []
            original_sol = {
                "id": 0,
                "solution": original_line
            }
            all_solutions.append(original_sol)
            
            # Zip the solution filenames with their corresponding lists for better error messages.
            current_id = 1
            for sf, sol_file_lines in zip(solution_files, solutions_per_file):
                if i < len(sol_file_lines):
                    candidate = sol_file_lines[i]
                    if dataset_type != "MBPP":
                        if candidate.get("prompt") != original_prompt:
                            print(f"Prompt mismatch in file {sf} at line {i} for solution id {current_id}.")
                    else:
                        if candidate.get("text") != original_prompt:
                            if candidate.get("text").startswith(original_prompt):
                                candidate["text"] = original_prompt
                            else:
                                print(f"Prompt mismatch in file {sf} at line {i} for solution id {current_id}.")
                    all_solutions.append({
                        "id": current_id,
                        "solution": candidate
                    })
                current_id += 1

            # ---------------------------------------------------
            #  Filter solutions that have all stderr lines non-empty
            #  If forced original (id=0) meets this condition, crash
            # ---------------------------------------------------
            filtered_solutions = []
            for s in all_solutions:
                if all_stderrs_nonempty(s):
                    if s["id"] == 0:
                        raise RuntimeError(
                            "Original solution (id=0) has all stderr lines non-empty. "
                            "Crashing as requested."
                        )
                    tossed_solutions += 1
                else:
                    filtered_solutions.append(s)

            all_solutions = filtered_solutions

            # -----------------
            # 1) Unranked file
            # -----------------
            unranked_entry = dict(original_line)
            unranked_entry.pop("base_input", None)
            unranked_entry.pop("plus_input", None)

            unranked_entry["all_solutions"] = all_solutions

            clean_solution(unranked_entry)
            for sol_obj in unranked_entry["all_solutions"]:
                tmp = {"solution": sol_obj["solution"]}
                clean_solution(tmp)
                sol_obj["solution"] = tmp["solution"]

            f_unranked.write(json.dumps(unranked_entry) + "\n")
            
            # -----------------
            # 2) Ranked file
            # -----------------
            base_ranks = rank_dimension(all_solutions, "base", original_id=0)
            if dataset_type != "MBPP":
                plus_ranks = rank_dimension(all_solutions, "plus", original_id=0)
            else:
                plus_ranks = {}
            
            final_solutions = []
            for s in all_solutions:
                sid = s["id"]
                s_base_rank = base_ranks.get(sid, None)
                s_plus_rank = plus_ranks.get(sid, None)
                final_solutions.append({
                    "rank": {
                        "base_execution": s_base_rank,
                        "plus_execution": s_plus_rank
                    },
                    "solution": s["solution"]
                })
            
            ranked_entry = dict(original_line)
            ranked_entry.pop("base_input", None)
            ranked_entry.pop("plus_input", None)
            ranked_entry.pop("base_execution_result", None)
            ranked_entry.pop("plus_execution_result", None)
            ranked_entry["all_solutions"] = final_solutions
            
            clean_solution(ranked_entry)
            for sol_dict in ranked_entry["all_solutions"]:
                tmp = {"solution": sol_dict["solution"]}
                clean_solution(tmp)
                sol_dict["solution"] = tmp["solution"]

                if dataset_type != "MBPP":
                    sol_dict["average_test_score"] = {
                        "base_execution": sol_dict["solution"]["base_execution_result"]["average_test_score"],
                        "plus_execution": (sol_dict["solution"]["plus_execution_result"]["average_test_score"] * len(sol_dict["solution"]["plus_input"]) + sol_dict["solution"]["base_execution_result"]["average_test_score"] * len(sol_dict["solution"]["base_input"])) / (len(sol_dict["solution"]["plus_input"]) + len(sol_dict["solution"]["base_input"]))
                    }
                    sol_dict["average_time_taken"] = {
                        "base_execution": sol_dict["solution"]["base_execution_result"]["average_time_taken"],
                        "plus_execution": sol_dict["solution"]["plus_execution_result"]["average_time_taken"]
                    }
                    del sol_dict["solution"]["base_execution_result"]
                    del sol_dict["solution"]["plus_execution_result"]
                else:
                    sol_dict["average_test_score"] = {
                        "base_execution": sol_dict["solution"]["base_execution_result"]["average_test_score"],
                        "plus_execution": None
                    }
                    sol_dict["average_time_taken"] = {
                        "base_execution": sol_dict["solution"]["base_execution_result"]["average_time_taken"],
                        "plus_execution": None
                    }
                    del sol_dict["solution"]["base_execution_result"]

            f_ranked.write(json.dumps(ranked_entry) + "\n")

            # -------------------------------------------------------------
            # 3) Additional base/plus files with filtered all_solutions
            # -------------------------------------------------------------
            # Base file
            base_entry = {k: v for k, v in ranked_entry.items() if k != "all_solutions"}
            base_solutions = []
            for sol_dict in ranked_entry["all_solutions"]:
                b_rank = sol_dict["rank"]["base_execution"]
                if b_rank is not None:
                    new_sol = {
                        "rank": b_rank,
                        "average_test_score": round(sol_dict["average_test_score"]["base_execution"], 2),
                        "average_time_taken": sol_dict["average_time_taken"]["base_execution"],
                        "solution": sol_dict["solution"]
                    }
                    base_solutions.append(new_sol)

            base_solutions = sorted(base_solutions, key=lambda x: x["rank"])
            base_entry["all_solutions"] = base_solutions
            f_base.write(json.dumps(base_entry) + "\n")

            # Plus file
            plus_entry = {k: v for k, v in ranked_entry.items() if k != "all_solutions"}
            plus_solutions = []
            for sol_dict in ranked_entry["all_solutions"]:
                p_rank = sol_dict["rank"]["plus_execution"]
                if p_rank is not None:
                    new_sol = {
                        "rank": p_rank,
                        "average_test_score": round(sol_dict["average_test_score"]["plus_execution"], 2),
                        "average_time_taken": sol_dict["average_time_taken"]["plus_execution"],
                        "solution": sol_dict["solution"]
                    }
                    plus_solutions.append(new_sol)

            plus_solutions = sorted(plus_solutions, key=lambda x: x["rank"])
            plus_entry["all_solutions"] = plus_solutions
            f_plus.write(json.dumps(plus_entry) + "\n")

        print(f"Done! Tossed {tossed_solutions} solutions due to all non-empty stderrs.")

if __name__ == "__main__":
    main()
