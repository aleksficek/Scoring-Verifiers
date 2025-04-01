# SPDX-License-Identifier: MIT

import json

FILE_PATH = ''

# Prepare accumulators for input lengths and scores.
base_lengths = {'MBPP_base': [], 'HE_base': []}
plus_lengths = {'MBPP_plus': [], 'HE_plus': []}
ground_scores = {'MBPP_base': [], 'HE_base': [], 'MBPP_plus': [], 'HE_plus': []}

with open(FILE_PATH, 'r') as f:
    for line in f:
        entry = json.loads(line)
        dataset = entry.get('dataset')
        score = entry.get('ground_average_test_score')

        # Bucket ground scores by dataset type.
        if dataset in ground_scores:
            ground_scores[dataset].append(score)

        # For base datasets, count length of "base_inputs"

        # base_inputs = entry["all_solutions"][0].get('base_input', [])
        # base_lengths[dataset].append(len(base_inputs))
        # For plus datasets, count length of "plus_inputs"
        plus_inputs = entry.get('plus_input') + entry.get('base_input')
        print(plus_inputs)
        break
        plus_lengths['MBPP_plus'].append(len(plus_inputs))

print(plus_lengths)

def average(lst):
    return sum(lst) / len(lst) if lst else 0

print("=== Average Base Input Lengths ===")
for key, lengths in base_lengths.items():
    avg = average(lengths)
    print(f"{key}: {avg:.2f} (n={len(lengths)})")

print("\n=== Average Plus Input Lengths ===")
for key, lengths in plus_lengths.items():
    avg = average(lengths)
    print(f"{key}: {avg:.2f} (n={len(lengths)})")

print("\n=== Ground Average Test Score (Bucketed) ===")
for key, scores in ground_scores.items():
    avg = average(scores)
    print(f"{key}: {avg:.2f} (n={len(scores)})")
