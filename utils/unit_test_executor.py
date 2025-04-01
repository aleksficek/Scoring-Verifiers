# SPDX-License-Identifier: MIT

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from code_execution_handler import local_code_execution
from mbpp_handler import mbpp_deserialize_inputs
import sys

sys.set_int_max_str_digits(5000000)

TIMEOUT_MULTIPLE = 4.0
TIMEOUT_MIN = 0.1

# Function to process a single data item
def process_line(index, line, dataset_type="HE", add_prompt=True, timeout=30, timeouts_list=False):
    
    start_time = time.time()
    if type(line) == dict:
        data = line
    else:
        data = json.loads(line.strip())

    if dataset_type == "MBPP":

        combined_solution = data["code"] + "\n" + data["test_setup_code"] + "\n\n"
        base_input = data["test_list"] + data["challenge_test_list"]

        base_unit_tests = base_input
        plus_unit_tests = []
    else:
        prompt = data["prompt"]
        canonical_solution = data["canonical_solution"]

        # Extract function signature: take everything before the first """
        func_signature = prompt.strip()

        if add_prompt:
            combined_solution = f"{func_signature}\n{canonical_solution}"
        else: 
            combined_solution = canonical_solution

        # Step 2: Execute base_input and plus_input
        base_input = data.get("base_input", [])
        plus_input = data.get("plus_input", [])

    if dataset_type == "HE":
        base_unit_tests = [
            f"\nprint({data['entry_point']}(*{inputs}))"
            for inputs in base_input
        ]
        plus_unit_tests = [
            f"\nprint({data['entry_point']}(*{inputs}))"
            for inputs in plus_input
        ]

    if dataset_type == "MBPP+":
        base_input = mbpp_deserialize_inputs(data["task_id"], base_input)
        plus_input = mbpp_deserialize_inputs(data["task_id"], plus_input)

        base_unit_tests = [
            f"\nprint({data['entry_point']}(*{inputs}))"
            for inputs in base_input
        ]
        plus_unit_tests = [
            f"\nprint({data['entry_point']}(*{inputs}))"
            for inputs in plus_input
        ]


        if data["task_id"] == "Mbpp/404":
            base_unit_tests = [ "\nfrom math import inf\n" + inputs
                for inputs in base_unit_tests
            ]
            plus_unit_tests = [ "\nfrom math import inf\n" + inputs
                for inputs in plus_unit_tests
            ]

        if data["task_id"] in ["Mbpp/737", "Mbpp/787", "Mbpp/794"]:
            base_unit_tests = [
                f"\nprint(bool({data['entry_point']}(*{inputs})))"
                for inputs in base_input
            ]
            plus_unit_tests = [
                f"\nprint(bool({data['entry_point']}(*{inputs})))"
                for inputs in plus_input
            ]

    if timeouts_list:
        base_timeouts_list = [max(TIMEOUT_MIN, x * TIMEOUT_MULTIPLE) for x in data['base_execution_result']['time_taken']]
        plus_timeouts_list = [max(TIMEOUT_MIN, x * TIMEOUT_MULTIPLE) for x in data['plus_execution_result']['time_taken']]
    else:
        base_timeouts_list = None
        plus_timeouts_list = None

    # Execute base_input tests
    base_results = local_code_execution(combined_solution, base_unit_tests, timeout=timeout, timeouts_list=base_timeouts_list)
    data["base_execution_result"] = base_results

    # Execute plus_input tests
    plus_results = local_code_execution(combined_solution, plus_unit_tests, timeout=timeout, timeouts_list=plus_timeouts_list)
    data["plus_execution_result"] = plus_results

    # Check for errors and flag the task_id if needed
    if any(base_results.get("unit_test_stderrs", [])) or any(plus_results.get("unit_test_stderrs", [])):
        print(f"Error in task_id: {data['task_id']}")

    if dataset_type == "MBPP":
        data["text"] = data["text"] + '\n' + data["test_list"][0] + '\n'

    return index, json.dumps(data), time.time() - start_time

# Main function to process data with parallelization
def process_data(lines, output_file, dataset_type="HE", add_prompt=True, timeout=30, timeouts_list=False):

    count = 0
    results = []

    with ProcessPoolExecutor(max_workers=8) as executor:
        future_to_index = {executor.submit(process_line, i, line, dataset_type, add_prompt, timeout, timeouts_list): i for i, line in enumerate(lines)}

        for future in as_completed(future_to_index):
            index = future_to_index[future]
            exec_time = 0
            try:
                result, data, exec_time = future.result()
                results.append((result, data))
            except Exception as e:
                print(f"Error processing line {index}: {e}")

            count += 1
            print(f"Executed prompt {count} in {exec_time:.2f} seconds")

    # Sort results by the original order
    results.sort(key=lambda x: x[0])

    # Write sorted results to the output file
    with open(output_file, 'w') as f_out:
        for _, data in results:
            f_out.write(data + '\n')

    print(f"Completed executing {count} prompts")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process some data.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output file")
    parser.add_argument("--dataset_type", type=str, default="HE", help="Type of the dataset")
    parser.add_argument("--timeout", type=int, default=30, help="Timeout for each execution")
    parser.add_argument("--add_prompt", type=bool, default=True, help="Whether to add prompt")
    parser.add_argument("--timeouts_list", type=bool, default=False, help="Whether to use timeouts list")

    args = parser.parse_args()

    input_file = args.input_file
    output_file = args.output_file
    dataset_type = args.dataset_type
    timeout = args.timeout
    add_prompt = args.add_prompt
    timeouts_list = args.timeouts_list

    print("Processing data...")
    with open(input_file, 'r') as f:
        lines = f.readlines()

    process_data(lines, output_file, dataset_type, timeout)
