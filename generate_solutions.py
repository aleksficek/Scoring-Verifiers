# SPDX-License-Identifier: MIT

import json
import os
from openai import OpenAI
from utils.unit_test_executor import process_data

prompt = """
Using only Python code, write a somewhat incorrect solution to the given coding problem. Do not provide any hints as to what is the mistake. Here are other guideliens for completing this task:

1. Enclose the code in a python code block ```python. 
2. Do not include any unit tests in your answer, only generate the function. 
3. The code must still compile, the only errors in the code should be logical and should only fail via assertion error. 
4. Include any necessary imports with your code, only import libraries included in the standard Python library.
5. Do not add any hints as to the error you made.

Here are some suggestions:
- do not handle negative numbers
- do not handle duplicate values
- introduce rounding errors
- ignore the last element in a list
- only handle specific values, like even or odd numbers
- only works for certain ranges of values or lengths

Question:
{instruction}

Answer:
"""

def load_input_file(input_file):
    with open(input_file, 'r') as f:
        return [json.loads(line) for line in f]

def initialize_openai_client(api_key):
    return OpenAI(api_key=api_key)

def format_prompt(instruction, initial_prompt):
    return initial_prompt.format(instruction=instruction)

def perform_inference(input_lines, client, initial_prompt, dataset_type, limit, seed, temperature):
    outputs = []
    for count, line in enumerate(input_lines):
        if count % 1 == 0:
            print("Processing prompt", count)

        instruction = line['text'] if dataset_type == "MBPP" else line['prompt']
        formatted_input = format_prompt(instruction, initial_prompt)

        response = client.chat.completions.create(
            model="gpt-4o-2024-11-20",
            messages=[{"role": "user", "content": formatted_input}],
            max_tokens=1048,
            temperature=1.0,
            n=1,
            top_p=1.0,
            seed=seed
        )

        content = extract_code_block(response.choices[0].message.content)
        outputs.append({"canonical_solution": content})

        if count >= limit:
            break

    print(f"Completed processing {count + 1} prompts")
    return outputs

def extract_code_block(content):
    if '```' in content:
        first_index = content.find('```')
        second_index = content.find('```', first_index + 1)
        content = content[first_index + 3:second_index].strip()

        if content.startswith('python'):
            content = content[len('python'):].strip()
    return content

def merge_with_ground_truth(input_file, outputs, dataset_type, limit):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    updated_lines = []
    for i, line in enumerate(lines):
        data = json.loads(line.strip())

        if dataset_type == "MBPP":
            data['code'] = outputs[i]['canonical_solution']
        else:
            data['canonical_solution'] = outputs[i]['canonical_solution']

        updated_lines.append(json.dumps(data))

        if i >= limit:
            break

    return updated_lines

def write_output_file(output_file, updated_lines):
    with open(output_file, 'w') as f:
        for line in updated_lines:
            f.write(line + '\n')

def execute_code(updated_lines, output_file_exec, dataset_type, timeout, timeouts_list=False):

    process_data(updated_lines, output_file_exec, dataset_type=dataset_type, add_prompt=False, timeout=timeout, timeouts_list=timeouts_list)

    with open(output_file_exec, 'r') as f:
        lines_predicted = f.readlines()

    if dataset_type == "MBPP":
        rewrite_mbpp_results(output_file_exec, lines_predicted)
    else:
        rewrite_other_results(output_file_exec, lines_predicted, updated_lines)

def rewrite_mbpp_results(output_file_exec, lines_predicted):
    with open(output_file_exec, 'w') as f:
        for line in lines_predicted:
            line = json.loads(line.strip())
            line['base_execution_result']['average_test_score'] = sum(line['base_execution_result']['correct_tests']) / len(line['base_execution_result']['correct_tests'])
            del line['plus_execution_result']
            f.write(json.dumps(line) + '\n')

def rewrite_other_results(output_file_exec, lines_predicted, updated_lines):
    with open(output_file_exec, 'w') as f:
        count = 0
        for line_predicted, line_expected in zip(lines_predicted, updated_lines):
            data_predicted = json.loads(line_predicted.strip())
            data_expected = line_expected
            count += 1


            for type in ['base_execution_result', 'plus_execution_result']:
                data_predicted[type]['correct_tests'] = []
                for test_predicted, test_expected in zip(data_predicted[type]['unit_test_stdouts'], data_expected[type]['unit_test_stdouts']):
                    test_predicted, test_expected = process_value(test_predicted), process_value(test_expected)

                    if data_predicted['atol'] == 0:
                        try:
                            is_correct = test_predicted == test_expected
                        except Exception:
                            is_correct = False
                    else:
                        try:
                            is_correct = abs(float(test_predicted) - float(test_expected)) <= data_predicted['atol']
                        except Exception:
                            is_correct = False

                    data_predicted[type]['correct_tests'].append(is_correct)

                if len(data_predicted[type]['correct_tests']) != 0:
                    data_predicted[type]['average_test_score'] = sum(data_predicted[type]['correct_tests']) / len(data_predicted[type]['correct_tests'])

            data_predicted['average_time_taken'] = sum(data_predicted['base_execution_result']['time_taken'] + data_predicted['plus_execution_result']['time_taken']) / len(data_predicted['base_execution_result']['time_taken'] + data_predicted['plus_execution_result']['time_taken'])

            f.write(json.dumps(data_predicted) + '\n')

def process_value(value):
    value = value.strip()
    if value.isdigit():
        return int(value)
    try:
        return float(value)
    except ValueError:
        return value

def main():

    dataset_type = ""
    input_file = ""
    output_file = ""

    seed = i
    limit = 2000
    temperature = 1.0
    timeout = 1
    timeouts_list = False
    mode = "both" # infer, exec, both

    output_file_infer = output_file.replace(f"{seed}.jsonl", f"infer_{seed}.jsonl")
    output_file_exec = output_file.replace(f"{seed}.jsonl", f"exec_{seed}.jsonl")

    input_lines = load_input_file(input_file)
    client = initialize_openai_client(api_key=os.environ['OPENAI_API_KEY'])

    print(f"Mode selected: {mode} out of ['infer', 'exec', 'both']")
    print(f"Dataset type: {dataset_type}")    

    if mode in ["infer", "both"]:
        outputs = perform_inference(input_lines, client, prompt, dataset_type, limit, seed, temperature)
        updated_lines = merge_with_ground_truth(input_file, outputs, dataset_type, limit)
        write_output_file(output_file_infer, updated_lines)

    if mode in ["exec", "both"]:
        updated_lines = load_input_file(output_file_infer)
        execute_code(updated_lines, output_file_exec, dataset_type, timeout, timeouts_list)

if __name__ == "__main__":
    main()
