# SPDX-License-Identifier: MIT

import argparse
from openai import OpenAI
import json
import os

client = OpenAI(
    base_url = "https://integrate.api.nvidia.com/v1",
    api_key = os.environ['NVIDIA_API_KEY'],
    
)

def reward_request(prompt, solution):

    user_prompt = f"""You are given a coding problem for which you need to generate/complete a solution that is as accurate as possible.

Please complete the function with the Python programming language. 

This is the problem you must solve:
{prompt}
"""

    assistant_prompt = f"""Here is the solution to the given problem:

{solution}
"""   

    completion = client.chat.completions.create(
        model="nvdev/nvidia/llama-3.1-nemotron-70b-reward",
        messages=[{"role": "user", "content": user_prompt}, {"role": "assistant", "content": assistant_prompt}],
    )
    return completion.choices[0].message[0].content

def main():

    parser = argparse.ArgumentParser(description="Generate reward scores for synthetic code.")
    parser.add_argument("--input_path", type=str, help="Path to the input JSONL file.")
    parser.add_argument("--output_path", type=str, help="Path to the output JSONL file.")
    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path


    with open(input_path, "r", encoding="utf-8") as fin, \
            open(output_path, "w", encoding="utf-8") as fout:
        
        for i, line in enumerate(fin):
            record = json.loads(line)
            solutions = record["all_solutions"]
            
            for sol in solutions:
                response = reward_request(sol["solution"]["prompt"], sol["solution"]["canonical_solution"])
                
                sol["reward"] = {
                    "reward_score": float(response.split(':')[1].strip()),
                    "response": response,
                }
            response = reward_request(record["instruction"], record["output"])

            record["reward"] = {
                "reward_score": response,
                "reward_score":  float(response.split(':')[1].strip()),
            }
            
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            print("Completed processing", i + 1, "lines")


if __name__ == "__main__":
    main()