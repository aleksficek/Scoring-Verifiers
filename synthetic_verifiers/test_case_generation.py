# SPDX-License-Identifier: MIT

import argparse
import json
from openai import OpenAI
import re
import os
from concurrent.futures import ThreadPoolExecutor

instruction_only_format = '''
You are an expert at writing assertion test cases and below is a question with function signature and test cases. 
You must generate 10 assert test cases that will be used to evaluate the code solution's correctness. You must adhere to the provided function signature and test case format.
Here are some examples that you should use as a reference:

Question: 
from typing import Optional

def first_repeated_char(s: str) -> Optional[str]:
    """ 
    Find the first repeated character in a given string.
    
    >>> first_repeated_char("abbac")
    'a'
    """
        
Test Cases:
<assertion>
assert first_repeated_char("ccccc") == "c"
</assertion>
<assertion>
assert first_repeated_char("xvvdf") == "v"
</assertion>
<assertion>
assert first_repeated_char("egeagea") == "e"
</assertion>
<assertion>
assert first_repeated_char("rrrrea") == "r"
</assertion>
<assertion>
assert first_repeated_char("fa") == "None"
</assertion>
<assertion>
assert first_repeated_char("!@#$%^&*!") == "!"
</assertion>
<assertion>
assert first_repeated_char("abcdedcba") == "d"
</assertion>
<assertion>
assert first_repeated_char("") == "None"
</assertion>
<assertion>
assert first_repeated_char("aaaa") == "a"
</assertion>
<assertion>
assert first_repeated_char("a") == "None"
</assertion>

Question: 
def reverse_words(s: str) -> str:
    """ 
    Reverse words in a given string.
    
    >>> reverse_words("hi this is bob.")
    'bob. is this hi'
    """

Test Cases:
<assertion>
assert reverse_words("the") == "the"
</assertion>
<assertion>
assert reverse_words("no way, really?") == "really? way, no"
</assertion>
<assertion>
assert reverse_words("one two three four") == "four three two one"
</assertion>
<assertion>
assert reverse_words("fire away, questions please!!") == "please!! questions away, fire"
</assertion>
<assertion>
assert reverse_words("live, laughter and life.") == "life. and laughter live,"
</assertion>
<assertion>
assert reverse_words("     ") == ""
</assertion>
<assertion>
assert reverse_words("123 456 !@#") == "!@# 456 123"
</assertion>
<assertion>
assert reverse_words("hello
world") == "world hello"
</assertion>
<assertion>
assert reverse_words("  hello   world  ") == "world hello"
</assertion>
<assertion>
assert reverse_words("hello") == "hello"
</assertion>

Here are guidelines for writing the assertion test cases:
1. You must wrap each assertion test case with tags <assertion> and </assertion>.
2. Do not start the assert with any indents or spaces.
3. You must not import any unit testing libraries for the assertions such as "unittest" or "pytest".
4. Each assertion must be complete and immediately executable. Assume the code solution is provided, do not repeat it.
5. Avoid unnecessary string literals, incorrect escaping, wrapping in "```python" or other redundancies.
6. Remember, it is your responsibility to carefully read the question and generate test cases that will evaluate the correctness of the solution.

Here is the question you must provide assertion test cases for:



Question: {input}
Test Cases:
'''


instruction_solution_format = '''
You are an expert at writing assertion test cases and below is a question with function signature and completed code solution. 
You must generate 10 assert statements that will be used to evaluate the code solution's correctness which may or may not be correct.
Here are some examples that you should use as a reference:

Question: 
from typing import Optional

def first_repeated_char(s: str) -> Optional[str]:
    """ 
    Find the first repeated character in a given string.
    
    >>> first_repeated_char("abbac")
    'a'
    """
        
Solution:
from typing import Optional

def first_repeated_char(s: str) -> Optional[str]:
    """ 
    Find the first repeated character in a given string.
    
    >>> first_repeated_char("abbac")
    'a'
    """
    for index, c in enumerate(s):
        if s[:index + 1].count(c) > 1:
            return c
    return None
Test Cases:
<assertion>
assert first_repeated_char("ccccc") == "c"
</assertion>
<assertion>
assert first_repeated_char("xvvdf") == "v"
</assertion>
<assertion>
assert first_repeated_char("egeagea") == "e"
</assertion>
<assertion>
assert first_repeated_char("rrrrea") == "r"
</assertion>
<assertion>
assert first_repeated_char("fa") == "None"
</assertion>
<assertion>
assert first_repeated_char("!@#$%^&*!") == "!"
</assertion>
<assertion>
assert first_repeated_char("abcdedcba") == "d"
</assertion>
<assertion>
assert first_repeated_char("") == "None"
</assertion>
<assertion>
assert first_repeated_char("aaaa") == "a"
</assertion>
<assertion>
assert first_repeated_char("a") == "None"
</assertion>

Question: 
def reverse_words(s: str) -> str:
    """ 
    Reverse words in a given string.
    
    >>> reverse_words("hi this is bob.")
    'bob. is this hi'
    """

Solution:
def reverse_words(s: str) -> str:
    """ 
    Reverse words in a given string.
    
    >>> reverse_words("hi this is bob.")
    'bob. is this hi'
    """
    return ' '.join(reversed(s.split()))
Test Cases:
<assertion>
assert reverse_words("the") == "the"
</assertion>
<assertion>
assert reverse_words("no way, really?") == "really? way, no"
</assertion>
<assertion>
assert reverse_words("one two three four") == "four three two one"
</assertion>
<assertion>
assert reverse_words("fire away, questions please!!") == "please!! questions away, fire"
</assertion>
<assertion>
assert reverse_words("live, laughter and life.") == "life. and laughter live,"
</assertion>
<assertion>
assert reverse_words("     ") == ""
</assertion>
<assertion>
assert reverse_words("123 456 !@#") == "!@# 456 123"
</assertion>
<assertion>
assert reverse_words("hello
world") == "world hello"
</assertion>
<assertion>
assert reverse_words("  hello   world  ") == "world hello"
</assertion>
<assertion>
assert reverse_words("hello") == "hello"
</assertion>

Here are guidelines for writing the assertion test cases:
1. You must wrap each assertion test case with tags <assertion> and </assertion>.
2. Do not start the assert with any indents or spaces.
3. You must not import any unit testing libraries for the assertions such as "unittest" or "pytest".
4. Each assertion must be complete and immediately executable. Assume the code solution is provided, do not repeat it.
5. Avoid unnecessary string literals, incorrect escaping, wrapping in "```python" or other redundancies.
6. Remember, it is your responsibility to carefully read the question and generate test cases that will evaluate the correctness of the solution.

Here is the question and code solution you must provide assertion test cases for:



Question: {input}
Solution:
{code}
Test Cases:
'''

def load_input_file(input_file):
    with open(input_file, 'r') as f:
        return [json.loads(line) for line in f]

def initialize_openai_client(api_key):
    return OpenAI(api_key=api_key)

def extract_code_block(content):
    if "</think>" in content:
        content = content.split("</think>")[-1]
    pattern = r"<assertion>(.*?)</assertion>"
    matches = re.findall(pattern, content, re.DOTALL)
    return matches

def process_prompt(args):
    idx, line, client, prompt_format, model = args
    print("Processing prompt", idx)
    
    if prompt_format == "instruction_only":
        formatted_input = instruction_only_format.format(input=line['instruction'])
    elif prompt_format == "instruction_solution":
        formatted_input = instruction_solution_format.format(input=line['instruction'], code=line['output'])
    else:
        raise ValueError("Invalid prompt format")
    
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": formatted_input}],
        max_completion_tokens=16000,
        temperature=1.0,
        n=1,
        top_p=1.0
    )
    
    content = extract_code_block(response.choices[0].message.content)
    line['unit_test_responses'] = response.choices[0].message.content
    line['unit_tests'] = content
    print(f"Completed processing {idx + 1} prompts")
    return json.dumps(line, ensure_ascii=False) + "\n"

def perform_inference(input_file, client, prompt_format, target_file, model):
    input_lines = load_input_file(input_file)
    args_list = [(i, line, client, prompt_format, model) for i, line in enumerate(input_lines)]
    buffer = []
    
    with open(target_file, "w", encoding="utf-8") as fout:
        # Increase max_workers as appropriate since these calls are I/O-bound.
        with ThreadPoolExecutor(max_workers=10) as executor:
            for i, result in enumerate(executor.map(process_prompt, args_list)):
                buffer.append(result)
                # Write out every 100 results to limit memory usage.
                if (i + 1) % 100 == 0:
                    fout.write("".join(buffer))
                    fout.flush()
                    buffer = []
        if buffer:
            fout.write("".join(buffer))
            fout.flush()
    
    print("Completed processing all prompts, written to", target_file)

def main():
    parser = argparse.ArgumentParser(description="Generate test cases for synthetic code scoring")
    parser.add_argument("--prompt_format", type=str, required=True, help="Prompt format to use, either instruction_only or instruction_solution")
    parser.add_argument("--model", type=str, required=True, help="Model to use for inference")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output file")
    args = parser.parse_args()

    prompt_format = args.prompt_format
    model = args.model
    input_file = args.input_file
    output_file = args.output_file

    client = initialize_openai_client(api_key=os.environ['OPENAI_API_KEY'])
    perform_inference(input_file, client, prompt_format, output_file, model)

if __name__ == "__main__":
    main()