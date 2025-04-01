# SPDX-License-Identifier: MIT

import json
from evalplus.evaluate import evaluate


input_file = ''
output_file = ''
is_mbpp = True

base_number_correct = 0
plus_number_correct = 0

total_number = 0.0

output = open(output_file, 'w')

with open(input_file, 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = json.loads(line.strip())

        if line['base_execution_result']['average_test_score'] == 1.0:
            base_number_correct += 1.0

        try:
            if line['plus_execution_result']['average_test_score'] == 1.0:
                plus_number_correct += 1.0
        except Exception:
            pass

        total_number += 1.0

        try:
            line['solution'] = line['canonical_solution']
        except Exception:
            line['solution'] = line['code']

        if is_mbpp:
            line['task_id'] = "Mbpp/" + str(line['task_id'])

        output.write(json.dumps(line) + '\n')


print(f"Base number correct: {float(base_number_correct) / total_number}")
print(f"Plus number correct: {float(plus_number_correct) / total_number}")


print(evaluate(
    dataset="mbpp",
    samples=output_file,
    min_time_limit=0.1,
))