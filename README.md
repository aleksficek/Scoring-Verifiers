# Scoring Verifiers: Evaluating Synthetic Verification in Code and Reasoning

Check out our paper for more details: [Scoring Verifiers: Evaluating Synthetic Verification in Code and Reasoning](https://arxiv.org/abs/2502.13820)

## Datasets
- HE-R
- HE-R+
- MBPP-R
- MBPP-R+


## Code

We provide the scripts we used to generate the scoring and ranking augmented benchmarks in our paper:

1. `generate_solutions.py` makes inference requests to OpenAI and executes the solutions to determine their ground truth fraction of predefined tests passed.
2. `combine_solutions.py` aggregates all of the candidates solutions generated in all `exec_{}.jsonl` files for each sample into one file.
3. `filter_solutions.py` filter solutions to generate k evenly spaced solutiosn for each sample.
4. `evaluate.py` evaluates the Top-1, Bottom-1, Spearman's, Kendall's Tau, MAE and R^2 for a target file.


## Evaluation

The file used to evaluate should include both the original ranks at key `rank` and expected test score at `ground_average_test_score` as found in the ranking datasets. Each solution is seperated as its own entry and is grouped based on keys `dataset` and `task_id`.

To compare test case generation set `method` to `utg` and the generated test scores at `average_test_score`.

To compare reward scoring set `method` to `reward` and the generated test scores at `reward`, `reward_score`.


## Requirements

```
pip install -r requirements.txt
```


## Citation
```
@misc{ficek2025scoringverifiersevaluatingsynthetic,
      title={Scoring Verifiers: Evaluating Synthetic Verification in Code and Reasoning}, 
      author={Aleksander Ficek and Somshubra Majumdar and Vahid Noroozi and Boris Ginsburg},
      year={2025},
      eprint={2502.13820},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2502.13820}, 
}
```

## Acknowledgements

- [EvalPlus](https://github.com/evalplus/evalplus/tree/master)
- [HumanEval](https://github.com/openai/human-eval)
- [MBPP](https://github.com/google-research/google-research/tree/master/mbpp)