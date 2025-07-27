import json

def load_local_gsm8k(parquet_path):
    from datasets import load_dataset
    dataset = load_dataset("parquet", data_files=parquet_path)
    return dataset["train"]

def format_prompt(problem, template_path="prompts/prompt_template_gsm8k.txt"):
    with open(template_path, 'r') as f:
        template = f.read()
    return template.replace("{problem}", problem.strip())

def equation_reward_func(predicted, answer):
    predicted = predicted.strip()
    answer = answer.strip()
    return 1.0 if predicted == answer else 0.0

