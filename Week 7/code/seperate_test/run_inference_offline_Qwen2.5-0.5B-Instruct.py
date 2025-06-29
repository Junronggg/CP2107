from vllm import LLM, SamplingParams

model_name = "/home/junrong/llmInference-vllm/Qwen2.5-0.5B-Instruct"
llm = LLM(
    model="/home/junrong/llmInference-vllm/Qwen2.5-0.5B-Instruct",
    gpu_memory_utilization=0.8  # or 0.7, 0.6
)

prompts = [
    "How many positive whole-number divisors does 196 have?",
    "The capital of Singapore is",
    "Who are you?",
    "What is the range of the output of tanh?"
]

temperatures = [0, 0.6, 1, 1.5]
num_trials = 3

"""
for temp in temperatures:
    print(f"\n--- Temperature: {temp} ---\n")
    sampling_params = SamplingParams(temperature=temp, max_tokens=4096)
    outputs = llm.generate(prompts, sampling_params)
    
    for output in outputs:
        print(f"Prompt: {output.prompt}")
        print(f"Output: {output.outputs[0].text.strip()}\n")i

with open("results.txt", "w") as f:
    for temp in temperatures:
        sampling_params = SamplingParams(temperature=temp, max_tokens=4096)
        outputs = llm.generate(prompts, sampling_params)
        for output in outputs:
            f.write(f"Temperature: {temp}\n")
            f.write(f"Prompt: {output.prompt}\n")
            f.write(f"Output: {output.outputs[0].text.strip()}\n")
            f.write("-" * 60 + "\n")
"""

with open("results.txt", "w", encoding="utf-8") as f:
    f.write(f"Model: {model_name}\n")
    f.write("=" * 60 + "\n")

    for trial in range(1, num_trials + 1):
        f.write(f"\n=== Trial {trial} ===\n")
        for temp in temperatures:
            f.write(f"\n--- Temperature: {temp} ---\n")
            sampling_params = SamplingParams(temperature=temp, max_tokens=4096)
            outputs = llm.generate(prompts, sampling_params)
            for output in outputs:
                f.write(f"\nPrompt: {output.prompt}\n")
                f.write(f"Output: {output.outputs[0].text.strip()}\n")
                f.write("-" * 40 + "\n")
