from vllm import LLM, SamplingParams

# List of Qwen models to evaluate
model_list = [
    "/home/junrong/llmInference-vllm/Qwen2.5-0.5B",
    "/home/junrong/llmInference-vllm/Qwen2.5-0.5B-Instruct",
    "/home/junrong/llmInference-vllm/Qwen3-0.6B-Base",
    "/home/junrong/llmInference-vllm/Qwen3-0.6B",
]

# Prompts to test
prompts = [
    "How many positive whole-number idivisors does 196 have?",
    "The capital of Singapore is",
    "Who are you?",
    "What is the range of the output of tanh?",
]

# Temperatures to compare
temperatures = [0, 0.6, 1, 1.5]

# Number of trials
num_trials = 3

# Open the output file
with open("results_all.txt", "w", encoding="utf-8") as f:
    for model_name in model_list:
        f.write(f"\n\n{'='*80}\nMODEL: {model_name}\n{'='*80}\n")
        
        # Load the model
        llm = LLM(model=model_name, gpu_memory_utilization=0.7, max_model_len=2048)

        for trial in range(1, num_trials + 1):
            f.write(f"\n--- Trial {trial} ---\n")
            
            for temp in temperatures:
                f.write(f"\n>>> Temperature: {temp}\n")
                sampling_params = SamplingParams(temperature=temp, max_tokens=4096)
                outputs = llm.generate(prompts, sampling_params)

                for output in outputs:
                    f.write(f"\nPrompt: {output.prompt.strip()}\n")
                    f.write(f"Output: {output.outputs[0].text.strip()}\n")
                    f.write("-" * 40 + "\n")

