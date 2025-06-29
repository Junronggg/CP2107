from vllm import LLM, SamplingParams

model_name = "Qwen/Qwen2.5-0.5B-Instruct"  # You will loop through other models later
llm = LLM(model=model_name)

prompts = [
    "How many positive whole-number divisors does 196 have?",
    "The capital of Singapore is",
    "Who are you?",
    "What is the range of the output of tanh?",
]

# Try different temperatures
temperatures = [0, 0.6, 1, 1.5]

for temp in temperatures:
    print(f"\n\n--- Temperature = {temp} ---")
    sampling_params = SamplingParams(temperature=temp, max_tokens=4096)

    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        print(f"\nPrompt: {output.prompt}\nOutput: {output.outputs[0].text.strip()}")

