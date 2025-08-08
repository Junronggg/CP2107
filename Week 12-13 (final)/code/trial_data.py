from datasets import load_dataset  # <-- You were missing this line

dataset = load_dataset("parquet", data_files="GSM-8k/data/train-00000-of-00001.parquet")

print(dataset)
print(dataset["train"][0])

