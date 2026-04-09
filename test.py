from datasets import load_dataset

code = load_dataset("nvidia/OpenCodeInstruct", split="train", streaming=True)
print(next(iter(code)))