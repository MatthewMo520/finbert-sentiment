from datasets import load_dataset
from transformers import AutoTokenizer
from collections import Counter

# 1. Load dataset
dataset = load_dataset("zeroshot/twitter-financial-news-sentiment")
dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)

# 2. Check the data
print(dataset)
print(dataset["train"][0])
print(Counter(dataset["train"]["label"]))

# 3. Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")

# 4. Tokenize
def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

tokenized = dataset.map(tokenize, batched=True)

# 5. Format for PyTorch
tokenized = tokenized.rename_column("label", "labels")
tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

train_dataset = tokenized["train"]
val_dataset = tokenized["test"]

print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")