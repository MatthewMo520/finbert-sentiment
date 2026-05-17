from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load val dataset
dataset = load_dataset("zeroshot/twitter-financial-news-sentiment")
dataset = DatasetDict({"train": dataset["train"], "test": dataset["validation"]})

tokenizer = AutoTokenizer.from_pretrained("./model/finbert-finetuned")
model = AutoModelForSequenceClassification.from_pretrained("./model/finbert-finetuned")
model.eval()

def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

tokenized = dataset["test"].map(tokenize, batched=True)
tokenized = tokenized.rename_column("label", "labels")
tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# 2. Run inference
all_preds = []
all_labels = []

with torch.no_grad():
    for i in range(0, len(tokenized), 16):
        batch = tokenized[i:i+16]
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )
        preds = np.argmax(outputs.logits.numpy(), axis=-1)
        all_preds.extend(preds)
        all_labels.extend(batch["labels"].numpy())

# 3. Print metrics
label_names = ["Bearish", "Neutral", "Bullish"]
print("\n=== Classification Report ===")
print(classification_report(all_labels, all_preds, target_names=label_names))

# 4. Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=label_names,
    yticklabels=label_names
)
plt.title("FinBERT Confusion Matrix")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
print("\nConfusion matrix saved to confusion_matrix.png")