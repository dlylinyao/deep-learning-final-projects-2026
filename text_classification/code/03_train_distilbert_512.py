import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset

# 1. Load data and downsample
print("Loading cleaned data...")
df = pd.read_pickle("reuters_clean_all.pkl")


df = df.sample(n=50000, random_state=42).reset_index(drop=True)
print(f"Sampling done, current data size: {len(df)}")

# Prepare features and labels
texts = df["text"].tolist()

labels = df["labels_vec"].apply(lambda x: [float(i) for i in x]).tolist()

# Split train and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)


# 2. Hugging Face dataset and tokenizer

print("Loading DistilBERT tokenizer...")
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Build Dataset objects
train_dataset = Dataset.from_dict({"text": train_texts, "labels": train_labels})
val_dataset = Dataset.from_dict({"text": val_texts, "labels": val_labels})


def tokenize_function(examples):

    return tokenizer(
        examples["text"], padding="max_length", truncation=True, max_length=512  # 256
    )


print("Tokenizing text...")
tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_val = val_dataset.map(tokenize_function, batched=True)


# 3. Define metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # Convert logits to probabilities with sigmoid
    probs = 1 / (1 + np.exp(-logits))
    # Threshold = 0.5
    predictions = (probs > 0.5).astype(int)

    micro_f1 = f1_score(labels, predictions, average="micro", zero_division=0)
    macro_f1 = f1_score(labels, predictions, average="macro", zero_division=0)

    return {"micro_f1": micro_f1, "macro_f1": macro_f1}


# 4. Load model and training arguments
print("Loading DistilBERT to GPU...")
num_labels = len(labels[0])

model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=num_labels, problem_type="multi_label_classification"
)

training_args = TrainingArguments(
    output_dir="./results_512",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    fp16=True,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)


# 5. Start fine-tuning
trainer.train()

print("Training finished! Model saved.")
