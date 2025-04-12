import json
import pandas as pd
from datasets import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, TrainingArguments, Trainer
import re
import os
import torch

# read the CSV file into a pandas DataFrame
df = pd.read_csv("./data/train_clean_processed.csv")

# Convert to HuggingFace dataset
dataset = Dataset.from_pandas(df)
dataset = dataset.train_test_split(test_size=0.1)

# Load tokenizer and model
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

# Preprocessing function
def preprocess(example):
    model_input = tokenizer(example["input_text"], max_length=64, truncation=True, padding="max_length")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(example["target_text"], max_length=64, truncation=True, padding="max_length")
    model_input["labels"] = labels["input_ids"]
    return model_input

# Apply preprocessing
tokenized_dataset = dataset.map(preprocess, batched=True)

# Training setup
training_args = TrainingArguments(
    output_dir="./t5-sarcasm-remover",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    logging_dir="./logs",
    save_total_limit=2,
    weight_decay=0.01,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained("./t5-sarcasm-remover")
tokenizer.save_pretrained("./t5-sarcasm-remover")
