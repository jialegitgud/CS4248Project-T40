import os
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
)
from peft import LoraConfig, get_peft_model
import evaluate
from tqdm import tqdm

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
MODEL_NAME = "google/flan-t5-base"
CSV_PATH   = "train_clean_processed.csv"
HUF_CSV    = "huff_titles.csv"   # optional
MAX_SOURCE_LEN = 64
MAX_TARGET_LEN = 64
OUTPUT_DIR = "./flan_t5_onion2serious_lora"
BATCH_SIZE = 4
GRAD_ACC   = 4
EPOCHS     = 6
LR         = 5e-5

# ------------------------------------------------------------
# 1) LOAD DATA
# ------------------------------------------------------------
print("Loading dataset …")
raw_df = pd.read_csv(CSV_PATH)
assert {"input_text", "target_text"}.issubset(raw_df.columns)

# ------------------------------------------------------------
# 2) OPTIONAL RETRIEVAL AUGMENTATION
# ------------------------------------------------------------
if os.path.exists(HUF_CSV):
    from sentence_transformers import SentenceTransformer
    from sklearn.neighbors import NearestNeighbors
    huff_titles = pd.read_csv(HUF_CSV)["title"].dropna().tolist()
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    huff_embs = embedder.encode(huff_titles, batch_size=128, show_progress_bar=True, normalize_embeddings=True)
    nn = NearestNeighbors(n_neighbors=1, metric="cosine").fit(huff_embs)

    def add_ctx(h):
        idx = nn.kneighbors(embedder.encode([h], normalize_embeddings=True), return_distance=False)[0][0]
        return f"{h} [CONTEXT] {huff_titles[idx]}"

    tqdm.pandas(desc="Retrieving")
    raw_df["src"] = raw_df["input_text"].progress_apply(add_ctx)
else:
    raw_df["src"] = raw_df["input_text"]
raw_df["tgt"] = raw_df["target_text"]

# ------------------------------------------------------------
# 3) DATASET & TOKENISER
# ------------------------------------------------------------
dataset = Dataset.from_pandas(raw_df[["src", "tgt"]])
dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_ds, val_ds = dataset["train"], dataset["test"]

tok = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess(batch):
    # encode source
    model_inputs = tok(batch["src"], max_length=MAX_SOURCE_LEN, padding="max_length", truncation=True)
    # encode target using `text_target` (new API, no deprecation warning)
    labels = tok(text_target=batch["tgt"], max_length=MAX_TARGET_LEN,
                 padding="max_length", truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

train_tok = train_ds.map(preprocess, batched=True, remove_columns=train_ds.column_names)
val_tok   = val_ds.map(preprocess,   batched=True, remove_columns=val_ds.column_names)

# ------------------------------------------------------------
# 4) LOAD MODEL WITH 8‑BIT QUANTISATION + LoRA
# ------------------------------------------------------------
print("Loading Flan‑T5‑base with 8‑bit quantisation …")
bnb_cfg = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)
base = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, quantization_config=bnb_cfg, device_map="auto")

lora_cfg = LoraConfig(r=8, lora_alpha=32, lora_dropout=0.05, target_modules=["q", "v"], task_type="SEQ_2_SEQ_LM")
model = get_peft_model(base, lora_cfg)
model.print_trainable_parameters()
model.config.use_cache = False

# ------------------------------------------------------------
# 5) TRAINER SETUP
# ------------------------------------------------------------
rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore", model_type="bert-base-uncased")  # use lightweight model to avoid roberta-large weights


def compute_metrics(p):
    preds, labels = p
    preds = preds[0] if isinstance(preds, tuple) else preds
    labels = np.where(labels != -100, labels, tok.pad_token_id)
    preds = np.where(preds < 0, tok.pad_token_id, preds)  # clip negative ids
    dec_preds = tok.batch_decode(preds, skip_special_tokens=True)
    dec_labels = tok.batch_decode(labels, skip_special_tokens=True)
    dec_preds = [s.strip() for s in dec_preds]
    dec_labels = [s.strip() for s in dec_labels]
    r = rouge.compute(predictions=dec_preds, references=dec_labels, use_stemmer=True)
    b = bertscore.compute(predictions=dec_preds, references=dec_labels, lang="en")
    return {"rougeL": r["rougeL"], "bertscore_f1": float(np.mean(b["f1"]))}

collator = DataCollatorForSeq2Seq(tok, model=model, padding="longest")

args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACC,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    fp16=False,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    logging_dir="./logs",
    predict_with_generate=True,
    load_best_model_at_end=True,
    metric_for_best_model="bertscore_f1",
    greater_is_better=True,
    max_grad_norm=0.5
)

trainer = Seq2SeqTrainer(    model=model,
    args=args,
    train_dataset=train_tok,
    eval_dataset=val_tok,
    data_collator=collator,
    processing_class=tok,
    compute_metrics=compute_metrics
)

# ------------------------------------------------------------
# 6) TRAIN
# ------------------------------------------------------------
print("Starting training …")
trainer.train()

# ------------------------------------------------------------
# 7) SAVE ADAPTER
# ------------------------------------------------------------
model.save_pretrained(OUTPUT_DIR)
tok.save_pretrained(OUTPUT_DIR)
print("Model saved to", OUTPUT_DIR)
