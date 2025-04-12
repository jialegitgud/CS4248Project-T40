import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

# --------- Configuration ---------
MODEL_NAME = "google/flan-t5-base"
OUTPUT_DIR = "./flan_t5_onion2serious_lora"
TEST_CSV = "test_processed.csv"
HUF_CSV = "huff_titles.csv"
OUTPUT_CSV = "test_with_generated.csv"
MAX_NEW_TOKENS = 32

# --------- Load LoRA model ---------
print("Loading LoRA model and tokenizer…")
base = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, device_map="auto")
model = PeftModel.from_pretrained(base, OUTPUT_DIR)
tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
model.eval()

# --------- Load retriever ---------
print("Loading HuffPost titles and building retriever…")
huff_titles = pd.read_csv(HUF_CSV)["title"].dropna().tolist()
embedder = SentenceTransformer("all-MiniLM-L6-v2")
huff_embs = embedder.encode(huff_titles, batch_size=128, show_progress_bar=True, normalize_embeddings=True)
retriever = NearestNeighbors(n_neighbors=1, metric="cosine").fit(huff_embs)

# --------- Define prediction function ---------
def predict_with_context(text):
    emb = embedder.encode([text], normalize_embeddings=True)
    idx = retriever.kneighbors(emb, return_distance=False)[0][0]
    context = huff_titles[idx]
    full_input = f"{text} [CONTEXT] {context}"
    input_ids = tokenizer(full_input, return_tensors="pt", truncation=True).to(model.device)
    with torch.no_grad():
        output = model.generate(**input_ids, max_new_tokens=MAX_NEW_TOKENS)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# --------- Run prediction on CSV ---------
print(f"Reading test data from {TEST_CSV}…")
df = pd.read_csv(TEST_CSV)
assert "input_text" in df.columns

print("Generating predictions…")
tqdm.pandas(desc="Generating")
df["generated"] = df["input_text"].progress_apply(predict_with_context)

print(f"Saving predictions to {OUTPUT_CSV}…")
# Save generated column separately
only_gen_path = OUTPUT_CSV.replace(".csv", "_only_generated.txt")
df["generated"].to_csv(only_gen_path, index=False, header=False)
print(f"✓ Also saved only generated headlines to {only_gen_path}")
df.to_csv(OUTPUT_CSV, index=False)
print("✓ Done!")
