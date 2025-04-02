import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

# NOTE: NOT VERY USEFUL WITHOUT TRAINING FOR SPECIALIZED TASKS
# DIFFICULT DUE TO MEMORY CONSUMPTION REQUIREMENTS

# Convert sigmoid output to binary class
def step(prob):
    return 0 if prob < 0.5 else 1

class DistillBERT:
    def __init__(self):
        # Load custom BERT model from HuggingFace
        model_name = "ayushdh96/Sarcasm_Detection_Distill_Bert_Fine_Tuned"
        self.model = TFAutoModelForSequenceClassification.from_pretrained(model_name, cache_dir="./bert_cache/")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="./bert_cache/")

    def predict(self, texts):
        input = self.tokenizer(texts, truncation=True, padding='max_length', max_length=64, return_tensors="tf")
        input_ids, attention_mask = input["input_ids"], input["attention_mask"]

        logits = self.model({"input_ids": input_ids, "attention_mask": attention_mask}).logits

        # Flatten over softmax output
        probabilities = tf.nn.softmax(logits, axis=-1).numpy()
        probabilities = probabilities[:, 1]

        return [step(p) for p in probabilities]