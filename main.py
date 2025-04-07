from collections import Counter
import json
import re
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report, confusion_matrix

from LSTM import BinaryLSTMModel
from GRU import BinaryGRUModel

# Preprocess text (custom behaviour)
def preprocess_text(text, clean=False):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    
    if clean:
        tokens = text.split()
        text = " ".join([token for token in tokens if token not in ENGLISH_STOP_WORDS])

    return text

# For simple visualization of dataset
def print_data_stats(path, title_key='headline', label_key='is_sarcastic'):
    headlines = []
    labels = []

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            entry = json.loads(line)
            headlines.append(entry[title_key])
            labels.append(entry[label_key])

    print("Sarcastic labels: ", labels.count(1))
    print("Non-Sarcastic labels: ", labels.count(0))

    headline_lengths = [len(headline) for headline in headlines]
    headline_lengths.sort()
    print("Average headline lengths: ", sum(headline_lengths) / len(headline_lengths))
    print("Min headline lengths: ", headline_lengths[0])
    print("Max headline lengths: ", headline_lengths[-1])

    cleaned_headlines = [preprocess_text(seq, True) for seq in headlines]
    vocab = [word for seq in cleaned_headlines for word in seq.split()]
    unique_words = set(vocab)

    word_freq = Counter(vocab)
    for word, count in word_freq.most_common(20):
        print(f"{word}: {count}")

    print("Number of unique words in dataset: ", len(unique_words))

# Read data from JSON
def read_data(path, title_key='headline', label_key='is_sarcastic'):
    headlines = []
    labels = []

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            entry = json.loads(line)
            headlines.append(entry[title_key])
            labels.append(entry[label_key])

    return headlines[200:], labels[200:], headlines[:200], labels[:200]

def main():
    model = BinaryLSTMModel(load=False)

    # Prepare data
    X_train, y_train, X_test, y_test = read_data("./Sarcasm_Headlines_Dataset.json")
    X_train2, y_train2, X_test2, y_test2 = read_data("./Sarcasm_Headlines_Dataset_v2.json")
    X_train.extend(X_train2)
    y_train.extend(y_train2)
    X_test.extend(X_test2)
    y_test.extend(y_test2)
    print(f"Starting with {len(X_train)} train data and {len(X_test)} test data...")

    # For control outside of default tokenizer preprocessing
    # X_train = [preprocess_text(text, True) for text in X_train]
    # X_test = [preprocess_text(text, True) for text in X_test]

    # For LSTM prep
    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(X_train)
    sequences = tokenizer.texts_to_sequences(X_train)
    test_sequences = tokenizer.texts_to_sequences(X_test)
    input_sequences = pad_sequences(sequences, maxlen=128, padding="post")
    test_sequences = pad_sequences(test_sequences, maxlen=128, padding="post")

    # New unseen sequences
    sarcastic_sequences = []
    with open("sample_sarcastic_headlines.txt") as f:
        for line in f.readlines():
            if line.startswith("#"): # Ignore the source list
                continue
            else:
                sarcastic_sequences.append(line)
    sarcastic_sequences = tokenizer.texts_to_sequences(sarcastic_sequences)
    sarcastic_sequences = pad_sequences(sarcastic_sequences, maxlen=128, padding="post")
    
    # Prep labels
    labels = np.array(y_train)

    # Train Model
    model.fit(input_sequences, labels)

    # Attempt predictions
    score = 0
    pred_labels = model.predict(test_sequences)

    for i in range(len(pred_labels)):
        if pred_labels[i] == y_test[i]:
            score += 1
    print("Percentage of correct predictions: ", score/len(pred_labels))

    unseen_predictions = model.predict(sarcastic_sequences)
    print("Predictions on unseen data: ", unseen_predictions)

    # Evaluate model
    print(confusion_matrix(y_test, pred_labels))
    print(classification_report(y_test, pred_labels))

# print_data_stats("./Sarcasm_Headlines_Dataset.json")
main()
