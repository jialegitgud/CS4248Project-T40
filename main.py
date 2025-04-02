import json
import re
import numpy as np

from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report, confusion_matrix

from LSTM import BinaryLSTMModel
from GRU import BinaryGRUModel

# Read data from JSON
def read_data(path, title_key='headline', label_key='is_sarcastic'):
    headlines = []
    labels = []

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            entry = json.loads(line)
            headlines.append(entry[title_key])
            labels.append(entry[label_key])

    return headlines[100:], labels[100:], headlines[:100], labels[:100]

# Preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)

    return text

def main():
    model = BinaryGRUModel()

    # Prepare data
    X_train, y_train, X_test, y_test = read_data("./Sarcasm_Headlines_Dataset.json")
    X_train = [preprocess_text(text) for text in X_train]
    X_test = [preprocess_text(text) for text in X_test]

    # For LSTM prep
    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(X_train)
    sequences = tokenizer.texts_to_sequences(X_train)
    test_sequences = tokenizer.texts_to_sequences(X_test)
    input_sequences = pad_sequences(sequences, maxlen=50, padding="post")
    test_sequences = pad_sequences(test_sequences, maxlen=50, padding="post")
    
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

    # Evaluate model
    print(confusion_matrix(y_test, pred_labels))
    print(classification_report(y_test, pred_labels))

main()
