import json
import re
import numpy as np

from keras import Sequential, layers, optimizers, utils
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report, confusion_matrix

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

# Classify Sigmoid Scores
def step(prob):
    return 0 if prob < 0.5 else 1

# LSTM model for sarcasm detection
class BinaryLSTMModel:
    def __init__(self, vocab_size=10000, output_dim=128, embedding_matrix=None):
        # nn = Sequential(
        #     [
        #         layers.Embedding(input_dim=vocab_size, output_dim=output_dim),
        #         layers.Bidirectional(layers.LSTM(128, return_sequences=True)),
        #         layers.Bidirectional(layers.LSTM(32)),
        #         layers.Dropout(0.4),
        #         layers.Dense(32, activation='relu'),
        #         layers.Dropout(0.4),
        #         layers.Dense(1, activation='sigmoid')
        #     ]
        # )
        nn = Sequential(
            [
                layers.Embedding(input_dim=vocab_size, output_dim=output_dim, weights=[embedding_matrix], trainable=False) if embedding_matrix is not None else layers.Embedding(input_dim=vocab_size, output_dim=output_dim),
                layers.Bidirectional(layers.LSTM(128, return_sequences=True)),
                layers.Bidirectional(layers.LSTM(64)),
                layers.Dropout(0.5),
                layers.Dense(64, activation='relu', kernel_regularizer='l2'),
                layers.Dropout(0.5),
                layers.Dense(1, activation='sigmoid')
            ]
        )
        nn.compile(optimizer=optimizers.Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

        self.model = nn
        self.epochs = 30
        self.batch_size = 32
    
    # Train
    def fit(self, X_train, X_labels):
        self.model.fit(X_train, X_labels, batch_size=self.batch_size, epochs=self.epochs, validation_split=0.2)

    # Predict
    def predict(self, X_test):
        return self.model.predict(X_test)
    
def main():
    model = BinaryLSTMModel()

    # Prepare data
    X_train, y_train, X_test, y_test = read_data("./Sarcasm_Headlines_Dataset.json")
    X_train = [preprocess_text(text) for text in X_train]
    X_test = [preprocess_text(text) for text in X_test]

    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(X_train)
    sequences = tokenizer.texts_to_sequences(X_train)
    test_sequences = tokenizer.texts_to_sequences(X_test)
    padded_sequences = pad_sequences(sequences, maxlen=50, padding="post")
    padded_test_sequences = pad_sequences(test_sequences, maxlen=50, padding="post")
    labels = np.array(y_train)

    # Train Model
    model.fit(padded_sequences, labels)

    # Attempt predictions
    # score = 0
    # pred = model.predict(padded_test_sequences)
    # for i in range(len(pred)):
    #     if step(pred[i]) == y_test[i]:
    #         score += 1
    
    # print(score/len(pred))

    # Attempt predictions
    score = 0
    pred = model.predict(padded_test_sequences)
    pred_labels = [step(p) for p in pred]

    for i in range(len(pred_labels)):
        if pred_labels[i] == y_test[i]:
            score += 1
    print("Percentage of correct predictions: ", score/len(pred_labels))

    # Evaluate model
    print(confusion_matrix(y_test, pred_labels))
    print(classification_report(y_test, pred_labels))

main()
