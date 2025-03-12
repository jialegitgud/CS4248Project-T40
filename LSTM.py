import json
import numpy as np

from keras import Sequential, layers, optimizers, utils
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences

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

# Classify Sigmoid Scores
def step(prob):
    return 0 if prob < 0.5 else 1

# LSTM model for sarcasm detection
class BinaryLSTMModel:
    def __init__(self, vocab_size=10000, output_dim=128):
        nn = Sequential(
            [
                layers.Embedding(input_dim=vocab_size, output_dim=output_dim),
                layers.Bidirectional(layers.LSTM(128, return_sequences=True)),
                layers.Bidirectional(layers.LSTM(32)),
                layers.Dropout(0.4),
                layers.Dense(32, activation='relu'),
                layers.Dropout(0.4),
                layers.Dense(1, activation='sigmoid')
            ]
        )
        nn.compile(optimizer=optimizers.Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy'])

        self.model = nn
        self.epochs = 25
        self.batch_size = 64
    
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
    score = 0
    pred = model.predict(padded_test_sequences)
    for i in range(len(pred)):
        if step(pred[i]) == y_test[i]:
            score += 1
    
    print(score/len(pred))

# main()