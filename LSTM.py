import numpy as np

from keras import Sequential, layers, optimizers, callbacks

# Classify Sigmoid Scores
def step(prob):
    return 0 if prob < 0.5 else 1

# LSTM model for sarcasm detection
class BinaryLSTMModel:
    def __init__(self, vocab_size=10000, output_dim=64, embedding_matrix=None):
        nn = Sequential(
            [
                layers.Embedding(input_dim=vocab_size, output_dim=output_dim, weights=[embedding_matrix], trainable=False) if embedding_matrix is not None else layers.Embedding(input_dim=vocab_size, output_dim=output_dim),
                layers.Bidirectional(layers.LSTM(64, recurrent_dropout=0.1, return_sequences=True)),
                layers.Bidirectional(layers.LSTM(32)),
                layers.Dropout(0.2),
                layers.Dense(32, activation='relu'),
                layers.Dense(1, activation='sigmoid')
            ]
        )
        nn.compile(optimizer=optimizers.Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

        self.model = nn
        self.epochs = 15
        self.batch_size = 128
    
    # Train
    def fit(self, X_train, X_labels):
        self.model.fit(X_train, X_labels, batch_size=self.batch_size, epochs=self.epochs, validation_split=0.2, callbacks=[callbacks.EarlyStopping(patience=2, restore_best_weights=True)])

    # Predict
    def predict(self, X_test):
        return [step(p) for p in self.model.predict(X_test)]