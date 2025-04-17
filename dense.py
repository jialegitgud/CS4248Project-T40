import numpy as np

from keras import Sequential, layers, optimizers, callbacks, saving

# Classify Sigmoid Scores
def step(prob):
    return 0 if prob < 0.5 else 1

# Dense model for sarcasm detection
class BinaryDenseModel:
    def __init__(self, vocab_size=15000, output_dim=128, embedding_matrix=None, load=False):
        if load:
            self.model = saving.load_model("./model_cp/best_dense_model.keras")
        else:
            self.model = Sequential([
                layers.Embedding(
                    input_dim=vocab_size,
                    output_dim=output_dim,
                    weights=[embedding_matrix],
                    trainable=False
                ) if embedding_matrix is not None else layers.Embedding(
                    input_dim=vocab_size,
                    output_dim=output_dim
                ),
                layers.Flatten(),
                layers.Dense(128, activation='relu'),
                layers.Dropout(0.4),
                layers.Dense(32, activation='relu'),
                layers.Dense(1, activation='sigmoid')
            ])

            self.model.compile(
                optimizer=optimizers.AdamW(learning_rate=1e-4),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )

            self.epochs = 30
            self.batch_size = 128

    # Train
    def fit(self, X_train, X_labels):
        self.model.fit(
            X_train,
            X_labels,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=0.2,
            callbacks=[callbacks.EarlyStopping(patience=2, restore_best_weights=True)]
        )
        # self.model.save("./model_cp/best_dense_model.keras")

    # Predict
    def predict(self, test):
        return [step(p) for p in self.model.predict(test)]
