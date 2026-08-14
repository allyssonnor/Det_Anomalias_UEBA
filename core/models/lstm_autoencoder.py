import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from core.models.base_model import BaseModel

class LSTMAutoencoder(BaseModel):
    """
    Autoencoder que usa redes recorrentes LSTM.
    HERANÇA E CONTRATO DEFINITIVOS (Compatível com GPUs genéricas/Apple Silicon).
    """
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        model_cfg = config.get("model", {}).get("lstm_autoencoder", {})
        train_cfg = config.get("training", {})

        self.latent_dim = model_cfg.get("latent_dim", 32)
        self.lstm_units = model_cfg.get("lstm_units", [64, 32])
        self.dropout = model_cfg.get("dropout", 0.1)

        self.epochs = train_cfg.get("epochs", 20)
        self.batch_size = train_cfg.get("batch_size", 256)
        self.learning_rate = train_cfg.get("learning_rate", 1e-3)
        self.validation_split = train_cfg.get("validation_split", 0.1)
        self.patience = train_cfg.get("early_stopping_patience", 5)
        self.seed = train_cfg.get("seed", 42)

        self.model = None
        self.encoder = None
        self.timesteps = None
        self.n_features = None

    def _set_seed(self):
        os.environ["PYTHONHASHSEED"] = str(self.seed)
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)

        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass

    def _build(self):
        inputs = keras.Input(shape=(self.timesteps, self.n_features))
        x = inputs

        # ENCODER
        for units in self.lstm_units[:-1]:
            # Substituído layers.LSTM por layers.RNN(layers.LSTMCell) para compatibilidade universal de GPU
            x = layers.RNN(layers.LSTMCell(units), return_sequences=True)(x)
            if self.dropout > 0:
                x = layers.Dropout(self.dropout)(x)

        # Última camada do Encoder
        # Substituído layers.LSTM por layers.RNN(layers.LSTMCell) para compatibilidade universal de GPU
        x = layers.RNN(layers.LSTMCell(self.lstm_units[-1]), return_sequences=False)(x)
        latent = layers.Dense(self.latent_dim, activation="linear", name="latent")(x)

        # REPETIDOR DE VETOR
        x = layers.RepeatVector(self.timesteps)(latent)

        # DECODER
        for units in reversed(self.lstm_units):
            # Substituído layers.LSTM por layers.RNN(layers.LSTMCell) para compatibilidade universal de GPU
            x = layers.RNN(layers.LSTMCell(units), return_sequences=True)(x)
            if self.dropout > 0:
                x = layers.Dropout(self.dropout)(x)

        outputs = layers.TimeDistributed(
            layers.Dense(self.n_features, activation="linear")
        )(x)

        self.model = keras.Model(inputs, outputs)
        self.encoder = keras.Model(inputs, latent)

        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)

        self.model.compile(
            optimizer=optimizer,
            loss="mse"
        )

    def fit(self, X):
        self._set_seed()
        X = np.asarray(X).astype(np.float32)

        if X.ndim != 3:
            raise ValueError(f"Expected 3D input, got {X.shape}")

        if len(X) == 0:
            raise ValueError("Empty training data")

        self.timesteps = X.shape[1]
        self.n_features = X.shape[2]

        self._build()

        callbacks = []

        if self.patience > 0:
            callbacks.append(
                keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=self.patience,
                    restore_best_weights=True
                )
            )

        self.model.fit(
            X, X,
            epochs=self.epochs,
            batch_size=self.batch_size,
            shuffle=True,
            validation_split=self.validation_split,
            verbose=0,
            callbacks=callbacks
        )

    def encode(self, X):
        X = np.asarray(X).astype(np.float32)
        return self.encoder.predict(
            X,
            batch_size=self.batch_size,
            verbose=0
        )

    def score(self, X):
        """
        Padronização Absoluta: Retorna vetor 1D com direção de anomalia normalizada.
        """
        X = np.asarray(X).astype(np.float32)

        recon = self.model.predict(
            X,
            batch_size=self.batch_size,
            verbose=0
        )

        errors = (X - recon) ** 2
        scores = np.mean(errors, axis=(1, 2))
        
        return self._ensure_higher_is_anomaly(scores)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        self.model.save(os.path.join(path, "model.keras"))
        self.encoder.save(os.path.join(path, "encoder.keras"))

    def load(self, path):
        self.model = keras.models.load_model(
            os.path.join(path, "model.keras"),
            compile=False
        )
        self.encoder = keras.models.load_model(
            os.path.join(path, "encoder.keras"),
            compile=False
        )
