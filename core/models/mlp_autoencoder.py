
##### ✅✅✅✅✅ core/models/base_model.py
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from core.models.base_model import BaseModel

class MLPAutoencoder(BaseModel):
    """
    Autoencoder MLP otimizado. 
    Inclui BatchNormalization e Gradient Clipping.
    Herda de BaseModel e utiliza a interface padronizada de score.
    """
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        model_cfg = config.get("model", {}).get("mlp_autoencoder", {})
        train_cfg = config.get("training", {})

        self.latent_dim = model_cfg.get("latent_dim", 16)
        self.dense_units = model_cfg.get("dense_units", [128, 64])
        self.dropout = model_cfg.get("dropout", 0.1)

        self.epochs = train_cfg.get("epochs", 20)
        self.batch_size = train_cfg.get("batch_size", 256)
        self.learning_rate = train_cfg.get("learning_rate", 1e-3)
        self.validation_split = train_cfg.get("validation_split", 0.1)
        self.patience = train_cfg.get("early_stopping_patience", 5)
        self.seed = train_cfg.get("seed", 42)

        self.model = None
        self.encoder = None

    def _set_seed(self):
        os.environ["PYTHONHASHSEED"] = str(self.seed)
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)

    def _build(self):
        inputs = keras.Input(shape=(self.timesteps, self.n_features))
        
        # Achata a janela temporal para passar no MLP
        x = layers.Flatten()(inputs)

        # Encoder
        for units in self.dense_units:
            x = layers.Dense(units, activation="relu")(x)
            # INTELIGÊNCIA ADAPTADA: BatchNormalization estabiliza os pesos
            x = layers.BatchNormalization()(x) 
            if self.dropout > 0:
                x = layers.Dropout(self.dropout)(x)

        latent = layers.Dense(self.latent_dim, activation="linear", name="latent")(x)

        # Decoder
        x = latent
        for units in reversed(self.dense_units):
            x = layers.Dense(units, activation="relu")(x)
            # INTELIGÊNCIA ADAPTADA: BatchNormalization no decodificador
            x = layers.BatchNormalization()(x)
            if self.dropout > 0:
                x = layers.Dropout(self.dropout)(x)

        # Reconstrução
        flat_output_dim = self.timesteps * self.n_features
        x = layers.Dense(flat_output_dim, activation="linear")(x)
        outputs = layers.Reshape((self.timesteps, self.n_features))(x)

        self.model = keras.Model(inputs, outputs)
        self.encoder = keras.Model(inputs, latent)

        # INTELIGÊNCIA ADAPTADA: clipnorm=1.0 para evitar explosão de gradiente
        self.model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=self.learning_rate, 
                clipnorm=1.0
            ),
            loss="mse"
        )

    def fit(self, X):
        self._set_seed()
        X = np.asarray(X).astype(np.float32)
        
        self.timesteps = X.shape[1]
        self.n_features = X.shape[2]

        self._build()

        callbacks = []
        if self.patience > 0:
            callbacks.append(
                keras.callbacks.EarlyStopping(monitor="val_loss", patience=self.patience, restore_best_weights=True)
            )

        # Ajuste: verbose=1 para exibir a barra de progresso no terminal
        self.model.fit(
            X, X,
            epochs=self.epochs,
            batch_size=self.batch_size,
            shuffle=True,
            validation_split=self.validation_split,
            verbose=1,
            callbacks=callbacks
        )

    # =========================================================
    # SCORE (Interface Padronizada)
    # =========================================================
    def score(self, X):
        """
        Substitui o antigo predict() incorreto. Calcula a perda (MSE) 
        entre a entrada e a reconstrução e devolve como anomaly score 1D.
        """
        X = np.asarray(X).astype(np.float32)
        recon = self.model.predict(X, batch_size=self.batch_size, verbose=0)
        errors = (X - recon) ** 2
        scores = np.mean(errors, axis=(1, 2))
        return self._ensure_higher_is_anomaly(scores)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        # Atualizado para formato nativo do Keras para evitar warnings
        self.model.save(os.path.join(path, "model.keras"))
        self.encoder.save(os.path.join(path, "encoder.keras"))

    def load(self, path):
        self.model = keras.models.load_model(os.path.join(path, "model.keras"), compile=False)
        self.encoder = keras.models.load_model(os.path.join(path, "encoder.keras"), compile=False)

