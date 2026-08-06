##### ✅✅✅✅✅ core/models/autoencoder_model.py
#NOVO MODELO DO AUTOENCODER
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from core.models.base_model import BaseModel


class AutoencoderModel(BaseModel):
    """
    Autoencoder padrão baseado em MLP (Keras/TensorFlow).
    Adaptado para aceitar tanto tensores 3D de sequências temporais (achatando-os)
    quanto tensores 2D convencionais.
    """

    def __init__(self, config):
        super().__init__(config)

        model_cfg = config.get("model", {}).get("autoencoder", {})
        train_cfg = config.get("training", {})

        self.layers_config = model_cfg.get("layers", [64, 32, 16])
        self.activation = model_cfg.get("activation", "relu")
        self.dropout = model_cfg.get("dropout", 0.0)

        self.epochs = train_cfg.get("epochs", 20)
        self.batch_size = train_cfg.get("batch_size", 256)
        self.learning_rate = train_cfg.get("learning_rate", 1e-3)
        self.validation_split = train_cfg.get("validation_split", 0.1)
        self.patience = train_cfg.get("early_stopping_patience", 5)

        self.seed = train_cfg.get("seed", 42)
        self.feature_weights = train_cfg.get("feature_weights", None)

        self.model = None
        self.input_dim = None

    # =====================================================
    # REPRODUTIBILIDADE
    # =====================================================

    def _set_seed(self, seed):
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        tf.keras.utils.set_random_seed(seed)
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass

    # =====================================================
    # CONSTRUÇÃO DO MODELO
    # =====================================================

    def _build_model(self, input_dim):
        if len(self.layers_config) < 2:
            raise ValueError(
                "O Autoencoder requer pelo menos duas camadas (encoder + bottleneck)"
            )

        inputs = keras.Input(shape=(input_dim,))
        x = inputs

        # Encoder
        for units in self.layers_config:
            x = layers.Dense(units, activation=self.activation)(x)
            if self.dropout > 0:
                x = layers.Dropout(self.dropout)(x)

        # Decoder
        for units in reversed(self.layers_config[:-1]):
            x = layers.Dense(units, activation=self.activation)(x)

        outputs = layers.Dense(input_dim, activation="linear")(x)

        model = keras.Model(inputs, outputs)
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss="mse")

        return model

    # =====================================================
    # TREINAMENTO
    # =====================================================

    def fit(self, X):
        self._set_seed(self.seed)
        X = np.asarray(X).astype(np.float32)

        # Trata entrada 3D do TemporalBuilder (n_samples, sequence_length, n_features)
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)

        self.input_dim = X.shape[1]
        self.model = self._build_model(self.input_dim)

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
            X,
            X,
            epochs=self.epochs,
            batch_size=self.batch_size,
            shuffle=True,
            validation_split=self.validation_split,
            verbose=0,
            callbacks=callbacks
        )

    # =====================================================
    # CÁLCULO DE SCORE DE ANOMALIA
    # =====================================================

    def score(self, X):
        X = np.asarray(X).astype(np.float32)

        # Trata entrada 3D do TemporalBuilder
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)

        recon = self.model.predict(
            X,
            batch_size=self.batch_size,
            verbose=0
        )

        errors = (X - recon) ** 2

        if self.feature_weights is not None:
            weights = np.asarray(self.feature_weights)
            if weights.shape[0] != errors.shape[1]:
                raise ValueError(
                    "O tamanho de feature_weights deve bater com o número de features"
                )
            errors = errors * weights

        scores = np.mean(errors, axis=1)

        # Contrato: valores maiores significam maior anomalia
        return self._ensure_higher_is_anomaly(scores, invert=False)
