import numpy as np
import tensorflow as tf


class DenseSOFLayer(tf.keras.layers.Layer):
    def __init__(self, features=32, **kwargs):
        super().__init__(**kwargs)
        self.features = features

    def build(self, input_shape):
        self.projection = self.add_weight(
            shape=(self.features, input_shape[-1]),
            initializer=tf.keras.initializers.VarianceScaling(
                scale=1., mode="fan_out", distribution="uniform", seed=None
            ),
            trainable=True,
        )
        self.scale_diag = self.add_weight(
            shape=(self.features, input_shape[-1]),
            initializer="ones",
            trainable=True,
        )

    @staticmethod
    def _log_prob(mu, scale_diag, x, unnormalized):
        log_unnormalized = -0.5 * tf.math.squared_difference(
            x / scale_diag, mu / scale_diag
        ) 
        if unnormalized:
            return tf.reduce_sum(log_unnormalized)
        log_normalization = tf.constant(
            0.5 * np.log(2.0 * np.pi), dtype=mu.dtype
        ) + tf.math.log(scale_diag)
        return tf.reduce_sum(log_unnormalized - log_normalization)

    def log_prob(self, mu, scale_diag, x, unnormalized=True):
        batch_log_prob = tf.vectorized_map(
            lambda _x: tf.vectorized_map(
                lambda _params: self._log_prob(
                    _params[0], _params[1], _x, unnormalized=unnormalized
                ),
                (mu, scale_diag),
            ),
            x,
        )
        return batch_log_prob

    def call(self, inputs):
        log_probs = self.log_prob(self.projection, self.scale_diag, inputs)
        return log_probs