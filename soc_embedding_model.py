import math

import keras_cv
import tensorflow as tf


class SOCModel(tf.keras.Model):
    def __init__(
        self,
        image_encoder,
        class_encoder,
        n_classes,
        embedding_dims,
        learn_scale=False,
        label_smoothing=0.1,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.image_encoder = image_encoder
        self.class_encoder = class_encoder
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.accuracy_tracker = tf.keras.metrics.Mean(name="acc")
        self.dummy_classes = tf.range(n_classes, dtype=tf.int32)
        self.n_classes = n_classes
        self.embedding_dims = embedding_dims
        self.learn_scale = learn_scale
        self.label_smoothing = label_smoothing
        self.unnormalized_log_probs = False

    @property
    def metrics(self):
        return [self.loss_tracker, self.accuracy_tracker]

    @staticmethod
    def _log_prob(mu, scale_diag, x, unnormalized):
        log_unnormalized = -0.5 * tf.math.squared_difference(
            x / scale_diag, mu / scale_diag
        )
        if unnormalized:
            return tf.reduce_sum(log_unnormalized)
        log_normalization = tf.constant(
            0.5 * tf.math.log(2.0 * math.pi), dtype=mu.dtype
        ) + tf.math.log(scale_diag)
        return tf.reduce_sum(log_unnormalized - log_normalization)

    def log_prob(self, mu, scale_diag, x, unnormalized=False):
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

    def loss_fn(self, y_true, img_log_probs, label_smoothing=0.0):
        loss = tf.reduce_mean(
            tf.keras.losses.categorical_crossentropy(
                y_true=tf.one_hot(y_true, self.n_classes),
                y_pred=img_log_probs,
                from_logits=True,
                label_smoothing=label_smoothing,
                axis=1,
            )
        )
        return loss

    def accuracy(self, y_true, img_means, log_probs):
        pred_inds = tf.argmax(log_probs, axis=1, output_type=tf.int32)
        batch_acc = tf.reduce_mean(
            tf.cast(tf.equal(tf.cast(y_true, tf.int32), pred_inds), tf.float32)
        )
        return batch_acc

    def train_step(self, data):
        images, y_true = data
        training = True

        with tf.GradientTape() as tape:
            img_embs = self.image_encoder(images, training=training)

            class_embs = self.class_encoder(self.dummy_classes, training=training)

            if self.learn_scale:
                class_mu, class_scale_diag = (
                    class_embs[:, : self.embedding_dims],
                    class_embs[:, self.embedding_dims:],
                )
            else:
                class_mu = class_embs
                class_scale_diag = tf.ones_like(class_scale_diag)

            img_log_probs = self.log_prob(
                class_mu,
                class_scale_diag,
                img_embs,
                unnormalized=self.unnormalized_log_probs,
            )

            class_log_probs = self.log_prob(
                class_mu,
                class_scale_diag,
                class_mu,
                unnormalized=self.unnormalized_log_probs,
            )

            loss = tf.reduce_mean(
                self.loss_fn(y_true, img_log_probs, self.label_smoothing)
            )
            loss += tf.reduce_mean(
                self.loss_fn(
                    tf.range(self.n_classes), class_log_probs, self.label_smoothing
                )
            )
            loss = loss / 2

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.loss_tracker.update_state(loss)
        self.accuracy_tracker.update_state(
            self.accuracy(y_true, img_embs, img_log_probs)
        )
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        images, y_true = data
        training = False

        img_embs = self.image_encoder(images, training=training)

        class_embs = self.class_encoder(self.dummy_classes, training=training)

        if self.learn_scale:
            class_mu, class_scale_diag = (
                class_embs[:, : self.embedding_dims],
                class_embs[:, self.embedding_dims:],
            )
        else:
            class_mu = class_embs
            class_scale_diag = tf.ones_like(class_scale_diag)

        img_log_probs = self.log_prob(
            class_mu,
            class_scale_diag,
            img_embs,
            unnormalized=self.unnormalized_log_probs,
        )

        class_log_probs = self.log_prob(
            class_mu,
            class_scale_diag,
            class_mu,
            unnormalized=self.unnormalized_log_probs,
        )

        loss = tf.reduce_mean(
            self.loss_fn(y_true, img_log_probs, self.label_smoothing)
        )
        loss += tf.reduce_mean(
            self.loss_fn(
                tf.range(self.n_classes), class_log_probs, self.label_smoothing
            )
        )
        loss = loss / 2

        self.loss_tracker.update_state(loss)
        self.accuracy_tracker.update_state(
            self.accuracy(y_true, img_embs, img_log_probs)
        )
        return {m.name: m.result() for m in self.metrics}


def get_class_encoder(n_classes, embedding_dims):
    inputs = tf.keras.layers.Input((0,))
    means = tf.keras.layers.Embedding(
        input_dim=n_classes,
        output_dim=embedding_dims,
        embeddings_initializer=tf.keras.initializers.VarianceScaling(
            scale=0.1, mode="fan_out", distribution="uniform", seed=None
        ),
    )(inputs)
    diags = tf.keras.layers.Embedding(
        input_dim=n_classes,
        output_dim=embedding_dims,
        embeddings_initializer="ones",
    )(inputs)
    out = tf.keras.layers.Concatenate(axis=1)([means, tf.abs(diags)])
    return tf.keras.Model(inputs, out)


def get_img_encoder(image_shape, embedding_dims):
    inputs = tf.keras.layers.Input(image_shape)
    x = keras_cv.models.ResNetV2Backbone.from_preset(
        "resnet18_v2",
    )(inputs)
    x = tf.keras.layers.Flatten()(x)
    means = tf.keras.layers.Dense(embedding_dims, activation=None)(x)

    return tf.keras.Model(inputs=inputs, outputs=means)


def get_soc_model(image_shape, n_classes, embedding_dims, learn_scale=False):
    image_encoder = get_img_encoder(image_shape, embedding_dims)
    class_encoder = get_class_encoder(n_classes, embedding_dims)

    image_encoder.build((1,) + image_shape)
    class_encoder.build((1,))

    soc_model = SOCModel(
        image_encoder=image_encoder,
        class_encoder=class_encoder,
        n_classes=n_classes,
        embedding_dims=embedding_dims,
        learn_scale=learn_scale,
    )

    return soc_model