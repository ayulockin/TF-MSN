import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models


class ProjectionHead(layers.Layer):
    """Build the projection head for the encoder."""

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config

        self.fc1 = layers.Dense(config.proj_hidden_size)
        if config.proj_use_bn:
            self.bn1 = layers.BatchNormalization()
        self.gelu1 = tf.keras.activations.gelu

        self.fc2 = layers.Dense(config.proj_hidden_size)
        if config.proj_use_bn:
            self.bn2 = layers.BatchNormalization()
        self.gelu2 = tf.keras.activations.gelu

        self.fc3 = layers.Dense(config.proj_output_dim)

    def call(self, inputs):
        x = self.fc1(inputs)
        if self.config.proj_use_bn:
            x = self.bn1(x)
        x = self.gelu1(x)

        x = self.fc2(x)
        if self.config.proj_use_bn:
            x = self.bn2(x)
        x = self.gelu2(x)

        x = self.fc3(x)

        return x
