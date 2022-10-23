import tensorflow as tf


class PrototypeLayer(tf.keras.layers.Layer):
    def __init__(self, config):
        super(PrototypeLayer, self).__init__()
        self.config = config

        # Initialize the prototype layer weights
        _sqrt_k = tf.math.sqrt(1.0 / config.model_config.num_prototypes)
        initializer = tf.keras.initializers.RandomUniform(
            minval=-_sqrt_k, maxval=_sqrt_k
        )

        # Initialize the prototype layer
        self.prototype = tf.keras.layers.Dense(
            config.model_config.num_prototypes,
            kernel_initializer=initializer,
            use_bias=False,
        )

    def call(self, inputs, tau):
        # Scale the prototype layer with tau
        scaled_prototype = tf.math.divide(self.prototype(inputs), tau)

        # Softmax on the scaled prototype layer
        outputs = tf.keras.layers.Softmax()(scaled_prototype)

        return outputs
