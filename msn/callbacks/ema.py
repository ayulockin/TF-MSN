import tensorflow as tf
import wandb


class EMA(tf.keras.callbacks.Callback):
    """Use this class to update the weights of a target model
    with exponential moving average of the weights of an anchor
    model.

    The model should have the structure:
        ```
        def siamese_network():
            inputs = layers.Input(shape=(...))
            # Init anchor model
            anchor_model = build_anchor_model(name='anchor_model')
            # Init target model without trainable params.
            target_model = build_target_model(name='target_model')
            target_model.trainable = False
            ...
            return models.Model(inputs, outputs=[...])
        ```

    Args:
        decay (float): The decay/momentum parameter.
    """

    def __init__(self, decay: float = 0.999):
        super(EMA, self).__init__()
        # TODO (ayulockin): make decay a tf.Variable instance and increase it
        # linearly as the training progesses.
        self.decay = decay

        # Create an ExponentialMovingAverage object
        self.ema = tf.train.ExponentialMovingAverage(decay=self.decay_var)

    def on_train_begin(self, logs: dict = None) -> None:
        # TODO (ayulockin): add assertion for anchor_model and target_model.
        self.ema.apply(self.model.get_layer("anchor_model").trainable_variables)

    def on_train_batch_end(self, batch: int, logs: dict = None) -> None:
        # Get exponential moving average of anchor model weights.
        train_vars = self.model.get_layer("anchor_model").trainable_variables
        averages = [self.ema.average(var) for var in train_vars]

        # Assign the average weights to target model
        target_model_vars = self.model.get_layer("target_model").non_trainable_variables
        assert len(target_model_vars) == len(averages)
        for i, var in enumerate(target_model_vars):
            var.assign(averages[i])

        self.ema.apply(self.model.get_layer("anchor_model").trainable_variables)
