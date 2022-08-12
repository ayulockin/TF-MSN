import tensorflow as tf
import wandb


class EMA(tf.keras.callbacks.Callback):
    def __init__(self, decay=0.999):
        super(EMA, self).__init__()
        # TODO (ayulockin): make decay a tf.Variable instance and increase it
        # linearly as the training progesses.
        self.decay = decay

        # Create an ExponentialMovingAverage object
        self.ema = tf.train.ExponentialMovingAverage(decay=self.decay_var)

    def on_train_begin(self, logs=None):
        # TODO (ayulockin): add assertion for anchor_model and target_model.
        self.ema.apply(self.model.get_layer('anchor_model').trainable_variables)

    def on_train_batch_end(self, batch, logs=None):
        # Get exponential moving average of anchor model weights.
        train_vars = self.model.get_layer('anchor_model').trainable_variables
        averages = [self.ema.average(var) for var in train_vars]

        # Assign the average weights to target model
        target_model_vars = self.model.get_layer('target_model').non_trainable_variables
        assert len(target_model_vars) == len(averages)
        for i, var in enumerate(target_model_vars):
            var.assign(averages[i])

        self.ema.apply(self.model.get_layer('anchor_model').trainable_variables)
