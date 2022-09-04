import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import pandas as pd
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags

import wandb
from wandb.keras import WandbCallback
import tensorflow as tf

from classifier.data import download_and_get_dataset
from classifier.data import GetDataloader
from classifier.model import get_model
from classifier.callbacks import *

# Modules
from msn.data import download_dataset, preprocess_dataframe
from msn.model import get_model

# Config
FLAGS = flags.FLAGS
CONFIG = config_flags.DEFINE_config_file("config")
flags.DEFINE_bool("wandb", False, "MLOps pipeline for our classifier.")


def main(_):
    # Get configs from the config file.
    config = CONFIG.value
    print(config)

    CALLBACKS = []
    # Initialize a Weights and Biases run.
    if FLAGS.wandb:
        run = wandb.init(
            project=CONFIG.value.wandb_config.project,
            job_type="train",
            config=config.to_dict(),
        )
        # WandbCallback for experiment tracking
        CALLBACKS += [WandbCallback(save_model=False)]

    # Download the dataset
    train_df = download_dataset("train", "labelled-dataset")

    # Preprocess the DataFrames
    train_paths, train_labels = preprocess_dataframe(train_df)
    print(train_paths[:5], train_labels[:5])

    # Get dataloader

    # Get model
    tf.keras.backend.clear_session()
    model = get_model(config)
    # TODO: REMOVE
    pixel_values = tf.random.normal((2, 224, 224, 3))
    model(pixel_values=pixel_values)
    print(model.summary())

    # # Compile the model
    # model.compile(
    #     optimizer = config.train_config.optimizer,
    #     loss = config.train_config.loss,
    #     metrics = config.train_config.metrics
    # )

    # # Train the model
    # model.fit(
    #     trainloader,
    #     validation_data = validloader,
    #     epochs = config.train_config.epochs,
    #     callbacks=CALLBACKS
    # )


if __name__ == "__main__":
    app.run(main)
