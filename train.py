import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import pandas as pd
import tensorflow as tf
import wandb
from absl import app, flags
from ml_collections.config_flags import config_flags
from wandb.keras import WandbCallback

# Modules
from msn.data import GetMSNDataloader, download_dataset, preprocess_dataframe
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
    train_paths, _ = preprocess_dataframe(train_df)
    print(train_paths[:5])

    # Get dataloader
    dataloader = GetMSNDataloader(config).get_dataloader(train_paths)
    # TODO: REMOVE
    samples = next(iter(dataloader))
    print(
        samples[0].shape,
        samples[1].shape,
        samples[2].shape,
        samples[3].shape,
        samples[4].shape,
    )

    # Get model
    tf.keras.backend.clear_session()
    model = get_model(config)
    # TODO: REMOVE
    pixel_values = tf.random.normal((2, 224, 224, 3))
    model(pixel_values=pixel_values)
    print(model.summary())


if __name__ == "__main__":
    app.run(main)
