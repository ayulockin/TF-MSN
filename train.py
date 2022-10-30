# python train.py --config configs/config.py --wandb --log_model
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
from msn.model import TFMAEViTModelWithProjection
from msn.utils import build_and_clone_model

# Config
FLAGS = flags.FLAGS
CONFIG = config_flags.DEFINE_config_file("config")
flags.DEFINE_bool("wandb", False, "MLOps pipeline for our classifier.")


def main(_):
    # Get configs from the config file.
    config = CONFIG.value

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

    # The parameters of the target model are updated with an Exponential Moving Average
    # of the parameters of the anchor model.
    tf.keras.backend.clear_session()

    # Create the anchor and target model.
    anchor = TFMAEViTModelWithProjection(config)
    target = TFMAEViTModelWithProjection(config)

    print(anchor, target)
    anchor_model, target_model = build_and_clone_model(
        anchor, target, config
    )

    # TODO: REMOVE
    pixel_values = tf.random.normal((8, 224, 224, 3))
    anchor_output = anchor(
        pixel_values=pixel_values, training=True
    )
    target_output = target(
        pixel_values=pixel_values, training=True
    )

    print(
        anchor_output.shape, target_output.shape
    )


    print(anchor.summary())
    print(target.summary())


if __name__ == "__main__":
    app.run(main)
