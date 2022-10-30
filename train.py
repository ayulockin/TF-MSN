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
from msn.model import PrototypeLayer, get_model
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
    anchor_model = get_model(config)
    target_model = get_model(config)
    encoder_model, target_model = build_and_clone_model(
        anchor_model, target_model, config
    )

    # # # TODO: REMOVE
    # pixel_values = samples[0]
    # anchor_output = anchor_model(
    #     pixel_values=pixel_values, return_dict=True, training=True
    # )
    # target_output = target_model(
    #     pixel_values=pixel_values, return_dict=True, training=True
    # )

    # print(
    #     anchor_output["cls_token_output"].shape, target_output["cls_token_output"].shape
    # )


    print(anchor_model.summary())
    # print(target_model.summary())

    # # Initialize a prototype model
    # prototype_layer = PrototypeLayer(config)

    # prototype_anchor_out = prototype_layer(
    #     anchor_output["cls_token_output"], tau=config.model_config.anchor_tau
    # )
    # prototype_target_out = prototype_layer(
    #     target_output["cls_token_output"], tau=config.model_config.target_tau
    # )

    # print(prototype_anchor_out.shape, prototype_target_out.shape)


if __name__ == "__main__":
    app.run(main)
