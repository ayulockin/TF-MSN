import os

import ml_collections
from ml_collections import config_dict


def get_wandb_configs() -> ml_collections.ConfigDict:
    configs = ml_collections.ConfigDict()
    configs.project = "tf-msn"

    return configs


def get_dataset_configs() -> ml_collections.ConfigDict:
    configs = ml_collections.ConfigDict()
    configs.use_options = True
    configs.size_crops = [224, 96]
    configs.num_crops = [2, 5]
    configs.shuffle_buffer = 100
    configs.batch_size = 8

    return configs


def get_model_configs() -> ml_collections.ConfigDict:
    configs = ml_collections.ConfigDict()
    configs.model_img_height = 224
    configs.model_img_width = 224
    configs.model_img_channels = 3
    configs.backbone = "ViT"
    configs.dropout_rate = 0.5
    configs.post_gap_dropout = False

    # ViT MAE configs
    configs.hidden_size = 768
    configs.num_hidden_layers = 12
    configs.num_attention_heads = 12
    configs.intermediate_size = 3072
    configs.hidden_act = "gelu"
    configs.hidden_dropout_prob = 0.0
    configs.attention_probs_dropout_prob = 0.0
    configs.initializer_range = 0.02
    configs.layer_norm_eps = 1e-12
    configs.is_encoder_decoder = False
    configs.patch_size = 16
    configs.num_channels = 3
    configs.qkv_bias = True
    configs.decoder_num_attention_heads = 16
    configs.decoder_hidden_size = 512
    configs.decoder_num_hidden_layers = 8
    configs.decoder_intermediate_size = 2048
    configs.mask_ratio = 0.75
    configs.norm_pix_loss = False

    # Projection configs
    configs.num_prototypes = 10
    configs.anchor_tau = 0.01
    configs.target_tau = 0.1

    return configs


def get_callback_configs() -> ml_collections.ConfigDict:
    configs = ml_collections.ConfigDict()
    # Early stopping
    configs.use_earlystopping = True
    configs.early_patience = 6
    # Reduce LR on plateau
    configs.use_reduce_lr_on_plateau = False
    configs.rlrp_factor = 0.2
    configs.rlrp_patience = 3
    # Model checkpointing
    configs.checkpoint_filepath = "wandb/model_{epoch}"
    configs.save_best_only = True
    # Model Prediction Viz
    configs.viz_num_images = 100

    return configs


def get_train_configs() -> ml_collections.ConfigDict:
    configs = ml_collections.ConfigDict()
    configs.epochs = 3
    configs.use_augmentations = False
    configs.use_class_weights = False
    configs.optimizer = "adam"
    configs.sgd_momentum = 0.9
    configs.loss = "categorical_crossentropy"
    configs.metrics = ["accuracy"]

    return configs


def get_config() -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()
    config.seed = 0
    config.wandb_config = get_wandb_configs()
    config.dataset_config = get_dataset_configs()
    config.model_config = get_model_configs()
    config.callback_config = get_callback_configs()
    config.train_config = get_train_configs()

    return config
