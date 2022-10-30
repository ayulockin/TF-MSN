import tensorflow as tf
from tensorflow.keras import models

from transformers.models.vit_mae.configuration_vit_mae import ViTMAEConfig

# Module
from .encoder import TFViTMAEMainModel
from .projection import get_projection


def get_vit_mae_configs(args):
    custom_config = ViTMAEConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        intermediate_size=args.intermediate_size,
        hidden_act=args.hidden_act,
        hidden_dropout_prob=args.hidden_dropout_prob,
        attention_probs_dropout_prob=args.attention_probs_dropout_prob,
        initializer_range=args.initializer_range,
        layer_norm_eps=args.layer_norm_eps,
        is_encoder_decoder=args.is_encoder_decoder,
        image_size=(args.model_img_width, args.model_img_height),
        patch_size=(args.patch_size, args.patch_size),
        num_channels=args.model_img_channels,
        qkv_bias=args.qkv_bias,
        mask_ratio=args.mask_ratio,
        norm_pix_loss=args.norm_pix_loss,
    )

    return custom_config


def get_model(args):
    # Get the config at the `model_config` level.
    args = args.model_config

    custom_config = get_vit_mae_configs(args)
    encoder = TFViTMAEMainModel(config=custom_config)
    projection_head = get_projection(args)

    return models.Sequential([encoder, projection_head])
