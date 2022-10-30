import numpy as np
from typing import Optional, Tuple, Union

import tensorflow as tf
from tensorflow.keras import models

from transformers.models.vit_mae.configuration_vit_mae import ViTMAEConfig
from transformers.modeling_tf_utils import TFModelInputType

# Module
from msn.model.encoder import ViTMAEEmbeddings
from msn.model.encoder import TFViTMAEEncoder
from msn.model.projection import ProjectionHead


def get_vit_mae_configs(args):
    """Get the configs for the ViT-MAE and the Projection model."""

    args = args.model_config
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
        proj_hidden_size = args.proj_hidden_size,
        proj_output_dim = args.proj_output_dim,
        proj_use_bn = args.proj_use_bn
    )

    return custom_config


class TFMAEViTModelWithProjection(tf.keras.Model):
    """The encoder model with projection head."""

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.config = get_vit_mae_configs(config)

        self.embeddings = ViTMAEEmbeddings(self.config, name="embeddings")
        self.encoder = TFViTMAEEncoder(self.config, name="encoder")
        self.layernorm = tf.keras.layers.LayerNormalization(
            epsilon=self.config.layer_norm_eps, name="layernorm"
        )
        self.projection_head = ProjectionHead(self.config, name="projection_head")

    def call(
        self,
        pixel_values = None,
        noise: tf.Tensor = None,
        head_mask = None,
        output_attentions = None,
        output_hidden_states = None,
        training: bool = False,
    ) -> tf.Tensor:
        embedding_output, mask, ids_restore = self.embeddings(
            pixel_values=pixel_values, training=training, noise=noise
        )

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            raise NotImplementedError
        else:
            head_mask = [None] * self.config.num_hidden_layers
        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(inputs=sequence_output)
        
        # Pass the [CLS] token to the projection head
        projection_output = self.projection_head(sequence_output[:, 0, :])

        return projection_output