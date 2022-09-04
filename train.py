from transformers.models.vit_mae.configuration_vit_mae import ViTMAEConfig
from msn.layers import TFViTMAEMainModel

import tensorflow as tf


custom_config = ViTMAEConfig(
    hidden_size = 768,
    num_hidden_layers = 12,
    num_attention_heads = 12,
    intermediate_size = 3072,
    hidden_act = 'gelu',
    hidden_dropout_prob = 0.0,
    attention_probs_dropout_prob = 0.0,
    initializer_range = 0.02,
    layer_norm_eps = 1e-12,
    is_encoder_decoder = False,
    image_size = (224, 224),
    patch_size = (16, 16),
    num_channels = 3,
    qkv_bias = True,
    decoder_num_attention_heads = 16,
    decoder_hidden_size = 512,
    decoder_num_hidden_layers = 8,
    decoder_intermediate_size = 2048,
    mask_ratio = 0.75,
    norm_pix_loss = False
)

model = TFViTMAEMainModel(
    config=custom_config
)

pixel_values = tf.random.normal((2, 224, 224, 3))
model(pixel_values=pixel_values)

print(model.summary())