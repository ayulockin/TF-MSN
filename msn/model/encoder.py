import math
from typing import Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from transformers.activations_tf import get_tf_activation
from transformers.file_utils import ModelOutput
from transformers.modeling_tf_outputs import TFBaseModelOutput
from transformers.modeling_tf_utils import TFModelInputType, get_initializer
from transformers.tf_utils import shape_list, stable_softmax

from .embeddings import ViTMAEEmbeddings

###############################################################################
# Model Output Data Class
###############################################################################


class TFViTMAEModelOutput(ModelOutput):
    """
    Class for TFViTMAEModel's outputs, with potential hidden states and attentions.
    Args:
        cls_token_output (`tf.Tensor` of shape `(batch_size, hidden_size)`):
            This is the output from the CLS Token.
        last_hidden_state (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        mask (`tf.Tensor` of shape `(batch_size, sequence_length)`):
            Tensor indicating which patches are masked (1) and which are not (0).
        ids_restore (`tf.Tensor` of shape `(batch_size, sequence_length)`):
            Tensor containing the original index of the (shuffled) masked patches.
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer plus
            the initial embedding outputs.
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
    """

    cls_token_output: tf.Tensor = None
    last_hidden_state: tf.Tensor = None
    mask: tf.Tensor = None
    ids_restore: tf.Tensor = None
    hidden_states: Optional[Tuple[tf.Tensor]] = None
    attentions: Optional[Tuple[tf.Tensor]] = None


###############################################################################
# Self Attention and Attention
###############################################################################


class TFViTMAESelfAttention(tf.keras.layers.Layer):
    """The multi head self attention module."""

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        # The hidden size must be divisible by the number of heads to facilitate
        # multi-heads attention.
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number "
                f"of attention heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_att_head_size = math.sqrt(self.attention_head_size)

        self.query = tf.keras.layers.Dense(
            units=self.all_head_size,
            kernel_initializer=get_initializer(config.initializer_range),
            name="query",
        )
        self.key = tf.keras.layers.Dense(
            units=self.all_head_size,
            kernel_initializer=get_initializer(config.initializer_range),
            name="key",
        )
        self.value = tf.keras.layers.Dense(
            units=self.all_head_size,
            kernel_initializer=get_initializer(config.initializer_range),
            name="value",
        )
        self.dropout = tf.keras.layers.Dropout(rate=config.attention_probs_dropout_prob)

    def transpose_for_scores(self, tensor: tf.Tensor, batch_size: int) -> tf.Tensor:
        # Reshape from [batch_size, seq_length, all_head_size] to [batch_size, seq_length, num_attention_heads, attention_head_size]
        tensor = tf.reshape(
            tensor=tensor,
            shape=(batch_size, -1, self.num_attention_heads, self.attention_head_size),
        )

        # Transpose the tensor from [batch_size, seq_length, num_attention_heads, attention_head_size] to
        # [batch_size, num_attention_heads, seq_length, attention_head_size]
        return tf.transpose(tensor, perm=[0, 2, 1, 3])

    def call(
        self,
        hidden_states: tf.Tensor,
        head_mask: tf.Tensor,
        output_attentions: bool,
        training: bool = False,
    ) -> Tuple[tf.Tensor]:
        batch_size = shape_list(hidden_states)[0]

        mixed_query_layer = self.query(inputs=hidden_states)
        mixed_key_layer = self.key(inputs=hidden_states)
        mixed_value_layer = self.value(inputs=hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer, batch_size)
        key_layer = self.transpose_for_scores(mixed_key_layer, batch_size)
        value_layer = self.transpose_for_scores(mixed_value_layer, batch_size)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
        dk = tf.cast(self.sqrt_att_head_size, dtype=attention_scores.dtype)
        attention_scores = tf.divide(attention_scores, dk)

        # Normalize the attention scores to probabilities.
        attention_probs = stable_softmax(logits=attention_scores, axis=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(inputs=attention_probs, training=training)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = tf.multiply(attention_probs, head_mask)

        attention_output = tf.matmul(attention_probs, value_layer)
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])

        # (batch_size, seq_len_q, all_head_size)
        attention_output = tf.reshape(
            tensor=attention_output, shape=(batch_size, -1, self.all_head_size)
        )
        outputs = (
            (attention_output, attention_probs)
            if output_attentions
            else (attention_output,)
        )

        return outputs


class TFViTMAESelfOutput(tf.keras.layers.Layer):
    """
    The residual connection is defined in TFViTMAELayer instead of here
    (as is the case with other models), due to the layernorm applied
    before each block.
    """

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.dense = tf.keras.layers.Dense(
            units=config.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            name="dense",
        )
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)

    def call(self, hidden_states: tf.Tensor, training: bool = False) -> tf.Tensor:
        hidden_states = self.dense(inputs=hidden_states)
        hidden_states = self.dropout(inputs=hidden_states, training=training)

        return hidden_states


class TFViTMAEAttention(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.self_attention = TFViTMAESelfAttention(config, name="attention")
        self.dense_output = TFViTMAESelfOutput(config, name="output")

    def call(
        self,
        input_tensor: tf.Tensor,
        head_mask: tf.Tensor,
        output_attentions: bool,
        training: bool = False,
    ) -> Tuple[tf.Tensor]:
        self_outputs = self.self_attention(
            hidden_states=input_tensor,
            head_mask=head_mask,
            output_attentions=output_attentions,
            training=training,
        )
        attention_output = self.dense_output(
            hidden_states=self_outputs[0], training=training
        )
        outputs = (attention_output,) + self_outputs[1:]

        return outputs


###############################################################################
# Intermediate Layers
###############################################################################


class TFViTMAEIntermediate(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.dense = tf.keras.layers.Dense(
            units=config.intermediate_size,
            kernel_initializer=get_initializer(config.initializer_range),
            name="dense",
        )

        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = get_tf_activation(config.hidden_act)
        else:
            self.intermediate_act_fn = config.hidden_act

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        hidden_states = self.dense(inputs=hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states


class TFViTMAEOutput(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.dense = tf.keras.layers.Dense(
            units=config.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            name="dense",
        )
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)

    def call(
        self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = False
    ) -> tf.Tensor:
        hidden_states = self.dense(inputs=hidden_states)
        hidden_states = self.dropout(inputs=hidden_states, training=training)
        hidden_states = hidden_states + input_tensor

        return hidden_states


###############################################################################
# MAE Layer
###############################################################################


class TFViTMAELayer(tf.keras.layers.Layer):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.attention = TFViTMAEAttention(config, name="attention")
        self.intermediate = TFViTMAEIntermediate(config, name="intermediate")
        self.vit_output = TFViTMAEOutput(config, name="output")

        self.layernorm_before = tf.keras.layers.LayerNormalization(
            epsilon=config.layer_norm_eps, name="layernorm_before"
        )
        self.layernorm_after = tf.keras.layers.LayerNormalization(
            epsilon=config.layer_norm_eps, name="layernorm_after"
        )

    def call(
        self,
        hidden_states: tf.Tensor,
        head_mask: tf.Tensor,
        output_attentions: bool,
        training: bool = False,
    ) -> Tuple[tf.Tensor]:
        attention_outputs = self.attention(
            # in ViTMAE, layernorm is applied before self-attention
            input_tensor=self.layernorm_before(inputs=hidden_states),
            head_mask=head_mask,
            output_attentions=output_attentions,
            training=training,
        )
        attention_output = attention_outputs[0]

        # first residual connection
        hidden_states = attention_output + hidden_states

        # in ViTMAE, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(inputs=hidden_states)

        intermediate_output = self.intermediate(hidden_states=layer_output)

        # second residual connection is done here
        layer_output = self.vit_output(
            hidden_states=intermediate_output,
            input_tensor=hidden_states,
            training=training,
        )
        outputs = (layer_output,) + attention_outputs[
            1:
        ]  # add attentions if we output them

        return outputs


###############################################################################
# Encoder Layer
###############################################################################


class TFViTMAEEncoder(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.layer = [
            TFViTMAELayer(config, name=f"layer_._{i}")
            for i in range(config.num_hidden_layers)
        ]

    def call(
        self,
        hidden_states: tf.Tensor,
        head_mask: tf.Tensor,
        output_attentions: bool,
        output_hidden_states: bool,
        return_dict: bool,
        training: bool = False,
    ) -> Union[TFBaseModelOutput, Tuple[tf.Tensor]]:
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states=hidden_states,
                head_mask=head_mask[i],
                output_attentions=output_attentions,
                training=training,
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, all_hidden_states, all_attentions]
                if v is not None
            )

        return TFBaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


###############################################################################
# Main Model
###############################################################################


class TFViTMAEMainModel(tf.keras.Model):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.config = config

        self.embeddings = ViTMAEEmbeddings(config, name="embeddings")
        self.encoder = TFViTMAEEncoder(config, name="encoder")
        self.layernorm = tf.keras.layers.LayerNormalization(
            epsilon=config.layer_norm_eps, name="layernorm"
        )

    def call(
        self,
        pixel_values: Optional[TFModelInputType] = None,
        noise: tf.Tensor = None,
        head_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[TFViTMAEModelOutput, Tuple[tf.Tensor]]:
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

        if not return_dict:
            return (sequence_output, mask, ids_restore) + encoder_outputs[1:]

        return TFViTMAEModelOutput(
            cls_token_output=sequence_output[:, 0, :],
            last_hidden_state=sequence_output,
            mask=mask,
            ids_restore=ids_restore,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
