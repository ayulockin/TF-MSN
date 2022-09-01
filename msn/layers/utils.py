from typing import List, Union, Optional
import tensorflow as tf
import numpy as np

def shape_list(tensor: Union[tf.Tensor, np.ndarray]) -> List[int]:
    """
    Deal with dynamic shape in tensorflow cleanly.
    Args:
        tensor (`tf.Tensor` or `np.ndarray`): The tensor we want the shape of.
    Returns:
        `List[int]`: The shape of the tensor as a list.
    """
    if isinstance(tensor, np.ndarray):
        return list(tensor.shape)

    dynamic = tf.shape(tensor)

    if tensor.shape == tf.TensorShape(None):
        return dynamic

    static = tensor.shape.as_list()

    return [dynamic[i] if s is None else s for i, s in enumerate(static)]


def get_initializer(initializer_range: float = 0.02) -> tf.initializers.TruncatedNormal:
    """
    Creates a `tf.initializers.TruncatedNormal` with the given range.
    Args:
        initializer_range (*float*, defaults to 0.02): Standard deviation of the initializer range.
    Returns:
        `tf.initializers.TruncatedNormal`: The truncated normal initializer.
    """
    return tf.keras.initializers.TruncatedNormal(stddev=initializer_range)


def stable_softmax(logits: tf.Tensor, axis: Optional[int] = None, name: Optional[str] = None) -> tf.Tensor:
    """
    Stable wrapper that returns the same output as `tf.nn.softmax`, but that works reliably with XLA on CPU. It is
    meant as a workaround for the [following issue](https://github.com/tensorflow/tensorflow/issues/55682), and will be
    removed after it gets fixed. The arguments and outputs are the same as `tf.nn.softmax`, and relies on the fact that
    `softmax(x) = softmax(x + c)` (see https://ogunlao.github.io/2020/04/26/you_dont_really_know_softmax.html).
    Args:
        logits (`tf.Tensor`):
            Must be one of the following types: half, float32, float64.
        axis (`int`, *optional*):
            The dimension softmax would be performed on. The default is -1 which indicates the last dimension.
        name (`str`, *optional*):
            A name for the operation.
    Returns:
        `tf.Tensor`:
            A Tensor. Has the same type and shape as logits.
    """
    # TODO: When the issue linked above gets sorted, add a check on TF version here and use the original function if
    # it has the fix. After we drop the support for unfixed versions, remove this function.
    return tf.nn.softmax(logits=logits + 1e-9, axis=axis, name=name)


class BaseModelOutput:
    def __init__(self, last_hidden_state, hidden_states, attentions):
        self.last_hidden_state=last_hidden_state
        self.hidden_states=hidden_states
        self.attentions=attentions