import tensorflow as tf


def test_weights_equality(encoder, target):
    """Test the equality of the weights of the encoder and target models.

    Arguments:
        encoder: A subclassed encoder model.
        target: A subclassed target model of the same instance as
            the encoder model.
    """
    for encoder_layer, target_layer in zip(encoder.layers, target.layers):
        for encoder_weight, target_weight in zip(encoder_layer.weights, target_layer.weights):
            assert tf.reduce_all(encoder_weight == target_weight), "The weights of the encoder and target models are not equal."


def build_and_clone_model(encoder, target, args):
    """Build the encoder and target models and clone the weights
    of the encoder model to the target model.

    Arguments:
        encoder: A subclassed encoder model.
        target: A subclassed target model of the same instance as
            the encoder model.
        args : ml-collection ConfigDict object.
    """
    args = args.model_config
    input_shape = (
        1,
        args.model_img_height,
        args.model_img_width,
        args.model_img_channels,
    )

    assert encoder.__class__ == target.__class__, "The encoder and target models must be of the same instance."

    # Build the encoder model
    encoder.build(input_shape=input_shape)

    # Build the target model
    target.build(input_shape=input_shape)

    # Clone the weights of the encoder model to the target model
    target.set_weights(encoder.get_weights())

    # Test the equality of the weights of the encoder and target models
    test_weights_equality(encoder, target)

    return encoder, target
