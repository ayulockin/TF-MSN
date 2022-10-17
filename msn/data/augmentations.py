# TODO (aritra): Replace this with KerasCV based augmentations

import random

import tensorflow as tf

# Reference: https://github.com/ayulockin/SwAV-TF/blob/master/utils/multicrop_dataset.py


@tf.function
def gaussian_blur(image, kernel_size=23, padding="SAME"):
    sigma = tf.random.uniform((1,)) * 1.9 + 0.1

    radius = tf.cast(kernel_size / 2, tf.int32)
    kernel_size = radius * 2 + 1
    x = tf.cast(tf.range(-radius, radius + 1), tf.float32)
    blur_filter = tf.exp(
        -tf.pow(x, 2.0) / (2.0 * tf.pow(tf.cast(sigma, tf.float32), 2.0))
    )
    blur_filter /= tf.reduce_sum(blur_filter)
    # One vertical and one horizontal filter.
    blur_v = tf.reshape(blur_filter, [kernel_size, 1, 1, 1])
    blur_h = tf.reshape(blur_filter, [1, kernel_size, 1, 1])
    num_channels = tf.shape(image)[-1]
    blur_h = tf.tile(blur_h, [1, 1, num_channels, 1])
    blur_v = tf.tile(blur_v, [1, 1, num_channels, 1])
    expand_batch_dim = image.shape.ndims == 3
    if expand_batch_dim:
        image = tf.expand_dims(image, axis=0)
    blurred = tf.nn.depthwise_conv2d(
        image, blur_h, strides=[1, 1, 1, 1], padding=padding
    )
    blurred = tf.nn.depthwise_conv2d(
        blurred, blur_v, strides=[1, 1, 1, 1], padding=padding
    )
    if expand_batch_dim:
        blurred = tf.squeeze(blurred, axis=0)
    return blurred


@tf.function
def color_jitter(x, s=0.5):
    x = tf.image.random_brightness(x, max_delta=0.8 * s)
    x = tf.image.random_contrast(x, lower=1 - 0.8 * s, upper=1 + 0.8 * s)
    x = tf.image.random_saturation(x, lower=1 - 0.8 * s, upper=1 + 0.8 * s)
    x = tf.image.random_hue(x, max_delta=0.2 * s)
    return x


@tf.function
def color_drop(x):
    x = tf.image.rgb_to_grayscale(x)
    x = tf.tile(x, [1, 1, 3])
    return x


@tf.function
def resize_view(x, mode="random"):
    if mode == "random":
        resize_dim = [224, 224]
    elif mode == "focal":
        resize_dim = [96, 96]
    else:
        NotImplementedError("wrong mode passed!")

    # resize the image to the desired size
    x = tf.image.resize(x, resize_dim, method="bicubic", preserve_aspect_ratio=False)

    return x


@tf.function
def random_apply(func, x, p):
    return tf.cond(
        tf.less(
            tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32),
            tf.cast(p, tf.float32),
        ),
        lambda: func(x),
        lambda: x,
    )


@tf.function
def custom_augment(image, mode="random"):
    # Resize view
    # The official implementation uses RandomResizedCrop for resizing the view to specific size.
    # But resize operation is done before any other augmentation.
    # https://github.com/facebookresearch/msn/blob/4388dc1eadbe3042b85d3296d41b9b207656e043/src/data_manager.py#L86
    image = resize_view(image, mode)
    # Random flips
    image = random_apply(tf.image.flip_left_right, image, p=0.5)
    # Randomly apply gausian blur
    image = random_apply(gaussian_blur, image, p=0.5)
    # Randomly apply transformation (color distortions) with probability p.
    image = random_apply(color_jitter, image, p=0.8)
    # Randomly apply grayscale
    image = random_apply(color_drop, image, p=0.2)
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image
