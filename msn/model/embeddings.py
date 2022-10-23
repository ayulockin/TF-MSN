import tensorflow as tf
from transformers.tf_utils import shape_list

###############################################################################
# Positional Embedding Utilities
###############################################################################


def get_2d_sincos_pos_embed(
    embed_dim: int, grid_size: int, add_cls_token: bool = False
) -> tf.Tensor:
    """
    Create 2D sin/cos positional embeddings.

    Args:
        embed_dim (`int`):
            Embedding dimension.
        grid_size (`int`):
            The grid height and width.
        add_cls_token (`bool`, *optional*, defaults to `False`):
            Whether or not to add a classification (CLS) token.

    Returns:
        (`tf.Tensor` of shape (grid_size*grid_size, embed_dim) or (1+grid_size*grid_size, embed_dim): the position
        embeddings (with or without classification token)
    """
    grid_h = tf.range(grid_size, dtype=tf.float32)
    grid_w = tf.range(grid_size, dtype=tf.float32)
    grid = tf.meshgrid(grid_h, grid_w)
    grid = tf.stack(grid, axis=0)
    grid = tf.reshape(grid, [2, 1, grid_size, grid_size])  # adding a new dimension

    # Get the 2d sin pos embedding
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if add_cls_token:
        pos_embed = tf.concat([tf.zeros((1, embed_dim)), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: tf.Tensor) -> tf.Tensor:
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even")

    # use half of dimensions to encode
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = tf.concat([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: tf.Tensor):
    """
    Args:
        embed_dim (`int`):
            Output dimension for each position
        pos (int): a list of positions to be encoded
    """
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even")

    omega = tf.range(embed_dim // 2, dtype="float32")
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = tf.reshape(pos, [-1])  # (M,)
    out = tf.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    # half of the positions get sinusoidal pattern and the rest gets
    # cosine pattern and then they are concatenated
    emb_sin = tf.sin(out)  # (M, D/2)
    emb_cos = tf.cos(out)  # (M, D/2)

    emb = tf.concat([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


###############################################################################
# Positional and Patch Embeddings
###############################################################################


class ViTMAEPatchEmbeddings(tf.keras.layers.Layer):
    """
    This class turns `pixel_values` of shape `(batch_size, height, width, num_channels)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        image_size = config.image_size
        patch_size = config.patch_size
        hidden_size = config.hidden_size
        self.num_patches = (image_size[0] // patch_size[0]) * (
            image_size[1] // patch_size[1]
        )
        self.config = config

        # Initialize the patchification and projection layer.
        self.projection = tf.keras.layers.Conv2D(
            filters=hidden_size,
            kernel_size=patch_size,
            strides=patch_size,
            padding="valid",
            kernel_initializer="glorot_uniform",
            bias_initializer="zeros",
            name="projection",
        )
        # Initialize the layer that reshapes a collection of pathces
        # to the temporal dimension.
        self.flatten = tf.keras.layers.Reshape(
            target_shape=(self.num_patches, hidden_size)
        )

    def call(self, pixel_values: tf.Tensor, training: bool = False) -> tf.Tensor:
        # pixel_values.shape = (batch_size, height, width, num_channels)
        # Patchify and project the pixel_values
        projection = self.projection(pixel_values)
        # projection.shape = (batch_size, image_size[0] // patch_size[0], image_size[1] // patch_size[1], hidden_size)

        # Change the 2D spatial dimensions to a single temporal dimension.
        x = self.flatten(projection)
        # x.shape = (batch_size, num_patches, hidden_size)
        return x


class ViTMAEEmbeddings(tf.keras.layers.Layer):
    """
    Construct the CLS token, position and patch embeddings.
    """

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.patch_embeddings = ViTMAEPatchEmbeddings(config, name="patch_embeddings")
        self.num_patches = (
            self.patch_embeddings.num_patches
        )  # num_patches are calculated in the patch_embedding layer

        self.config = config

    def build(self, input_shape: tf.TensorShape):
        self.cls_token = self.add_weight(
            shape=(1, 1, self.config.hidden_size),
            initializer=tf.random_normal_initializer(
                stddev=self.config.initializer_range
            ),
            trainable=True,
            name="cls_token",
        )
        self.position_embeddings = self.add_weight(
            shape=(1, self.num_patches + 1, self.config.hidden_size),
            initializer="zeros",
            trainable=False,  # fixed sin-cos embedding
            name="position_embeddings",
        )
        pos_embed = get_2d_sincos_pos_embed(
            self.position_embeddings.shape[-1],
            int(self.patch_embeddings.num_patches**0.5),
            add_cls_token=True,
        )[None, ...]
        self.position_embeddings.assign(pos_embed)

        super().build(input_shape)

    def random_masking(self, sequence: tf.Tensor):
        """
        Perform per-sample random masking by per-sample shuffling. Per-sample shuffling is done by argsort random
        noise.

        Args:
            sequence (`tf.Tensor` of shape `(batch_size, sequence_length, dim)`)
        """
        # TODO @ariG23498:
        # This might create a problem when training, the `batch_size` might be `None`
        batch_size, seq_length, dim = shape_list(sequence)

        # Calculate the length of the sequence to keep
        len_keep = int(seq_length * (1 - self.config.mask_ratio))

        # Build a uniform distribution
        noise = tf.random.uniform(
            shape=(batch_size, seq_length), minval=0.0, maxval=1.0
        )

        # `ids_shuffle` are the indicies that will sort the uniform distribution
        # as this is a uniform distribution, the sorted distirbution will have
        # more values close to 0 and less values close to 1
        # small is keep, large is remove to mimic keeping central portions of the
        # image and removing parts of the image that are close to the edges
        ids_shuffle = tf.argsort(noise, axis=1)

        # `ids_resotre` are the indicies that will help resotre the uniform distribution
        # from its sorted form.
        ids_restore = tf.argsort(ids_shuffle, axis=1)

        # `ids_keep` are the indices of the sequence that will be kept as is
        # these indices are used to get us the unmasked sequences
        ids_keep = ids_shuffle[:, :len_keep]
        sequence_unmasked = tf.gather(
            sequence,
            axis=1,
            batch_dims=1,
            indices=ids_keep,
        )

        # generate the binary mask: 0 is keep, 1 is remove
        # this hack is needed because TF's EagerTensors don't support
        # assignment
        mask_keep = tf.zeros((batch_size, len_keep))
        mask_remove = tf.ones((batch_size, seq_length - len_keep))
        mask = tf.concat([mask_keep, mask_remove], axis=-1)

        # the mask is a binary mask that at this stage is a sorted array (mask_keep, mask_reomove)
        # we would need to shuffle the sorted mask in a way to align it upon the uniform distribution
        # this is where the `ids_restore` comes to play
        mask = tf.gather(mask, axis=1, batch_dims=1, indices=ids_restore)

        return sequence_unmasked, mask, ids_restore

    def call(self, pixel_values: tf.Tensor, noise: tf.Tensor = None) -> tf.Tensor:
        # embeddings.shape = (batch_size, num_patches, hidden_size)
        embeddings = self.patch_embeddings(pixel_values)

        # add position embeddings w/o cls token
        embeddings = embeddings + self.position_embeddings[:, 1:, :]

        # masking: length -> length * config.mask_ratio
        embeddings, mask, ids_restore = self.random_masking(
            embeddings,
        )

        # append cls token
        cls_token = self.cls_token + self.position_embeddings[:, :1, :]
        cls_tokens = tf.tile(cls_token, (shape_list(embeddings)[0], 1, 1))
        embeddings = tf.concat([cls_tokens, embeddings], axis=1)

        return embeddings, mask, ids_restore
