from functools import partial

import numpy as np
import tensorflow as tf

from .augmentations import custom_augment

AUTOTUNE = tf.data.AUTOTUNE


class GetMSNDataloader:
    def __init__(self, args):
        self.args = args.dataset_config

    def get_dataloader(self, image_paths):
        # Load the images
        img_file_loader = tf.data.Dataset.from_tensor_slices((image_paths))

        # Preprocess images
        dataloader = img_file_loader.map(
            self.preprocess_image, num_parallel_calls=AUTOTUNE
        )

        # Get options
        if self.args.use_options:
            options = self.get_options()

        # Get multi view dataloaders
        loaders = self.multiview_dataloaders(
            dataloader, self.args.size_crops, self.args.num_crops, options=options
        )

        # Zip the multi view dataloaders together
        loaders_zipped = tf.data.Dataset.zip(loaders)

        # Final trainloader
        loaders_zipped = (
            loaders_zipped.shuffle(self.args.shuffle_buffer)
            .batch(self.args.batch_size)
            .prefetch(AUTOTUNE)
        )

        return loaders_zipped

    def preprocess_image(self, path):
        # Parse Image
        image = tf.io.read_file(path)
        image = self.decode_image(image)

        return image

    def decode_image(self, img):
        # Decode
        img = tf.image.decode_jpeg(img, channels=3)
        # Normalize image
        img = tf.image.convert_image_dtype(img, dtype=tf.float32)

        return img

    def multiview_dataloaders(self, dataloader, size_crops, num_crops, options=None):
        loaders = tuple()
        for i, num_crop in enumerate(num_crops):
            if size_crops[i] == 224:
                mode = "random"
            else:
                mode = "focal"
            for _ in range(num_crop):
                loader = (
                    dataloader
                    # TODO: Add augmentations using KerasCV
                    .map(
                        lambda x: custom_augment(x, mode=mode),
                        num_parallel_calls=AUTOTUNE,
                    )
                )
                if options is not None:
                    loader = loader.with_options(options)
                loaders += (loader,)

        return loaders

    def get_options(self):
        # Experimental options
        options = tf.data.Options()
        options.experimental_optimization.noop_elimination = True
        options.experimental_optimization.apply_default_optimizations = True
        options.experimental_deterministic = True
        options.threading.max_intra_op_parallelism = 1

        return options
