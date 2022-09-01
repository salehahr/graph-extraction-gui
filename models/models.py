from typing import Tuple

import numpy as np
import tensorflow as tf


def fp_to_grayscale_img(fp: tf.Tensor) -> tf.Tensor:
    raw_img = tf.io.read_file(fp)
    unscaled_img = tf.image.decode_png(raw_img, channels=1, dtype=tf.uint8)
    return tf.image.convert_image_dtype(unscaled_img, tf.float32)


def classify(mask: np.ndarray) -> Tuple[np.ndarray, bool]:
    """Returns mask with integer classes."""
    is_binary = mask.shape[-1] <= 2

    if is_binary:
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
        mask = mask.astype(np.uint8)
    else:
        mask = tf.argmax(mask, axis=-1)
        mask = mask[..., tf.newaxis].numpy()

    return tf.cast(mask, tf.uint8)


class SavedModel(object):
    def __init__(self, filepath):
        self._model = tf.keras.models.load_model(filepath)

    @property
    def keras_model(self):
        return self._model


class NodesNN(SavedModel):
    def __init__(self, filepath):
        super(NodesNN, self).__init__(filepath)

    def predict_from_fp(self, filepath) -> Tuple:
        skel = fp_to_grayscale_img(filepath)

        pred_input = tf.expand_dims(skel, 0)
        pos, deg, types = self._model.predict(pred_input)

        skel = tf.squeeze(skel)
        pos = tf.squeeze(classify(pos))
        deg = tf.squeeze(classify(deg))

        return skel, pos, deg


class EdgeNN(SavedModel):
    def __init__(self, filepath):
        super(EdgeNN, self).__init__(filepath)
