from typing import Tuple

import numpy as np
import tensorflow as tf


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


def get_edgenn_caller(
    model: tf.keras.models.Model,
) -> tf.types.experimental.ConcreteFunction:
    def evaluate(
        model: tf.keras.models.Model,
        skel_img: tf.Tensor,
        node_pos: tf.Tensor,
        combo_img: tf.Tensor,
    ):
        return model((skel_img, node_pos, combo_img), training=False)

    return tf.function(evaluate).get_concrete_function(
        model=model,
        skel_img=tf.TensorSpec(shape=(None, 256, 256), dtype=tf.float32),
        node_pos=tf.TensorSpec(shape=(None, 256, 256), dtype=tf.uint8),
        combo_img=tf.TensorSpec(shape=(None, 256, 256), dtype=tf.int64),
    )


class SavedModel(object):
    def __init__(self, filepath):
        self._model = tf.keras.models.load_model(filepath)

    @property
    def keras_model(self):
        return self._model


class NodesNN(SavedModel):
    def __init__(self, filepath):
        super(NodesNN, self).__init__(filepath)

    def predict_from_skel(self, skel: tf.Tensor) -> Tuple:
        pred_input = tf.expand_dims(skel, 0)
        pos, deg, types = self._model.predict(pred_input)

        skel = tf.squeeze(skel)
        pos = tf.squeeze(classify(pos))
        deg = tf.squeeze(classify(deg))

        return skel, pos, deg


class EdgeNN(SavedModel):
    def __init__(self, filepath):
        super(EdgeNN, self).__init__(filepath)
