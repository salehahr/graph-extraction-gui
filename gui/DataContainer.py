from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import tensorflow as tf
from PyQt5.QtGui import QImage, QPixmap

if TYPE_CHECKING:
    import numpy as np
    from .Viewer import Viewer

from .config import DEFAULT_FILEPATH, IMAGE_SIZE


def fp_to_grayscale_img(fp: str) -> tf.Tensor:
    raw_img = tf.io.read_file(fp)
    unscaled_img = tf.image.decode_png(raw_img, channels=1, dtype=tf.uint8)
    return tf.image.convert_image_dtype(unscaled_img, tf.float32)


def _img_to_pixmap(im_: tf.Tensor) -> QPixmap:
    img_dims = im_.shape
    if len(img_dims) < 3:
        im = tf.expand_dims(im_, -1)
    else:
        im = im_

    if im.dtype == tf.uint8:
        if im.numpy().max() == 1:
            im = 255 * im

    im = tf.image.grayscale_to_rgb(im)
    im = tf.image.convert_image_dtype(im, dtype=tf.uint8).numpy()

    height, width = img_dims[0], img_dims[1]
    qim = QImage(im.data, width, height, im.strides[0], QImage.Format_RGB888)
    return QPixmap.fromImage(qim)


class DataContainer(object):
    def __init__(self, viewer: Viewer):
        self.image_size = IMAGE_SIZE
        self._viewer = viewer

        self._current_image_filepath: str = DEFAULT_FILEPATH
        self._node_pos_tensor: Optional[tf.Tensor] = None
        self._adjacency_matrix: Optional[np.ndarray] = None

    @property
    def current_image_filepath(self) -> str:
        return self._current_image_filepath

    @current_image_filepath.setter
    def current_image_filepath(self, filepath: str) -> None:
        self._current_image_filepath = filepath
        self._viewer.update_skel_image()

    @property
    def skel_image(self) -> QPixmap:
        return QPixmap(self.current_image_filepath)

    @property
    def skel_image_tensor(self) -> tf.Tensor:
        return fp_to_grayscale_img(self.current_image_filepath)

    @property
    def node_pos_tensor(self) -> tf.Tensor:
        return self._node_pos_tensor

    @node_pos_tensor.setter
    def node_pos_tensor(self, value: tf.Tensor) -> None:
        self._node_pos_tensor = value
        self._viewer.update_node_pos_image()

    @property
    def node_pos_image(self) -> QPixmap:
        return _img_to_pixmap(self.node_pos_tensor)

    @property
    def adjacency_matrix(self) -> np.ndarray:
        return self._adjacency_matrix

    @adjacency_matrix.setter
    def adjacency_matrix(self, value: np.ndarray):
        self._adjacency_matrix = value
