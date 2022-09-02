import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PyQt5.QtGui import QImage, QPixmap

marker_size = 2
line_thickness = 2

MARKER_COLOUR = (232, 212, 111)
LINE_COLOUR = (214, 120, 172)


def fp_to_grayscale_img(fp: str) -> tf.Tensor:
    raw_img = tf.io.read_file(fp)
    unscaled_img = tf.image.decode_png(raw_img, channels=1, dtype=tf.uint8)
    return tf.image.convert_image_dtype(unscaled_img, tf.float32)


def draw_graph(
    base_img: np.ndarray, pos_list_xy: np.ndarray, adj_matrix: np.ndarray
) -> QPixmap:
    img = _lighten_skel_img(base_img)
    img = _draw_edges(img, pos_list_xy, adj_matrix)
    img = _draw_circles(img, pos_list_xy)
    return np_img_to_pixmap(img)


def _lighten_skel_img(img_skel: np.ndarray, black_to_grey: float = 0.7) -> np.ndarray:
    assert img_skel.dtype == np.float32
    assert np.max(img_skel) <= 1

    img = img_skel.copy() / 2

    ids = np.argwhere(img_skel == 0)
    img[ids[:, 0], ids[:, 1]] = black_to_grey
    ids = np.argwhere(img_skel == 1)
    img[ids[:, 0], ids[:, 1]] = np.mean([1 - black_to_grey, 1 / black_to_grey])

    cmap = plt.get_cmap("gray")
    return np.squeeze(cmap(img)).astype(np.float32)


def _draw_edges(
    base_img: np.ndarray, pos_list_xy: np.ndarray, adj_matrix: np.ndarray
) -> np.ndarray:
    assert base_img.dtype == np.float32
    assert np.max(base_img) <= 1

    img = rgba2rgb(base_img)
    assert img.dtype == np.uint8
    assert np.max(img) > 1

    A_triu = np.triu(adj_matrix, 0)
    combos = np.where(A_triu)
    combos = np.vstack(combos).T

    for (vi, vj) in combos:
        vi_xy = pos_list_xy[vi]
        vj_xy = pos_list_xy[vj]
        cv2.line(img, vi_xy, vj_xy, LINE_COLOUR, thickness=line_thickness)

    return img


def _draw_circles(base_img: np.ndarray, pos_list_xy: np.ndarray) -> np.ndarray:
    """
    Draws circles on the image colour coded according to the unique values
    in the classifier matrix.
    Returns BGR image.
    """
    img = base_img
    assert img.dtype == np.uint8
    assert np.max(img) > 1

    for (x, y) in pos_list_xy:
        cv2.circle(img, (x, y), marker_size, MARKER_COLOUR, -1)

    return img


def float2uintrgb(img) -> np.ndarray:
    assert img.dtype == np.float32
    assert np.max(img) <= 1
    assert img.shape[-1] == 1 or img.shape[-1] == 3

    return (255 * img.copy()).astype(np.uint8)


def rgba2rgb(rgba, background=(255, 255, 255)):
    assert rgba.dtype == np.float32
    assert np.max(rgba) <= 1
    assert rgba.shape[-1] == 4

    r, g, b, a = rgba[:, :, 0], rgba[:, :, 1], rgba[:, :, 2], rgba[:, :, 3]
    if np.max(a) > 1:
        a = np.asarray(a, dtype="float32") / 255.0

    rgb = np.zeros((*rgba.shape[0:2], 3), dtype="float32")
    R, G, B = background
    rgb[:, :, 0] = r * a + (1.0 - a) * R
    rgb[:, :, 1] = g * a + (1.0 - a) * G
    rgb[:, :, 2] = b * a + (1.0 - a) * B

    return np.asarray(255 * rgb, dtype="uint8")


def np_img_to_pixmap(im_: np.ndarray) -> QPixmap:
    assert im_.dtype == np.uint8
    assert np.max(im_) > 1

    img_dims = im_.shape
    if len(img_dims) < 3:
        im = np.expand_dims(im_, -1)
    else:
        assert img_dims[-1] == 3
        im = im_

    height, width = img_dims[:2]
    qim = QImage(im.data, width, height, im.strides[0], QImage.Format_RGB888)
    return QPixmap.fromImage(qim)


def tf_img_to_pixmap(im_: tf.Tensor) -> QPixmap:
    if len(im_.shape) < 3:
        im = tf.expand_dims(im_, -1)
    else:
        im = im_

    if im.dtype == tf.uint8:
        if im.numpy().max() == 1:
            im = 255 * im

    im = tf.image.grayscale_to_rgb(im)
    im = tf.image.convert_image_dtype(im, dtype=tf.uint8).numpy()

    return np_img_to_pixmap(im)
