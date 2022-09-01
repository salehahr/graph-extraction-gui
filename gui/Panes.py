import os
from enum import Enum
from typing import Optional

import tensorflow as tf
from PyQt5.QtCore import QSize
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QLabel, QVBoxLayout


def _img_to_pixmap(im_):
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


class PanesEnum(Enum):
    INPUT = 0
    OUTPUT = 1


class Panes(QLabel):
    def __init__(self, _id: int, image_size: int, *args, **kwargs):
        super(Panes, self).__init__(*args, **kwargs)
        self.type = PanesEnum(_id)
        self.id = _id

        self._descriptor = "Input" if self.type == PanesEnum.INPUT else "Output"
        self._image_height: int = image_size
        self._text_height: int = 20

        # ui elements
        self._image: Optional[QLabel] = None
        self._text: Optional[QLabel] = None

        self._init_layout()
        self.setFixedWidth(self.width)
        self.setFixedHeight(self.height)

    def _init_layout(self) -> None:
        self._image = QLabel()
        self._image.setFixedWidth(self._image_height)
        self._image.setFixedHeight(self._image_height)

        self._text = QLabel()
        self._text.setFixedHeight(self._text_height)

        descriptor = QLabel(self._descriptor)
        descriptor.setFixedHeight(self._text_height)

        layout = QVBoxLayout()
        layout.addWidget(descriptor)
        layout.addWidget(self._image)
        layout.addWidget(self._text)
        self.setLayout(layout)

    def display(self, filepath: str) -> None:
        self.display_image(filepath)
        self._display_text(filepath)

    def display_image(self, img) -> None:
        if isinstance(img, str):
            self._image.setPixmap(QPixmap(img))
        else:
            self._image.setPixmap(_img_to_pixmap(img))

    def _display_text(self, text):
        text = os.path.split(text)[1]
        text = os.path.splitext(text)[0]

        self._text.setText(text)

    def sizeHint(self) -> QSize:
        width = self._image_height
        height = self._image_height + self._text_height
        return QSize(width, height)

    @property
    def width(self) -> int:
        return self.layout().sizeHint().width()

    @property
    def height(self) -> int:
        return self.layout().sizeHint().height()
