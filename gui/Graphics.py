from typing import Optional

from PyQt5.QtWidgets import QHBoxLayout, QLabel

from .config import DEFAULT_FILEPATH, IMAGE_SIZE
from .Panes import Panes


class Graphics(QLabel):
    def __init__(self, *args, **kwargs):
        super(Graphics, self).__init__(*args, **kwargs)

        # defaults
        self._image_size = IMAGE_SIZE
        self._current_filepath = DEFAULT_FILEPATH

        # ui elements
        self._input_pane: Optional[Panes] = None
        self._output_pane: Optional[Panes] = None

        self._init_ui()

    def _init_ui(self) -> None:
        self._input_pane = Panes(0, self._image_size)
        self._output_pane = Panes(1, self._image_size)

        layout = QHBoxLayout()
        layout.addWidget(self._input_pane)
        layout.addWidget(self._output_pane)
        self.setLayout(layout)

    def display(self, filepath_: Optional = None) -> None:
        filepath = filepath_ if filepath_ else self._current_filepath
        self._input_pane.display(filepath)

    @property
    def height(self) -> int:
        return self.layout().sizeHint().height()

    @property
    def width(self) -> int:
        return self.layout().sizeHint().width()

    @property
    def current_filepath(self):
        return self._current_filepath

    @current_filepath.setter
    def current_filepath(self, filepath):
        self._current_filepath = filepath
        self.display(filepath)
