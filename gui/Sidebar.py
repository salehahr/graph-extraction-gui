from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDockWidget,
    QFileDialog,
    QGridLayout,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from .DataContainer import DataContainer


class Sidebar(QDockWidget):
    def __init__(self, data_container: DataContainer, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._file_browser = FileBrowser(data_container)

        self._init_layout()

        self.width: int = self.sizeHint().width()
        self.height: int = self.sizeHint().height()

    def _init_layout(self) -> None:
        # remove title bar
        self.setTitleBarWidget(QWidget())
        self.setFeatures(QDockWidget.NoDockWidgetFeatures)

        layout = QVBoxLayout()
        layout.addWidget(self._file_browser)

        multi_widget = QWidget()
        multi_widget.setLayout(layout)

        self.setWidget(multi_widget)

        layout.setAlignment(Qt.AlignTop)


class FileBrowser(QWidget):
    def __init__(self, data_container: DataContainer, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._data_container: DataContainer = data_container

        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(self._browse)

        layout = QGridLayout()
        layout.addWidget(browse_button, 0, 0)

        self.setLayout(layout)

    def _browse(self):
        dialog = QFileDialog()
        dialog.setFileMode(QFileDialog.ExistingFile)
        dialog.setDirectory("./img/")
        dialog.setNameFilter("Images (*.png)")
        dialog.setViewMode(QFileDialog.Detail)

        if dialog.exec_():
            filenames = dialog.selectedFiles()
            self.current_filepath = filenames[0]

    @property
    def current_filepath(self):
        return self._data_container.current_image_filepath

    @current_filepath.setter
    def current_filepath(self, filepath):
        self._data_container.current_image_filepath = filepath
