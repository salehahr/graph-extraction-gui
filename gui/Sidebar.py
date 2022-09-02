from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDockWidget,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from .DataContainer import DataContainer

from config import IMAGE_DIRECTORY


class Sidebar(QDockWidget):
    def __init__(self, data_container: DataContainer, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._file_browser = FileBrowser("File browser", data_container)

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


class SideBarWidget(QGroupBox):
    def __init__(self, title: str, data_container: DataContainer, *args, **kwargs):
        super().__init__(title, *args, **kwargs)

        self._data_container: DataContainer = data_container

        self._init_layout()

    def _init_layout(self):
        layout = QGridLayout()
        self.setLayout(layout)


class FileBrowser(SideBarWidget):
    def __init__(self, *args, **kwargs):
        self._directory_field: Optional[QLineEdit] = None
        self._current_directory: str = IMAGE_DIRECTORY

        super().__init__(*args, **kwargs)

    def _init_layout(self):
        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(self._browse)

        self._directory_field = QLineEdit(self.current_directory)
        self._directory_field.returnPressed.connect(self._update_directory)

        layout = QGridLayout()
        layout.addWidget(QLabel("Directory:"), 0, 0)
        layout.addWidget(self._directory_field, 1, 0)
        layout.addWidget(browse_button, 2, 0)

        self.setLayout(layout)

    def _browse(self):
        self._update_directory()

        dialog = QFileDialog()
        dialog.setFileMode(QFileDialog.ExistingFile)
        dialog.setDirectory(os.path.abspath(self.current_directory))
        dialog.setNameFilter("Images (*.png)")
        dialog.setViewMode(QFileDialog.Detail)

        if dialog.exec_():
            filenames = dialog.selectedFiles()
            self.current_filepath = filenames[0]

    def _update_directory(self):
        new_dir = self._directory_field.text()

        if os.path.exists(new_dir):
            self.current_directory = new_dir

    @property
    def current_directory(self):
        return self._current_directory

    @current_directory.setter
    def current_directory(self, new_dir):
        if os.path.abspath(self._current_directory) == os.path.abspath(new_dir):
            return

        self._current_directory = new_dir
        self._directory_field.setText(self.current_directory)

    @property
    def current_filepath(self):
        return self._data_container.current_image_filepath

    @current_filepath.setter
    def current_filepath(self, filepath):
        self._data_container.current_image_filepath = filepath
        self.current_directory = os.path.split(
            os.path.relpath(self.current_filepath, start="./")
        )[0]
