"""Entry point for the LiDAR PCD editor + occupancy-grid exporter."""

from __future__ import annotations

import sys

from PyQt5 import QtWidgets

from .main_window import MainWindow


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    if len(sys.argv) > 1:
        w.load_path(sys.argv[1])
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
