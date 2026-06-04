"""Semi-transparent loading overlay with a rotating arc spinner.

The overlay is a child of the central widget (the EmbeddedViewer). It tracks
its parent's size via an event filter and blocks mouse input on the canvas
while visible so the user cannot orbit / crop / drag-marker during work.
"""
from __future__ import annotations

from PyQt5 import QtCore, QtGui, QtWidgets


class LoadingOverlay(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget):
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WA_NoSystemBackground, True)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        self.setFocusPolicy(QtCore.Qt.NoFocus)
        # Eat mouse events so they don't reach the canvas while busy.
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, False)

        self._angle = 0
        self._message = "Loading…"
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._tick)

        parent.installEventFilter(self)
        self.resize(parent.size())
        self.hide()

    # -- public API --------------------------------------------------------

    def start(self, message: str | None = None) -> None:
        if message:
            self._message = message
        if not self.isVisible():
            self.resize(self.parentWidget().size())
            self.show()
            self.raise_()
        self._timer.start(40)

    def stop(self) -> None:
        self._timer.stop()
        self.hide()

    def set_message(self, message: str) -> None:
        self._message = message
        if self.isVisible():
            self.update()

    # -- behaviour ---------------------------------------------------------

    def eventFilter(self, obj, event):
        if obj is self.parentWidget() and event.type() == QtCore.QEvent.Resize:
            self.resize(self.parentWidget().size())
        return False

    def mousePressEvent(self, ev):  # swallow clicks while busy
        ev.accept()

    def mouseMoveEvent(self, ev):
        ev.accept()

    def mouseReleaseEvent(self, ev):
        ev.accept()

    def _tick(self):
        self._angle = (self._angle + 8) % 360
        self.update()

    def paintEvent(self, _ev):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)

        # Dim backdrop
        p.fillRect(self.rect(), QtGui.QColor(0, 0, 0, 130))

        # Spinner: arc that rotates
        cx, cy = self.width() / 2.0, self.height() / 2.0
        r = 32.0
        rect = QtCore.QRectF(cx - r, cy - r, 2 * r, 2 * r)

        # Faint full ring (track)
        pen_track = QtGui.QPen(QtGui.QColor(255, 255, 255, 50), 4)
        pen_track.setCapStyle(QtCore.Qt.RoundCap)
        p.setPen(pen_track)
        p.drawArc(rect, 0, 360 * 16)

        # Bright sweeping arc
        pen_arc = QtGui.QPen(QtGui.QColor(255, 200, 0), 4)
        pen_arc.setCapStyle(QtCore.Qt.RoundCap)
        p.setPen(pen_arc)
        # Qt arc angles are 1/16 degree. Start at current angle, sweep 270°.
        p.drawArc(rect, int(-self._angle * 16), int(270 * 16))

        # Message
        p.setPen(QtGui.QColor(235, 235, 235))
        font = p.font()
        font.setPointSize(11)
        font.setBold(True)
        p.setFont(font)
        text_rect = QtCore.QRectF(0, cy + r + 12, self.width(), 30)
        p.drawText(text_rect, QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop, self._message)
        p.end()
