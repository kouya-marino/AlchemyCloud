"""Reusable Qt widgets for the LiDAR PCD App."""
from __future__ import annotations

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets


class ToggleSwitch(QtWidgets.QAbstractButton):
    """A modern iOS-style toggle switch.

    - Rounded track interpolates colour between *off* (grey) and *on* (green)
      as the knob slides.
    - Knob position is animated via QPropertyAnimation; the same property is
      also driven directly by the toggled state for instant feedback.
    - Use `toggled` (inherited) to react to state changes.
    """

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        *,
        width: int = 46,
        height: int = 22,
        off_color: QtGui.QColor | None = None,
        on_color: QtGui.QColor | None = None,
    ):
        super().__init__(parent)
        self.setCheckable(True)
        self.setCursor(QtCore.Qt.PointingHandCursor)
        self.setFixedSize(width, height)
        self._w = width
        self._h = height
        self._knob_pos = 0.0
        self._off_color = off_color or QtGui.QColor(100, 100, 100)
        self._on_color = on_color or QtGui.QColor(70, 170, 110)
        self._knob_color = QtGui.QColor(245, 245, 245)

        self._anim = QtCore.QPropertyAnimation(self, b"knobPos", self)
        self._anim.setDuration(140)
        self._anim.setEasingCurve(QtCore.QEasingCurve.OutCubic)
        self.toggled.connect(self._on_toggled)

    # -- animated property -------------------------------------------------

    @QtCore.pyqtProperty(float)
    def knobPos(self) -> float:
        return self._knob_pos

    @knobPos.setter
    def knobPos(self, value: float) -> None:
        self._knob_pos = float(value)
        self.update()

    def _on_toggled(self, checked: bool) -> None:
        self._anim.stop()
        self._anim.setStartValue(self._knob_pos)
        self._anim.setEndValue(1.0 if checked else 0.0)
        self._anim.start()

    # -- painting ----------------------------------------------------------

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(self._w, self._h)

    def hitButton(self, pos: QtCore.QPoint) -> bool:
        return self.rect().contains(pos)

    def paintEvent(self, _ev):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)

        radius = self._h / 2.0
        track = QtCore.QRectF(1.0, 1.0, self._w - 2.0, self._h - 2.0)
        t = self._knob_pos
        bg = self._lerp(self._off_color, self._on_color, t)
        p.setBrush(bg)
        p.setPen(QtCore.Qt.NoPen)
        p.drawRoundedRect(track, radius - 1, radius - 1)

        knob_r = (self._h - 4.0) / 2.0
        knob_x = radius + t * (self._w - 2.0 * radius)
        knob_y = self._h / 2.0
        # Soft shadow under the knob
        shadow = QtGui.QColor(0, 0, 0, 60)
        p.setBrush(shadow)
        p.drawEllipse(QtCore.QPointF(knob_x, knob_y + 1.0), knob_r, knob_r)
        # Knob itself
        p.setBrush(self._knob_color)
        p.drawEllipse(QtCore.QPointF(knob_x, knob_y), knob_r, knob_r)
        p.end()

    @staticmethod
    def _lerp(c1: QtGui.QColor, c2: QtGui.QColor, t: float) -> QtGui.QColor:
        return QtGui.QColor(
            int(c1.red() + (c2.red() - c1.red()) * t),
            int(c1.green() + (c2.green() - c1.green()) * t),
            int(c1.blue() + (c2.blue() - c1.blue()) * t),
        )


# ---------------------------------------------------------------------------
# Colorbar legend (RViz-style)
# ---------------------------------------------------------------------------

class ColorbarLegend(QtWidgets.QWidget):
    """A vertical viridis (or arbitrary matplotlib) colorbar with min/mid/max
    tick labels and an axis label at the top. Designed to be a mouse-
    transparent child overlay on top of a 3D viewer."""

    BAR_WIDTH = 18
    MARGIN_TOP = 22
    MARGIN_BOTTOM = 22
    BAR_X = 6
    TEXT_GUTTER = 56

    def __init__(self, parent: QtWidgets.QWidget | None = None, *, cmap_name: str = "viridis"):
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(QtCore.Qt.WA_NoSystemBackground, True)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        self._cmap_name = cmap_name
        self._axis = "Z"
        self._units = "m"
        self._vmin = 0.0
        self._vmax = 1.0
        self._cached_image: QtGui.QImage | None = None
        self._cached_bar_h = -1
        self.setFixedWidth(self.BAR_X + self.BAR_WIDTH + self.TEXT_GUTTER)
        self.hide()

    def set_range(self, axis: str, vmin: float, vmax: float, units: str = "m") -> None:
        self._axis = axis.upper()
        self._units = units
        self._vmin = float(vmin)
        self._vmax = float(vmax)
        self.update()

    def show_legend(self) -> None:
        self.show()
        self.raise_()

    def hide_legend(self) -> None:
        self.hide()

    # -- painting ----------------------------------------------------------

    def _make_gradient_image(self, h: int) -> QtGui.QImage:
        if self._cached_image is not None and self._cached_bar_h == h:
            return self._cached_image
        from matplotlib import colormaps

        # Top of bar = max value, bottom = min value. linspace from 1→0.
        normed = np.linspace(1.0, 0.0, max(h, 2))
        rgba = colormaps[self._cmap_name](normed)
        rgb255 = (rgba[:, :3] * 255).astype(np.uint8)
        img = QtGui.QImage(1, len(rgb255), QtGui.QImage.Format_RGB888)
        for i, (r, g, b) in enumerate(rgb255):
            img.setPixel(0, i, QtGui.qRgb(int(r), int(g), int(b)))
        self._cached_image = img
        self._cached_bar_h = h
        return img

    def paintEvent(self, _ev):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)

        # Faint backdrop so the labels are readable over a bright cloud
        backdrop = QtCore.QRectF(2, 2, self.width() - 4, self.height() - 4)
        p.setBrush(QtGui.QColor(0, 0, 0, 110))
        p.setPen(QtCore.Qt.NoPen)
        p.drawRoundedRect(backdrop, 6, 6)

        bar_h = max(self.height() - self.MARGIN_TOP - self.MARGIN_BOTTOM, 1)
        bar_x = self.BAR_X
        bar_y = self.MARGIN_TOP
        bar_rect = QtCore.QRectF(bar_x, bar_y, self.BAR_WIDTH, bar_h)

        img = self._make_gradient_image(bar_h)
        p.drawImage(bar_rect, img)
        p.setPen(QtGui.QPen(QtGui.QColor(200, 200, 200), 1))
        p.setBrush(QtCore.Qt.NoBrush)
        p.drawRect(bar_rect)

        # Title (axis + units)
        p.setPen(QtGui.QColor(235, 235, 235))
        font = p.font()
        font.setBold(True)
        font.setPointSize(9)
        p.setFont(font)
        title = f"{self._axis} ({self._units})"
        p.drawText(
            QtCore.QRectF(0, 2, self.width(), self.MARGIN_TOP - 2),
            QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter,
            title,
        )

        # Tick labels: max at top, mid in middle, min at bottom
        font.setBold(False)
        font.setPointSize(8)
        p.setFont(font)
        ticks = [
            (0.0, self._vmax),
            (0.5, (self._vmin + self._vmax) / 2.0),
            (1.0, self._vmin),
        ]
        for frac, val in ticks:
            ty = bar_y + frac * bar_h
            p.setPen(QtGui.QPen(QtGui.QColor(210, 210, 210), 1))
            p.drawLine(QtCore.QPointF(bar_x + self.BAR_WIDTH, ty),
                       QtCore.QPointF(bar_x + self.BAR_WIDTH + 3, ty))
            p.setPen(QtGui.QColor(235, 235, 235))
            text_rect = QtCore.QRectF(
                bar_x + self.BAR_WIDTH + 5, ty - 8,
                self.width() - bar_x - self.BAR_WIDTH - 5, 16,
            )
            p.drawText(text_rect, QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter, f"{val:.2f}")

        p.end()
