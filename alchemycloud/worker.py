"""Tiny QRunnable-based worker wrapper for off-main-thread work.

Open3D operations release the GIL in C++, so heavy work (load_pcd, voxel
downsample, RANSAC, etc.) runs on a worker thread without blocking the Qt
event loop — that's what keeps the loading spinner animating smoothly.

Signals are emitted via a separate `QObject` (`WorkerSignals`) because
`QRunnable` is not a `QObject` in Qt.
"""

from __future__ import annotations

import traceback
from typing import Any, Callable

from PyQt5 import QtCore


class WorkerSignals(QtCore.QObject):
    success = QtCore.pyqtSignal(object)
    error = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal()


class Worker(QtCore.QRunnable):
    def __init__(self, fn: Callable[..., Any], *args, **kwargs):
        super().__init__()
        # Don't let QThreadPool auto-delete us. With autoDelete=True the C++
        # QRunnable (and the WorkerSignals child) can be freed before queued
        # cross-thread `success`/`finished` signals are delivered to the
        # main thread, leading to a "WorkerSignals has been deleted" crash.
        # Caller is responsible for dropping the Python reference after the
        # `finished` signal fires.
        self.setAutoDelete(False)
        self._fn = fn
        self._args = args
        self._kwargs = kwargs
        self.signals = WorkerSignals()

    @QtCore.pyqtSlot()
    def run(self) -> None:
        try:
            result = self._fn(*self._args, **self._kwargs)
        except Exception as e:
            self.signals.error.emit(f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
        else:
            self.signals.success.emit(result)
        finally:
            self.signals.finished.emit()


def run_async(
    fn: Callable[..., Any],
    on_success: Callable[[Any], None],
    on_error: Callable[[str], None] | None = None,
    on_finished: Callable[[], None] | None = None,
) -> Worker:
    """Schedule `fn` on the global QThreadPool. Returns the Worker so the
    caller can keep a reference (signals stay alive)."""
    worker = Worker(fn)
    worker.signals.success.connect(on_success)
    if on_error is not None:
        worker.signals.error.connect(on_error)
    if on_finished is not None:
        worker.signals.finished.connect(on_finished)
    QtCore.QThreadPool.globalInstance().start(worker)
    return worker
