"""Error reporting and small fixes from the November review (#11).

* A failed operation shows its one-line reason; the traceback is under
  "Show Details…" and in the log.
* Every background failure is logged with its traceback.
* Undo / Redo / Reset change the history only once their worker succeeds.
* The map YAML is written as UTF-8 whatever the platform default.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import pytest
import yaml
from PyQt5 import QtWidgets

from alchemycloud import occupancy, pcd_ops
from alchemycloud.main_window import MainWindow
from alchemycloud.worker import split_error


def _cloud(n=3000, seed=0):
    pts = np.random.default_rng(seed).uniform(0, 3, (n, 3))
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))


def _window(pcd):
    w = MainWindow()
    w.prompt_unsaved_on_close = False
    w.current_pcd = pcd
    w.original_pcd = pcd
    w.viewer.set_cloud(pcd)
    w._refresh_state()
    return w


def _wait_idle(qapp, w, timeout_s=10.0):
    t0 = time.monotonic()
    while w._busy and time.monotonic() - t0 < timeout_s:
        qapp.processEvents()
        time.sleep(0.01)
    qapp.processEvents()
    assert not w._busy


def _worker_error_text(exc: Exception) -> str:
    """The exact string Worker emits for `exc`."""
    import traceback

    try:
        raise exc
    except Exception as e:
        return f"{type(e).__name__}: {e}\n{traceback.format_exc()}"


# ---------------------------------------------------------------------------
# readable error dialogs
# ---------------------------------------------------------------------------


def test_split_error_separates_the_reason_from_the_traceback():
    text = _worker_error_text(ValueError("No near-horizontal plane found"))
    summary, details = split_error(text)
    assert summary == "ValueError: No near-horizontal plane found"
    assert details == text and "Traceback (most recent call last)" in details


def test_split_error_leaves_a_plain_message_alone():
    assert split_error("Operation produced an empty cloud — ignored.") == (
        "Operation produced an empty cloud — ignored.",
        None,
    )


def test_failure_dialog_shows_the_reason_with_the_traceback_under_details(qapp, monkeypatch):
    shown = []

    def fake_exec(box):
        shown.append((box.windowTitle(), box.text(), box.detailedText()))
        return QtWidgets.QMessageBox.Ok

    monkeypatch.setattr(QtWidgets.QMessageBox, "exec_", fake_exec)
    w = _window(_cloud())

    def boom():
        raise ValueError("crop to the floor area first")

    w._apply_async("Level ground", boom)
    _wait_idle(qapp, w)
    assert len(shown) == 1
    title, text, details = shown[0]
    assert title == "Level ground failed"
    assert text == "ValueError: crop to the floor area first"
    assert "Traceback (most recent call last)" not in text
    assert "Traceback (most recent call last)" in details and "boom" in details
    w.close()


def test_plain_warnings_still_use_a_simple_message_box(qapp, monkeypatch):
    shown = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *a, **k: shown.append(a[1:3]))
    w = _window(_cloud())
    w._warn("Tilt measurement", "Floor 0.2° from horizontal")
    assert shown == [("Tilt measurement", "Floor 0.2° from horizontal")]
    w.close()


# ---------------------------------------------------------------------------
# background failures are logged
# ---------------------------------------------------------------------------


def test_background_failure_is_logged_with_its_traceback(qapp, monkeypatch, caplog):
    w = _window(_cloud())
    monkeypatch.setattr(w, "_warn", lambda *a: None)

    def boom():
        raise RuntimeError("simulated failure")

    with caplog.at_level(logging.WARNING, logger="alchemycloud.worker"):
        w._run_async(boom, lambda _r: None, message="x")
        _wait_idle(qapp, w)
    records = [r for r in caplog.records if r.name == "alchemycloud.worker"]
    assert records, "the failure was not logged"
    assert "simulated failure" in records[0].getMessage()
    assert records[0].exc_info is not None  # traceback attached
    w.close()


# ---------------------------------------------------------------------------
# undo / redo / reset keep the stacks intact when their worker fails
# ---------------------------------------------------------------------------


def _with_history(qapp):
    """Window with one completed edit: history [a], current b."""
    a = _cloud()
    w = _window(a)
    w._apply_async("Voxel", lambda: pcd_ops.voxel_downsample(a, 0.2))
    _wait_idle(qapp, w)
    assert w.history == [a]
    return w, a, w.current_pcd


def _fail_next_prepare(w, monkeypatch):
    def boom(_pcd):
        raise RuntimeError("prepare failed")

    monkeypatch.setattr(w.viewer, "prepare_cloud", boom)
    monkeypatch.setattr(w, "_warn", lambda *a: None)


def test_failed_undo_keeps_the_history(qapp, monkeypatch):
    w, a, b = _with_history(qapp)
    _fail_next_prepare(w, monkeypatch)
    w.action_undo()
    _wait_idle(qapp, w)
    assert w.current_pcd is b
    assert w.history == [a] and w.redo_stack == []
    w.close()


def test_failed_redo_keeps_both_stacks(qapp, monkeypatch):
    w, a, b = _with_history(qapp)
    w.action_undo()
    _wait_idle(qapp, w)
    assert w.current_pcd is a and w.redo_stack == [b]
    _fail_next_prepare(w, monkeypatch)
    w.action_redo()
    _wait_idle(qapp, w)
    assert w.current_pcd is a
    assert w.history == [] and w.redo_stack == [b]
    w.close()


def test_failed_reset_adds_no_history_entry(qapp, monkeypatch):
    w, a, b = _with_history(qapp)
    _fail_next_prepare(w, monkeypatch)
    w.action_reset()
    _wait_idle(qapp, w)
    assert w.current_pcd is b
    assert w.history == [a]
    w.close()


@pytest.mark.parametrize("action", ["undo", "redo", "reset"])
def test_successful_undo_redo_reset_still_move_the_stacks(qapp, action):
    w, a, b = _with_history(qapp)
    if action == "undo":
        w.action_undo()
        _wait_idle(qapp, w)
        assert (w.current_pcd, w.history, w.redo_stack) == (a, [], [b])
    elif action == "redo":
        w.action_undo()
        _wait_idle(qapp, w)
        w.action_redo()
        _wait_idle(qapp, w)
        assert (w.current_pcd, w.history, w.redo_stack) == (b, [a], [])
    else:
        w.action_reset()
        _wait_idle(qapp, w)
        assert w.current_pcd is a and w.history == [a, b]
    w.close()


# ---------------------------------------------------------------------------
# UTF-8 map YAML
# ---------------------------------------------------------------------------


def test_map_yaml_is_written_as_utf8_explicitly(tmp_path, monkeypatch):
    """Linux defaults to UTF-8 anyway, so check the encoding is asked for,
    not just that the bytes happen to decode."""
    encodings = []
    real_open = Path.open

    def spy(self, mode="r", *a, **k):
        if str(self).endswith(".yaml") and "w" in mode:
            encodings.append(k.get("encoding"))
        return real_open(self, mode, *a, **k)

    monkeypatch.setattr(Path, "open", spy)
    pcd = _cloud()
    params = occupancy.GridParams(z_min=0.0, z_max=3.0)
    grid = occupancy.build(pcd, params)
    pgm, yml = occupancy.save(grid, str(tmp_path / "3階_地図.pgm"), params)
    assert encodings == ["utf-8"]
    meta = yaml.safe_load(yml.read_bytes().decode("utf-8"))
    assert meta["image"] == "3階_地図.pgm"
