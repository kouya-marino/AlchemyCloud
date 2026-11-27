"""Quality-of-life features (#13).

* Last folder, recent files and window layout survive a restart.
* A point-cloud file dropped on the window opens (after the unsaved-edits
  prompt).
* E / C / M toggle eraser / crop / marker; Esc leaves the active tool.
* Ctrl+S writes straight to the file last saved this session; the first save
  asks, so the opened scan is never overwritten silently.
* ``alchemycloud-cli`` runs the pipeline headless, steps in command-line order,
  without importing Qt.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

import numpy as np
import open3d as o3d
import pytest
import yaml
from PyQt5 import QtCore, QtGui, QtTest, QtWidgets

from alchemycloud import cli, pcd_ops
from alchemycloud.main_window import MainWindow

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _cloud(n=3000, seed=0):
    pts = np.random.default_rng(seed).uniform(0, 3, (n, 3))
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))


def _write(path, n=3000, seed=0):
    o3d.io.write_point_cloud(str(path), _cloud(n, seed))
    return str(path)


def _window():
    w = MainWindow()
    w.prompt_unsaved_on_close = False
    return w


def _wait_idle(qapp, w, timeout_s=10.0):
    t0 = time.monotonic()
    while w._busy and time.monotonic() - t0 < timeout_s:
        qapp.processEvents()
        time.sleep(0.01)
    qapp.processEvents()
    assert not w._busy


def _load(qapp, w, path):
    w.load_path(path)
    _wait_idle(qapp, w)


# ---------------------------------------------------------------------------
# remembered settings
# ---------------------------------------------------------------------------


def test_recent_files_and_last_folder_survive_a_restart(qapp, tmp_path):
    a, b = _write(tmp_path / "a.pcd"), _write(tmp_path / "b.pcd", seed=1)
    w = _window()
    _load(qapp, w, a)
    _load(qapp, w, b)
    _load(qapp, w, a)  # re-opening moves it to the top, no duplicate
    w.close()
    w2 = _window()
    assert w2.recent_files() == [os.path.abspath(a), os.path.abspath(b)]
    assert w2.last_dir() == str(tmp_path)
    labels = [act.text() for act in w2._recent_menu.actions() if not act.isSeparator()]
    assert labels[:2] == ["&1  a.pcd", "&2  b.pcd"] and labels[-1] == "Clear Recent"
    w2.close()


def test_recent_list_is_capped(qapp, tmp_path):
    w = _window()
    for i in range(MainWindow.RECENT_MAX + 3):
        w._remember_file(str(tmp_path / f"f{i}.pcd"))
    recent = w.recent_files()
    assert len(recent) == MainWindow.RECENT_MAX
    assert recent[0].endswith(f"f{MainWindow.RECENT_MAX + 2}.pcd")
    w.close()


def test_open_recent_drops_a_missing_file_without_loading(qapp, tmp_path):
    w = _window()
    gone = str(tmp_path / "gone.pcd")
    w._remember_file(gone)
    w.open_recent(gone)
    assert not w._busy and w.current_pcd is None
    assert gone not in w.recent_files()
    assert "removed from Recent" in w.statusBar().currentMessage()
    w.close()


def test_open_recent_loads_the_file(qapp, tmp_path):
    p = _write(tmp_path / "r.pcd")
    w = _window()
    w._remember_file(p)
    w.open_recent(os.path.abspath(p))
    _wait_idle(qapp, w)
    assert w.current_path == os.path.abspath(p) and len(w.current_pcd.points) == 3000
    w.close()


def test_open_dialog_starts_in_the_last_folder(qapp, tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(
        QtWidgets.QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (seen.append(a[2]), ("", ""))[1])
    )
    w = _window()
    w._remember_dir(str(tmp_path / "x.pcd"))
    w.action_open()
    assert seen == [str(tmp_path)]
    w.close()


def test_window_layout_survives_a_restart(qapp):
    w = _window()
    w.show()
    w.resize(1000, 640)
    marker_tb = w.findChild(QtWidgets.QToolBar, "MarkerToolbar")
    marker_tb.hide()
    qapp.processEvents()
    w.close()
    w2 = _window()
    w2.show()
    qapp.processEvents()
    assert w2.size() == QtCore.QSize(1000, 640)
    assert not w2.findChild(QtWidgets.QToolBar, "MarkerToolbar").isVisible()
    w2.close()


# ---------------------------------------------------------------------------
# quick save
# ---------------------------------------------------------------------------


@pytest.fixture
def save_dialog(monkeypatch):
    """Patch the Save dialog: returns the queued answers, records each call."""
    state = {"answers": [], "calls": 0}

    def fake(*_a, **_k):
        state["calls"] += 1
        return (state["answers"].pop(0) if state["answers"] else ""), ""

    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", staticmethod(fake))
    return state


def test_first_save_asks_then_ctrl_s_writes_straight_there(qapp, tmp_path, save_dialog):
    src = _write(tmp_path / "scan.pcd")
    out = str(tmp_path / "scan_clean.pcd")
    w = _window()
    _load(qapp, w, src)
    save_dialog["answers"] = [out]
    w.action_save()  # first save: asks
    _wait_idle(qapp, w)
    assert save_dialog["calls"] == 1 and os.path.isfile(out)

    w._apply_async("Voxel", lambda: pcd_ops.voxel_downsample(w.current_pcd, 0.3))
    _wait_idle(qapp, w)
    assert w._dirty
    w.action_save()  # Ctrl+S: no dialog
    _wait_idle(qapp, w)
    assert save_dialog["calls"] == 1
    assert not w._dirty
    assert len(o3d.io.read_point_cloud(out).points) == len(w.current_pcd.points) < 3000
    assert len(o3d.io.read_point_cloud(src).points) == 3000  # the scan is untouched
    w.close()


def test_save_as_always_asks(qapp, tmp_path, save_dialog):
    w = _window()
    _load(qapp, w, _write(tmp_path / "scan.pcd"))
    save_dialog["answers"] = [str(tmp_path / "one.pcd"), str(tmp_path / "two.pcd")]
    w.action_save_as()
    _wait_idle(qapp, w)
    w.action_save_as()
    _wait_idle(qapp, w)
    assert save_dialog["calls"] == 2
    assert w._save_target == (str(tmp_path / "two.pcd"), False)
    w.close()


def test_quick_save_keeps_the_ascii_format(qapp, tmp_path, save_dialog):
    out = str(tmp_path / "ascii.pcd")
    w = _window()
    _load(qapp, w, _write(tmp_path / "scan.pcd"))
    save_dialog["answers"] = [out]
    w.action_save_as(ascii_format=True)
    _wait_idle(qapp, w)
    os.remove(out)
    w.action_save()
    _wait_idle(qapp, w)
    with open(out, "rb") as f:
        assert b"DATA ascii" in f.read(400)
    w.close()


def test_opening_another_file_forgets_the_save_target(qapp, tmp_path, save_dialog):
    w = _window()
    _load(qapp, w, _write(tmp_path / "a.pcd"))
    save_dialog["answers"] = [str(tmp_path / "a_clean.pcd")]
    w.action_save()
    _wait_idle(qapp, w)
    _load(qapp, w, _write(tmp_path / "b.pcd", seed=1))
    assert w._save_target is None
    w.action_save()  # asks again rather than writing b over a_clean.pcd
    assert save_dialog["calls"] == 2
    w.close()


# ---------------------------------------------------------------------------
# drag and drop
# ---------------------------------------------------------------------------


def _mime(*paths):
    m = QtCore.QMimeData()
    m.setUrls([QtCore.QUrl.fromLocalFile(p) for p in paths])
    return m


def _drag_enter(w, mime):
    ev = QtGui.QDragEnterEvent(
        QtCore.QPoint(5, 5), QtCore.Qt.CopyAction, mime, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier
    )
    w.dragEnterEvent(ev)
    return ev.isAccepted()


def test_drag_accepts_point_clouds_only(qapp, tmp_path):
    w = _window()
    assert _drag_enter(w, _mime(str(tmp_path / "scan.PCD")))
    assert _drag_enter(w, _mime(str(tmp_path / "notes.txt"), str(tmp_path / "scan.ply")))
    assert not _drag_enter(w, _mime(str(tmp_path / "notes.txt")))
    w._busy = True
    assert not _drag_enter(w, _mime(str(tmp_path / "scan.pcd")))
    w._busy = False
    w.close()


def test_dropping_a_file_opens_it_after_the_unsaved_prompt(qapp, tmp_path, monkeypatch):
    asked = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "warning",
        staticmethod(lambda *a, **k: (asked.append(1), QtWidgets.QMessageBox.Discard)[1]),
    )
    w = _window()
    _load(qapp, w, _write(tmp_path / "a.pcd"))
    w._dirty = True
    w.prompt_unsaved_on_close = True
    dropped = _write(tmp_path / "b.pcd", n=1234)
    mime = _mime(dropped)  # the event keeps only a raw pointer: hold a reference
    ev = QtGui.QDropEvent(QtCore.QPointF(5, 5), QtCore.Qt.CopyAction, mime, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier)
    w.dropEvent(ev)
    assert ev.isAccepted()
    t0 = time.monotonic()
    while w.current_path != dropped and time.monotonic() - t0 < 10:
        qapp.processEvents()
        time.sleep(0.01)
    _wait_idle(qapp, w)
    assert asked == [1]
    assert w.current_path == dropped and len(w.current_pcd.points) == 1234
    w.prompt_unsaved_on_close = False
    w.close()


# ---------------------------------------------------------------------------
# tool shortcuts
# ---------------------------------------------------------------------------


def test_tool_shortcuts_are_assigned():
    w = _window()
    assert w._a_eraser.shortcut() == QtGui.QKeySequence("E")
    assert w._a_crop.shortcut() == QtGui.QKeySequence("C")
    assert w._a_marker.shortcut() == QtGui.QKeySequence("M")
    assert w._a_leave_tool.shortcut() == QtGui.QKeySequence(QtCore.Qt.Key_Escape)
    w.close()


def test_keys_toggle_tools_and_escape_leaves_them(qapp):
    w = _window()
    pcd = _cloud()
    w.current_pcd = pcd
    w.viewer.set_cloud(pcd)
    w._refresh_state()
    w.show()
    w.activateWindow()
    if not QtTest.QTest.qWaitForWindowActive(w, 2000):
        w.close()
        pytest.skip("window cannot become active on this platform")
    QtTest.QTest.keyClick(w, QtCore.Qt.Key_E)
    assert w._a_eraser.isChecked() and w.viewer._eraser_mode
    QtTest.QTest.keyClick(w, QtCore.Qt.Key_C)  # switching tools is exclusive
    assert w._a_crop.isChecked() and not w._a_eraser.isChecked()
    QtTest.QTest.keyClick(w, QtCore.Qt.Key_Escape)
    assert not w._a_crop.isChecked() and not w.viewer.is_crop_mode()
    QtTest.QTest.keyClick(w, QtCore.Qt.Key_M)
    assert w._a_marker.isChecked() and w.viewer.is_marker_mode()
    QtTest.QTest.keyClick(w, QtCore.Qt.Key_Escape)
    assert not w._a_marker.isChecked() and not w.viewer.is_marker_mode()
    w.close()


def test_escape_cancels_a_crop_drag(qapp):
    w = _window()
    pcd = _cloud()
    w.current_pcd = pcd
    w.viewer.set_cloud(pcd)
    w._refresh_state()
    w._a_crop.setChecked(True)
    w._on_crop_toggled()
    w.viewer._crop_origin = QtCore.QPoint(3, 3)  # mid-drag
    w.action_leave_tool()
    assert not w._a_crop.isChecked()
    assert w.viewer._crop_origin is None and not w.viewer.is_crop_mode()
    w.close()


# ---------------------------------------------------------------------------
# alchemycloud-cli
# ---------------------------------------------------------------------------


def test_cli_steps_run_in_command_line_order():
    args = cli._parser().parse_args(
        ["in.pcd", "--voxel", "0.1", "--zclip", "0", "1", "--level", "--remove-ground", "--voxel", "0.2"]
    )
    assert args.steps == [
        ("voxel", [0.1]),
        ("zclip", [0.0, 1.0]),
        ("level", []),
        ("remove_ground", [0.05]),
        ("voxel", [0.2]),
    ]


def test_cli_pipeline_writes_cloud_and_map(tmp_path, capsys):
    src = _write(tmp_path / "in.pcd", n=20000)
    out, mp = tmp_path / "out.pcd", tmp_path / "map.pgm"
    rc = cli.main(
        [src, "--zclip", "0.5", "2.5", "--voxel", "0.1", "-o", str(out), "--map", str(mp), "--map-threshold", "1.0"]
    )
    assert rc == 0
    want = pcd_ops.voxel_downsample(pcd_ops.z_clip(pcd_ops.load_pcd(src), 0.5, 2.5), 0.1)
    assert len(o3d.io.read_point_cloud(str(out)).points) == len(want.points)
    meta = yaml.safe_load((tmp_path / "map.yaml").read_text(encoding="utf-8"))
    assert meta["image"] == "map.pgm" and meta["resolution"] == 0.05
    text = capsys.readouterr().out
    assert text.index("--zclip 0.5 2.5") < text.index("--voxel 0.1") < text.index("saved") < text.index("map ")


def test_cli_reports_bad_parameters_as_an_error(tmp_path, capsys):
    src = _write(tmp_path / "in.pcd")
    assert cli.main([src, "--voxel", "0"]) == 1
    assert "voxel_size must be > 0" in capsys.readouterr().err


def test_cli_with_nothing_to_do_is_a_usage_error(tmp_path):
    with pytest.raises(SystemExit) as exc:
        cli.main([_write(tmp_path / "in.pcd")])
    assert exc.value.code == 2


def test_cli_never_imports_qt(tmp_path):
    src = _write(tmp_path / "in.pcd")
    code = (
        "import sys; from alchemycloud import cli; "
        f"rc = cli.main([{src!r}, '--voxel', '0.2', '--map', {str(tmp_path / 'm.pgm')!r}]); "
        "assert rc == 0; assert 'PyQt5' not in sys.modules, 'Qt was imported'"
    )
    env = {**os.environ, "PYTHONPATH": ROOT}
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=120)
    assert res.returncode == 0, res.stderr
