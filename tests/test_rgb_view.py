"""RGB colour mode: showing a cloud's own colours instead of an axis colormap.

Covers the viewer plumbing (`set_color_axis('rgb')`, tone mapping, the
source-RGB snapshot that survives an axis round-trip) and the MainWindow
toolbar wiring (combo entry, gating of the Bright / Auto-level controls).
"""

from __future__ import annotations

import numpy as np
import open3d as o3d
import pytest

from alchemycloud.embedded_viewer import EmbeddedViewer
from alchemycloud.main_window import MainWindow

# Viridis never emits a fully saturated primary, so pure R/G/B in the output
# is proof the file's own colours reached the scatter.
PURE = {(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)}


@pytest.fixture
def rgb_cloud():
    """900 points whose colour is deliberately uncorrelated with position:
    pure red / green / blue in round-robin."""
    rng = np.random.default_rng(0)
    pts = rng.uniform(0, 3, (900, 3))
    cols = np.zeros((900, 3))
    cols[0::3, 0] = 1.0
    cols[1::3, 1] = 1.0
    cols[2::3, 2] = 1.0
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    pcd.colors = o3d.utility.Vector3dVector(cols)
    return pcd


@pytest.fixture
def viewer(qapp):
    v = EmbeddedViewer()
    v.resize(640, 480)
    return v


def _raw_rgb(viewer, pcd):
    """prepare_cloud with tone mapping neutralised, so colours pass through."""
    viewer.set_rgb_autolevel(False)
    viewer.set_rgb_gamma(1.0)
    return viewer.prepare_cloud(pcd)


# -- colour mode selection -------------------------------------------------


def test_set_color_axis_accepts_rgb(viewer):
    viewer.set_color_axis("rgb")
    assert viewer._color_axis == "rgb"


def test_set_color_axis_still_rejects_nonsense(viewer):
    viewer.set_color_axis("z")
    for bad in ("w", "none", "original", "", "xyz"):
        viewer.set_color_axis(bad)
        assert viewer._color_axis == "z"


# -- the core behaviour ----------------------------------------------------


def test_axis_mode_replaces_file_colours(viewer, rgb_cloud):
    """Regression guard for the old behaviour: Z mode must still colormap."""
    viewer.set_color_axis("z")
    _, _, rgba = viewer.prepare_cloud(rgb_cloud)
    got = {tuple(np.round(c, 3)) for c in np.unique(rgba[:, :3], axis=0)}
    assert not (PURE & got)
    assert len(got) > 100  # a continuous viridis ramp, not 3 flat colours


def test_rgb_mode_shows_the_files_own_colours(viewer, rgb_cloud):
    viewer.set_color_axis("rgb")
    _, _, rgba = _raw_rgb(viewer, rgb_cloud)
    got = {tuple(np.round(c, 3)) for c in np.unique(rgba[:, :3], axis=0)}
    assert got == PURE
    src = np.asarray(rgb_cloud.colors, dtype=np.float32)
    assert np.abs(rgba[:, :3] - src).max() < 1e-6
    assert np.all(rgba[:, 3] == 1.0)


def test_rgb_mode_does_not_mutate_the_source_cloud(viewer, rgb_cloud):
    viewer.set_color_axis("rgb")
    viewer.prepare_cloud(rgb_cloud)
    after = np.unique(np.round(np.asarray(rgb_cloud.colors), 2), axis=0)
    assert len(after) == 3  # still the three pure colours


def test_axis_mode_does_not_mutate_the_source_cloud(viewer, rgb_cloud):
    viewer.set_color_axis("z")
    viewer.prepare_cloud(rgb_cloud)
    after = np.unique(np.round(np.asarray(rgb_cloud.colors), 2), axis=0)
    assert len(after) == 3


def test_rgb_survives_a_round_trip_through_an_axis_mode(viewer, rgb_cloud):
    """RGB -> Z -> RGB must come back to the file's colours.

    colorize_by_axis mutates the display cloud in place, so this only works
    because prepare_cloud snapshots the source RGB before it runs.
    """
    viewer.set_color_axis("rgb")
    viewer.set_rgb_autolevel(False)
    viewer.set_rgb_gamma(1.0)
    viewer.set_cloud(rgb_cloud)

    viewer.set_color_axis("z")
    viewer._refresh_scatter_colors(slab_mask=None)
    mid = {tuple(np.round(c, 3)) for c in np.unique(viewer._cached_rgba[:, :3], axis=0)}
    assert not (PURE & mid)

    viewer.set_color_axis("rgb")
    viewer._refresh_scatter_colors(slab_mask=None)
    back = {tuple(np.round(c, 3)) for c in np.unique(viewer._cached_rgba[:, :3], axis=0)}
    assert back == PURE


def test_colourless_cloud_in_rgb_mode_falls_back_to_grey(viewer, cube_cloud):
    assert not cube_cloud.has_colors()
    viewer.set_color_axis("rgb")
    _, pts, rgba = viewer.prepare_cloud(cube_cloud)
    assert len(rgba) == len(pts)
    assert np.allclose(rgba[:, :3], 0.6)
    assert not viewer.has_source_colors()


def test_has_source_colors_tracks_the_displayed_cloud(viewer, rgb_cloud, cube_cloud):
    viewer.set_cloud(rgb_cloud)
    assert viewer.has_source_colors()
    viewer.set_cloud(cube_cloud)
    assert not viewer.has_source_colors()
    viewer.clear_cloud()
    assert not viewer.has_source_colors()


# -- tone mapping ----------------------------------------------------------


def test_gamma_below_one_brightens_and_above_darkens(viewer):
    viewer.set_color_axis("rgb")
    viewer.set_rgb_autolevel(False)
    mid = np.full((10, 3), 0.25, dtype=np.float32)

    viewer.set_rgb_gamma(1.0)
    neutral = viewer._tone_map(mid)
    viewer.set_rgb_gamma(0.5)
    brighter = viewer._tone_map(mid)
    viewer.set_rgb_gamma(2.0)
    darker = viewer._tone_map(mid)

    assert np.allclose(neutral, 0.25)
    assert brighter.mean() > neutral.mean()
    assert darker.mean() < neutral.mean()


def test_gamma_is_clamped_to_a_safe_range(viewer):
    viewer.set_rgb_gamma(0.0)
    assert viewer.rgb_gamma() >= 0.05
    viewer.set_rgb_gamma(1e6)
    assert viewer.rgb_gamma() <= 5.0


def test_autolevel_stretches_a_dim_cloud_towards_full_range(viewer):
    """A cloud confined to the bottom of the range should end up spanning it."""
    viewer.set_color_axis("rgb")
    viewer.set_rgb_gamma(1.0)
    dim = np.linspace(0.10, 0.30, 300, dtype=np.float32)[:, None].repeat(3, axis=1)

    viewer.set_rgb_autolevel(False)
    assert viewer._tone_map(dim).max() < 0.31

    viewer.set_rgb_autolevel(True)
    stretched = viewer._tone_map(dim)
    assert stretched.max() > 0.98
    assert stretched.min() < 0.02


def test_neutral_settings_reproduce_the_file_exactly(viewer, rgb_cloud):
    viewer.set_color_axis("rgb")
    viewer.set_rgb_autolevel(False)
    viewer.set_rgb_gamma(1.0)
    _, _, rgba = viewer.prepare_cloud(rgb_cloud)
    assert np.abs(rgba[:, :3] - np.asarray(rgb_cloud.colors, dtype=np.float32)).max() < 1e-6


def test_tone_controls_repaint_without_touching_geometry(viewer):
    """Changing gamma re-uploads colour only; positions are untouched.

    Uses midtone colours on purpose: 0.0 and 1.0 are fixed points of x**gamma,
    so a cloud of pure primaries would be gamma-invariant and prove nothing.
    """
    rng = np.random.default_rng(1)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(rng.uniform(0, 3, (400, 3))))
    pcd.colors = o3d.utility.Vector3dVector(rng.uniform(0.25, 0.75, (400, 3)))

    viewer.set_color_axis("rgb")
    viewer.set_cloud(pcd)
    viewer._refresh_scatter_colors(slab_mask=None)
    pts_before = viewer._cached_pts.copy()
    rgba_before = viewer._cached_rgba.copy()

    viewer.set_rgb_gamma(2.5)
    assert np.array_equal(viewer._cached_pts, pts_before)
    assert not np.allclose(viewer._cached_rgba[:, :3], rgba_before[:, :3])


def test_tone_controls_are_inert_in_axis_modes(viewer, rgb_cloud):
    viewer.set_color_axis("z")
    viewer.set_cloud(rgb_cloud)
    viewer._refresh_scatter_colors(slab_mask=None)
    before = viewer._cached_rgba.copy()
    viewer.set_rgb_gamma(2.5)
    viewer.set_rgb_autolevel(False)
    assert np.allclose(viewer._cached_rgba, before)


# -- legend ----------------------------------------------------------------


def test_legend_hidden_in_rgb_mode_and_shown_for_an_axis(viewer, rgb_cloud):
    viewer.set_cloud(rgb_cloud)
    viewer.set_color_axis("z")
    assert viewer._legend.isVisible() or not viewer.isVisible()
    viewer.set_color_axis("rgb")
    assert not viewer._legend.isVisible()


# -- MainWindow wiring -----------------------------------------------------


def test_toolbar_offers_rgb_and_gates_the_tone_controls(qapp, rgb_cloud):
    w = MainWindow()
    assert [w.color_combo.itemText(i) for i in range(w.color_combo.count())] == [
        "Z",
        "X",
        "Y",
        "RGB",
    ]

    w.current_pcd = rgb_cloud
    w.original_pcd = rgb_cloud
    w.viewer.set_cloud(rgb_cloud)
    w._refresh_state()
    # Axis mode: tone controls are dead.
    assert not w.rgb_gamma_spin.isEnabled()
    assert not w.rgb_auto_check.isEnabled()

    w.color_combo.setCurrentText("RGB")
    assert w.viewer._color_axis == "rgb"
    assert w.rgb_gamma_spin.isEnabled()
    assert w.rgb_auto_check.isEnabled()

    w.color_combo.setCurrentText("Z")
    assert not w.rgb_gamma_spin.isEnabled()


def test_toolbar_spins_drive_the_viewer(qapp):
    w = MainWindow()
    w.color_combo.setCurrentText("RGB")
    w.rgb_gamma_spin.setValue(1.40)
    assert w.viewer.rgb_gamma() == pytest.approx(1.40)
    w.rgb_auto_check.setChecked(False)
    assert w.viewer.rgb_autolevel() is False


def test_status_bar_warns_when_the_cloud_has_no_colour(qapp, cube_cloud, rgb_cloud):
    w = MainWindow()
    w.current_pcd = cube_cloud
    w.original_pcd = cube_cloud
    w.viewer.set_cloud(cube_cloud)
    w.color_combo.setCurrentText("RGB")
    assert "no colour" in w.statusBar().currentMessage()

    w.color_combo.setCurrentText("Z")
    w.current_pcd = rgb_cloud
    w.viewer.set_cloud(rgb_cloud)
    w.color_combo.setCurrentText("RGB")
    assert "no colour" not in w.statusBar().currentMessage()
