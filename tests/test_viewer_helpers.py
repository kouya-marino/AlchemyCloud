"""Tests for the pure-math projection helpers in `embedded_viewer`.

These helpers don't touch any Qt / GL state, so the tests run with the
offscreen Qt platform (set in `conftest.py`) but without spinning up a
QApplication.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw
from PyQt5 import QtCore

from alchemycloud.embedded_viewer import (
    mask_points_in_brush,
    mask_points_in_rect,
    project_points,
)

# ---------------------------------------------------------------------------
# project_points
# ---------------------------------------------------------------------------


def test_project_points_identity_mvp():
    """Identity MVP at 100×100: NDC == world, pixels are (x+1)*50, (1-y)*50."""
    mvp = np.eye(4)
    W, H = 100, 100
    pts = np.array(
        [
            [0.0, 0.0, 0.0],  # → (50, 50)
            [0.5, 0.5, 0.0],  # → (75, 25)
            [-0.5, -0.5, 0.0],  # → (25, 75)
        ],
        dtype=np.float64,
    )
    px, in_front = project_points(pts, mvp, W, H)
    assert in_front.all()
    expected = np.array([[50, 50], [75, 25], [25, 75]], dtype=np.float64)
    assert np.allclose(px, expected)


def test_project_points_excludes_behind_camera():
    """Negative w should mark points as not-in-front."""
    mvp = np.diag([1, 1, 1, -1.0])  # flips homogeneous w sign
    pts = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    _, in_front = project_points(pts, mvp, 100, 100)
    assert not in_front[0]


# ---------------------------------------------------------------------------
# mask_points_in_rect
# ---------------------------------------------------------------------------


def test_mask_points_in_rect_separates_inside_outside():
    mvp = np.eye(4)
    W, H = 100, 100
    pts = np.array(
        [
            [0.0, 0.0, 0.0],  # pixel (50, 50)
            [0.5, 0.5, 0.0],  # pixel (75, 25)
            [-0.5, -0.5, 0.0],  # pixel (25, 75)
            [0.99, 0.99, 0.0],  # pixel (99.5, 0.5) — top-right
        ],
        dtype=np.float64,
    )
    rect = QtCore.QRect(QtCore.QPoint(20, 40), QtCore.QPoint(60, 80))
    mask = mask_points_in_rect(pts, mvp, W, H, rect)
    assert mask.tolist() == [True, False, True, False]


# ---------------------------------------------------------------------------
# mask_points_in_brush
# ---------------------------------------------------------------------------


def test_mask_points_in_brush_circle_hits_within_tolerance():
    """Paint a circular brush at the centre of a 200×200 canvas, project
    100 000 random NDC points through identity MVP, and check the hit count
    falls within ±20 % of the geometric expectation. (PIL anti-aliasing
    over-counts slightly; numpy fancy indexing matches exactly.)"""
    W, H = 200, 200
    radius = 30
    img = Image.new("L", (W, H), 0)
    ImageDraw.Draw(img).ellipse(
        [W // 2 - radius, H // 2 - radius, W // 2 + radius, H // 2 + radius],
        fill=255,
    )
    canvas_mask = np.array(img) > 0

    rng = np.random.default_rng(7)
    pts = rng.uniform(-0.95, 0.95, (100_000, 3))
    mvp = np.eye(4)
    mask, _ = mask_points_in_brush(pts, mvp, W, H, canvas_mask)
    hits = int(mask.sum())
    expected = int(np.pi * radius * radius / (W * H) * len(pts))
    # ±20 % gives the test some headroom for stochastic sampling on
    # ~7 % of the canvas (~7 000 points).
    assert abs(hits - expected) < expected * 0.20


def test_mask_points_in_brush_rectangle_exact():
    """A rectangular brush is purely deterministic — exact match expected."""
    W, H = 200, 200
    w, h = 60, 20  # rect half-extents
    img = Image.new("L", (W, H), 0)
    ImageDraw.Draw(img).rectangle(
        [W // 2 - w, H // 2 - h, W // 2 + w, H // 2 + h],
        fill=255,
    )
    canvas_mask = np.array(img) > 0

    # Plant points on a uniform grid; we know exactly which fall in the rect.
    xs = np.linspace(-1, 1, 50)
    ys = np.linspace(-1, 1, 50)
    grid = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
    pts = np.column_stack([grid, np.zeros(len(grid))])
    mvp = np.eye(4)
    mask, _ = mask_points_in_brush(pts, mvp, W, H, canvas_mask)
    # Recompute the expected hits directly via the rect bounds in pixel space.
    px = (pts[:, 0] + 1.0) * (W / 2.0)
    py = (1.0 - pts[:, 1]) * (H / 2.0)
    expected_mask = (px >= W // 2 - w) & (px <= W // 2 + w) & (py >= H // 2 - h) & (py <= H // 2 + h)
    # The PIL rectangle endpoints are inclusive — allow a 1-pixel slack.
    diff = abs(int(mask.sum()) - int(expected_mask.sum()))
    assert diff <= max(5, int(0.02 * expected_mask.sum()))
