# Test Suite Guide

The LiDAR PCD App ships a pytest suite that validates every public
operation, every occupancy-grid path, the projection math behind the
canvas crop/eraser tools, and that the GUI wires up correctly. This file
is the field guide: how to run the tests, what each one means, and how
to add new ones.

---

## 1. Setup (once)

The project uses the `pcd_env` pyenv virtualenv. Dev-only dependencies
live in a separate file so production installs stay slim.

```bash
cd /home/prashant/Workspace/LiDAR_PCD_App
pyenv exec pip install -r requirements-dev.txt
```

This installs `pytest`, `pytest-cov` and `ruff` — the last because CI
gates on `ruff check` **and** `ruff format --check`, so you want to be able
to reproduce that locally. Everything else (Open3D, PyQt5, pyqtgraph, PIL,
NumPy, …) is already in the project's main `requirements.txt`.

Before pushing, run what CI runs:

```bash
ruff check alchemycloud tests
ruff format --check alchemycloud tests
QT_QPA_PLATFORM=offscreen pyenv exec pytest tests/ -q
```

---

## 2. Running the tests

### All tests

```bash
pyenv exec pytest tests/
```

Expected output:

```
============================== 43 passed in ~3s ==============================
```

The runtime is dominated by Open3D's I/O round-trip tests and the
five-second timeouts in the GUI smoke tests (which only kick in if
something genuinely hangs).

### Verbose

```bash
pyenv exec pytest tests/ -v
```

Each test prints its own pass/fail line. Useful when you want to see
exactly which assertions ran.

### One file at a time

```bash
pyenv exec pytest tests/test_pcd_ops.py -v
pyenv exec pytest tests/test_occupancy.py -v
pyenv exec pytest tests/test_viewer_helpers.py -v
pyenv exec pytest tests/test_gui_smoke.py -v
```

### One specific test

```bash
pyenv exec pytest tests/test_pcd_ops.py::test_level_ground_corrects_10deg_tilt -v
```

### Stop at the first failure

```bash
pyenv exec pytest tests/ -x
```

### Print stdout / debug output

```bash
pyenv exec pytest tests/ -s
```

By default pytest captures stdout; `-s` disables that so `print()`
calls show up immediately.

---

## 3. What's covered

**109 tests across seven files**, split by concern. Pure-math tests have
no Qt dependency and run instantly; the GUI tests share a single
session-scoped `QApplication` (started by the `qapp` fixture) and run
under `QT_QPA_PLATFORM=offscreen` so they don't need an X server.

| File | Tests | Scope |
|---|---|---|
| `test_pcd_ops.py` | 24 | Every public `pcd_ops` function, plus floor-vs-wall RANSAC selection |
| `test_occupancy.py` | 9 | Both policies over both methods; PGM/YAML round trip |
| `test_occupancy_geometry.py` | 21 | Map **geometry**, `voxel3d` exactness, size guards, writer edge cases |
| `test_viewer_helpers.py` | 5 | `project_points` / `mask_points_in_*` against identity MVPs |
| `test_rgb_view.py` | 19 | The RGB colour mode and its tone mapping |
| `test_regressions.py` | 21 | Audit fixes in the Qt layer, one test per named defect |
| `test_gui_smoke.py` | 10 | `MainWindow` construction and the async pipeline |

Two things worth knowing before you add tests here:

- **A hang is not a failure.** `MainWindow.closeEvent` shows a modal
  unsaved-changes prompt, and under offscreen Qt nobody answers it. Any test
  that edits a cloud and then closes the window must set
  `w.prompt_unsaved_on_close = False` first, or the run blocks forever
  instead of failing.
- **The GL paints throw and are swallowed.** Under offscreen Qt every
  `paintGL` raises `OpenGL.error.GLError`, which pyqtgraph catches and prints.
  Nothing is actually rendered, so these tests verify wiring and state, never
  pixels. `test_regressions.py` asserts the scatter's GL *options* for that
  reason — it is the closest we get to checking how the cloud is drawn.

### 3.1 `test_pcd_ops.py` — point-cloud operations (24 tests)

Every public function in `pcd_ops.py` is exercised against a synthetic
cloud built by a fixture in `conftest.py`.

| Test | What it asserts |
|---|---|
| `test_load_save_roundtrip[.pcd]` / `[.ply]` | Save a cloud, load it back, coordinates match within 1 e-5. |
| `test_load_pcd_rejects_empty` | `load_pcd` on an empty PLY raises `ValueError` with message "No points". |
| `test_voxel_downsample_reduces_count` | `voxel_downsample(p, 0.2)` returns a non-empty cloud with fewer points than the input. |
| `test_voxel_downsample_rejects_nonpositive` | `voxel_size = 0` or `voxel_size < 0` raises `ValueError`. |
| `test_statistical_outlier_removal_drops_outliers` | SOR removes the 50 injected outliers without destroying the dense cluster. |
| `test_radius_outlier_removal_drops_outliers` | Same scenario, radius method. |
| `test_radius_outlier_rejects_bad_args` | `nb_points <= 0` or `radius <= 0` raises `ValueError`. |
| `test_z_clip_inclusive_bounds` | Points exactly at `z_min` / `z_max` are **kept** (inclusive). Points just outside are dropped. *Hand-crafted points used so the test catches the boundary mutation `>=` → `>`.* |
| `test_z_clip_rejects_inverted_range` | `z_max <= z_min` raises `ValueError`. |
| `test_remove_ground_ransac_finds_floor` | A flat z = 0 floor yields plane equation with `\|c\| ≈ 1`, `\|d\| ≈ 0`, and > 4 000 of the 5 000 points classified as ground inliers. |
| `test_transform_to_marker_frame_translation` | Marker at `(1, 2, 0)` with yaw 0 → every point shifted by `−(1, 2, 0)`. |
| `test_transform_to_marker_frame_rotation` | Marker at origin with yaw 90° → world +X maps to local −Y (matches the docstring). |
| `test_level_ground_corrects_10deg_tilt` | Cloud injected with 10° tilt + 1.5 m lift → returned `tilt_deg ≈ 10` (within 0.5°), floor `\|z\| < 0.01` after correction. |
| `test_level_ground_idempotent` | A second `level_ground` call on the already-flat result returns `tilt_deg < 0.05` and unchanged points. |
| `test_colorize_by_axis_writes_normalised_colors` | `pcd.colors` gets a length-N float array with values in `[0, 1]`. |
| `test_colorize_by_axis_rejects_bad_axis` | `axis="w"` raises `ValueError`. |
| `test_display_downsample_under_target` | 800 k input + target 200 k → output ≤ 250 k. |
| `test_display_downsample_passthrough_for_small_clouds` | Cloud smaller than `max_points` is returned **by identity** (no copy). |
| `test_stats_matches_input` | `stats(pcd)` reports the right point count and bounding box; `describe()` mentions the formatted count. |

### 3.2 `test_occupancy.py` — occupancy grid build + save (9 tests)

Covers both **policies** (binary vs tristate) and both **methods**
(`projection` vs `voxel3d`), plus the file I/O round-trip.

| Test | What it asserts |
|---|---|
| `test_grid_params_defaults` | Dataclass defaults match the documented values (resolution 0.05, projection, threshold `None`, …). |
| `test_binary_projection_marks_floor_occupied` | A floor cloud → over 30 % of cells occupied (binary policy fills cells that have any point). |
| `test_binary_voxel3d_produces_grid` | Voxel3D method produces a non-empty grid with at least some occupied cells. |
| `test_unknown_method_raises` | `params.method = "bogus"` raises `ValueError`. |
| `test_empty_z_range_raises` | No points in `[z_min, z_max]` → `ValueError` with "No points". |
| `test_tristate_separates_free_and_occupied` | Floor + chair + threshold 0.2 m → cells under the chair are **occupied** (10 < n < 100), floor-only cells are **free** (> 100), padding cells are **unknown**, and the three classes partition the image. |
| `test_tristate_with_voxel3d` | Same tristate logic also applies in the voxel3d code path. |
| `test_save_roundtrip_pgm_yaml` | `occupancy.save` writes a PGM that PIL re-loads to **exactly** the same `numpy` array; the sidecar YAML parses to a dict with the right keys and values. |

### 3.3 `test_viewer_helpers.py` — projection math (5 tests)

The crop rectangle and eraser brush both project the *full* cloud
through the live MVP matrix and look points up in a 2-D mask. These
helpers are pure NumPy and reusable; the tests pin their semantics.

| Test | What it asserts |
|---|---|
| `test_project_points_identity_mvp` | With identity MVP at 100 × 100, world (0, 0, 0) → pixel (50, 50), (0.5, 0.5) → (75, 25), etc. |
| `test_project_points_excludes_behind_camera` | An MVP that flips `w → −w` flags points as not-in-front (the `in_front` mask is `False`). |
| `test_mask_points_in_rect_separates_inside_outside` | Four hand-placed points + a known rectangle → exact `[True, False, True, False]` mask. |
| `test_mask_points_in_brush_circle_hits_within_tolerance` | A 30-px circular PIL brush at the canvas centre, 100 k random points → hit count within ±20 % of `π · r² / (W · H) · N`. |
| `test_mask_points_in_brush_rectangle_exact` | A rectangular PIL brush on a uniform grid of points → hit count matches the analytical answer within 5 cells. |

### 3.4 `test_gui_smoke.py` — end-to-end GUI flow (10 tests)

These create a `MainWindow` under offscreen Qt, plant a synthetic
cloud, and validate the async pipeline. Each test uses the shared
`qapp` fixture so we don't spin up multiple `QApplication` instances.

| Test | What it asserts |
|---|---|
| `test_main_window_constructs` | Window opens, all **four** toolbars are present (`EditToolbar`, `MarkerToolbar`, `ToolsToolbar`, `OccupancyToolbar`), Color combo defaults to Z, Display switch is off (= LOD on). |
| `test_load_path_populates_state` | Writing a synthetic cloud to disk then calling `load_path` clears `_busy`, sets `current_pcd`, and stashes `current_path`. |
| `test_apply_filter_updates_history` | Running a voxel downsample through `_apply_async` grows `history` by 1 and shrinks `current_pcd`. |
| `test_undo_redo_round_trip` | After a filter: `undo` restores the original point count and pushes onto `redo_stack`; `redo` replays forward. |
| `test_occupancy_preview_renders_initial_grid` | `OccupancyPreviewDialog` builds its initial preview within 5 s, in tristate mode by default. |
| `test_marker_coord_entry_moves_marker` | Typing X / Y / yaw into the Marker toolbar spin boxes moves the marker live to that pose. |
| `test_marker_drag_syncs_back_to_spins` | Moving the marker via the viewer API (the mouse-drag / Reset path) mirrors back into the spin boxes; Reset Marker zeroes them. |
| `test_marker_spin_ranges_clamp_to_cloud_scope` | X / Y spin ranges track the loaded cloud's bounds; yaw is fixed to ±180°. |
| `test_point_size_spin_updates_viewer` | Point-size spin sets `viewer.point_size()` / the scatter's `size`, and the value persists across a filter re-prepare. |
| `test_brush_shape_combo_mirrors_viewer` | Selecting Rectangle in the Shape combo flips `viewer.brush_shape()` to `"rectangle"` and enables the Aspect spin; switching to Square disables it again. |

---

## 4. Fixtures (`conftest.py`)

When you write a new test, **reuse a fixture** instead of building a
cloud inline. They're already tuned for the operations they support.

| Fixture | Scope | Returns |
|---|---|---|
| `qapp` | session | A shared `QApplication`. Inject this into any GUI test. |
| `cube_cloud` | function | 5 000 points uniform in `[-1, 1]³`. General-purpose. |
| `cluster_with_outliers` | function | 5 000-point Gaussian cluster + 50 isolated outliers far away — for SOR / radius tests. |
| `floor_with_chair` | function | Floor at z = 0 + a "chair top" cluster at z = 0.5 — for occupancy / tristate tests. |
| `tilted_cloud` | function | 12 000-point cloud rotated 10° about +X and lifted 1.5 m — for `level_ground` tests. |
| `busy_wait` | function | `wait_for(predicate, timeout_s=5)` — drains Qt events while polling. Returns `True` if the predicate became true. |

Adding a new fixture? Put it in `conftest.py`, mark it `@pytest.fixture`,
and any test in any file under `tests/` can use it by name.

---

## 5. Why some warnings show up (and are fine)

Under `QT_QPA_PLATFORM=offscreen` the platform plugin reports a stub
OpenGL context, which pyqtgraph's `GLViewWidget` interprets as
"requires ≥ OpenGL 2.1; found 0.0" and raises a `RuntimeError` inside
`initializeGL`. Qt swallows the exception (printing the traceback) and
moves on — the widget stays in a degraded state but every CPU-side
operation we test (event handling, signals, internal state) still
works. So the test output will include a few of these tracebacks; they
are **not** failures, and `pytest` reports the suite as green.

If we ever start testing actual rendered pixels we'd need to swap to a
real headless GL stack (`xvfb-run`, EGL, …) — not in scope today.

---

## 6. Adding a new test

1. Pick the right file by **concern**:
   * Pure math on a point cloud → `test_pcd_ops.py`.
   * Occupancy grid logic → `test_occupancy.py`.
   * Camera / projection helpers → `test_viewer_helpers.py`.
   * Anything that touches `MainWindow` or a dialog → `test_gui_smoke.py`.
2. Reuse a fixture if it fits. Otherwise add a new one to
   `conftest.py` — keep the seed deterministic (`np.random.default_rng(K)`).
3. For async ops, use the `busy_wait` fixture to drain Qt events
   instead of `time.sleep` loops.
4. Name the test `test_<verb>_<expected_behaviour>` — pytest discovers
   anything starting with `test_`.
5. Run just your new test with
   `pyenv exec pytest tests/test_X.py::test_Y -v` while iterating.
6. When green, run the whole suite once
   (`pyenv exec pytest tests/`) to make sure you didn't regress an
   unrelated path.

---

## 7. Mutation check (sanity)

The suite is designed to catch real bugs, not be rubber-stamped. A
quick sanity ritual: temporarily break `pcd_ops.z_clip` (`>=` → `>`),
re-run `tests/test_pcd_ops.py::test_z_clip_inclusive_bounds`, confirm
it fails with `assert 1 == 3`, then revert. If you ever rework the
boundary semantics you'll see this test light up — that's the point.
