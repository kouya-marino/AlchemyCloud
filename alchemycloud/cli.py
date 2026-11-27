"""Headless pipeline: ``alchemycloud-cli INPUT [steps…] [-o OUT] [--map MAP.pgm]``.

Runs the same operations as the GUI, without Qt, in the order they are given
on the command line, then optionally saves the cloud and/or exports a ROS
map_server PGM + YAML pair. Useful for batches and for scripting a pipeline
that was worked out interactively.

    alchemycloud-cli scan.pcd --zclip -1 3 --voxel 0.05 --sor 30 1.5 --level \\
        -o scan_clean.pcd --map scan_map.pgm --map-threshold 0.3
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

from . import __version__, occupancy, pcd_ops

log = logging.getLogger(__name__)


class _Step(argparse.Action):
    """Record a processing step in command-line order (argparse alone keeps
    one value per option and forgets the order)."""

    def __call__(self, parser, namespace, values, option_string=None):
        steps = list(getattr(namespace, "steps", None) or [])
        if values is None:
            values = []
        elif not isinstance(values, list):
            values = [values]
        steps.append((self.dest, values))
        namespace.steps = steps


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="alchemycloud-cli",
        description=(
            "Clean a point cloud and export a ROS occupancy map without the GUI. "
            "Processing steps run in the order you give them and may repeat."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "example:\n"
            "  alchemycloud-cli scan.pcd --zclip -1 3 --voxel 0.05 --sor 30 1.5 --level \\\n"
            "      -o scan_clean.pcd --map scan_map.pgm --map-threshold 0.3"
        ),
    )
    p.add_argument("input", help="point cloud to read (.pcd / .ply / .xyz)")
    p.add_argument("--version", action="version", version=f"alchemycloud {__version__}")
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging to stderr")
    p.set_defaults(steps=[])

    s = p.add_argument_group("processing steps (run in the order given)")
    s.add_argument(
        "--zclip", nargs=2, type=float, metavar=("ZMIN", "ZMAX"), action=_Step, help="keep ZMIN <= z <= ZMAX"
    )
    s.add_argument("--voxel", type=float, metavar="SIZE", action=_Step, help="voxel downsample (m)")
    s.add_argument(
        "--sor",
        nargs=2,
        type=float,
        metavar=("NEIGHBOURS", "STD_RATIO"),
        action=_Step,
        help="statistical outlier removal (GUI default: 30 1.5)",
    )
    s.add_argument(
        "--radius",
        nargs=2,
        type=float,
        metavar=("MIN_POINTS", "RADIUS"),
        action=_Step,
        help="radius outlier removal (GUI default: 30 0.10); ALCHEMYCLOUD_GPU=1 uses the GPU",
    )
    s.add_argument(
        "--remove-ground",
        nargs="?",
        type=float,
        const=0.05,
        metavar="DIST",
        action=_Step,
        help="remove the floor plane (RANSAC, inlier distance DIST m, default 0.05)",
    )
    s.add_argument("--level", nargs=0, action=_Step, help="rotate the floor to horizontal and put it at z = 0")
    s.add_argument(
        "--tilt", nargs=2, type=float, metavar=("ROLL", "PITCH"), action=_Step, help="manual trim in degrees"
    )
    s.add_argument(
        "--origin",
        nargs=4,
        type=float,
        metavar=("X", "Y", "Z", "YAW"),
        action=_Step,
        help="re-frame so this pose (m, degrees) becomes the origin, like Apply Marker as Origin",
    )

    o = p.add_argument_group("output")
    o.add_argument("-o", "--output", metavar="OUT", help="write the processed cloud (.pcd / .ply / .xyz)")
    o.add_argument("--ascii", action="store_true", help="write the cloud as ASCII instead of binary")

    m = p.add_argument_group("occupancy map")
    m.add_argument("--map", metavar="MAP.pgm", help="export a map_server PGM + YAML pair")
    m.add_argument("--map-resolution", type=float, default=0.05, metavar="M", help="metres per cell (0.05)")
    m.add_argument(
        "--map-z", nargs=2, type=float, metavar=("ZMIN", "ZMAX"), help="height band to use (default: whole cloud)"
    )
    m.add_argument(
        "--map-threshold",
        type=float,
        metavar="H",
        help="tristate map: cells with returns above H are occupied, observed cells below are free "
        "(omit for a binary map)",
    )
    m.add_argument("--map-method", choices=["projection", "voxel3d"], default="projection")
    m.add_argument("--map-min-points", type=int, default=1, metavar="N", help="returns needed to mark a cell occupied")
    m.add_argument("--map-padding", type=int, default=5, metavar="CELLS", help="empty margin around the data (5)")
    m.add_argument(
        "--map-unobserved",
        choices=["unknown", "free"],
        default="unknown",
        help="tristate only: what never-observed cells become (unknown)",
    )
    return p


def _run_step(pcd, name: str, values: list):
    """Apply one step; returns (new_cloud, note) where note adds detail."""
    if name == "zclip":
        return pcd_ops.z_clip(pcd, values[0], values[1]), ""
    if name == "voxel":
        return pcd_ops.voxel_downsample(pcd, values[0]), ""
    if name == "sor":
        return pcd_ops.statistical_outlier_removal(pcd, int(values[0]), values[1]), ""
    if name == "radius":
        return pcd_ops.radius_outlier_removal(pcd, int(values[0]), values[1]), ""
    if name == "remove_ground":
        non_ground, _ground, plane = pcd_ops.remove_ground_ransac(pcd, distance_threshold=values[0])
        return non_ground, "plane {:.3f}x + {:.3f}y + {:.3f}z + {:.3f} = 0".format(*plane)
    if name == "level":
        out, _plane, tilt = pcd_ops.level_ground(pcd)
        return out, f"levelled {tilt:.2f}°"
    if name == "tilt":
        return pcd_ops.apply_tilt(pcd, values[0], values[1]), ""
    if name == "origin":
        import numpy as np

        x, y, z, yaw_deg = values
        return pcd_ops.transform_to_marker_frame(pcd, (x, y, z), float(np.radians(yaw_deg))), ""
    raise ValueError(f"unknown step {name!r}")  # pragma: no cover - parser prevents it


def _label(name: str, values: list) -> str:
    flag = "--" + name.replace("_", "-")
    return " ".join([flag, *(f"{v:g}" for v in values)])


def run(args: argparse.Namespace, out=None) -> int:
    out = sys.stdout if out is None else out  # looked up now, so redirection works
    t0 = time.perf_counter()
    pcd, dropped = pcd_ops.load_pcd_counted(args.input)
    note = f", dropped {dropped:,} non-finite" if dropped else ""
    print(f"loaded {args.input}: {len(pcd.points):,} points{note}", file=out)

    for name, values in args.steps:
        t = time.perf_counter()
        before = len(pcd.points)
        pcd, detail = _run_step(pcd, name, values)
        if len(pcd.points) == 0:
            raise ValueError(f"{_label(name, values)} left no points")
        extra = f" — {detail}" if detail else ""
        print(
            f"{_label(name, values)}: {before:,} → {len(pcd.points):,} points ({time.perf_counter() - t:.2f} s){extra}",
            file=out,
        )

    if args.output:
        pcd_ops.save_pcd(pcd, args.output, ascii_format=args.ascii)
        print(f"saved {args.output}", file=out)

    if args.map:
        s = pcd_ops.stats(pcd)
        z_lo, z_hi = args.map_z if args.map_z else (float(s.min_bound[2]), float(s.max_bound[2]))
        params = occupancy.GridParams(
            resolution=args.map_resolution,
            z_min=z_lo,
            z_max=z_hi,
            method=args.map_method,
            min_points_per_cell=args.map_min_points,
            padding_cells=args.map_padding,
            height_threshold=args.map_threshold,
            tristate_empty_unknown=args.map_unobserved == "unknown",
        )
        grid = occupancy.build(pcd, params)
        pgm, yml = occupancy.save(grid, args.map, params)
        print(f"map {grid.width}×{grid.height} cells @ {grid.resolution:g} m → {pgm} + {yml.name}", file=out)

    print(f"done in {time.perf_counter() - t0:.2f} s", file=out)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if not args.steps and not args.output and not args.map:
        parser.error("nothing to do: give processing steps, -o OUT and/or --map MAP.pgm")
    try:
        return run(args)
    except (ValueError, OSError) as exc:  # bad input, parameters or paths
        log.debug("failed", exc_info=True)
        print(f"alchemycloud-cli: error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
