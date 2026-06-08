#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot EuROC trajectory from state_groundtruth_estimate0/data.csv"
    )
    parser.add_argument(
        "csv_path",
        type=Path,
        help="Path to EuROC groundtruth csv file",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional output image path (e.g. trajectory.png)",
    )
    return parser.parse_args()


def load_xyz(csv_path: Path) -> tuple[list[float], list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []

    with csv_path.open("r", encoding="utf-8") as file:
        reader = csv.reader(file)
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            if len(row) < 4:
                continue
            xs.append(float(row[1]))
            ys.append(float(row[2]))
            zs.append(float(row[3]))

    if not xs:
        raise ValueError(f"No trajectory rows found in {csv_path}")

    return xs, ys, zs


def main() -> None:
    args = parse_args()

    if not args.csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {args.csv_path}")

    xs, ys, zs = load_xyz(args.csv_path)

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required. Install with: pip install matplotlib"
        ) from exc

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(projection="3d")
    ax.plot(xs, ys, zs, linewidth=1.0)
    ax.set_title("EuROC trajectory")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
