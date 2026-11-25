"""Training history from CSV logs, keeping the last row for a repeated step."""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

try:
    from . import _common as common
except ImportError:
    import _common as common


def run(name, description, column, ylabel, power, argv=None):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--stats",
        nargs="+",
        type=Path,
        required=True,
        help="train_stats.csv files, or run directories.",
    )
    parser.add_argument("--n-particles", type=common.positive_int, required=True)
    parser.add_argument("--labels", nargs="+", help="One legend label per input log.")
    parser.add_argument(
        "--bin-size",
        type=common.positive_int,
        default=100,
        help="Consecutive logged rows per plotted bin.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("observable-results") / name
    )
    args = parser.parse_args(argv)
    try:
        if args.labels and len(args.labels) != len(args.stats):
            raise ValueError("Supply one --labels value per --stats file")
        frames = []
        plt = common.style()
        fig, ax = plt.subplots(figsize=(6.5, 4.8))
        for i, path in enumerate(args.stats):
            if path.is_dir():
                path = path / "train_stats.csv"
            data = pd.read_csv(path)
            if not {"step", column} <= set(data):
                raise ValueError(f"{path} needs columns step and {column}")
            data = data.drop_duplicates("step", keep="last").sort_values("step")
            if (
                not len(data)
                or not np.isfinite(data[["step", column]].to_numpy(float)).all()
            ):
                raise ValueError(f"{path} has empty or non-finite training history")
            if column == "ewvar" and (data[column] < 0).any():
                raise ValueError("Logged variances must be nonnegative")
            label = args.labels[i] if args.labels else path.parent.name
            rows = []
            for start in range(0, len(data), args.bin_size):
                part = data.iloc[start : start + args.bin_size]
                values = part[column].to_numpy(float) / args.n_particles**power
                rows.append(
                    dict(
                        step=part.step.mean(),
                        step_first=part.step.iloc[0],
                        step_last=part.step.iloc[-1],
                        value=values.mean(),
                        temporal_std=values.std(),
                        rows=len(part),
                        source=str(path),
                        label=label,
                        n_particles=args.n_particles,
                        quantity=f"{column}/N^{power}",
                    )
                )
            frame = pd.DataFrame(rows)
            frames.append(frame)
            ax.plot(frame.step, frame.value, label=label)
            if column == "energy":
                ax.fill_between(
                    frame.step,
                    frame.value - frame.temporal_std,
                    frame.value + frame.temporal_std,
                    alpha=0.15,
                )
        table = pd.concat(frames, ignore_index=True)
        if column == "ewvar" and (table.value > 0).all():
            ax.set_yscale("log")
        ax.set(xlabel="Optimization step", ylabel=ylabel)
        ax.legend(fontsize=11)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(str(args.output) + ".csv", index=False)
        common.save_figure(fig, args.output)
        print(f"Saved {args.output}.csv, .pdf, .png", flush=True)
        return table
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
