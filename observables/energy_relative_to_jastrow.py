"""Relative energy improvement (E-E_J)/E_J against an energy-optimized pure Jastrow."""

import numpy as np
import pandas as pd

try:
    from . import _common as common, _jastrow as jastrow
except ImportError:
    import _common as common
    import _jastrow as jastrow


def calculate(state, positions, coupling, args):
    result = jastrow.comparison(state, positions, coupling, args)
    energy, baseline = result["energy_mean"], result["baseline_energy_mean"]
    if baseline == 0:
        raise ValueError("Cannot normalize by zero baseline energy")
    result["relative_energy_difference"] = energy / baseline - 1
    result["relative_energy_difference_se"] = np.hypot(
        result["energy_se"] / baseline,
        energy * result["baseline_energy_se"] / baseline**2,
    )
    return pd.DataFrame([result])


def plot(table, args):
    common.curve_plot(
        table,
        args,
        "relative_energy_difference",
        r"$(E-E_J)/E_J$",
        error="relative_energy_difference_se",
    )


def main(argv=None):
    return common.run(
        "energy_relative_to_jastrow",
        __doc__,
        calculate,
        plot,
        configure=jastrow.configure,
        argv=argv,
    )


if __name__ == "__main__":
    main()
