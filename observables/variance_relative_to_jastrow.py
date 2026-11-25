"""Local-energy variance divided by that of an energy-optimized pure-Jastrow state."""

import numpy as np
import pandas as pd

try:
    from . import _common as common, _jastrow as jastrow
except ImportError:
    import _common as common
    import _jastrow as jastrow


def calculate(state, positions, coupling, args):
    result = jastrow.comparison(state, positions, coupling, args)
    variance, baseline = result["energy_variance"], result["baseline_energy_variance"]
    if baseline <= 0:
        raise ValueError("Cannot normalize by a nonpositive baseline variance")
    result["variance_ratio"] = variance / baseline
    result["variance_ratio_se"] = np.hypot(
        result["energy_variance_se"] / baseline,
        variance * result["baseline_energy_variance_se"] / baseline**2,
    )
    return pd.DataFrame([result])


def plot(table, args):
    common.curve_plot(
        table,
        args,
        "variance_ratio",
        r"$\mathrm{var}(E_L)/\mathrm{var}(E_{L,J})$",
        error="variance_ratio_se",
        log=True,
    )


def main(argv=None):
    return common.run(
        "variance_relative_to_jastrow",
        __doc__,
        calculate,
        plot,
        configure=jastrow.configure,
        argv=argv,
    )


if __name__ == "__main__":
    main()
