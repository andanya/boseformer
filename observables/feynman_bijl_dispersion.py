"""Feynman-Bijl estimate omega(k) = k^2/[2 S(k)], in units hbar = m = 1.

S(k) is averaged over nonzero reciprocal shells before taking its inverse.
This is a single-mode estimate from the static structure factor.
"""

import numpy as np

try:
    from . import _common as common
    from .structure_factor_radial import calculate as structure_factor, configure
except ImportError:
    import _common as common
    from structure_factor_radial import calculate as structure_factor, configure


def calculate(state, positions, coupling, args):
    table = structure_factor(state, positions, coupling, args)
    if np.any(table.structure_factor <= 0):
        raise ValueError("Feynman-Bijl dispersion requires positive S(k)")
    table["omega"] = table.k**2 / (2 * table.structure_factor)
    table["omega_se"] = table.omega * table.structure_factor_se / table.structure_factor
    table["error_method"] = (
        "first-order propagation of the chain-mean S(k) standard error"
    )
    return table


def plot(table, args):
    common.curve_plot(
        table,
        args,
        "omega",
        r"$\omega_{\mathrm{FB}}(k)$",
        error="omega_se",
        x="k",
        xlabel=r"$k$",
    )


def main(argv=None):
    return common.run(
        "feynman_bijl_dispersion",
        __doc__,
        calculate,
        plot,
        configure=configure,
        argv=argv,
    )


if __name__ == "__main__":
    main()
