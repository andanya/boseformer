"""Logged exponentially weighted variance of batch-mean training energy, divided by N^2.

This is a training-history diagnostic, distinct from the variance of individual
local energies measured by local_energy_variance.py.
"""

try:
    from ._history import run
except ImportError:
    from _history import run


def main(argv=None):
    return run(
        "training_variance",
        __doc__,
        "ewvar",
        r"$\mathrm{EWvar}(E_{\mathrm{batch}})/N^2$",
        2,
        argv,
    )


if __name__ == "__main__":
    main()
