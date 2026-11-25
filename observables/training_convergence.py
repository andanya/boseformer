"""Training energy per particle versus step; shading is temporal spread within bins."""

try:
    from ._history import run
except ImportError:
    from _history import run


def main(argv=None):
    return run("training_convergence", __doc__, "energy", r"$E/N$", 1, argv)


if __name__ == "__main__":
    main()
