"""Project-scoped AI guide for Neural Activity Toolkit."""


def main() -> int:
    from .gui import main as run_gui

    return run_gui()

__all__ = ["main"]
