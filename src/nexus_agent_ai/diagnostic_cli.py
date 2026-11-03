"""Run only the frozen synthetic command under explicit diagnostic activation."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .agent.budget_diagnostics import ROWS, activate_scope, frozen_argv


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", required=True, type=Path)
    parser.add_argument("--row", required=True, choices=ROWS)
    try:
        separator = arguments.index("--")
    except ValueError:
        parser.error("the frozen CLI arguments must follow --")
    options = parser.parse_args(arguments[:separator])
    command = arguments[separator + 1:]
    if command != frozen_argv(options.row):
        parser.error("command does not match the exact synthetic row")
    with activate_scope(options.scope, options.row) as scope:
        if scope is None:
            parser.error("scope does not match the isolated synthetic environment")
        # Keep the installed normal Typer entrypoint, streams, and SystemExit.
        # Import only after wrapper validation, never initialize an alternate agent.
        from .cli.app import app
        return app(args=command, prog_name="nexus-agent")


if __name__ == "__main__":
    main()
