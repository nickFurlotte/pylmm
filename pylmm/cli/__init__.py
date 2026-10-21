# pylmm is a python-based linear mixed-model solver with applications to GWAS
# Copyright (C) 2013-2026 Nicholas A. Furlotte (nick.furlotte@gmail.com)

# SPDX-License-Identifier: MIT
# See the LICENSE file in the project root for full license text.

"""The ``pylmm`` command line: ``pylmm gwas`` and ``pylmm kinship``.

Installed as the ``pylmm`` console script; ``python -m pylmm`` runs the
same thing.  `main` can also be called in-process with an argument list and
returns the exit code instead of exiting:

* 0 -- success;
* 2 -- a usage or input-validation error (bad options, a missing or
  malformed input file), reported as one ``pylmm: error:`` line;
* 1 -- an unexpected error.

Add ``--debug`` to a subcommand to see the Python traceback of an error.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from pylmm import __version__
from pylmm.cli._common import (
    EXIT_OK,
    EXIT_RUNTIME_ERROR,
    EXIT_USAGE,
    UsageError,
    configured_logging,
    debug_arguments,
    log,
)
from pylmm.samples import InputError

__all__ = ["build_parser", "main"]

_EPILOG = """\
examples:
  pylmm kinship --bfile data/snps.132k.clean.noX -o snps.kin
  pylmm gwas --bfile data/snps.132k.clean.noX --kinship snps.kin \\
      --pheno data/snps.132k.clean.noX.fake.phenos -o results.tsv

Run `pylmm <command> --help` for the options of a command.
"""


def build_parser() -> argparse.ArgumentParser:
    """The ``pylmm`` argument parser, with the ``gwas`` and ``kinship`` subcommands.

    Returns
    -------
    argparse.ArgumentParser
        Parsing sets ``command``, ``run`` (the function that runs the
        command, called with the namespace) and the options of the command.
    """
    parser = argparse.ArgumentParser(
        prog="pylmm",
        description="PyLMM: a modern, readable, validated mixed-model toolkit for statistical\n"
        "genetics, built to be used by humans and agents.",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Imported here, not at the top: `python -m pylmm.cli.gwas` (the removed
    # invocation, a tombstone) would otherwise find the module already imported
    # by this package and print a runpy RuntimeWarning first.
    from pylmm.cli import gwas, kinship

    parser.add_argument("--version", action="version", version="%(prog)s " + __version__)
    commands = parser.add_subparsers(dest="command", metavar="COMMAND", title="commands")
    commands.required = True
    gwas.add_parser(commands)
    kinship.add_parser(commands)
    return parser


def _exit_code(code: object) -> int:
    """The exit status for a `SystemExit` code (None is 0, a message is 2)."""
    if code is None:
        return EXIT_OK
    if isinstance(code, int):
        return code
    return EXIT_USAGE


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ``pylmm`` command line.

    Parameters
    ----------
    argv : sequence of str, optional
        The arguments, without the program name (default: ``sys.argv[1:]``).

    Returns
    -------
    int
        The exit code: 0 on success, 2 for a usage or input error, 1 for an
        unexpected error.  ``--help`` and ``--version`` return 0.

    Notes
    -----
    Each call configures process-global logging for its duration (handlers
    on the ``pylmm`` logger, ``warnings.showwarning``) and restores it on
    return, so calls in sequence are fine but `main` is not thread-safe:
    do not run two calls at once in one process.
    """
    parser = build_parser()
    try:
        args = parser.parse_args(None if argv is None else list(argv))
        args.argv = list(sys.argv[1:] if argv is None else argv)
        args.check(args.parser, args)
    except SystemExit as e:  # --help, --version, or a usage error argparse printed
        return _exit_code(e.code)
    try:
        with configured_logging(args.log_level, args.log_file, args.debug):
            return _run(args)
    except UsageError as e:  # the log file could not be opened
        sys.stderr.write("pylmm: error: %s\n" % e)
        return EXIT_USAGE


def _run(args: argparse.Namespace) -> int:
    """Run the parsed command and turn its errors into exit codes (logging them)."""
    log.debug("arguments: %s", debug_arguments(args))
    try:
        args.run(args)
    except (UsageError, InputError) as e:
        # InputError: a malformed input file, also one found while reading the
        # SNPs (a bad TPED allele code, a truncated BED file), after the
        # up-front checks.
        log.error("%s", e, exc_info=args.debug)
        return EXIT_USAGE
    except OSError as e:  # a missing or unreadable input file, an unwritable output
        log.error("%s", e, exc_info=args.debug)
        return EXIT_USAGE
    except KeyboardInterrupt:
        log.error("interrupted")
        return 130
    except Exception as e:
        if args.debug:
            log.exception("unexpected error: %s: %s", type(e).__name__, e)
        else:
            log.error(
                "unexpected error: %s: %s (rerun with --debug for the traceback)",
                type(e).__name__,
                e,
            )
        return EXIT_RUNTIME_ERROR
    return EXIT_OK
