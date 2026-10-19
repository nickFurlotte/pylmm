# pylmm is a python-based linear mixed-model solver with applications to GWAS
# Copyright (C) 2013-2026 Nicholas A. Furlotte (nick.furlotte@gmail.com)

# SPDX-License-Identifier: MIT
# See the LICENSE file in the project root for full license text.

"""Pieces shared by the ``pylmm`` subcommands: errors, exit codes and logging.

Every message of the command line goes through the ``pylmm`` logger to
stderr, as ``pylmm: <level>: <message>``.  ``warnings.warn`` calls of the
library are logged as warnings while a command runs.
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import sys
import warnings
from collections.abc import Iterator
from typing import TextIO

LOGGER_NAME = "pylmm"
log = logging.getLogger(LOGGER_NAME)

EXIT_OK = 0
"""The command succeeded."""
EXIT_RUNTIME_ERROR = 1
"""An unexpected error (a bug, or a failure pylmm did not anticipate)."""
EXIT_USAGE = 2
"""A usage or input-validation error (the argparse convention)."""


class UsageError(Exception):
    """An expected user error: bad arguments or invalid input.

    The command line reports it as one ``pylmm: error:`` line, without a
    traceback (unless ``--debug``), and exits with `EXIT_USAGE`.
    """


class _ConsoleFormatter(logging.Formatter):
    """``pylmm: <level>: <message>``, the level in lower case."""

    def format(self, record: logging.LogRecord) -> str:
        text = "%s: %s: %s" % (LOGGER_NAME, record.levelname.lower(), record.getMessage())
        if record.exc_info:
            text += "\n" + self.formatException(record.exc_info)
        return text


def add_logging_options(parser: argparse.ArgumentParser) -> None:
    """Add ``-v/--verbose``, ``-q/--quiet``, ``--log-file`` and ``--debug``.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The subcommand parser.  The options set ``log_level``, ``log_file``
        and ``debug`` on the parsed namespace.
    """
    group = parser.add_argument_group("logging")
    level = group.add_mutually_exclusive_group()
    level.add_argument(
        "-v",
        "--verbose",
        dest="log_level",
        action="store_const",
        const=logging.INFO,
        default=logging.WARNING,
        help="also log progress messages (INFO); the default logs warnings and errors",
    )
    level.add_argument(
        "-q",
        "--quiet",
        dest="log_level",
        action="store_const",
        const=logging.ERROR,
        help="log errors only",
    )
    group.add_argument(
        "--log-file",
        metavar="FILE",
        help="also write the log, with timestamps and INFO messages whatever -v/-q say, to FILE",
    )
    group.add_argument(
        "--debug",
        action="store_true",
        help="log DEBUG messages and show the Python traceback of an error",
    )


def _log_warning(
    message: Warning | str,
    category: type[Warning],
    filename: str,
    lineno: int,
    file: TextIO | None = None,
    line: str | None = None,
) -> None:
    """A `warnings.showwarning` replacement: log the message as one line."""
    log.warning("%s", message)


@contextlib.contextmanager
def configured_logging(
    level: int, log_file: str | None = None, debug: bool = False
) -> Iterator[None]:
    """Send the ``pylmm`` logger (and library warnings) to stderr for one command.

    The handlers are removed again on exit, so `pylmm.cli.main` can be called
    more than once in a process (as the tests do).  ``sys.stderr`` is looked
    up when the context is entered, so ``contextlib.redirect_stderr`` works.

    Parameters
    ----------
    level : int
        The console level (``logging.INFO``, ``WARNING`` or ``ERROR``).
    log_file : str, optional
        Also log to this file (overwritten), at INFO or above.
    debug : bool, default False
        Log DEBUG messages too, on the console and in the log file.

    Raises
    ------
    UsageError
        If the log file cannot be opened.
    """
    handlers: list[logging.Handler] = []
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(logging.DEBUG if debug else level)
    console.setFormatter(_ConsoleFormatter())
    handlers.append(console)
    if log_file:
        try:
            file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        except OSError as e:
            raise UsageError("--log-file %s: cannot open it (%s)" % (log_file, e)) from e
        file_handler.setLevel(logging.DEBUG if debug else min(level, logging.INFO))
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
        )
        handlers.append(file_handler)
    saved = (log.level, log.propagate)
    log.setLevel(logging.DEBUG)
    log.propagate = False
    for handler in handlers:
        log.addHandler(handler)
    try:
        with warnings.catch_warnings():
            warnings.showwarning = _log_warning
            yield
    finally:
        for handler in handlers:
            log.removeHandler(handler)
            handler.close()
        log.setLevel(saved[0])
        log.propagate = saved[1]
