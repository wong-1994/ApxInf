#!/usr/bin/env python3
"""Compatibility entry point for the client-only LIBERO evaluator.

Prefer ``python -m evaluation.libero.client`` for new automation.
"""

from __future__ import annotations

import pathlib
import sys


REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from evaluation.libero.client import main  # noqa: E402


if __name__ == "__main__":
    main()
