
"""Utility package helpers.

This module exposes convenience imports for modules that use
non-standard file names.  In particular, :mod:`utils.42` defines the
function :func:`handle`, but the numeric module name prevents a regular
``from utils.42 import handle`` import.  To keep the public API simple we
re-export :func:`handle` here so other modules can just ``from utils
import handle``.
"""

from importlib import import_module

# ``42.py`` cannot be imported using a normal statement because the
# module name starts with a digit.  ``import_module`` allows us to load it
# dynamically and re-export the ``handle`` coroutine.
_module_42 = import_module(".42", package=__name__)
handle = _module_42.handle  # type: ignore[attr-defined]

__all__ = ["handle"]

