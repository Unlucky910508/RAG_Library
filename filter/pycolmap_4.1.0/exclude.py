"""Prefixes to drop from the pycolmap API records.

Every list of strings in this file is read and the lists are merged, so
group them however reads best - by reason, by subsystem, one list is
fine too. Variable names are not looked at.

A prefix matches the record of that exact name and everything under it:
"pycolmap.logging" drops pycolmap.logging itself and
pycolmap.logging.info, but not pycolmap.loggingX.

Nothing is excluded by default. pycolmap is small enough that the whole
surface is worth keeping; this file exists as the place to put exclusions
when they are wanted, and as a shape to copy for larger libraries where
most of the module tree is internal.
"""

INTERNAL = [
    # e.g. "pycolmap.cost_functions",
]
