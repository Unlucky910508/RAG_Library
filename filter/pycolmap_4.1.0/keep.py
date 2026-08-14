"""Prefixes to keep in the pycolmap API records, dropping everything else.

Same format as exclude.py: every list of strings is read and merged.

Used with --keep, this is the opposite policy - useful when a library
exposes far more than you want indexed and naming the wanted part is
shorter than naming the rest. Note that keeping only "lib.sub" also drops
the root module record for "lib".
"""

CORE = [
    "pycolmap.Reconstruction",
    "pycolmap.Camera",
    "pycolmap.Image",
    "pycolmap.Database",
]
