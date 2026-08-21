"""Template: prefixes to drop from a library's API records.

This is the checked-in example for filter_api_records.py --exclude. To
use it for a real library, copy this whole directory to
filter/<your_module>_<your_version>/ - the name must match
parsed_module_name / parsed_module_version in config/config.py exactly,
since that is how the step finds the file (no path to pass, no way for
the two to disagree).

Every list of strings in this file is read and the lists are merged, so
group them however reads best - by reason, by subsystem, one list is
fine too. Variable names are not looked at (they only show up in the
step's own reporting).

A prefix matches the record of that exact name and everything beneath
it: "your_module.internal" drops your_module.internal itself and
your_module.internal.helper, but not your_module.internalX.

This file is parsed with ast, never executed - only top-level
assignments of a list/tuple of string literals are understood, so
anything else written here (imports, function calls, f-strings) is
silently ignored rather than run.

Nothing is excluded by default below; that's deliberate for the
template. Delete this file entirely if you only need keep.py, or vice
versa - only whichever policy you actually pass on the command line
(--exclude / --keep) needs a file to exist.
"""

INTERNAL = [
    # e.g. "your_module.internal",
    # e.g. "your_module.tests",
]

VENDORED = [
    # e.g. "your_module._vendor",
]
