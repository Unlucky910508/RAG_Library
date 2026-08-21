"""Template: prefixes to keep in a library's API records, dropping
everything else.

This is the checked-in example for filter_api_records.py --keep. To use
it for a real library, copy this whole directory to
filter/<your_module>_<your_version>/ - the name must match
parsed_module_name / parsed_module_version in config/config.py exactly,
since that is how the step finds the file (no path to pass, no way for
the two to disagree).

Same format as exclude.py: every list of strings at the top level is
read and merged, variable names are ignored, and the file is parsed with
ast rather than executed - only literal lists of string prefixes are
understood.

Used with --keep, this is the opposite policy from exclude.py - useful
when a library exposes far more than you want indexed and naming the
wanted part is shorter than naming the rest. A prefix matches the record
of that exact name and everything beneath it, so keeping only
"your_module.sub" also drops the root module record for "your_module"
itself.

Delete this file if you only need exclude.py, or vice versa - only
whichever policy you actually pass on the command line (--exclude /
--keep) needs a file to exist.
"""

CORE = [
    # e.g. "your_module.CoreClass",
    # e.g. "your_module.core_function",
]
