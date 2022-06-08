import os
import glob as globlib


def open_globs(globs, exclusions):
    """Open either a collection of globs or an individual glob and return the union of their filenames, with exclusions
        applied.
    
    Args:
        globs: Either a set/list of globs or a single glob string.
        exclusions: A collection of excluded filenames.
    
    Returns:
        The union of all full file paths in all the globs, with the base filenames appearing in exclusions removed.
    """
    if type(globs) is set:
        globs = list(globs)
    if type(globs) is list:
        globs = [fn for glob in globs for fn in globlib.glob(glob)]
    if type(globs) is str:
        globs = globlib.glob(globs)
    return list(filter(lambda fn: os.path.basename(fn) not in exclusions, globs))
