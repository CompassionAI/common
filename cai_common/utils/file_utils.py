import os
import glob as globlib


def open_globs(globs, exclusions):
    if type(globs) is set:
        globs = list(globs)
    if type(globs) is list:
        globs = [fn for glob in globs for fn in globlib.glob(glob)]
    if type(globs) is str:
        globs = globlib.glob(globs)
    return list(filter(lambda fn: os.path.basename(fn) not in exclusions, globs))
