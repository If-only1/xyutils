__all__ = ['mkdir', 'mkdirs', 'add_new_ext']

import os


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mkdirs(paths):
    assert isinstance(paths, (list, tuple))
    for path in paths:
        mkdir(path)


def add_new_ext(path, new_ext):
    assert new_ext.startswith('.'), new_ext
    path = os.path.splitext(path)[0] + new_ext
    return path
