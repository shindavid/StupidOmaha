import datetime
import os


def timed_print(s, *args, **kwargs):
    t = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
    print(f'{t} {s}', *args, **kwargs)


def make_hidden_filename(filename):
    """
    Returns a filename formed by prepending a '.' to the filename part of filename.
    """
    head, tail = os.path.split(filename)
    return os.path.join(head, '.' + tail)
