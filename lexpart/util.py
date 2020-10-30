import re
from pathlib import Path

import numpy


def doc_iter(path):
    for f in Path(path).iterdir():
        yield f.read_text(encoding='utf-8')


def docs_tokens(docpath, _t_rex=re.compile(r'\w+')):
    for d in doc_iter(docpath):
        yield _t_rex.findall(d.lower())


def random_window_gen(mean, std, stop, block_size=1000):
    start = 0
    while True:
        block = start + numpy.random.normal(mean, std, block_size).cumsum()
        block = block.astype(int)
        block_max = block.max()
        if block_max < stop:
            yield from block
            start = block[-1]
        else:
            block = block[block < stop]
            yield from block
            yield stop
            return
