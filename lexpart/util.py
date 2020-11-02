import re
import tarfile
import tempfile
from pathlib import Path
from contextlib import contextmanager

import pkg_resources
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


@contextmanager
def temp_test_corpus():
    en8path = pkg_resources.resource_filename(
            'lexpart',
            'data/test/enwiki8.tar.bz2')
    try:
        tmpdir = tempfile.TemporaryDirectory()
        data = tarfile.open(en8path, 'r:bz2')
        data.extractall(tmpdir.name)
        docs = Path(tmpdir.name) / Path('enwiki8')
        for f in docs.iterdir():
            if f.stem.startswith('.'):
                f.unlink()
        yield docs
    finally:
        data.close()
        tmpdir.cleanup()
