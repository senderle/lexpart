import argparse

import numpy

from util import docs_tokens, random_window_gen
from vocab import VocabTable
from fastcount import fast_count_batch


def array_join(arrs):
    out_len = sum(len(a) for a in arrs)
    out = numpy.empty(out_len, dtype=arrs[0].dtype)
    start = 0
    for a in arrs:
        end = start + len(a)
        out[start: end] = a
        start = end
    return out


def array_join_cumsum(arrs):
    out_len = sum(len(a) for a in arrs)
    out = numpy.empty(out_len, dtype=arrs[0].dtype)
    start = 0
    lastval = 0
    for a in arrs:
        end = start + len(a)
        out[start: end] = a + lastval
        start = end
        lastval = out[end - 1]
    return out


class RagBag:
    def __init__(self, doc, ends, counts, vocab):
        self.doc = doc
        self.ends = ends
        self.counts = counts
        self.vocab = vocab
        self.potentials = self.vocab.potential[self.doc]

    @classmethod
    def from_corpus_doc(cls, corpus_doc, vocab, window=50, window_sigma=0.5):
        encoded_doc = vocab[corpus_doc]
        raw_bag_ends = numpy.fromiter(
            random_window_gen(window, window_sigma, len(encoded_doc)),
            dtype=numpy.int64
        )
        doc, ends, counts = fast_count_batch(encoded_doc, raw_bag_ends)
        return cls(doc, ends, counts, vocab)

    @classmethod
    def from_corpus(cls, corpus, vocab):
        ragbags = [RagBag.from_corpus_doc(d, vocab)
                   for d in docs_tokens(corpus)]
        return RagBag.join(ragbags)

    @classmethod
    def from_numpy(cls, numpy_path):
        data = numpy.load(numpy_path)
        vocab = VocabTable.from_export(data)
        return cls(data['RagBag_doc'],
                   data['RagBag_ends'],
                   data['RagBag_counts'],
                   vocab)

    def save_numpy(self, numpy_path, projection=False):
        numpy.savez_compressed(
                numpy_path,
                RagBag_doc=self.doc,
                RagBag_ends=self.ends,
                RagBag_counts=self.counts,
                **self.vocab.export_numpy(projection))

    @classmethod
    def join(cls, ragbags):
        # Take multiple ragbags, check that they are all
        # based on the same VocabTable, throw an error if not, and
        # join them into one big megabag if so.
        vocab = ragbags[0].vocab
        for rb in ragbags[1:]:
            if rb.vocab != vocab:
                raise ValueError(
                    'RagBags based on different VocabTables '
                    'cannot be joined.'
                )
        doc = array_join([rb.doc for rb in ragbags])
        ends = array_join_cumsum([rb.ends for rb in ragbags])
        counts = array_join([rb.counts for rb in ragbags])
        return RagBag(doc, ends, counts, vocab)


def corpus_argparser(subparser=None):
    if subparser is None:
        parser = argparse.ArgumentParser(
                description='Encode a corpus using a given vocabulary table.')
    else:
        parser = subparser

    parser.add_argument(
            'vocab_file',
            help="The name of the vocabulary file.",
            type=str)
    parser.add_argument(
            'docs',
            help="A path to a folder containing plain text files.",
            type=str)
    parser.add_argument(
            'corpus_file',
            help="The name of the corpus file.")

    return parser


def main(args):
    if args.vocab_file.endswith('.npz'):
        vocab = VocabTable.from_numpy(args.vocab_file)
    else:
        vocab = VocabTable.from_csv(args.vocab_file)

    megabag = RagBag.from_corpus(args.docs, vocab)
    megabag.save_numpy(args.corpus_file)


if __name__ == '__main__':
    parser = corpus_argparser()
    main(parser.parse_args())
