import argparse

import numpy

from .util import docs_tokens, random_window_gen, temp_test_corpus
from .vocab import VocabTable
from .fastcount import fast_count_batch


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
    """
    A ragged array of bags of words. Words are represented as indices into
    a VocabTable. `doc` is a sequence of the word indices contained by
    all the bags; `ends` describes the endpoints of each of the bags;
    `counts` contains the counts for each word in a given bag; `vocab`
    contains the VocabTable to be used for word attribute lookup.
    """
    def __init__(self, doc, ends, counts, vocab):
        self.doc = doc
        self.ends = ends
        self.counts = counts
        self.vocab = vocab
        self.potentials = self.vocab.potential[self.doc]

    @classmethod
    def from_text(cls, text, vocab, window=50, window_sigma=0.5,
                  downsample=1.0):
        """
        Generate a RagBag from a single text document (represented as a list
        of token strings).
        """
        encoded_doc = vocab[text]
        raw_bag_ends = numpy.fromiter(
            random_window_gen(window, window_sigma, len(encoded_doc)),
            dtype=numpy.int64
        )
        doc, ends, counts = fast_count_batch(encoded_doc, raw_bag_ends)

        ragbag = cls(doc, ends, counts, vocab)
        if downsample < 1.0:
            ragbag.downsample_common_words(downsample)

        return ragbag

    @classmethod
    def from_corpus(cls, corpus, vocab, window=50, window_sigma=0.5,
                    downsample=1.0):
        """
        Generate a RagBag from a collection of text documents by creating
        separate RagBags for each document, and then joining them together.
        """
        ragbags = [RagBag.from_text(d, vocab, window, window_sigma)
                   for d in docs_tokens(corpus)]
        megabag = RagBag.join_all(ragbags)

        # Faster to downsample all at once.
        if downsample < 1.0:
            megabag.downsample_common_words(downsample)
        return megabag

    @classmethod
    def from_numpy(cls, numpy_path):
        """
        Load a RagBag from a saved numpy archive.
        """
        data = numpy.load(numpy_path)
        vocab = VocabTable.from_export(data)
        return cls(data['RagBag__doc'],
                   data['RagBag__ends'],
                   data['RagBag__counts'],
                   vocab)

    def save_numpy(self, numpy_path, projection=False):
        """
        Save a RagBag to a numpy archive.
        """
        numpy.savez_compressed(
                numpy_path,
                RagBag__doc=self.doc,
                RagBag__ends=self.ends,
                RagBag__counts=self.counts,
                **self.vocab.export_numpy(projection))

    @classmethod
    def join_all(cls, ragbags):
        """
        Join together multiple RagBags based on the same VocabTable.
        If the RagBags are not based on the same table, throw a ValueError.
        """
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

    def downsample_common_words(self, threshold):
        # Identify words to downsample.
        downsample = self.vocab.frequency > threshold

        # Identify doc indices to downsample, repeating count times.
        downsample_doc_index = numpy.array([
            i for i in range(len(self.doc))
            for c in range(int(self.counts[i]))
            if downsample[self.doc[i]]
        ])

        # Fix downsample probability for each instance.
        downsample_probability = numpy.array([
            1 - (threshold / self.vocab.frequency[self.doc[i]]) ** 0.5
            for i in downsample_doc_index
        ])

        # Randomly select instances to downsample.
        rand = numpy.random.random(len(downsample_probability))
        to_downsample = downsample_doc_index[rand < downsample_probability]

        # Reduce count by one for each selected instance.
        for i in to_downsample:
            self.counts[i] -= 1

        # Avoid div by zero in log-based calculations.
        self.counts[self.counts == 0] = 1e-10


def corpus_argparser(subparser=None):
    if subparser is None:
        parser = argparse.ArgumentParser(
                description='Encode a corpus using a given vocabulary table.')
    else:
        parser = subparser

    parser.add_argument(
            'corpus_file',
            help="The name of the corpus file to be saved.")
    parser.add_argument(
            'vocab_file',
            help="The name of the input vocabulary file.",
            type=str)
    parser.add_argument(
            'docs',
            help="The name of an input folder containing plain text files.",
            type=str)
    parser.add_argument(
            '--downsample-threshold',
            help="A threshold for downsampling common words. Words that "
                 "appear in the corpus with a probability greater than this "
                 "value will be randomly downsampled with probability "
                 "dp = 1 - (t / p) ** 0.5.",
            type=float,
            default=1.0)
    parser.add_argument(
            '--window_size',
            help="The window size to use for creating \"sentences\". (This "
                 "simple model creates them by randomly slicing input text "
                 "into chunks roughly this many words long.)",
            type=int,
            default=50)

    return parser


def main(args):
    if args.vocab_file.endswith('.npz'):
        vocab = VocabTable.from_numpy(args.vocab_file)
    else:
        vocab = VocabTable.from_csv(args.vocab_file)

    if args.docs == '-':
        with temp_test_corpus() as docs:
            megabag = RagBag.from_corpus(
                docs,
                vocab,
                window=args.window_size,
                downsample=args.downsample_threshold
            )
    else:
        megabag = RagBag.from_corpus(
            args.docs,
            vocab,
            window=args.window_size,
            downsample=args.downsample_threshold
        )
    megabag.save_numpy(args.corpus_file)


if __name__ == '__main__':
    parser = corpus_argparser()
    main(parser.parse_args())
