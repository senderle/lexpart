import argparse
import csv
from collections import Counter
from pathlib import Path


import numpy
from pyhash import city_64

from util import docs_tokens


def stable_random_matrix(words, dimension, _hashfunc=city_64(0)):
    """
    Generate a random projection vector for each word in a given
    vocabulary. The projection vector for two identical strings will
    be identical (but randomly selected from among all possible vectors,
    subject to the limits of the RNG's state space).

    This borrows heavily from Ben Schmidt's Stable Random Projection
    scheme (https://culturalanalytics.org/article/11033), but instead
    of directly using the binary hash vectors, it passes them to a
    RNG as a seed, and uses the RNG to generate samples from a gaussian.
    Gaussian projection seems to produce better results for this use
    case.
    """
    seeds = [_hashfunc(str(w)) & (2 ** 32 - 1) for w in words]
    out = []
    for s in seeds:
        numpy.random.seed(s)
        out.append(numpy.random.normal(0.0, 1 / 3, size=dimension))
    return numpy.array(out)


class VocabIndex:
    def __init__(self, vocab):
        self.vocab = vocab
        self.index = {w: i for i, w in enumerate(self.vocab)}

    def __getitem__(self, word):
        if isinstance(word, str):
            return self.index[word] if word in self.index else -1
        else:
            return numpy.array([self.index[w] for w in word
                                if w in self.index])

    def __eq__(self, other):
        # Take a shortcut; identity is sufficient for equality.
        if self is other:
            return True
        else:
            return ((self.vocab == other.vocab) and
                    (self.index == other.index))


# TODO: Add a mechanism for selecting a potential function.
class VocabTable:
    def __init__(self, word, count, potential, ndims=300):
        self.word = word
        self.count = count
        self.potential = potential
        self.index = VocabIndex(self.word)
        self.projection = None
        self.create_projection(ndims)

    def create_projection(self, ndims):
        self.projection = stable_random_matrix(self.word, ndims)

    @property
    def ndims(self):
        return self.projection.shape[1]

    @property
    def nwords(self):
        return len(self.word)

    @classmethod
    def from_csv(cls, csv_path):
        with open(csv_path, 'r') as ip:
            rows = list(csv.reader(ip))

        word = numpy.array([r[0] for r in rows])
        count = numpy.array([r[1] for r in rows])
        potential = numpy.array([r[2] for r in rows])
        return cls(word, count, potential)

    @classmethod
    def from_numpy(cls, numpy_path):
        data = numpy.load(numpy_path)
        word = data['word']
        count = data['count']
        potential = data['potential']
        if 'projection' in data:
            # Assume that the projection can't be recreaed using
            # stable random projection.
            projection = data['projection']
            instance = cls(word, count, potential)
            instance.projection = projection
            return instance
        elif 'ndims' in data:
            ndims = data['ndims'].item()
            return cls(word, count, potential, ndims)
        else:
            return cls(word, count, potential)

    @classmethod
    def from_corpus(cls, docpath, vocab_max, min_count, ndims=300):
        ct = Counter()
        for toks in docs_tokens(docpath):
            ct.update(toks)
        ct = Counter({w: c for w, c in ct.items() if c >= min_count})

        word, count = map(numpy.array, zip(*ct.most_common(vocab_max)))
        potential = numpy.ones(len(count))
        return cls(word, count, potential, ndims)

    def __getitem__(self, ix):
        return self.index[ix]

    def __eq__(self, other):
        # Take a shortcut; identity is sufficient for equality.
        if self is other:
            return True
        else:
            return ((self.word == other.word).all() and
                    (self.count == other.count).all() and
                    (self.potential == other.potential).all() and
                    (self.index == other.index))

    def save_numpy(self, numpy_path, save_projection=False):
        """
        Save the current VocabTable to a compressed numpy file. Most of
        the time, the projeciton can be recreated using the stable random
        projection scheme, but in situations where it can't, we can set
        `save_projeciton` to `True` to preserve it.
        """
        if save_projection:
            numpy.savez_compressed(
                    numpy_path,
                    word=self.word,
                    count=self.count,
                    potential=self.potential,
                    ndims=self.ndims)
        else:
            numpy.savez_compressed(
                    numpy_path,
                    word=self.word,
                    count=self.count,
                    potential=self.potential,
                    projection=self.projection)

    def save_csv(self, csv_path):
        csv_path = Path(csv_path)
        if csv_path.suffix != 'csv':
            csv_path = csv_path.with_suffix('.csv')
        with open(csv_path, 'w', encoding='utf-8', newline='') as op:
            wr = csv.writer(op)
            wr.writerows(zip(self.word, self.count, self.potential))


def vocab_argparser(subparser=None):
    if subparser is None:
        parser = argparse.ArgumentParser(
                description='Create a vocabulary file')
    else:
        parser = subparser

    parser.add_argument(
            'docs',
            help="A path to a folder containing plain text files.",
            type=str)
    parser.add_argument(
            'vocab_file',
            help="The name of the vocabulary file.",
            type=str)
    parser.add_argument(
            '--max-vocab',
            help="The maximum number of terms to include in the "
                 "vocabulary file.",
            type=int,
            default=300_000)
    parser.add_argument(
            '--min-count',
            help="The minimum number of times a word must appear in "
                 "the corpus to be included in the vocabulary file.",
            type=int,
            default=5)
    parser.add_argument(
            '--csv',
            help="Save the vocabulary as a CSV file. Default is false, "
                 "meaning that the vocabulary will be saved as a compressed "
                 ".npz file.",
            action="store_true",
            default=False)

    return parser


def main(args):
    vocab = VocabTable.from_corpus(args.docs, args.max_vocab, args.min_count)
    if args.csv:
        vocab.save_csv(args.vocab_file)
    else:
        vocab.save_numpy(args.vocab_file)


if __name__ == '__main__':
    parser = vocab_argparser()
    main(parser.parse_args())
