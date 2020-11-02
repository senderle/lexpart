import argparse
import re

import numpy

from .embed import Embedding


def wordsim_argparser(subparser=None):
    if subparser is None:
        parser = argparse.ArgumentParser(
                description='Create a word embedding model from a corpus.')
    else:
        parser = subparser

    parser.add_argument(
            'embedding_file',
            help='The name of the embedding file.',
            type=str)
    parser.add_argument(
            'query',
            help='Word similarity query.')
    return parser


def main(args):
    embed = Embedding.from_numpy(args.embedding_file)
    vecs = embed.get_word_vectors()

    query_words = re.split(r'([-+])', args.query)

    # Extremely quick-and-dirty query parsing.

    # If a sign precedes the first term, there will be an empty string at
    # the head. In that case, the sign is explicit. Otherwise, we assume
    # that the sign is positive.
    if query_words[0] == '':
        query_words = query_words[1:]
    else:
        query_words = ['+'] + query_words
    query_pairs = zip(query_words[0::2], query_words[1::2])
    query_vecs = [(1 if sign == '+' else -1) * vecs[embed.vocab.index[word]]
                  for sign, word in query_pairs]

    query_vecs = numpy.array(query_vecs)
    query_vecs = query_vecs.sum(axis=0)

    sim = vecs @ query_vecs.reshape(-1, 1)
    print(embed.vocab.word[(-sim).argsort(axis=0)[:10]])


if __name__ == '__main__':
    parser = wordsim_argparser()
    main(parser.parse_args())
