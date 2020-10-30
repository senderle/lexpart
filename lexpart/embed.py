import argparse

import numpy

from vocab import VocabTable
from corpus import RagBag
from sparsehess import hessian_projection


class Embedding:
    def __init__(self, vocab):
        self.vocab = vocab

        # Components of the embedding:

        # The sum of the partition function's polynomial terms.
        self.partition = None

        # The sum of the jacobians of the polynomial terms
        # with respect to all word types.
        self.jacobian = None

        # The sum of the hessians of the polynomial terms with respect to
        # pairs of word types. Since the hessian is prohibitively large
        # to store, the hessians are reduced using a stable random
        # projection scheme (https://culturalanalytics.org/article/11033).
        # Hence the `p_` prefix.
        self.p_hessian = None

        # As more embeddings are joined together, the risk of overflow
        # grows, and eventually we might want to rescale the embeddings.
        # We keep track of the current scale and pass it to
        # `hessian_projection`, which will scale its output accordingly
        # so that all corpora are equally weighted in the final sum.
        self.scale = 1.0

    @property
    def nwords(self):
        return self.vocab.nwords

    @property
    def ndims(self):
        return self.vocab.ndims

    def partition_update(self, corpus):
        if self.partition is None:
            self.partition = numpy.zeros(())

        if self.jacobian is None:
            self.jacobian = numpy.zeros(self.nwords)

        if self.p_hessian is None:
            self.p_hessian = numpy.zeros((self.nwords, self.ndims))

        hessian_projection(
                self.partition,
                self.jacobian,
                self.p_hessian,
                corpus.ends,
                corpus.doc,
                corpus.counts,
                corpus.potentials,
                self.vocab.projection,
                self.scale)

        # Insert stuff from around lines 703 in docembed.py
    def get_word_vectors(self):
        """
        Do the final calculations to convert the (randomly projected)
        hessian of the partition function into a (randomly projected)
        covariance matrix.

        The covariance matrix of the partition function is given by the
        hessian of the _logarithm_ of the partition function. And in
        general, the hessian of the log of any scalar-valued function of
        multiple variables is equal to the hessian of the function divided
        by the value of the function, minus the outer product of the
        jacobian divided by the squared value of the function. This
        function performs those operations, accounting for the fact
        that the hessian has been randomly projected, and returns
        the result.
        """

        # Normalize the hessian by the value of the partition function.
        print(self.partition)
        print(self.p_hessian[:5, :5])
        print(self.jacobian[:5])
        normed_p_hessian = self.p_hessian / self.partition

        # Normalize the jacobian by the value of the partition function.
        # When we take the outer product of this later, the partition
        # function value will be implicitly squared.
        normed_jacobian = self.jacobian / self.partition

        # The jacobian hasn't been randomly projected yet. If we were
        # to take the outer product here, we'd get a huge matrix, so
        # instead we perform the projection step first.
        normed_p_jacobian = normed_jacobian @ self.vocab.projection

        # The outer product here is now tractable. `normed_jacobian`
        # becomes a very long column vector; `normed_p_jacobian` becomes
        # relatively small row vector. The result has the same shape
        # as `normed_p_hessian`.
        normed_p_jacobian_outer = (normed_jacobian.reshape(-1, 1) @
                                   normed_p_jacobian.reshape(1, -1))

        # Finally, we subtract `normed_p_jacobian_outer` from
        # `normed_p_hessian` and return the result.
        return normed_p_hessian - normed_p_jacobian_outer


def postprocess_word_vectors(vecs):
    return vecs


def embed_argparser(subparser=None):
    if subparser is None:
        parser = argparse.ArgumentParser(
                description='Create a word embedding model from a corpus.')
    else:
        parser = subparser

    parser.add_argument(
            'vocab_file',
            help='The name of the vocabulary file.',
            type=str)
    parser.add_argument(
            'corpus_file',
            help='The name of the corpus file.',
            type=str)
    return parser


def main(args):
    if args.vocab_file.endswith('.npz'):
        vocab = VocabTable.from_numpy(args.vocab_file)
    else:
        vocab = VocabTable.from_csv(args.vocab_file)

    corpus = RagBag.from_numpy(args.corpus_file, args.vocab_file)

    embedding = Embedding(vocab)
    embedding.partition_update(corpus)
    vecs = embedding.get_word_vectors()
    vecs = postprocess_word_vectors(vecs)
    print(vecs.shape)


if __name__ == '__main__':
    parser = embed_argparser()
    main(parser.parse_args())
