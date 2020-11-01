import argparse

import numpy

from vocab import VocabTable
from corpus import RagBag
from sparsehess import hessian_projection


class Embedding:
    def __init__(self, vocab_or_corpus):
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

        if isinstance(vocab_or_corpus, VocabTable):
            self.vocab = vocab_or_corpus
        else:
            self.vocab = vocab_or_corpus.vocab
            self.partition_update(vocab_or_corpus)

    @classmethod
    def from_numpy(cls, numpy_path):
        data = numpy.load(numpy_path)
        vocab = VocabTable.from_export(data)
        emb = cls(vocab)
        emb.partition = data['Embedding_partition']
        emb.jacobian = data['Embedding_jacobian']
        emb.p_hessian = data['Embedding_p_hessian']
        emb.scale = data['Embedding_scale']
        return emb

    def save_numpy(self, numpy_path):
        numpy.savez_compressed(
                numpy_path,
                Embedding_partition=self.partition,
                Embedding_jacobian=self.jacobian,
                Embedding_p_hessian=self.p_hessian,
                Embedding_scale=self.scale,
                **self.vocab.export_numpy(projection=True))

    @property
    def nwords(self):
        return self.vocab.nwords

    @property
    def ndims(self):
        return self.vocab.ndims

    def partition_update(self, corpus):
        if self.vocab != corpus.vocab:
            raise ValueError(
                'Corpus VocabTable and Embedding VocabTable are incompatible.'
            )

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

        # Then we subtract `normed_p_jacobian_outer` from
        # `normed_p_hessian` and return the result.
        vecs = normed_p_hessian - normed_p_jacobian_outer

        # Cosine normalization
        mag = (vecs * vecs).sum(axis=1) ** 0.5
        return vecs / mag.reshape(-1, 1)


def embed_argparser(subparser=None):
    if subparser is None:
        parser = argparse.ArgumentParser(
                description='Create a word embedding model from a corpus.')
    else:
        parser = subparser

    parser.add_argument(
            'corpus_file',
            help='The name of the corpus file.',
            type=str)
    parser.add_argument(
            'embedding_file',
            help='The name of the embedding file.',
            type=str)
    return parser


def main(args):
    corpus = RagBag.from_numpy(args.corpus_file)

    embedding = Embedding(corpus)
    embedding.save_numpy(args.embedding_file)
    vecs = embedding.get_word_vectors()

    print(f'    Final embedding stats:')
    rows, cols = vecs.shape
    print(f'      {rows} {cols}-d vectors')
    vecs_rav = vecs.ravel()
    vecs_max = vecs_rav.max()
    vecs_min = vecs_rav.min()
    vecs_avg = vecs_rav.mean()
    print(f'      embed max: {vecs_max:.4g}  '
          f'min: {vecs_min:.4g}  avg: {vecs_avg:.4g}')


if __name__ == '__main__':
    parser = embed_argparser()
    main(parser.parse_args())
