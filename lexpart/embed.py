import argparse

import numpy

from .vocab import VocabTable
from .corpus import RagBag
from .sparsehess import hessian_projection


class Embedding:
    def __init__(self, vocab, partition=None, jacobian=None,
                 p_hessian=None, scale=None):
        """
        Create a new embedding. Only the associated VocabTable must be
        provided. If only the VocabTable is provided, the resulting
        embedding will be uninitialized, and will need to be updated with
        a corpus using the `partition_update` method before it can be
        used.

        If `partition`, `jacobian`, `p_hessian`, and `scale` are provided,
        this is effectively a copy constructor.

        To create an embedding directly from a corpus, use the `from_corpus`
        classmethod.
        """
        self.vocab = vocab

        # Components of the embedding:

        # The sum of the partition function's polynomial terms.
        if partition is None:
            self.partition = None
        else:
            # This must be a zero-dim numpy array to work with
            # the numba partition function calculating routines.
            self.partition = numpy.zeros(())
            self.partition[()] = partition

        # The sum of the jacobians of the polynomial terms
        # with respect to all word types.
        self.jacobian = jacobian

        # The sum of the hessians of the polynomial terms with respect to
        # pairs of word types. Since the hessian is prohibitively large
        # to store, the hessians are reduced using a stable random
        # projection scheme (https://culturalanalytics.org/article/11033).
        # Hence the `p_` prefix.
        self.p_hessian = p_hessian

        # As more embeddings are joined together, the risk of overflow
        # grows, and eventually we will need to rescale the embeddings.
        # We keep track of the current scale and pass it to
        # `hessian_projection`, which will scale its output accordingly
        # so that all corpora are equally weighted in the final sum.
        self.scale = scale

    @classmethod
    def from_corpus(cls, corpus):
        """
        Create an embedding directly from a corpus. This initializes the
        partition function values using the given corpus.
        """
        emb = cls(corpus.vocab)
        emb.partition_update(corpus)
        return emb

    @classmethod
    def from_numpy(cls, numpy_path):
        """
        Load an existing embedding from a numpy archive.
        """
        data = numpy.load(numpy_path)
        vocab = VocabTable.from_export(data)
        emb = cls(vocab)
        emb.partition = data['Embeddding__partition']
        emb.jacobian = data['Embeddding__jacobian']
        emb.p_hessian = data['Embeddding__p_hessian']
        emb.scale = data['Embeddding__scale']
        return emb

    def save_numpy(self, numpy_path):
        """
        Save the current embedding to a numpy archive.
        """
        numpy.savez_compressed(
                numpy_path,
                Embeddding__partition=self.partition,
                Embeddding__jacobian=self.jacobian,
                Embeddding__p_hessian=self.p_hessian,
                Embeddding__scale=self.scale,
                **self.vocab.export_numpy(projection=True))

    @property
    def nwords(self):
        return self.vocab.nwords

    @property
    def ndims(self):
        return self.vocab.ndims

    def partition_update(self, corpus):
        """
        Calculate the partition value, jacobian, and projected hessian of
        the given corpus, and initialize the embedding with the result, or
        add them to the current embedding values.
        """
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

        if self.scale is None:
            self.scale = 1.0

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

    def rescale(self, scale):
        """
        Rescale the current embedding. This is sometimes necessary to avoid
        overflow errors as embeddings are joined together. Current partition
        function values are rescaled to match the new value, and future
        updates to the partition function will be scaled by the new value.
        """
        if self.scale is None:
            raise ValueError(
                'Embedding is empty and cannot be rescaled. Initialize the '
                'embedding with at least one corpus using the '
                '`partition_update` method.'
            )

        if self.scale != scale:
            rescale_ratio = scale / self.scale
            self.partition *= rescale_ratio
            self.jacobian *= rescale_ratio
            self.p_hessian *= rescale_ratio
            self.scale = scale

    @classmethod
    def join_all(cls, embeddings):
        """
        Join together multiple Embeddings based on the same VocabTable.
        If the Embeddings are not based on the same VocabTable, throw a
        ValueError.
        """
        vocab = embeddings[0].vocab
        for emb in embeddings[1:]:
            if emb.vocab != vocab:
                raise ValueError(
                    'Embeddings based on different VocabTables '
                    'cannot be joined.'
                )

        min_scale = min(emb.scale for emb in embeddings)
        for emb in embeddings:
            emb.rescale(min_scale)

        partition = numpy.zeros(())
        partition[()] = sum(emb.partition for emb in embeddings)

        jacobian = numpy.array([emb.jacobian for emb in embeddings])
        jacobian = jacobian.sum(axis=0)

        p_hessian = numpy.array([emb.p_hessian for emb in embeddings])
        p_hessian = p_hessian.sum(axis=0)

        new_emb = cls(vocab, partition, jacobian, p_hessian, min_scale)

        # Rescale by 256x to leave lots of headroom for future joins.
        if new_emb.partition > 2 ** 36:
            new_emb.rescale(2 ** -8)
        return new_emb

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

        This is pretty easy because all the operators involved are
        linear.
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

        # Normalize vectors so that dot product is the same as cosine sim.
        mag = (vecs * vecs).sum(axis=1) ** 0.5
        # Avoid divide by zero for empty vectors.
        mag[mag == 0] = 1
        return vecs / mag.reshape(-1, 1)


def embed_argparser(subparser=None):
    if subparser is None:
        parser = argparse.ArgumentParser(
                description='Create a word embedding model from a corpus.')
    else:
        parser = subparser

    parser.add_argument(
            'embedding_file',
            help='The name of the embedding file to be saved.',
            type=str)
    parser.add_argument(
            'corpus_file',
            help='The name of the input corpus file.',
            type=str)
    return parser


def main(args):
    corpus = RagBag.from_numpy(args.corpus_file)

    embedding = Embedding.from_corpus(corpus)
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
