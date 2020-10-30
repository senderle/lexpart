import argparse

import vocab
import corpus
import embed


def main_argparser():
    parser = argparse.ArgumentParser(
            description='Create a word embedding model based on a '
                        'thermodynamic partition function.')
    subparsers = parser.add_subparsers(
            help='Available commands. Type -h after a command for '
                 'detailed help.')

    vocab_parser = subparsers.add_parser(
            'vocab',
            help='Read a corpus and create a vocabulary table.')
    vocab_parser = vocab.vocab_argparser(vocab_parser)
    vocab_parser.set_defaults(func=vocab.main)

    corpus_parser = subparsers.add_parser(
            'corpus',
            help='Encode a corpus based on a vocabulary table.')
    corpus_parser = corpus.corpus_argparser(corpus_parser)
    corpus_parser.set_defaults(func=corpus.main)

    embed_parser = subparsers.add_parser(
            'embed',
            help='Create a word embedding model.')
    embed_parser = embed.embed_argparser(embed_parser)
    embed_parser.set_defaults(func=embed.main)

    return parser


if __name__ == '__main__':
    parser = main_argparser()
    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()
