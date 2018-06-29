# -*- coding: utf-8 -*-

from __future__ import division, print_function

"""
Script to train an RTE LSTM.
"""

import sys
import argparse
import tensorflow as tf

import ioutils
import utils
import namespace_utils
from classifiers.lstm import LSTMClassifier
from classifiers.multimlp import MultiFeedForwardClassifier
from classifiers.decomposable import DecomposableNLIModel
from atec_data_process import AtecCorpus


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--embeddings', default='glove/vectors.txt',
                        help='Text or numpy file with word embeddings')
    parser.add_argument('--train_path', dest='train_path', default='data/train_90000.csv',
                        help='CSV file with training corpus')
    parser.add_argument('--dev_path', dest='dev_path', default='data/dev_12477.csv',
                        help='CSV file with validation corpus')
    parser.add_argument('--save', dest='save', default='saved-model',
                        help='Directory to save the model files')
    parser.add_argument('--model_type',dest='model_type', default='lstm',
                        help='Type of architecture',
                        choices=['lstm', 'mlp'])
    parser.add_argument('--vocab', help='Vocabulary file (only needed if numpy'
                                        'embedding file is given)')
    parser.add_argument('--num_epochs', dest='num_epochs', default=300, type=int,
                        help='Number of epochs')
    parser.add_argument('--batch_size', dest='batch_size', default=64, help='Batch size',
                        type=int)
    parser.add_argument('--num_units', dest='num_units', help='Number of hidden units',
                        default=200, type=int)
    parser.add_argument('--no-proj', help='Do not project input embeddings to '
                                          'the same dimensionality used by '
                                          'internal networks',
                        action='store_false', dest='no_project')
    parser.add_argument('--dropout', dest='dropout', help='Dropout keep probability',
                        default=0.8, type=float)
    parser.add_argument('--clip_norm', dest='clip_norm', help='Norm to clip training '
                                                     'gradients',
                        default=100, type=float)
    parser.add_argument('-lr', help='Learning rate', type=float, default=0.001,
                        dest='rate')
    parser.add_argument('--lang', choices=['en', 'pt'], default='en',
                        help='Language (default en; only affects tokenizer)')
    parser.add_argument('--lower', help='Lowercase the corpus (use it if the '
                                        'embedding model is lowercased)',
                        action='store_true')
    parser.add_argument('--use-intra', help='Use intra-sentence attention',
                        action='store_true', dest='use_intra')
    parser.add_argument('--l2', help='L2 normalization constant', type=float,
                        default=0.0)
    parser.add_argument('--report', help='Number of batches between '
                                         'performance reports',
                        default=100, type=int)
    parser.add_argument('-v', default=True, help='Verbose', action='store_true',
                        dest='verbose')
    parser.add_argument('--optim', help='Optimizer algorithm',
                        default='adagrad',
                        choices=['adagrad', 'adadelta', 'adam'])
    # config file
    parser.add_argument('--config_path', type=str, help='Configuration file.')
    args, unparsed = parser.parse_known_args()
    utils.config_logger(args.verbose)
    logger = utils.get_logger('train')
    if args.config_path is not None:
        logger.info('Loading the configuration from ' + args.config_path)
        args = namespace_utils.load_namespace(args.config_path)
    # 实时输出，无缓存
    sys.stdout.flush()
    logger.debug("参数设置：{}".format(args.__dict__))

    logger.debug('Training with following options: %s' % ' '.join(sys.argv))
    atec_corpus = AtecCorpus()
    train_pairs = atec_corpus.read_corpus(args.train)
    valid_pairs = atec_corpus.read_corpus(args.validation)

    # whether to generate embeddings for unknown, padding, null
    word_dict, embeddings = ioutils.load_embeddings(args.embeddings, args.vocab,
                                                    True, normalize=True)

    logger.info('Converting words to indices')
    # find out which labels are there in the data
    # (more flexible to different datasets)
    label_dict = utils.create_label_dict(train_pairs)
    train_data = utils.create_dataset(train_pairs, word_dict, label_dict)
    valid_data = utils.create_dataset(valid_pairs, word_dict, label_dict)

    ioutils.write_params(args.save, lowercase=None, language=None,
                         model=args.model)
    ioutils.write_label_dict(label_dict, args.save)
    ioutils.write_extra_embeddings(embeddings, args.save)

    msg = '{} sentences have shape {} (firsts) and {} (seconds)'
    logger.debug(msg.format('Training',
                            train_data.sentences1.shape,
                            train_data.sentences2.shape))
    logger.debug(msg.format('Validation',
                            valid_data.sentences1.shape,
                            valid_data.sentences2.shape))

    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

    logger.info('Creating model')
    vocab_size = embeddings.shape[0]
    logger.info('vocab_size: {}'.format(vocab_size))
    embedding_size = embeddings.shape[1]
    logger.info('embedding_size: {}'.format(embedding_size))

    if args.model == 'mlp':
        model = MultiFeedForwardClassifier(args.num_units, 3, vocab_size,
                                           embedding_size,
                                           use_intra_attention=args.use_intra,
                                           training=True,
                                           project_input=args.no_project,
                                           optimizer=args.optim)
    else:
        model = LSTMClassifier(args.num_units, 3, vocab_size,
                               embedding_size, training=True,
                               project_input=args.no_project,
                               optimizer=args.optim)

    model.initialize(sess, embeddings)

    # this assertion is just for type hinting for the IDE
    assert isinstance(model, DecomposableNLIModel)

    total_params = utils.count_parameters()
    logger.debug('Total parameters: %d' % total_params)

    logger.info('Starting training')
    model.train(sess, train_data, valid_data, args.save, args.rate,
                args.num_epochs, args.batch_size, args.dropout, args.l2,
                args.clip_norm, args.report)
