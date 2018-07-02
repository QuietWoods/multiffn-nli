# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals

import argparse
from itertools import izip
import tensorflow as tf

import utils
import ioutils

"""
Evaluate the performance of an NLI model on a dataset
"""


def print_errors(pairs, answers, label_dict):
    """
    Print the pairs for which the model gave a wrong answer,
    their gold label and the system one.
    """
    wrong_anser_result = open('wrong_anser_result.txt', 'w')
    for pair, answer in izip(pairs, answers):
        label_str = pair[2]
        label_number = label_dict[label_str]
        if answer != label_number:
            sent1 = ' '.join(pair[0])
            sent2 = ' '.join(pair[1])
            wrong_anser_result.write('Sent 1: {}\nSent 2: {}'.format(sent1, sent2))
            wrong_anser_result.write('System label: {}, gold label: {}'.format(answer,
                                                            label_number))
    wrong_anser_result.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', help='Directory with saved model')
    parser.add_argument('dataset',
                        help='JSONL or TSV file with data to evaluate on')
    parser.add_argument('embeddings', help='Numpy embeddings file')
    parser.add_argument('-vocabulary',
                        help='Text file with embeddings vocabulary')
    parser.add_argument('-v',
                        help='Verbose', action='store_true', dest='verbose')
    parser.add_argument('-e',
                        help='Print pairs and labels that got a wrong answer',
                        action='store_true', dest='errors')
    args = parser.parse_args()

    utils.config_logger(verbose=args.verbose)
    params = ioutils.load_params(args.model)
    sess = tf.InteractiveSession()

    model_class = utils.get_model_class(params)
    model = model_class.load(args.model, sess)
    word_dict, embeddings = ioutils.load_embeddings(args.embeddings,
                                                    args.vocabulary,
                                                    generate=False,
                                                    load_extra_from=args.model,
                                                    normalize=True)
    model.initialize_embeddings(sess, embeddings)
    label_dict = ioutils.load_label_dict(args.model)

    pairs = ioutils.read_corpus(args.dataset, None,
                                None)
    dataset = utils.create_dataset(pairs, word_dict, label_dict)
    loss, acc, answers = model.evaluate(sess, dataset, True, batch_size=500)
    print('Loss: %f' % loss)
    print('Accuracy: %f' % acc)

    if args.errors:
        print_errors(pairs, answers, label_dict)
