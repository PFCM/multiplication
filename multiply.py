"""See if we can get a tensor decomposition to learn to multiply.
Start with elementwise AND, and move on to reals."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import progressbar

import mrnn


flags = tf.app.flags
# meta params
flags.DEFINE_bool('binary', False, 'if we true, we are trying to learn to AND,'
                  ' otherwise mutliply scalars.')
flags.DEFINE_string('operation', 'multiply', 'what we are actually trying to '
                    'learn, if `multiply` just a class AND/Hadamard, if '
                    '`permute_multiply` then we shuffle one of them first.'
                    'If it contains the word correlate then we will also make '
                    'the random data smaller and blow it up through a random '
                    'sparse (possibly binary) matrix to introduce some '
                    'structure.')
flags.DEFINE_float('structure', 0.5, 'if correlate, the fraction (potentially)'
                   ' greater than one) which controls the size of the random '
                   'vector we are are blowing up or squishing down to create '
                   'inputs.')
flags.DEFINE_float('validation', 10.0, 'if non-zero, scale factor applied to '
                   'validation inputs.')

# model params
flags.DEFINE_bool('size', 50, 'how big to make the things')
flags.DEFINE_integer('rank', 5, 'the rank of the decomposition')
flags.DEFINE_string('decomposition', 'cp', 'how to decompose the tensor')

# training params
flags.DEFINE_float('learning_rate', 0.1, 'learning rate for sgd')
flags.DEFINE_integer('batch_size', 10, 'how many to do at once')
flags.DEFINE_integer('max_steps', 50000, 'how long to train for')

FLAGS = flags.FLAGS


def _cp_product(input_a, input_b):
    """CANDECOMP/PARAFAC"""
    tensor = mrnn.get_cp_tensor([FLAGS.size] * 3, FLAGS.rank, 'CP_tensor')
    return mrnn.bilinear_product_cp(input_a, tensor, input_b)


def bilinear_product(input_a, input_b):
    """do the product"""
    if FLAGS.decomposition == 'cp':
        result = _cp_product(input_a, input_b)
    else:
        raise ValueError(
            'Unknown decomposition `{}`'.format(FLAGS.decomposition))
    return result


def get_inputs():
    """Gets input tensors, with parameters from FLAGS"""
    with tf.variable_scope('inputs'):
        if "correlate" in FLAGS.operation:
            random_size = int(FLAGS.size * FLAGS.structure)
            permute_matrix = tf.Variable(
                tf.random_uniform([random_size, FLAGS.size]))
        else:
            random_size = FLAGS.size
            permute_matrix = None
        random_source = tf.random_uniform([FLAGS.batch_size * 2, random_size])
        if FLAGS.binary:
            random_source = tf.round(random_source)
            if permute_matrix is not None:
                permute_matrix = tf.round(permute_matrix)
        if permute_matrix is not None:
            random_source = tf.matmul(random_source, permute_matrix)

        return tf.split(0, 2, random_source)


def combine_inputs(input_a, input_b):
    """Does the appropriate business to get the targets for our training"""
    with tf.variable_scope('targets') as scope:
        result = input_a * input_b
        if 'permute' in FLAGS.operation:
            # we need a consistent permutation here
            # not convinced that this is working
            permutation = tf.get_variable(
                'permutation',
                dtype=tf.int32,
                initializer=tf.random_shuffle(tf.range(FLAGS.size)),
                trainable=False,
                regularizer=tf.no_regularizer)
            # slightly awkwardly transpose, gather from the first index then
            # transpose it again.
            result = tf.transpose(tf.gather(tf.transpose(result), permutation))
        return result


def mse(vec_a, vec_b):
    """get mean squared error"""
    return tf.reduce_mean(tf.squared_difference(vec_a, vec_b))


def get_train_step(loss):
    """gets a training step op"""
    opt = tf.train.MomentumOptimizer(FLAGS.learning_rate, 0.99)
    return opt.minimize(loss)


def count_trainable_params():
    """literally count the trainable parameters"""
    dims = [var.get_shape().as_list() for var in tf.trainable_variables()]
    total = 0
    for shape in dims:
        prod = 1
        for dim in shape:
            prod *= dim
        total += prod
    return total


def l1_regularizer(amount):
    """Return an l1 regulariser"""

    def _reg(var):
        return tf.reduce_sum(tf.abs(var)) * amount

    return _reg


def main(_):
    """do the thing"""
    input_a, input_b = get_inputs()

    with tf.variable_scope('model', initializer=tf.random_normal_initializer(
     stddev=0.1)) as scope:

        targets = combine_inputs(input_a, input_b)

        model_outputs = bilinear_product(input_a, input_b)
        print('got model')
        if FLAGS.validation != 0.0:
            scope.reuse_variables()
            vinput_a = input_a * FLAGS.validation
            vinput_b = input_b * FLAGS.validation
            vtargets = combine_inputs(vinput_a, vinput_b)
            valid_outputs = bilinear_product(vinput_a, vinput_b)
            valid_loss = mse(vtargets, valid_outputs)
        else:
            valid_loss = None

    loss_op = mse(model_outputs, targets)
    train_op = get_train_step(loss_op)

    print('Got model with {} params'.format(count_trainable_params()))

    bar = progressbar.ProgressBar(
        widgets=['[', progressbar.Percentage(), '] ',
                 '(๑•̀ㅁ•́๑)✧',
                 progressbar.Bar(
                     marker='✧',
                     left='',
                     right=''),
                 '(', progressbar.DynamicMessage('loss'), ')',
                 '(', progressbar.AdaptiveETA(), ')'],
        redirect_stdout=True)

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    with sess.as_default():
        bar.start(FLAGS.max_steps)
        for step in range(FLAGS.max_steps):
            a, b, batch_loss, _ = sess.run(
                [input_a, input_b, loss_op, train_op])
            bar.update(step, loss=batch_loss)
            # print('a: {}'.format(a))
            # print('b: {}'.format(b))
            # import time; time.sleep(1)
        bar.finish()

        print(sess.run(tf.reduce_sum(tf.add_n(tf.trainable_variables()))))
        if valid_loss is not None:
            vloss, = sess.run([valid_loss])
            print('Validation loss: {}'.format(vloss))
            print('   (inputs scaled by {})'.format(FLAGS.validation))
        # print(sess.run(tf.trainable_variables()))

if __name__ == '__main__':
    tf.app.run()
