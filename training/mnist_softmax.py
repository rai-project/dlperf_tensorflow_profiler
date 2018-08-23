from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

FLAGS = None

def main(_):
    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    # Create the model
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.matmul(x, W) + b

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10])

    # The raw formulation of cross-entropy,
    #
    #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
    #                                 reduction_indices=[1]))
    #
    # can be numerically unstable.
    #
    # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
    # outputs of 'y', and then average across the batch.
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # Profiler is created here.
        profiler = tf.profiler.Profiler(sess.graph)
        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        # Train
        n_steps = FLAGS.steps
        for i in range(n_steps):
            batch_xs, batch_ys = mnist.train.next_batch(100)

            run_metadata = tf.RunMetadata()
            sess.run(train_step, options=options, run_metadata=run_metadata, feed_dict={x: batch_xs, y_: batch_ys})
            # We collect profiling infos for each step.
            profiler.add_step(i, run_metadata)

        # Test trained model
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        run_metadata = tf.RunMetadata()
        print(sess.run(accuracy, options=options, run_metadata=run_metadata, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
        # I collect profiling infos for last step, too.
        profiler.add_step(n_steps, run_metadata)

        option_builder = tf.profiler.ProfileOptionBuilder
        opts = (option_builder(option_builder.time_and_memory()).
                with_step(-1). # with -1, should compute the average of all registered steps.
                with_file_output('test-%s.txt' % FLAGS.out).
                select(['micros','bytes','occurrence']).order_by('micros').
                build())
        # Profiling infos about ops are saved in 'test-%s.txt' % FLAGS.out
        profiler.profile_operations(options=opts)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    parser.add_argument('--steps', type=int, default=10, help='Number of steps to run.')
    parser.add_argument('--out', type=str, required=True, help='Output filename.')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
