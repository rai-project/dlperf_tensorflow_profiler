import tensorflow as tf
import os
import numpy as np
import os
import glob
import cv2
import sys
import argparse

# First, pass the path of the image
dir_path = os.path.dirname(os.path.realpath(__file__))
image_path = sys.argv[1]
filename = dir_path + '/' + image_path
image_size = 227
num_channels = 3
images = []
# Reading the image using OpenCV
image = cv2.imread(filename)
# Resizing the image to our desired size and preprocessing will be done exactly as done during training
image = cv2.resize(image, (image_size, image_size), cv2.INTER_LINEAR)
images.append(image)
images = np.array(images, dtype=np.uint8)
images = images.astype('float32')
images = np.multiply(images, 1.0/255.0)
# The input to the network is of shape [None image_size image_size num_channels]. Hence we reshape.
x_batch = images.reshape(1, image_size, image_size, num_channels)


frozen_graph = "./alexnet.pb"
with tf.gfile.GFile(frozen_graph, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

with tf.Graph().as_default() as graph:
    tf.import_graph_def(
        graph_def,
        input_map=None,
        return_elements=None,
        name="",
        # op_dict=None,
        # producer_op_list=None
    )

    y_pred = graph.get_tensor_by_name("prob:0")
    x = graph.get_tensor_by_name("data:0")

    sess = tf.Session(graph=graph)

    # Creating the feed_dict that is required to be fed to calculate y_pred
    feed_dict_testing = {x: x_batch}

    # Create profiler
    profiler = tf.profiler.Profiler(sess.graph)
    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()

    result = sess.run(y_pred, options=options,
                      run_metadata=run_metadata, feed_dict=feed_dict_testing)

    profiler.add_step(0, run_metadata)

    # Generate profile
    option_builder = tf.profiler.ProfileOptionBuilder
    opts = (option_builder(option_builder.time_and_memory()).
            # with -1, should compute the average of all registered steps.
            with_step(-1).
            with_file_output('profile_alexnet.out').
            select(['micros', 'bytes', 'occurrence']).order_by('micros').
            build())

    profiler.profile_operations(options=opts)


# result is of this format [probabiliy_of_cats probability_of_dogs]
    print(result)

