import tensorflow as tf
import os
import numpy as np
import cv2
import sys
from tensorflow.python.client import timeline

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


frozen_graph  = "../alexnet.pb"
#frozen_graph = "../ssd_inception_v2_coco_2018_01_28/frozen_inference_graph.pb"
with tf.device('/device:GPU:0'):

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
	    #y_pred = graph.get_tensor_by_name("detection_boxes:0")
	    x = graph.get_tensor_by_name("data:0")
	    #x = graph.get_tensor_by_name("image_tensor:0")

	    # Creating the feed_dict that is required to be fed to calculate y_pred
	    feed_dict_testing = {x: x_batch}

	sess = tf.Session(graph=graph, config=tf.ConfigProto(allow_soft_placement=False, log_device_placement=False, gpu_options = tf.GPUOptions(force_gpu_compatible=True)))

	result = sess.run(y_pred, feed_dict=feed_dict_testing)
	run_metadata = tf.RunMetadata()
	result = sess.run(y_pred, options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
		      run_metadata=run_metadata, feed_dict=feed_dict_testing)
	# Create the Timeline object, and write it to a json
	tl = timeline.Timeline(run_metadata.step_stats)
	ctf = tl.generate_chrome_trace_format(show_dataflow=True, show_memory=True)
	with open('timeline.json', 'w') as f:
		f.write(ctf)

	ProfileOptionBuilder = tf.profiler.ProfileOptionBuilder

	# profile the timing of the operations
	opts = (ProfileOptionBuilder(ProfileOptionBuilder.time_and_memory())
	    .select(['accelerator_micros', 'micros', 'device', 'bytes', 'float_ops'])
	    .with_file_output("alexnet_profile.out")
	    .build())

	tf.profiler.profile(
		graph,
		cmd='code',
		run_meta=run_metadata,
		options=opts)

	# generate a timeline
	opts = (ProfileOptionBuilder(ProfileOptionBuilder.time_and_memory())
	    .with_step(0)
	    .select(['accelerator_micros', 'micros', 'bytes', 'float_ops'])
	    .with_timeline_output("alexnet_profile.json")
	    .build())

	tf.profiler.profile(
graph,
run_meta=run_metadata,
options=opts)

# print(result)

