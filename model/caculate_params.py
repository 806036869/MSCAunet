import tensorflow as tf

def load_pb(pb):
    with tf.gfile.GFile(pb, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='prefix')
        return graph

# #
def stats_graph(graph):
    flops = tf.compat.v1.profiler.profile(graph, options=tf.compat.v1.profiler.ProfileOptionBuilder.float_operation())
    params = tf.compat.v1.profiler.profile(graph, options=tf.compat.v1.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    print('FLOPs: {};    Trainable params: {}'.format(flops.total_float_ops, params.total_parameters))

# with tf.compat.v1.Session() as sess:
#     sess.run(tf.compat.v1.global_variables_initializer())
#     graph = tf.compat.v1.get_default_graph()
#     stats_graph(graph)
#========================================================================================

#
# def get_flops_params():
#     sess = tf.compat.v1.Session()
#     graph = sess.graph
#     flops = tf.compat.v1.profiler.profile(graph, options=tf.compat.v1.profiler.ProfileOptionBuilder.float_operation())
#     params = tf.compat.v1.profiler.profile(graph, options=tf.compat.v1.profiler.ProfileOptionBuilder.trainable_variables_parameter())
#     print('FLOPs: {};    Trainable params: {}'.format(flops.total_float_ops, params.total_parameters))



#==================================================================
# from tensorflow.python.framework import graph_util
# import tensorflow as tf
# from tensorflow.contrib.layers import flatten
#
#
# def stats_graph(graph):
#     flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
#     params = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
#     print('GFLOPs: {};    Trainable params: {}'.format(flops.total_float_ops / 1000000000.0, params.total_parameters))
#
#
# def load_pb(pb):
#     with tf.gfile.GFile(pb, "rb") as f:
#         graph_def = tf.GraphDef()
#         graph_def.ParseFromString(f.read())
#     with tf.Graph().as_default() as graph:
#         tf.import_graph_def(graph_def, name='')
#         return graph
#
#
# with tf.Graph().as_default() as graph:
#     模型开始处××××××××××××××××××××××××××××
#     ***** (1) Create Graph *****
#
#     模型结束××××××××××××××××××××××××××××

#     print('stats before freezing')
#     stats_graph(graph)
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())
#         # ***** (2) freeze graph *****
#         output_graph = graph_util.convert_variables_to_constants(sess, graph.as_graph_def(), ['output'])
#         with tf.gfile.GFile('graph.pb', "wb") as f:
#             f.write(output_graph.SerializeToString())
# # ***** (3) Load frozen graph *****
# with tf.Graph().as_default() as graph:
#     graph = load_pb('./graph.pb')
#     print('stats after freezing')
#     stats_graph(graph)


#================================================

#
# from tensorflow.python.framework import graph_util
# from tensorflow.contrib.layers import flatten
# import numpy as np
# import tensorflow as tf
# # 自己函数需要用到的函数

# import core.utils as utils
# import core.common as common
# import core.backbone as backbone
# from core.config import cfg
#
#
# def stats_graph(graph):
#     flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
#     params = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
#     print('GFLOPs: {};    Trainable params: {}'.format(flops.total_float_ops / 1000000000.0, params.total_parameters))
#
#
# def load_graph(frozen_graph_filename):
#     # We load the protobuf file from the disk and parse it to retrieve the
#     # unserialized graph_def
#     with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
#         graph_def = tf.GraphDef()
#         graph_def.ParseFromString(f.read())
#
#     # Then, we import the graph_def into a new Graph and return it
#     with tf.Graph().as_default() as graph:
#         # The name var will prefix every op/nodes in your graph
#         # Since we load everything in a new graph, this is not needed
#         tf.import_graph_def(graph_def, name="prefix")
#     return graph


# with tf.Graph().as_default() as graph:
#     # 模型开始处××××××××××××××××××××××××××××
#     # ***** (1) Create Graph *****
#     input_data = tf.Variable(initial_value=tf.random_normal([1, 416,416,3]))
#     route_1, route_2, input_data = backbone.darknet53(input_data, True)
#
#     # 模型结束××××××××××××××××××××××××××××
#
#     print('stats before freezing')
#     stats_graph(graph)
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())
#         # ***** (2) freeze graph *****
#         output_graph = graph_util.convert_variables_to_constants(sess, graph.as_graph_def(), ['prefix/pred_lbbox/concat_2'])
#         with tf.gfile.GFile('graph.pb', "wb") as f:
#             f.write(output_graph.SerializeToString())
#
# def count_flops(graph):
#     flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
#     print('FLOPs: {}'.format(flops.total_float_ops))
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     graph = load_graph('./yolov3_coco.pb')
#     stats_graph(graph)
#======================================================================