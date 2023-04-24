import tensorflow as tf


#==========================================================
# def stats_graph(graph):
#     flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
#     params = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
#     print('FLOPs: {};    Trainable params: {}'.format(flops.total_float_ops, params.total_parameters))
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
# graph =tf.get_default_graph()
# stats_graph(graph)

#==============================================================
from tensorflow.python.framework import graph_util
import tensorflow as tf
# from tensorflow.contrib.layers import flatten


def stats_graph(graph):
    flops = tf.compat.v1.profiler.profile(graph, options=tf.compat.v1.profiler.ProfileOptionBuilder.float_operation())
    params = tf.compat.v1.profiler.profile(graph, options=tf.compat.v1.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    print('GFLOPs: {};    Trainable params: {}'.format(flops.total_float_ops / 1000000000.0, params.total_parameters))


def load_pb(pb):
    with tf.compat.v1.gfile.GFile(pb, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.compat.v1.Graph().as_default() as graph:
        tf.compat.v1.import_graph_def(graph_def, name='')
        return graph


# with tf.compat.v1.Graph().as_default() as graph:
#     # 模型开始处××××××××××××××××××××××××××××
#     # ***** (1) Create Graph *****
#
#     # 模型结束××××××××××××××××××××××××××××
#
#     print('stats before freezing')
#     stats_graph(graph)
#     with tf.compat.v1.Session() as sess:
#         sess.run(tf.compat.v1.global_variables_initializer())
#         # ***** (2) freeze graph *****
#         output_graph = graph_util.convert_variables_to_constants(sess, graph.as_graph_def(), ['output'])
#         with tf.compat.v1.gfile.GFile('graph.pb', "wb") as f:
#             f.write(output_graph.SerializeToString())
# # ***** (3) Load frozen graph *****
# with tf.compat.v1.Graph().as_default() as graph:
#     graph = load_pb('./graph.pb')
#     print('stats after freezing')
#     stats_graph(graph)
#=========================================================