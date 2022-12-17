import tensorflow as tf

from tensorflow.python.framework import graph_util


def simple_model():
  inputs = tf.compat.v1.placeholder(
      tf.float32, shape=[None, 224, 224, 3], name='inputs')
  dropout = tf.compat.v1.nn.dropout(inputs, keep_prob=0.6)
  outputs = tf.compat.v1.layers.flatten(dropout, name='outputs')


def save(pb_path):
  simple_model()

  with tf.compat.v1.Session() as sess:
    graph_def = tf.compat.v1.get_default_graph().as_graph_def()
    # for node in graph_def.node:
    #   print(node)
    output_graph_def = graph_util.convert_variables_to_constants(
        sess=sess, input_graph_def=graph_def,
        output_node_names=['outputs/Reshape']
    )

  with tf.io.gfile.GFile(pb_path, 'wb') as writer:
    serialized_graph = output_graph_def.SerializeToString()
    writer.write(serialized_graph)


if __name__ == '__main__':
  tf.compat.v1.disable_eager_execution()
  pb_path = '../models/test.pb'
  save(pb_path=pb_path)
