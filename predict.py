import os
import tensorflow as tf


class Predictor(object):
  """Load a pb file to do inference."""

  def __init__(self, pb_path: str, input_node_name: str,
               output_node_name: str):
    self.graph, self.sess = self._load_model(pb_path)
    self.inputs = self.graph.get_tensor_by_name(input_node_name)
    self.outputs = self.graph.get_tensor_by_name(output_node_name)

  def _load_model(self, pb_path: str):
    if not os.path.exists(pb_path):
      raise ValueError('`pb_path` does not exist.')

    graph = tf.compat.v1.Graph()
    with graph.as_default():
      with tf.io.gfile.GFile(pb_path, 'rb') as reader:
        graph_def = tf.compat.v1.GraphDef()
        serialized_graph = reader.read()
        graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(graph_def, name='')

    for node in graph_def.node:
      # if node.name == 'MobilenetV3/Conv/Conv2D_Fold':
      #   print(type(node))
      #   print(type(node.attr['T']))
      #   print(node.attr['T'])
      #   print('-----')
      #   print(node.attr.keys())
      #   print('-'*10)
      #   for key in node.attr:
      #     print(type(node.attr[key]))
      #     print(node.attr[key])

      # if node.name == 'Const_167':
      #   print(node)
      #   print(node.attr['value'].tensor.dtype)
      #   print(type(node.attr['value'].tensor.tensor_shape))

      if node.name == 'MobilenetV3/expanded_conv_2/depthwise/weights_quant/FakeQuantWithMinMaxVars':
        print(node)
        # print(type(node.attr['num_bits']))

      # if node.name == 'MobilenetV2/Conv/BatchNorm/FusedBatchNorm':
      #   print(type(node))
      #   print(node)
      # if node.name == 'MobilenetV2/Conv/Relu6':
      #   print(type(node))
      #   print(node)

    sess = tf.compat.v1.Session(graph=graph)
    return graph, sess

  def predict(self, inputs):
    outputs = self.sess.run(self.outputs, feed_dict={self.inputs: inputs})
    return outputs


if __name__ == '__main__':
  # pb_path = './models/mobilenet_v2_1.0_224/mobilenet_v2_1.0_224_frozen.pb'
  # input_node_name = 'input:0'
  # output_node_name = 'MobilenetV2/Predictions/Reshape_1:0'
  # predictor = Predictor(pb_path=pb_path, input_node_name=input_node_name,
  #                       output_node_name=output_node_name)

  pb_path = './models/v3-small_224_1.0_uint8/v3-small_224_1.0_uint8.pb'
  input_node_name = 'input:0'
  output_node_name = 'MobilenetV3/Predictions/Softmax:0'
  predictor = Predictor(pb_path=pb_path, input_node_name=input_node_name,
                        output_node_name=output_node_name)

  # pb_path = './models/resnet/resnet50_v1.pb'
  # input_node_name = 'input_tensor:0'
  # output_node_name = 'softmax_tensor:0'
  # predictor = Predictor(pb_path=pb_path, input_node_name=input_node_name,
  #                       output_node_name=output_node_name)

  # pb_path = './models/test.pb'
  # input_node_name = 'inputs:0'
  # output_node_name = 'outputs/Reshape:0'
  # predictor = Predictor(pb_path=pb_path, input_node_name=input_node_name,
  #                       output_node_name=output_node_name)
