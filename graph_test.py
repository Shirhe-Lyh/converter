import numpy as np
import os
import torch

import graph
import predict

from nets import alexnet
from nets import mobilenet
from nets import resnet


def test_create_graph():
  # Node configurations
  config = [
      {'op': 'placeholder',
       'name': 'input',
       'kwargs': {'shape': (1, 5, 5, 1)}
       },
      {
          'op': 'relu',
          'name': 'relu',
          'kwargs': {'inputs': ['input']}
      }
  ]

  # Create and save graph
  g = graph.Graph(config=config)
  pb_path = './test.pb'
  g.save(output_path=pb_path)

  # Test pb file
  input_node_name = 'input:0'
  output_node_name = 'relu:0'
  predictor = predict.Predictor(
      pb_path=pb_path, input_node_name=input_node_name,
      output_node_name=output_node_name)
  inputs = np.random.randn(1, 5, 5, 1)
  print('Input has negative element: ', 'Yes' if (inputs < 0).any() else 'No')
  outputs = predictor.predict(inputs)
  print('Output is non negative: ', 'Yes' if (outputs >= 0).all() else 'No')

  # Remove the temporary pb file
  os.remove(pb_path)


def cosin_similarity(a, b):
  sim = np.sum(a * b) / (np.linalg.norm(a, ord=2) * np.linalg.norm(b, ord=2))
  return sim


def test_convert_graph(net_name='resnet50'):
  # Define PyTorch's model
  if net_name == 'resnet50':
    pretrained_path = './models/resnet/resnet50-0676ba61.pth'
    net = resnet.resnet_50(pretrained_path=pretrained_path)
  elif net_name == 'mobilenet_v2':
    pretrained_path = './models/mobilenet/mobilenet_v2-7ebf99e0.pth'
    net = mobilenet.mobilenet_v2(pretrained_path=pretrained_path)
  elif net_name == 'mobilenet_v3_small':
    pretrained_path = './models/mobilenet/mobilenet_v3_small-047dcff4.pth'
    net = mobilenet.mobilenet_v3_small(pretrained_path=pretrained_path)
  elif net_name == 'alexnet':
    pretrained_path = './models/alexnet/alexnet-owt-7be5be79.pth'
    net = alexnet.alexnet(pretrained_path=pretrained_path)
  net.eval()

  # # Generate graph configurations
  graph_config = graph.GraphConfig()
  config = graph_config.generate(model=net)

  # # Convert to Tensorflow's graph
  # g = graph.Graph(config=config)

  # # # Save the graph
  output_path = './models/{}.pb'.format(net_name)
  # g.save(output_path=output_path)

  # Test pb file
  # np.random.seed(42)
  inputs = np.random.randn(1, 224, 224, 3).astype(np.float32)
  inptus_pth = torch.from_numpy(inputs.transpose(0, 3, 1, 2))
  with torch.no_grad():
    outputs_pth = net(inptus_pth).data.cpu().numpy().squeeze()
  input_node_name = '{}:0'.format(config[0]['name'])
  output_node_name = '{}:0'.format(config[-1]['name'])
  predictor = predict.Predictor(
      pb_path=output_path, input_node_name=input_node_name,
      output_node_name=output_node_name)
  outputs_tf = predictor.predict(inputs).squeeze()
  print('The class label is: [PyTorch {}] vs [TensorFlow {}]'.format(
      np.argmax(outputs_pth), np.argmax(outputs_tf)))
  print('The output (50/1000) of PyTorch is: \n', outputs_pth[:50])
  print('The output (50/1000) of TensorFlow is: \n', outputs_tf[:50])
  sim = cosin_similarity(outputs_pth, outputs_tf)
  print('The cosin similarity is: ', sim)


if __name__ == '__main__':
  # test_create_graph()

  # test_convert_graph(net_name='resnet50')
  # test_convert_graph(net_name='mobilenet_v2')
  test_convert_graph(net_name='mobilenet_v3_small')
  # test_convert_graph(net_name='alexnet')
