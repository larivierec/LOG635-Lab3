import numpy as np
import ipdb, sys, math
from functools import reduce

def loadExternalData(file):
  matrix = np.genfromtxt(file, names=True, delimiter=";")
  return matrix.view(np.float64).reshape(matrix.shape + (-1,))

def normalize(matrix, high=1.0, low=0.0, ignore=-1):
  mins = np.min(matrix, axis=0)
  maxs = np.max(matrix, axis=0)
  rng = maxs - mins
  return (high - (((high - low) * (maxs - matrix)) / rng)) * 1.0

def sigmoid(x):
  return 1.0 / (1.0 + math.exp(-x))

def sigmoidDeriv(x):
  # x must be the result of the sigmoid
  return x * (1.0 - x)

class Neuron(object):
  def __init__(self, nbAttributes):
    self.weights    = np.random.rand(nbAttributes)
    self.deltas     = np.zeros(nbAttributes)
    self.delta      = np.zeros(nbAttributes)
    self.activation = 0.0
    self.output     = 0.0

  def activate(self, vector):
    self.activation = np.sum(self.weights * vector)

  def transfer(self):
    self.output = sigmoid(self.activation)

  def __repr__(self):
    return "Neuron(output={})".format(self.output)

class NeuralNetwork(object):
  def __init__(self, domain, nbAttributes, nbLayers=1, nbNeurons=11, iterations=1, learningRate=0.8):
    self.domain       = domain
    self.nbAttributes = nbAttributes
    self.nbLayers     = nbLayers
    self.nbNeurons    = nbNeurons
    self.iterations   = iterations
    self.learningRate = learningRate

  def initializeNetwork(self):
    self.layers = []
    for _ in range(self.nbLayers):
      self.addLayer()
    self.addLayer(nbNeurons=1)

  def addLayer(self, nbNeurons=None, nbAttributes=None):
    nbNeurons    = nbNeurons or self.nbNeurons
    nbAttributes = nbAttributes or self.nbAttributes
    self.layers.append([Neuron(nbAttributes) for _ in range(nbNeurons)])

  def forwardPropagate(self, vector):
    for i, layer in enumerate(self.layers):
      if i > 0:
        vector = self.buildFromPreviousLayer(i)

      for neuron in layer:
        neuron.activate(vector)
        neuron.transfer()

    return self.layers[-1][0].output

  def buildFromPreviousLayer(self, index):
    return np.array([neuron.output for neuron in self.layers[index]])

  def backwardPropagateError(self, expected):
    for index in reversed(range(self.nbLayers + 1)):
      if index == self.nbLayers:
        # on est sur la couche output
        neuron = self.layers[index][0]
        error  = (expected - neuron.output)
        neuron.delta = error * sigmoidDeriv(neuron.output)
      else:
        pass
        # for (j, neuron) in enumerate(self.layers[index]):
        #   summ = 0
        #   for nextNeuron in self.layers[index+1]:
        #     summ += (nextNeuron.weights[j] * nextNeuron.delta)
        #   neuron.delta = error * sigmoidDeriv(neuron.output)

  def updateWeights(self):
    for layer in self.layers:
      for neuron in layer:
        neuron.weights += self.learningRate * neuron.output * neuron.delta

  def run(self):
    self.initializeNetwork()
    self.train()
    self.test()

  def train(self):
    for epoch in range(self.iterations):
      for pattern in self.domain:
        expected = pattern[-1]
        vector = pattern[:-1]
        output = self.forwardPropagate(vector)
        self.backwardPropagateError(expected)

        print("{} == {}".format(output, expected))

      self.updateWeights()
      print(self.layers)

  def test(self):
    pass

if __name__ == '__main__':
  domain = normalize(loadExternalData("input.csv"))
  nbAttributes = len(domain[0]) - 1

  np.random.seed(42)

  NeuralNetwork(domain, nbAttributes, iterations=5).run()
