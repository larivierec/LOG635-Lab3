import numpy as np
import ipdb, sys, math
from functools import reduce

def loadExternalData(file):
  vectors = np.genfromtxt(file, names=True, delimiter=";")
  return vectors.view(np.float64).reshape(vectors.shape + (-1,))

def normalize(vectors, high=1.0, low=0.0, ignore=-1):
  mins = np.min(vectors, axis=0)
  maxs = np.max(vectors, axis=0)
  mins[ignore] = 0
  maxs[ignore] = 1
  rng = maxs - mins
  return (high - (((high - low) * (maxs - vectors)) / rng)) * 1.0

class Neuron(object):
  def __init__(self, nbInputs):
    self.weights     = np.random.uniform(0,0.5, nbInputs + 1)
    self.lastDelta   = np.zeros(nbInputs + 1)
    self.derivatives = np.zeros(nbInputs + 1)
    self.delta       = 0.0

  def activate(self, vector):
    #summ = self.weights[-1] * 1.0
    #summ = 0
    summ = sum([self.weights[i] * elem for i, elem in enumerate(vector)])
    self.activation = summ

  def transfer(self):
    self.output = (1.0 / (1.0 + math.exp(-self.activation)))

  def transferDerivative(self, error):
    self.delta = error * (self.output * (1.0 - self.output))

  def __repr__(self):
    return str(self.__dict__)

class NeuralNetwork(object):
  def __init__(self, domain, nbInputs, learningRate=0.3, nbNodes=4,
               iterations=2000, seed=123456, momentum=0.8, nbLayers=1):
    self.domain        = domain
    self.nbInputs      = nbInputs
    self.learningRate  = learningRate
    self.nbNodes       = nbNodes
    self.iterations    = iterations
    self.momentum      = momentum
    self.nbLayers      = nbLayers

    np.random.seed(seed)

  def initializeNetwork(self):
    self.network = []
    self.appendLayer(nbAttrs=self.nbInputs)
    for _ in range(self.nbLayers):
      self.appendLayer()

    self.appendLayer(nbNeurons=1)
    print("Topologie du reseau : {} {}".format(self.nbInputs, reduce(lambda m,i: m + "{} ".format(str(len(i))), self.network, "")))

  def appendLayer(self, nbAttrs=None, nbNeurons=None):
    nbAttrs   = nbAttrs or len(self.network[-1])
    nbNeurons = nbNeurons or self.nbNodes
    neurons   = [Neuron(nbAttrs) for _ in range(nbNeurons)]
    self.network.append(neurons)

  def buildFromPreviousLayerOutput(self, i):
    return np.array([self.network[i-1][k].output for k in range(len(self.network[i-1]))])

  def propagate(self, vector):
    elem = vector
    for (i, layer) in enumerate(self.network):
      if i > 0:
        elem = self.buildFromPreviousLayerOutput(i)

      for neuron in layer:
        neuron.activate(elem)
        neuron.transfer()

    return self.network[-1][0].output

  def backwardPropagateError(self, expected):
    networkLength = len(self.network)
    for n in range(networkLength):
      index = networkLength - 1 - n

      if index == networkLength - 1:
        neuron = self.network[index][0] # only one node in output layer
        error  = (expected - neuron.output)
        neuron.transferDerivative(error)
      else:
        for (j, neuron) in enumerate(self.network[index]):
          summ = 0.0
          for nextNeuron in self.network[index+1]:
            summ += (nextNeuron.weights[j] * nextNeuron.delta)

          neuron.transferDerivative(summ)
          # print(neuron.delta)

  def calculateErrorDerivatives(self, vector):
    elem = vector
    for (i, layer) in enumerate(self.network):
      if i > 0:
        elem = self.buildFromPreviousLayerOutput(i)

      for neuron in layer:
        for (j, signal) in enumerate(elem):
          neuron.derivatives[j] += neuron.delta * signal

        neuron.derivatives[-1] += neuron.delta * 1.0

  def updateWeights(self):
    for layer in self.network:
      for neuron in layer:
        for i in range(len(neuron.weights)):
          delta = (self.learningRate * neuron.derivatives[i]) + (neuron.lastDelta[i] * self.momentum)
          neuron.weights[i] += delta
          neuron.lastDelta[i] = delta
          neuron.derivatives[i] = 0.0

  def trainNetwork(self):
    correct = 0
    for epoch in range(self.iterations):
      for (i, pattern) in enumerate(self.domain):
        vector   = pattern[0:-1]
        expected = round(pattern[-1])
        prop     = self.propagate(vector)
        output   = round(prop * 10)

        #if epoch > 50:

        if output == expected:
          correct += 1

        self.backwardPropagateError(expected / 10.0)
        self.calculateErrorDerivatives(vector)

      self.updateWeights()

      nextEpoch = epoch + 1
      if nextEpoch % 100 == 0:
        print("epoch={}, correct={}/{}".format(nextEpoch, correct, 100*len(self.domain)))
        correct = 0

  def testNetwork(self):
    correct = 0
    for pattern in self.domain:
      vector = pattern[0:-1]
      output = self.propagate(vector)
      if round(output) == round(pattern[-1]):
        correct += 1
    print("accuracy={}".format(correct / len(self.domain) * 100))

  def run(self):
    self.initializeNetwork()
    self.trainNetwork()
    self.testNetwork()

if __name__ == '__main__':
  # problem configuration
  domain = normalize(loadExternalData("input.csv"))

  print(domain)
  nbInputs = len(domain[0]) - 1

  network = NeuralNetwork(domain, nbInputs, iterations=2000, nbNodes=11, nbLayers=0)
  network.run()
