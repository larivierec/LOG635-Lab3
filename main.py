import numpy as np
import pdb, sys, math
from functools import reduce

def transferFunc(output, derivative=False):
  if derivative:
    return output * (1.0 - output)
  return 1.0 / (1.0 + math.exp(-output))

def loadVectors(file):
  vectors = np.genfromtxt(file, names=True, delimiter=";")
  return vectors.view(np.float64).reshape(vectors.shape + (-1,))

def normalizeVectors(vectors, high=1.0, low=0.0):
  mins = np.min(vectors, axis=0)
  maxs = np.max(vectors, axis=0)
  rng = maxs - mins
  return high - (((high - low) * (maxs - vectors)) / rng)

class Neuron:
  def __init__(self, nbInputs):
    self.weights     = np.random.uniform(-0.5, 0.5, nbInputs + 1)
    self.lastDelta   = np.zeros(nbInputs + 1)
    self.derivatives = np.zeros(nbInputs + 1)
    self.delta       = 0

  def activate(self, vector):
    summ = self.weights[-1] * 1.0
    for (i, elem) in enumerate(vector):
      summ += self.weights[i] * elem

    self.activation = summ

  def transfer(self):
    self.output = transferFunc(self.activation)

  def transferDerivative(self, error):
    self.delta = error * transferFunc(self.output, derivative=True)

class NeuralNetwork:
  def __init__(self, domain, nbInputs, learningRate=0.3, nbNodes=4, iterations=2000, seed=42, momentum=0.8):
    self.domain        = domain
    self.nbInputs      = nbInputs
    self.learningRate  = learningRate
    self.nbNodes       = nbNodes
    self.iterations    = iterations
    self.momentum      = momentum

    np.random.seed(seed)

  def initializeNetwork(self):
    self.network = []
    self.network.append([Neuron(self.nbInputs)] * self.nbNodes)
    self.network.append([Neuron(len(self.network[-1]))] * self.nbNodes)
    self.network.append([Neuron(len(self.network[-1]))])
    print("Topology : {} {}".format(self.nbInputs, reduce(lambda m,i: m + "{} ".format(str(len(i))), self.network, "")))

  def forwardPropagate(self, vector):
    elem = vector
    for (i, layer) in enumerate(self.network):
      if i > 0:
        elem = np.array([self.network[i-1][k].output for k in range(len(self.network[i-1]))])

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

  def calculateErrorDerivativesForWeights(self, vector):
    elem = vector
    for (i, layer) in enumerate(self.network):
      if i > 0:
        elem = np.array([self.network[i-1][k].output for k in range(len(self.network[i-1]))])

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
      for pattern in self.domain:
        vector = pattern[0:-1]
        expected = pattern[-1]
        output = self.forwardPropagate(vector)

        if round(output) == int(expected):
          correct += 1

        self.backwardPropagateError(expected)
        self.calculateErrorDerivativesForWeights(vector)

      self.updateWeights()
      #print("CORRECT = {0}".format(correct))
      nextEpoch = epoch + 1
      if nextEpoch % 100 == 0:
        print("epoch={}, correct={}/{}".format(nextEpoch, correct, 100*len(self.domain)))
        correct = 0

  def testNetwork(self):
    # TODO
    pass

  def run(self):
    self.initializeNetwork()
    self.trainNetwork()
    self.testNetwork()

if __name__ == '__main__':
  # problem configuration
  domain = np.array([
      [0.0, 0.0, 0],
      [0.0, 1.0, 1],
      [1.0, 0.0, 1],
      [1.0, 1.0, 0],
    ])

  # domain = normalizeVectors(loadVectors("input.csv"))
  nbInputs = len(domain[0]) - 1

  network = NeuralNetwork(domain, nbInputs, iterations=2000)
  network.run()
