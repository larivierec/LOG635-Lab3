import numpy as np
import pdb, sys, math

def transfer(activation, derivative=False):
  if derivative:
    return activation * (1.0 - activation)
  return 1.0 / (1.0 + math.exp(-activation))

class Neuron:
  def __init__(self, nbInputs):
    self.weights     = np.random.rand(nbInputs + 1)
    self.delta       = np.zeros(nbInputs + 1)
    self.derivatives = np.zeros(nbInputs + 1)
    self.lastDelta   = 0

  def activate(self, vector):
    pdb.set_trace()
    sum = self.weights[-1] * 1.0
    for (i, input) in enumerate(vector):
      sum += self.weights[i] * input

    self.activation = sum

  def transfer(self):
    self.output = transfer(self.activation)

class NeuralNetwork:
  def __init__(self, domain, nbInputs, learningRate = 0.3, nbNodes = 4, iterations = 2000):
    self.domain        = domain
    self.nbInputs      = nbInputs
    self.learningRate  = learningRate
    self.nbNodes       = nbNodes
    self.iterations    = iterations

  def initializeNetwork(self):
    self.network = []
    self.network.append([Neuron(self.nbInputs)] * self.nbNodes)
    self.network.append([Neuron(len(self.network[-1]))])

  def forwardPropagate(self, vector):
    input = vector
    for (i, layer) in enumerate(self.network):
      if i > 0:
        input = np.array([self.network[i-1][k].output for k in range(len(self.network[i-1]))])

      for neuron in layer:
        neuron.activate(input)
        neuron.transfer()

    return self.network[-1][0].output

  def backwardPropagateError(self, expected):
    networkLength = len(self.network)
    for i in range(networkLength):
      index = networkLength - 1 - i

      if index == networkLength - 1:
        neuron = self.network[index][0] # only one node in output layer
        error = (expected - neuron.output)
        neuron.delta = error * transfer(neuron.output, derivative=True)
      else:
        for (j, neuron) in enumerate(self.network[index]):
          sum = 0.0
          for nextNeuron in self.network[index+1]:
            sum += nextNeuron.weights[j] * nextNeuron.delta

          neuron.delta = sum * transfer(neuron.output, derivative=True)

  def calculateErrorDerivativesForWeights(self, vector):
    input = vector
    for (i, layer) in enumerate(self.network):
      if i > 0:
        input = np.array([self.network[i-1][k].output for k in range(len(self.network[i-1]))])

      for neuron in layer:
        for (j, signal) in enumerate(input):
          neuron.derivatives[j] += neuron.delta * signal

        neuron.derivatives[-1] += neuron.delta * 1.0

  def updateWeights(self, mom = 0.8):
    for layer in self.network:
      for neuron in layer:
        for (i, weight) in enumerate(neuron.weights):
          delta = (self.learningRate * neuron.derivatives[i]) + neuron.lastDelta[i] * mom
          neuron.weights[i] += delta
          neuron.lastDelta = delta
          neuron.derivatives = 0.0

  def trainNetwork(self):
    correct = 0
    for epoch in range(self.iterations):
      for pattern in self.domain:
        vector = pattern[0:-1]
        expected = pattern[-1]
        output = self.forwardPropagate(vector)

        # if round(expected) == expected:
        #   correct += 1

        # self.backwardPropagateError(expected)
        # self.calculateErrorDerivativesForWeights(vector)

      self.updateWeights()
      if (epoch + 1) % 100 == 0:
        print("epoch={}, correct={}/{}".format((epoch + 1), correct, 100*len(self.domain)))
        correct = 0

  def testNetwork(self):
    pass

  def run(self):
    self.initializeNetwork()
    self.trainNetwork()
    self.testNetwork()

if __name__ == '__main__':
  # problem configuration
  domain = np.array([
      [0.0, 0.0, 0.0],
      [0.0, 1.0, 1.0],
      [1.0, 0.0, 1.0],
      [1.0, 1.0, 0.0],
    ])
  nbInputs = len(domain[0]) - 1

  network = NeuralNetwork(domain, nbInputs)
  network.run()
