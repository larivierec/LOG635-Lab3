import numpy as np
import pdb, sys, math

def transfer(activation, derivative=False):
  if derivative:
    return activation * (1.0 - activation)
  return 1.0 / (1.0 + exp(-activation))

class Neuron:
  def __init__(self, nbInputs):
    self.weights    = np.random.rand(nbInputs + 1)
    self.delta      = np.zeros(nbInputs + 1)
    self.derivative = np.zeros(nbInputs + 1)

class NeuralNetwork:
  def __init__(self, domain, nbInputs, learningRate = 0.3, nbHiddenNodes = 4, iterations = 2000):
    self.domain        = domain
    self.nbInputs      = nbInputs
    self.learningRate  = learningRate
    self.nbHiddenNodes = nbHiddenNodes
    self.iterations    = iterations

  def initializeNetwork(self):
    self.network = []
    self.network.append([Neuron(self.nbInputs)] * self.nbHiddenNodes)
    self.network.append([Neuron(len(self.network))])

  def trainNetwork(self):
    pass

  def testNetwork(self):
    pass

  def run(self):
    self.initializeNetwork()
    self.trainNetwork()
    self.testNetwork()

if __name__ == '__main__':
  # problem configuration
  domain = np.array([
      [0, 0, 0],
      [0, 1, 1],
      [1, 0, 1],
      [1, 1, 0],
    ])
  nbInputs = len(domain[0]) - 1

  network = NeuralNetwork(domain, nbInputs)
  network.run()
