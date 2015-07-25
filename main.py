import csv, math, itertools
import numpy as np
from sortedcontainers import SortedListWithKey
from math import sqrt
import ipdb, sys

def euclideanDistance(p, q):
  return sum([pow(q[i] - p[i], 2) for i, value in enumerate(p)])

def majority(vectors):
  classes = list(map(lambda x: x[1], vectors))
  return max(set(classes), key=classes.count)

def ponderate(vectors):
  num   = sum(map(lambda x: x[1] / x[0], vectors))
  denum = sum(map(lambda x: 1 / x[0], vectors))
  return round(num / denum)

def normalize(vectors, high=1.0, low=0.0, ignore=-1):
  mins = np.min(vectors, axis=0)
  maxs = np.max(vectors, axis=0)
  mins[ignore] = 0
  maxs[ignore] = 1
  rng = maxs - mins
  return (high - (((high - low) * (maxs - vectors)) / rng)) * 1.0

def loadMatrix(file):
  matrix = np.genfromtxt(file, names=True, delimiter=";")
  return matrix.view(np.float64).reshape(matrix.shape + (-1,))

class Knn(object):
  def __init__(self, training, testing,
               distanceFunc = euclideanDistance,
               approximateFunc = ponderate):

    self.training        = training
    self.testing         = testing
    self.distanceFunc    = distanceFunc
    self.approximateFunc = approximateFunc

    self.preprocess()

  def preprocess(self):
    trainingMat = loadMatrix(self.training)
    testingMat  = loadMatrix(self.testing)
    allMat      = np.concatenate((trainingMat, testingMat))

    normalizedMat = normalize(allMat)
    end           = len(trainingMat)

    self.facts        = normalizedMat[0:end]
    self.examples     = normalizedMat[end:]
    self.nbAttributes = len(self.facts[0])

  def nearestsNeighbours(self, k, fact, examples):
    nearests = SortedListWithKey(key = lambda val: val[0])

    for example in examples:
      factV        = fact[0:-1]
      factClass    = fact[-1]
      exampleV     = example[0:-1]
      exampleClass = example[-1]

      distance = self.distanceFunc(factV, exampleV)
      nearests.add((distance, exampleClass, example))

    return nearests[-k:]

  def accuracy(self, k, facts, examples):
    correct = 0
    for fact in facts:
      expected = fact[-1]
      nn      = self.nearestsNeighbours(k, fact, examples)
      output  = self.approximateFunc(nn)

      classes = list(map(lambda x: x[1], nn))
      print("{} => {} == {}".format(classes, output, expected))

      if output == expected:
        correct += 1

    return correct / len(facts)

  def run(self, k):
    result = self.accuracy(k, self.facts, self.examples)
    print(result)

if __name__ == '__main__':
  knn = Knn('dataset.csv', 'sample.csv')
  knn.run(5)
