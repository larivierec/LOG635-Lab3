import csv, math, itertools
import numpy as np
from sortedcontainers import SortedListWithKey
from math import sqrt
import ipdb, sys

def euclideanDistance(p, q):
  return sum([pow(q[i] - p[i], 2) for i, value in enumerate(p)])

def majority(klasses):
  return max(set(klasses), key=klasses.count)

def ponderate(klasses):
  a = sum(map(lambda x: x[1].label/ x[0], klasses))
  b = sum(map(lambda x: 1 / x[0], klasses))
  return int(a/b)

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
               approximateFunc = majority):

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

  def nearestsClasses(self, k, fact, examples):
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
      nn      = self.nearestsClasses(k, fact, examples)
      classes = list(map(lambda x: x[1], nn))
      output  = self.approximateFunc(classes)

      ipdb.set_trace()
      if output == expected:
        correct += 1

    return correct / len(facts)

  def run(self, k):
    result = self.accuracy(k, self.facts, self.examples)
    print(result)

if __name__ == '__main__':
  knn = Knn('dataset.csv', 'sample.csv')
  knn.run(5)
