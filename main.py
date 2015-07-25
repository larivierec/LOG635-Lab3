import csv, math, itertools
import numpy as np
from sortedcontainers import SortedListWithKey
from math import sqrt
import pdb, sys

def euclideanDistance(v1, v2):
  return math.sqrt(sum([pow(v1[i] - v2[i], 2) for i in range(len(v1))]))

def majority(klasses):
  return max(set(klasses), key=klasses.count)

def normalize(vectors, high=1.0, low=0.0, ignore=-1):
  mins = np.min(vectors, axis=0)
  maxs = np.max(vectors, axis=0)
  mins[ignore] = 0
  maxs[ignore] = 1
  rng = maxs - mins
  return (high - (((high - low) * (maxs - vectors)) / rng)) * 1.0

def loadMatrix(file):
  matrix = np.genfromtxt(file, names=True, delimiter=";")
  return matrix.view(np.float64).reshape(vectors.shape + (-1,))

class Knn(object):
  def __init__(self, training, testing, distanceFunc = euclideanDistance):
    self.training     = training
    self.testing      = testing
    self.distanceFunc = distanceFunc

    self.preprocess()

  def preprocess(self):
    trainingMat = loadMatrix(self.training)
    testingMat  = loadMatrix(self.testing)
    allMat      = trainingMat + testingMat

    normalizedMat = normalize(allMat)
    end           = len(trainingMat)

    self.facts        = normalizedMat[0:end]
    self.examples     = normalizedMat[end:]
    self.nbAttributes = len(self.facts[0])

  def nearestsClasses(self, fact, examples):
    nearests = SortedListWithKey(key = lambda val: val[0])

    for example in examples:
      factV        = fact[0:-1]
      factClass    = fact[-1]
      exampleV     = example[0:-1]
      exampleClass = example[-1]

      distance = self.distanceFunc(factV, exampleV)
      nearests.add((distance, exampleClass, example))

    return nearests[-k:]

  def applyFeatureExtraction(self):
    pass

  def accuracy(self, facts, examples):
    for fact in facts:
      nn = self.nearestsClasses(fact, examples)
      

    pass

  def run(self, k):


if __name__ == '__main__':
  trainer = Trainer('dataset.csv', 'sample.csv')
  print(trainer.train())
