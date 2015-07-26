import csv, math, itertools
import numpy as np
from sortedcontainers import SortedListWithKey
from math import sqrt
import ipdb, sys
from multiprocessing import Pool

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
  def __init__(self, facts, examples,
               distanceFunc = euclideanDistance,
               approximateFunc = majority):

    self.facts           = facts
    self.examples        = examples
    self.distanceFunc    = distanceFunc
    self.approximateFunc = approximateFunc

    self.preprocess()

  def preprocess(self):
    factsMat    = loadMatrix(self.facts)
    examplesMat = loadMatrix(self.examples)
    allMat      = np.concatenate((factsMat, examplesMat))

    normalizedMat = normalize(allMat)
    end           = len(factsMat)

    self.facts        = normalizedMat[0:end]
    self.examples     = normalizedMat[end:]
    self.nbAttributes = len(self.facts[0])

  def nearestsNeighbours(self, k, facts, example):
    nearests = SortedListWithKey(key = lambda val: val[0])

    for fact in facts:
      factV        = fact[0:-1]
      factClass    = fact[-1]
      exampleV     = example[0:-1]
      exampleClass = example[-1]

      if fact is example:
        continue

      distance = self.distanceFunc(exampleV, factV)
      nearests.add((distance, factClass, factV))

    return nearests[:k]

  def accuracy(self, k, facts, examples, printApproximation=False):
    correct = 0
    for example in examples:
      expected = example[-1]
      nn       = self.nearestsNeighbours(k, facts, example)
      output   = self.approximateFunc(nn)

      if printApproximation:
        print("{} => {}".format(example, output))

      if expected != -1 and output == expected:
        correct += 1

    return correct / len(examples)

  def backwardElimination(self, k):
    accuracy = 0
    unrelevant = []
    print("Applying backward elimination to find unrelevant attributes")
    for i in range(self.nbAttributes - 1):
      print("{}% completed".format(int(((i / (self.nbAttributes - 1)) * 100))))
      unrelevant.append(i)

      facts = np.delete(self.facts, unrelevant, 1)
      output = self.accuracy(k, facts, facts)

      if output > accuracy:
        accuracy = output
      else:
        unrelevant.remove(i)

    print("Attributes at index {} seems unrelevant".format(unrelevant))
    return unrelevant

  def run(self, k):
    print("Using kNN with k={}".format(k))

    # Find the attributes that are not all that useful
    #unrelevant = self.backwardElimination(k)
    unrelevant = [0, 5, 6]

    # Remove the useless attributes
    examples = np.delete(self.examples, unrelevant, 1)
    facts    = np.delete(self.facts, unrelevant, 1)

    # We print the approximation only
    accuracy = self.accuracy(k, facts, examples, printApproximation=True)

    # If there is an actual quality to be evaluated, an accuracy will be > 0
    if accuracy > 0:
      print("Accuracy = {}".format(accuracy))

if __name__ == '__main__':
  knn = Knn('dataset.csv', 'sample.csv')
  knn.run(5)
