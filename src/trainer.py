import csv, math, itertools
from statistics import mean, pstdev
from sortedcontainers import SortedListWithKey
from collections import OrderedDict
from math import sqrt
import pdb

class Trainer:
  def __init__(self, trainingData, testingData):
    # References:
    # http://www.d.umn.edu/~deoka001/BackwardElimination.html
    # https://cours.etsmtl.ca/log635/private/notes-de-cours/Cours_10.pdf
    self.trainingData    = trainingData
    self.testingData     = testingData
    self.distance        = euclideanDistance
    self.trainingVectors = []
    self.testingVectors  = []
    self.numberOfAttrs   = 0

    self.preprocess()

  def preprocess(self):
    trainingVectors = loadVectors(self.trainingData)
    testingVectors  = loadVectors(self.testingData)

    allVectors = trainingVectors + testingVectors

    normalizedVectors = normalize(allVectors)

    end = len(trainingVectors)
    self.trainingVectors = normalizedVectors[0:end]
    self.testingVectors = normalizedVectors[end:]

  def train(self, k = 7):
    # Apply backward filter here.

    """
    accuracy = 0
    i = 0
    while i != len(self.trainingVectors[0].features):
      origTraining = list(self.trainingVectors)
      self.trainingVectors = list(map(lambda x: x.remove(i), self.trainingVectors))

      print(len(self.trainingVectors[0].features))
      bfAccuracy = self.findNN(k, self.trainingVectors, self.trainingVectors)
      if bfAccuracy > accuracy:
        accuracy = bfAccuracy
        print("BF : {}".format(bfAccuracy))
        i = 0
      else:
        print("BF : {}".format(bfAccuracy))
        self.trainingVectors = origTraining
        i += 1
    """

    # Accuracy here.
    accuracy = self.findNN(k, self.trainingVectors, self.testingVectors)
    return accuracy

  def findNN(self, k, dataset, examples):
    exact = 0
    for fact in dataset:
      vecWithDistance = SortedListWithKey(key = lambda val: val[0])

      for example in examples:
        d = self.distance(fact, example)
        vecWithDistance.add((d, example))

      # find K sample based on the closest distance
      nearests = list(map(lambda x: x[1].label, vecWithDistance[-k:]))

      # apply majority vote
      approximatedLabel = majority(list(nearests))

      if approximatedLabel == fact.label:
        exact += 1

    return exact / len(dataset)

def normalize(vectors):
  normal = []
  for c in range(len(vectors[0].features)):
    values    = list(map(lambda x: x.features[c], vectors))
    average   = mean(values)
    deviation = pstdev(values, average)
    normal.append((average, deviation))

  return list(map(lambda v: v.normalize(normal), vectors))

def loadVectors(file):
  vectors = []
  with open(file) as fileCsv:
    reader = csv.DictReader(fileCsv, delimiter= ';')

    for row in reader:
      if not row:
        continue

      vectors.append(LabeledVector.fromDict(row))

  return vectors

class LabeledVector:
  @classmethod
  def fromDict(cls, d):
    label = int(d.pop('quality'))
    features = [float(v) for k, v in sorted(d.items())]
    return cls(label, features)

  @classmethod
  def fromList(cls, l, label):
    return cls(label, l)

  def remove(self, i):
    del self.features[i]
    return self

  def __init__(self, label = "", features = []):
    self.label = label
    self.features = features

  """
  Normalize using an array of (average, stdev) for each attributes
  """
  def normalize(self, normal):
    newFeatures = [(x - normal[i][0]) / normal[i][1] for i, x in enumerate(self.features)]
    return LabeledVector.fromList(newFeatures, self.label)

  def __repr__(self):
    return "[{0} => {1}]".format(self.label, self.features)

  def __getitem__(self, key):
    return self.features[key]

  def __iter__(self):
    return self.features.iter()

def majority(labels):
  return max(set(labels), key=labels.count)

def euclideanDistance(v1, v2):
  return math.sqrt(sum([pow(v1[i] + v2[i], 2) for i in range(len(v1.features))]))
