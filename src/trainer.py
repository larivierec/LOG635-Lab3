import csv, math, itertools
from statistics import mean, stdev
from sortedcontainers import SortedDict
from collections import OrderedDict
from math import sqrt
import pdb

class Trainer:
  def __init__(self, trainingData):
    # References:
    # http://www.d.umn.edu/~deoka001/BackwardElimination.html
    # https://cours.etsmtl.ca/log635/private/notes-de-cours/Cours_10.pdf
    self.trainingData  = trainingData

  """
  Reads the entire training data and normalize it using and average and a
  standard deviation for each attributes (this might be a variance instead).

  Returns the KB obtained by normalization.
  """
  def train(self):
    vectors = []
    with open(self.trainingData) as trainingCsv:
      trainingReader = csv.DictReader(trainingCsv, delimiter=';')

      for row in trainingReader:
        if not row: # skip empty line
          continue

        vectors.append(LabeledVector.fromDict(row))

    #kb = map(lambda vec: normalize(vec, vectors), vectors)
    kb = list(map(lambda vec: vec.normalize, vectors))
    return kb

class LabeledVector:
  @classmethod
  def fromDict(cls, d):
    temp = OrderedDict({k: float(v) for k, v in sorted(d.items())})
    label = temp.pop('quality')
    features = list(temp.values())
    return cls(label, features)

  @classmethod
  def fromList(cls, l, label):
    return cls(label, l)

  def __init__(self, label = "", features = []):
    self.label = label
    self.features = features

  def normalize(self):
    length = sqrt(sum([pow(f, 2) for f in self.features]))
    return self.__class__.fromList(list(map(lambda x: x/length, self.features)), self.label)

  def label(self):
    return self.label

  def features(self):
    return self.features

  def __repr__(self):
    return "{0} => {1}".format(self.label, self.features)

#def normalize(vector, all):
#  ret = {}
#  for k, v in vector.items():
#    valuesForKey = list(map(lambda x: x[k], all))
#    average = mean(valuesForKey)
#    standardDeviation = stdev(valuesForKey, average) # stdev or variance?
#    ret[k] = (v - average) / standardDeviation
#
#  return ret

class Evaluator:
  def __init__(self, kb, output):
    self.kb        = kb                # Knowledgebase obtained from trainer
    self.output    = output            # stdout or file?
    self.k         = 3                 # k constant, defines the number of neighbors
    self.threshold = 0.8               # use to compare accuracy
    self.distance  = euclideanDistance # function used to evaluate the distance

  """
  Find the nearests neighbors in KB and try to find the closest.

  Returns the approximation for each tests.
  """
  def approximate(self, testingData):
    with open(testingData) as testingCsv:
      testReader = csv.DictReader(testingCsv, delimiter=';')

      self.applyBackwardElimination(testReader)

      for test in testReader:
        if not test: # skip empty line
          continue

        nearests = SortedDict()

        testVec = LabeledVector.fromDict(test)

        for fact in self.kb:
          d = self.distance(testVec.normalize(), fact)
          nearests[d] = fact.label()

        # find K sample based on the closest distance
        labels = list(nearests.values()[0:self.k])

        # apply majority vote
        approximatedLabel = majority(labels)
        print("approx: {0}, real: {1}".format(approximatedLabel, testVec.label()))

  def applyBackwardElimination(self, testReader):
    # TODO
    pass

  def accuracy(self):
    # TODO
    pass

def majority(labels):
  return max(set(labels), key=labels.count)

def euclideanDistance(v1, v2):
  #return math.sqrt(sum([pow(float(v1[i]) + float(v2[i]), 2) for i in list(v1.keys())[:-1]]))
  return math.sqrt(sum([pow(v1[i] + v2[i], 2) for i in v1.features()]))
