import csv, math
import pdb
from collections import OrderedDict

class Trainer:
  def __init__(self, dataset, sample):
    # http://www.d.umn.edu/~deoka001/BackwardElimination.html
    self.dataset  = dataset
    self.sample   = sample
    self.k        = 3
    self.distance = euclideanDistance

  def train(self):
    with open(self.dataset) as datasetCsv, open(self.sample) as sampleCsv:
      reader1 = csv.DictReader(datasetCsv, delimiter=';')
      reader2 = csv.DictReader(sampleCsv, delimiter=';')

      self.applyBackwardElimination(reader1, reader2)

      for row1 in reader1:
        if any(row1):
          continue

        nearests = OrderedDict()

        for row2 in reader2:
          if any(row2):
            continue

          d = self.distance(row1, row2)
          nearests[d] = row2

        distance, value = nearests.popitem()

  def applyBackwardElimination(self, reader1, reader2):
    pass

  def accuracy(self):
    pass

def euclideanDistance(v1, v2):
  return math.sqrt(sum([pow(float(v1[i]) + float(v2[i]), 2) for i in v1.keys()]))
