class Trainer:
  def __init__(self, input):
    self.input = input
    self.trained = False

  def isTrained(self):
    return self.trained

  def train(self):
    with open(self.input, 'r') as f:
      for line in f:
        self.process(line)

    self.trained = True
    return None

  def process(self, line):
    pass
