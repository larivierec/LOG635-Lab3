from src.trainer import *
import pdb, sys

if __name__ == '__main__':
  trainer = Trainer('dataset.csv')
  evaluator = Evaluator(trainer.train(), sys.stdout)
  evaluator.approximate('sample.csv')

