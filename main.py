from src.trainer import *
from src.evaluator import *
import pdb, sys

if __name__ == '__main__':
  trainer = Trainer('input.csv')
  evaluator = Evaluator(trainer.train(), 'test-input.csv', sys.stdout)
