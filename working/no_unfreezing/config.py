import os

TEST = '../input/test.csv'
INTENTS = '../input/intents.csv'
SPEAKERS = '../input/speakers'
OUTPUT = 'output'
PRED, CSV, SUBMISSION, LABELS = [os.path.join(OUTPUT, p) for p in ['preds.pkl', 'preds.csv', 'submission.csv', '1.csv']]
