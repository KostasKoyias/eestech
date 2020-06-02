import os

TEST = '../input/myinput/test.csv'
INTENTS = '../input/myinput/intents.csv'
SPEAKERS = '../input/myinput/speakers'
OUTPUT = 'output'
PRED, CSV, SUBMISSION, LABELS = [os.path.join(OUTPUT, p) for p in ['preds.pkl', 'preds.csv', 'submission.csv', '1.csv']]
