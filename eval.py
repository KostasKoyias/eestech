import pandas as pd

# configs
LABELS = '/home/kostas/Desktop/coding/contests/kaggle/eestech/input/2020-athens-eestech-challenge/data/1.csv'
SUBMISSION = 'sub.csv'

# load our predictions as well as the ground truth and sort by id
sub = pd.read_csv(SUBMISSION).sort_values(by='id')['category']
labels = pd.read_csv(LABELS).sort_values(by='id')['category']

# compare
correct = (sub == labels).sum()
total = labels.shape[0]
print(f'\033[1;32mAccuracy\033[0m: {correct/total:.6f}')