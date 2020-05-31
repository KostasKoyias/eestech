import os
import data
import torch
import models
import pandas as pd
import soundfile as sf
from tqdm import tqdm
from naive import naive_predict

OPTIMIZE = 0
UNSURE = 31 + OPTIMIZE
def predict(wav):
    signal, _ = sf.read(wav)
    signal = torch.tensor(signal, device=device).float().unsqueeze(0)
    label = model.decode_intents(signal)
    return label

def set_label(category, intents):
    category = intents.loc[intents.intent == category]
    return UNSURE if category.empty else category.category.item()

# configuration & file paths
TRAIN = 'input/2020-athens-eestech-challenge/train.csv'
TEST = 'input/2020-athens-eestech-challenge/test.csv'
INTENTS = 'input/2020-athens-eestech-challenge/intents.csv'
SPEAKERS = 'input/2020-athens-eestech-challenge/wavs/speakers'
PRED = 'preds.pkl'
CSV = 'preds.csv'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
config = data.read_config('experiments/no_unfreezing.cfg'); _,_,_=data.get_SLU_datasets(config)
model = models.Model(config).eval()
model.load_state_dict(torch.load('experiments/no_unfreezing/training/model_state.pth', map_location=device)) # load trained model

# predict label of each .wav file and store it as a pickle
test = pd.read_csv(TEST)
if not os.path.isfile(PRED):
    df, paths = list(), list()
    files = set(test['file'].apply(lambda f: f.replace('.png', '.wav')))
    for i, speaker in enumerate(os.listdir(SPEAKERS)):
        speaker = os.path.join(SPEAKERS, speaker)
        for wav in os.listdir(speaker):
            if wav not in files:
                continue
            wav = os.path.join(speaker, wav)
            paths.append(wav)

    df = pd.DataFrame({'file': paths})
    tqdm.pandas(desc='')
    df['category'] = df['file'].progress_apply(lambda f: predict(f))
    paths = pd.Series(paths)

    df = pd.DataFrame(df, columns=['file', 'category'])
    df['category'] = df['category'].apply(lambda l: ','.join(l[0]))
    df.to_pickle(PRED)
else: 
    df = pd.read_pickle(PRED)

# map intent list to category ID
if not os.path.isfile(CSV):
    intents = pd.read_csv(INTENTS)
    tqdm.pandas(desc='Mapping intent to category ID', total=df.shape[0])
    df['category'] = df['category'].progress_apply(lambda c: set_label(c, intents))
    df.to_csv(CSV, index=False)
else:
    df = pd.read_csv(CSV)

# keep labels for samples of the test set only
df['file'] = df['file'].apply(lambda file: os.path.basename(file).replace('.wav', '.png'))
tqdm.pandas(desc='Mapping files to category ID', total=test.shape[0])
test['category'] = test['file'].progress_apply(lambda file: df.loc[df.file == file]['category'].item())

# convert to the requested format requested
test['file'] = range(1, test['file'].shape[0] + 1)
test = test.rename(columns={'file': 'id'})

# predict unsure labels
if OPTIMIZE:
    test = pd.read_csv('sub.csv')
    unsure = test.loc[test.category == UNSURE]
    test = pd.concat(test.loc[test.category != UNSURE], naive_predict(pd.read_csv(TRAIN, dtype=str), unsure))

# submit result
test.to_csv('submission.csv', index=False)
print('Submission ready!!!')