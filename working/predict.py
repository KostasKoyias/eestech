import os
from shutil import copyfile as cp

for f in ['data.py', 'config.py', 'models.py']:
    cp(os.path.join('../input/myinput', f), f)

import data
import torch
import models
import pandas as pd
import soundfile as sf
from config import *
from tqdm import tqdm

UNSURE = 31
def predict(wav):
    signal, _ = sf.read(wav)
    signal = torch.tensor(signal, device=device).float().unsqueeze(0)
    label = model.decode_intents(signal)
    return label

def set_label(category, intents):
    category = intents.loc[intents.intent == category]
    return UNSURE if category.empty else category.category.item()

# make output directory
if not os.path.isfile(OUTPUT): os.makedirs(OUTPUT)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
config = data.read_config('../input/myinput/no_unfreezing/no_unfreezing.cfg'); _,_,_=data.get_SLU_datasets(config)
model = models.Model(config).eval()
model.load_state_dict(torch.load('../input/myinput/no_unfreezing/model_state.pth', map_location=device)) # load trained model

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
    tqdm.pandas(desc='Predicting command labels')
    df['category'] = df['file'].progress_apply(lambda f: predict(f))

    df = pd.DataFrame(df, columns=['file', 'category'])
    df['category'] = df['category'].apply(lambda l: ','.join(l[0]))
    df.to_pickle(PRED)
else:
    print(f' > Using existing predictions under {PRED}(delete to reproduce)') 
    df = pd.read_pickle(PRED)

# map intent list to category ID
if not os.path.isfile(CSV):
    intents = pd.read_csv(INTENTS)
    tqdm.pandas(desc='Mapping intent to category ID', total=df.shape[0])
    df['category'] = df['category'].progress_apply(lambda c: set_label(c, intents))
    df.to_csv(CSV, index=False)
else:
    print(f' > Using existing mapping {CSV}(delete to reproduce)') 
    df = pd.read_csv(CSV)

# keep labels for samples of the test set only
df['file'] = df['file'].apply(lambda file: os.path.basename(file).replace('.wav', '.png'))
tqdm.pandas(desc='Mapping files to category ID', total=test.shape[0])
test['category'] = test['file'].progress_apply(lambda file: df.loc[df.file == file]['category'].item())

# convert to the format requested
test['file'] = range(1, test['file'].shape[0] + 1)
test = test.rename(columns={'file': 'id'})

# submit result
test.to_csv(SUBMISSION, index=False)
print('Submission ready!!!')