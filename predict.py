import os
import data
import torch
import models
import pandas as pd
import soundfile as sf
from tqdm import tqdm

# configuration & file paths
TEST = '/home/kostas/Desktop/coding/contests/kaggle/eestech/input/2020-athens-eestech-challenge/test.csv'
INTENTS = '/home/kostas/Desktop/coding/contests/kaggle/eestech/input/2020-athens-eestech-challenge/intents.csv'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config = data.read_config("experiments/no_unfreezing.cfg"); _,_,_=data.get_SLU_datasets(config)
model = models.Model(config).eval()
model.load_state_dict(torch.load("experiments/no_unfreezing/training/model_state.pth", map_location=device)) # load trained model

speakers = "/home/kostas/Desktop/coding/contests/kaggle/eestech/input/2020-athens-eestech-challenge/wavs/speakers"

# predict label of each .wav file and store it as a pickle
df = list()
count = len(speakers)
test = pd.read_csv(TEST)
files = set(test['file'].apply(lambda f: f.replace('.png', '.wav')))
bar = tqdm(total=test.shape[0], desc=f'Predicting labels for test set samples only')
paths = list()
for i, speaker in enumerate(os.listdir(speakers)):
    speaker = os.path.join(speakers, speaker)
    for wav in os.listdir(speaker):
        if wav not in files:
            continue
        bar.update(1)
        wav = os.path.join(speaker, wav)
        paths.append(wav)
        signal, _ = sf.read(wav)
        signal = torch.tensor(signal, device=device).float().unsqueeze(0)
        label = model.decode_intents(signal)
        df.append((wav, label))

paths = pd.Series(paths)

bar.close()
df = pd.DataFrame(df, columns=['file', 'category'])
print(df.shape)
df.to_pickle('pred-test.pkl')

exit()
df.to_pickle('pred.pkl')

# map intent list to category ID
intents = pd.read_csv(INTENTS)
tqdm.pandas(desc='Mapping intent to category ID', total=df.shape[0])
df['category'] = df['category'].progress_apply(lambda c: intents.loc[intents.intent == ','.join(c[0])]['category'])
df.to_csv('pred.csv')

# keep labels for samples of the test set only
df['file'] = df['file'].apply(lambda file: os.path.basename(file).replace('.wav', '.png'))
tqdm.pandas(desc='Mapping files to category ID', total=test.shape[0])
test['category'] = test['file'].progress_apply(lambda file: df.loc[df.file == file]['category'].item())

# submit in the format requested
test['file'] = range(1, test['file'].shape[0])
test.rename(columns={'file': 'id'})
test.to_csv('sub.csv', index=False)
print('Submission ready!!!')