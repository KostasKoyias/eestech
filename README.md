# eestech-challenge 2020 in Athens, Greece

## Our Approach

We used a [pre-trained model](https://github.com/lorenlugosch/end-to-end-SLU),
which **we shared** in the Discussion section with other participants.
We mapped each 3-categories combo(action-object-location) to a single one
base on the `intents.csv` file provided in the competition and used
the most frequent category for unknown combos. As a result, we achieved a 0.95 accuracy,
the highest amongst all teams.

## How to run it

First off, fork or download and unzip the code under a Unix-like environment.
Then download the dataset from
[here](https://drive.google.com/file/d/1x2guEnhRjWxBlO0RRgxeq_loAsjUbBmt/view?usp=sharing)
and move it to the root directory of the repository.
Then, from repository root, run the following command

```bash
bash run.sh
```

This will download the dataset, labels and pre-trained model. It will take some time.
Make sure you have at least 3GB of available memory.
After that, the model will predict the labels for all samples
under `eestech/input/test.csv`. This should last roughly 8 minutes.
Lastly, the results will be compared to the ground truth, using accuracy as the metric of interest.

## The model

The pre-trained model can be found
[here](https://github.com/lorenlugosch/end-to-end-SLU).
Make sure to also check the corresponding papers of the creators.

- Loren Lugosch, Mirco Ravanelli, Patrick Ignoto, Vikrant Singh Tomar,
and Yoshua Bengio,
"Speech Model Pre-training for End-to-End Spoken Language Understanding",
Interspeech 2019.
- Loren Lugosch, Brett Meyer, Derek Nowrouzezahrai, and Mirco Ravanelli,
"Using Speech Synthesis to Train End-to-End Spoken Language Understanding Models",
ICASSP 2020.

## Dataset

The script will automatically download from
[here](http://users.uoa.gr/~sdi1500071/eestech/) the following

### input/speakers

A folder containing 10 sub-folders, one for each speaker
containing a .wav file for each command.

### input/data

The .wav file paths to train, test and validate on.

### no_unfreezing/pretraining -- no_unfreezing/model_state.pth

Binary files keeping the state of the pre-trained model serialized.
