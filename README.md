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
Once at the repository root, run the following command

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
