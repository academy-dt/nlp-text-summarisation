# nlp-text-summarisation

Retention of properties during text summarisation

## Get dataset

Download the 'FINISHED FILES' from: https://github.com/JafferWilson/Process-Data-of-CNN-DailyMail
Unzip them into the `dataset` directory. The following path should be a valid one: `./dataset/chunked/train_000.bin`.

## Steps to run

First, initialize the project once by running the following command:
```
source project-init.sh
```

This steps should handle multiple things, including, but not limited to:
- Initialize the 'pointer-generator' git submodule
- Create a Python virtual env
- Install Python prerequisites
- Run Python init script

### LDA

First, train an LDA model once (Takes around 30min on a MacBook Pro):
```
python lda_trainer.py './dataset/chunked/train_*.bin' "$PWD/model/xxx"
```