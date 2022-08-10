# nlp-text-summarisation

Retention of properties during text summarisation

## Get dataset

Download the 'FINISHED FILES' from: https://github.com/JafferWilson/Process-Data-of-CNN-DailyMail
Unzip them into the `dataset` directory. The following path should be a valid one: `./dataset/chunked/train_000.bin`.

## Prepare Python environment

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

Sanity check that pre-computation works:
```
python ./compute_resources.py './dataset/chunked/train_000.bin' ./vocab000 ./tf_idf000
```

Now, pre-compute the resources from the entire training set:
```
python ./compute_resources.py './dataset/chunked/train_*.bin' ./vocab ./tf_idf
```
NOTE: On my Macbook Pro laptop, this step takes ~15min.

The `lda.ipynb` notebook generates multiple models based on the grid search. The part that chooses the best one based on the perplexity is still manual.

### Summarization

TBD