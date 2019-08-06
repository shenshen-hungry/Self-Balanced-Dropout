#!/usr/bin/env bash

# An example to apply Self-Balanced Dropout to CNN on SST-1 dataset.
# Test results are predicted by the model which achieves the highest accuracy on Dev set.

python3 process_data.py GoogleNews-vectors-negative300.bin
python3 cnn.py sst1.p
