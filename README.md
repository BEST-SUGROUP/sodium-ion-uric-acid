# Sodium-Ion-Uric-Acid

## Overview

The main branch contains the code for the classification of sodium-ion, while the uric-acid branch contains the code for the classification of uric acid. Both branches follow the same organized structure:

## Structure

### `network_building.py`
Construct and build the network.

### `network_in_training_XX.pth`
Parameters checkpoint obtained during training.

### `process_data`
Data preprocessing and data loading.

### `test_output.py`
Test function for the final test.

### `main.py`
Starts the training, validation, and testing processes.
- Replace the `img_path` with the correct path of your data in line 7.
- Comment out line 12 if you want to run inference only.
- Pass the name of the `XX.pth` file to `trained_network_name` in line 16.
