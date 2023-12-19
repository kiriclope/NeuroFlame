# NeuroTorch

## Introduction
This package provides an implementation of a recurrent neural network trainer and simulator with pytorch.
The network can have multiple neural populations, different connectivity profiles (all to all, sparse, tuned, ...).
For more info look at the config files in ./conf/.

## Installation
Provide clear instructions on how to get your development environment running.
```bash
pip install -r requirements.txt
```
## Usage
Assuming the dependencies are installed, here is how to run the model (see notebooks folder or org folder for more doc)

```python
# import the network class
from src.model.network import Network

# initialize model
model = Network(config_file_name, output_file_name, path_to_repo, **kwargs)

# kwargs can be any of the args in the config file

# run the model
model.run()
```
There are two configs here:
- The first one is config_bump.py which is a continuous 1 population bump attractor model as in the NB stim paper.
- The second is config_EI.py which are standard parameters for a tuned bump attractor balance network with 2 populations.

## Contributing
Feel free to contribute.
```
MIT License
Copyright (c) [2023] [A. Mahrach]
```
