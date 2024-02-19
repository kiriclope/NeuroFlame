# NeuroTorch

## Introduction
This package provides an implementation of a recurrent neural network trainer and simulator with pytorch.
Networks can have multiple neural populations with different connectivity types (all to all, sparse) that can be structured (tuned, low rank).
Network weights can be trained in a unsupervised or supervised manner just as vanilla RNNs in torch.

For more info look at the notebooks in ./notebooks and the configuration files in ./conf. 

## Installation
```bash
pip install -r requirements.txt
```
or alternatively,
```bash
conda install --file conda_requirements.txt
```

## Usage
Assuming the dependencies are installed, here is how to run the model (see notebooks folder or org folder for more doc)

```python
# import the network class
from src.network import Network

# initialize model
model = Network(config_file_name, output_file_name, path_to_repo, **kwargs)

# kwargs can be any of the args in the config file

# run a forward pass
model()
```

There are two basic configuration files:
- 'config_bump.py' contains the parameters of a single continuous bump attractor network.
- 'config_EI.py' contains the parameters of a standard balanced network.

## Contributing
Feel free to contribute.
```
MIT License
Copyright (c) [2023] [A. Mahrach]
```
