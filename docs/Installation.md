## Installation

### Requirements

* Linux (tested on Ubuntu 24.04)
* Python 3.7
* Miniconda
* PyTorch
* CUDA 11.7

### Create Anaconda Environment from yml

in the directory of `BTS`:

```bash
cd coperception
conda env create -f environment.yml
conda activate coperception
```

### CUDA

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```

### Install CoPerception Library

This installs and links `coperception` library to code in `./coperception` directory.

```bash
pip install -e .
```

### Dataset Preparation

Please download and unzip the [parsed dataset](https://huggingface.co/datasets/liuzh594/CoSwarm/resolve/main/CoSwarm-det.tar.gz) of CoSwarm.

### Specifying Dataset

Link the test split of **CoSwarm** dataset in the default value of argument `data`

```python
# BTS/coperception/tools/det/BTS/BTS_util.py
parser.add_argument("-d", "--data", default="/{your location}/dataset/CoSwarm-det/test", type=str, help="The path to the preprocessed sparse BEV training data", )
```

in the `test` folder data are structured like:

```
test
├──agent_0
├──agent_1
├──agent_2
├──agent_3
├──agent_4
├──agent_5
    ├──8_0
	    ├──0.npy		
    ...
```



### Specifying Victim Detection Model Checkpoint

Link the checkpoint location in the default value of argument `resume`

Please download [pre-trained weights](https://huggingface.co/datasets/liuzh594/CoSwarm/resolve/main/epoch_50.pth) and save them in `BTS/coperception/tools/det/runs/resume/max/with_cross` folder.

```python
# BTS/coperception/tools/det/BTS/BTS_util.py
parser.add_argument("--resume", default="/{your location}/BTS/coperception/tools/det/runs/resume/max/with_cross/epoch_50.pth", type=str, help="The path to the saved model that is loaded to resume training", )
```

### Specifying The Log Path
If you need log, don't forget to specify the log path in addition to `--log`.
```python
# BTS/coperception/tools/det/BTS/BTS_util.py
parser.add_argument("--logpath", default="/{your location}/BTS/coperception/logs", help="The path to the output log file")
```