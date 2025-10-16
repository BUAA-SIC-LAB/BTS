# Detection on CoSwarm
You need to convert the raw dataset (CoSwarm) to the BEV

## Preparation

- Download CoSwarm datasets from [Hugging Face](https://huggingface.co/datasets/liuzh594/CoSwarm)
- Add nuscenes-devkit dependency: ```export PYTHONPATH=nuscenes-devkit/python-sdk/:PYTHONPATH```
- Run the code below to generate preprocessed data
```bash
python multiprocess_create.py
```
- This is a multi-process parallel processing data set program([multiprocess_create.py](coperception/tools/det/multiprocess_create.py)), you can click to view the relevant parameters and logic