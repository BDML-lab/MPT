

## Pre-trained Models

### Downloaded Models

1. Download the pre-trained models from the provided [links](https://drive.google.com/drive/folders/126NTICVEAVj0lpIKYvZGjza-WakYz4I8?usp=sharing).
2. Extract the downloaded files and place them in the `models` directory.

### Pre-training Custom MPT

1. Directly specify the model directory by passing its path to the `--path` argument in `finetune.py`, for example:

```bash
python finetune.py --path logs/...
```