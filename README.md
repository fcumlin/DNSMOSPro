# Dnsmos Pro

This is the official implementation of "DNSMOS Pro: A Reduced-Size DNN for Probabilistic MOS of Speech". DNSMOS Pro is a model that takes as input a speech clip, and outputs the mean opinion score (MOS).

Authors: Fredrik Cumlin, Xinyu Liang
Emails: fcumlin@gmail.com, hopeliang990504@gmail.com

## Inference

There are three pretrained models ready to be used. For inference, one can do the following (all paths are relative to this directory):
```
import numpy as np
import torch

import utils  # Python file containing the STFT.

model = torch.jit.load('runs/test_nisqa/model_best.pt', map_location=torch.device('cpu'))
samples = np.ones(160_000)
spec = torch.FloatTensor(utils.stft(samples))  # Defaults in `utils.stft` correspond to training values.
with torch.no_grad():
    prediction = model(spec[None, None, ...])
mean = prediction[:, 0]
variance = prediction[:, 1]
print(f'{mean=}, {variance=}')
```
## Dataset preparation
[VCC2018](https://github.com/unilight/LDNet/tree/main/data).
[BVCC](https://zenodo.org/records/6572573#.Yphw5y8RprQ).
[NISQA](https://github.com/gabrielmittag/NISQA/wiki/NISQA-Corpus).

## Training
The framework is Gin configurable, hence specifying model and dataset is done with a Gin config. See `configs/*.gin` for examples.
 
Example launch:
```
python train.py --gin_path "configs/vcc2018.gin" --save_path "runs/VCC2018"
```
