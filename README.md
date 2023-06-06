# BirdCLEF-2023-Identify-bird-calls-in-soundscapes 4th place solution

My writeup for this solution can be found on [kaggle](https://www.kaggle.com/competitions/birdclef-2023/discussion/412753).

## Hardware
- Intel(R) Core(TM) i7-8700 CPU @ 3.20GHz, CPU Core=12, CPU Memory=64GB, GPU= 1 x RTX 3090
## OS/platform
- Linux Ubuntu 20.04 LTS
- python==3.7.13

## Training

### Data preparation
- Download BirdCLEF data for 2021, 2022, and 2023
- Download additional datasets [here](https://www.kaggle.com/datasets/atsunorifujita/birdclef-2023-additional)
- Copy the no-call directory of ff1010bird_nocall to the BirdCLEF 2023 train_audio directory.

Directory structure example
```
/input/
    ┣ aicrowd2020_noise_30sec/
    ┣ birdclef-2021/
        └ train_short_audio
    ┣ birdclef-2022/
        └ train_audio
    ┣ birdclef-2023/
        ├ train_audio <- add no-call
        └ train_meta_pseudo.pickle
    ┣ esc50/
    ┣ ff1010bird_nocall/
        └  ff1010bird_metadata_v1_pseudo.pickle
    ┣ train_soundscapes/
    ┣ xeno-canto/
    ┣ xeno-canto_nd/
    ┣ zenodo_nocall_30sec/
    ┣ pretrain_metadata_10fold_pseudo.pickle
    ┣ xeno-canto_audio_meta_pseudo.pickle
    ┗ xeno-canto_nd_audio_meta_pseudo.pickle
/src/
    ┗ ...
```

### If you train on your own data
- Get predicted values from Kaggle Models like this [notebook](https://www.kaggle.com/code/atsunorifujita/extract-from-kaggle-models/notebook).
- Store the vector of predicted values as one column (teacher_preds) in the training data. like a ○○○.pickle.

### Run
```
# -C flag is used to specify a config file
# replace NAME_OF_CONFIG with an appropiate config file name such as exp105

python pretrain_net.py -C NAME_OF_CONFIG  # for pretraining using BirdCLEF 2021, 2022

python train_net.py -C NAME_OF_CONFIG  # for training using BirdCLEF 2021, 2022
```

### Weights of the trained model
- [BirdCLEF2023-4th-models](https://www.kaggle.com/datasets/atsunorifujita/birdclef2023-4th-models)

## Inference
Inference is published in a kaggle kernel [here](https://www.kaggle.com/code/atsunorifujita/4th-place-solution-inference-kernel).

### Ablation study
| Name | Public LB | Private LB |
| --- | --- |--- |
| BaseModel | 0.80603 | 0.70782 |
| BaseModel + Knowledge Distillation | 0.82073 | 0.72752 |
| BaseModel + Knowledge Distillation + Adding xeno-canto | 0.82905 | 0.74038 |
| BaseModel + Knowledge Distillation + Adding xeno-canto + Pretraining | 0.8312 | 0.74424 |
| BaseModel + Knowledge Distillation + Adding xeno-canto + Pretraining + Ensemble (4 models) | 0.84019 | 0.75688 |


## References
- [Bird CLEF 2021 2nd place solution](https://www.kaggle.com/competitions/birdclef-2021/discussion/243463)
- [BirdCLEF 2021 - Birdcall Identification 4th place solution](https://github.com/tattaka/birdclef-2021)
- [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)
- [Knowledge distillation: A good teacher is patient and consistent](https://arxiv.org/abs/2106.05237)
- [PyTorch-PCEN](https://github.com/daemon/pytorch-pcen)
