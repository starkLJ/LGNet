# Shooting Condition Insensitive Unmanned Aerial Vehicle Object Detection

This project contains the code for the paper "Shooting Condition Insensitive Unmanned Aerial Vehicle Object Detection" and relies on mmdetection 2.x.

## Environment Setup
After preparing the environment, you can download the following datasets:
- UAVDT dataset can be downloaded from [here](https://sites.google.com/view/grli-uavdt/%E9%A6%96%E9%A1%B5)
- VisDrone dataset can be downloaded from [here](https://github.com/VisDrone/VisDrone-Dataset)

You can also convert the dataset annotations to the COCO format. For your convenience, we have provided COCO format annotations in the project.

Please organize the datasets as follows:

```shell
DATA
├─ UAVDT
│  ├─ annotations
│  │  ├─ UAVDT_test_coco.json
│  │  ├─ UAVDT_train_coco.json
│  ├─ images
│  │  ├─ test
│  │  │  ├─ M0203
│  │  │  │  ├─ xxxx.jpg
│  │  │  │  └─ ...
│  │  │  └─ ...
│  │  ├─ train
│  │  │  ├─ M0101
│  │  │  │  ├─ xxxx.jpg
│  │  │  │  └─ ...
│  │  │  └─ ...
├─ visdrone_coco
│  ├─ annotations
│  │  ├─ instances_UAVval.json
│  │  ├─ instances_UAVtrain.json
│  ├─ images
│  │  ├─ instances_UAVtrain
│  │  │  ├─ xxxx.jpg
│  │  │  └─ ...
│  │  ├─ instances_UAVval
│  │  │  ├─ xxxx.jpg
│  │  │  └─ ...
```


## Text Prompt Embedding Fine-Tuning
1. Run `text_learner/gen_fix_prompts.py` to generate initial prompt features.
2. Then, run `text_learner/prompts_learner.py` to generate fine-tuned features.

## Training
Please note that training is supported on a single GPU:

```shell
python train.py configs/xxxx.py
```

## Citation
If you find this repository helpful, please consider citing our paper:

```shell
@article{LIU2024123221,
title = {Shooting Condition Insensitive Unmanned Aerial Vehicle Object Detection},
journal = {Expert Systems with Applications},
volume = {246},
pages = {123221},
year = {2024},
issn = {0957-4174},
doi = {https://doi.org/10.1016/j.eswa.2024.123221},
url = {https://www.sciencedirect.com/science/article/pii/S0957417424000861},
author = {Jie Liu and Jinzong Cui and Mao Ye and Xiatian Zhu and Song Tang},
}
```