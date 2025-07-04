# HyperAugment

## Overview
This repository contains the implementation of HyperAugment, a generative feature augmentation model for HNNs. This model is designed to improve HNNs' performance.

## Table of Contents
- [HyperAugment](#hyperaugment)
  - [Overview](#overview)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Pretraining](#pretraining)
  - [Training \& Evaluation](#training--evaluation)


## Installation
To install the necessary dependencies, run:

```bash
pip install -r requirements.txt
```

## Pretraining
```bash
python hyperaugment_cvae_pretrain.py
python hyperaugment_cvae_generate.py
```

## Training & Evaluation
```bash
python hyperaugment_main.py
```
