# LDIFS

[![arXiv](https://img.shields.io/badge/stat.ML-arXiv%3A2308.13320-B31B1B.svg)](https://arxiv.org/abs/2308.13320)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/omegafragger/ldifs_code/blob/main/LICENSE)

This repository contains the code for [*Fine-tuning can cripple your foundation model; preserving features may be the solution*](https://openreview.net/forum?id=kfhoeZCeW7).

## Dependencies

The code is based on PyTorch and the dependencies are in [requirements.txt](requirements.txt). To prepare the environment run `pip install -r requirements.txt`.


## Fine-tuning

In order to fine-tune a model, use the [finetune.py](finetune.py) training script. Following are the important parameters for fine-tuning a model:
```
--data-location: directory containing the fine-tuning datasets
--train-datasets: comma separated list of fine-tuning datasets
--model: model architecture to train, eg. ViT-B-32
--finetune-loss: Loss function for fine-tuning, options: ce, ls, l2sp, flyp_ce, flyp, ldifs
--zs-init: whether to use zero-shot initialization, setting false will do linear probe initialization
```

The full set of fine-tuning arguments can be found in [args.py](args.py). As an example, in order to fine-tune a `ViT-B/32` based CLIP model on the sequence SVHN, CIFAR-10 and RESISC45, on `LP-init-LDIFS` use the following:

```python
python finetune.py \
--data-location /location/to/training/datasets \
--train-datasets SVHN,CIFAR10,RESISC45 \
--model ViT-B-32 \
--finetune-loss ldifs \
--ldifs-alpha 10.0 \
--save /path/to/store/model/checkpoints \
```

### Evaluation

To evaluate trained models, use [evaluate.py](evaluate.py). Following are the most important parameters for the evaluation script.
```
--data-location: directory containing the datasets
--res-store-path: path to store results
--train-dataset: name of the train dataset
--eval-dataset: name of the eval dataset
--model: model architecture of checkpoint
--model-location: path of the directory of the saved model checkpoint
--finetune-loss: loss function used to finetune the model
--ldifs-alpha: alpha used for LDIFS regularisation (if finetune loss is ldifs)
--zs-init: whether ZS initialization was used, if not then by default LP initialization is assumed
--it-index: Iteration index which is a value in [0, 1, 2, ..., 100] which determines the model checkpoint to evaluate
--zs: Whether to do ZS evaluation
--lp: Whether to do LP evaluation
```

The full set of evaluation arguments can also be found in [args.py](args.py). As an example, to evaluate the 50th iteration index of a CLIP ViT-B/32 fine-tuned using LDIFS on CIFAR-10 but being evaluated on the Cars dataset, run the following:

```python
python evaluate.py \
--data-location /location/to/evaluation/datasets \
--res-store-path /path/to/directory/for/storing/results \
--train-dataset CIFAR10 \
--eval-dataset Cars \
--model-location /path/to/directory/containing/models \
--model ViT-B-32 \
--finetune-loss ldifs \
--zs \
--lp \
--ldifs-alpha 10.0 \
--it-index 50 \
```

The result is stored as a simple JSON file with the following format. The following is an example and not numbers from real results:

```json
{
    "zs_acc": 0.6744562,
    "zs_ece": 0.0889233,
    "lp_acc": 0.6453246,
    "lp_ece": 0.0284556
}
```


## Citation

If the code or the paper has been useful in your research, please add a citation to our work:

```
@article{
mukhoti2024finetuning,
title={Fine-tuning can cripple your foundation model; preserving features may be the solution},
author={Jishnu Mukhoti and Yarin Gal and Philip Torr and Puneet K. Dokania},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2024},
url={https://openreview.net/forum?id=kfhoeZCeW7},
note={Featured Certification}
}
```

## Questions

For any questions, please feel free to raise an issue or email us directly. Our emails can be found on the [paper](https://arxiv.org/abs/2308.13320).
