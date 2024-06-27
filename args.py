import os
import argparse

import torch


# Define a custom argument type for a list of strings
def list_of_strings(arg):
    return arg.split(',')


def parse_finetune_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-location",
        type=str,
        default='/home/jishnu/Projects/foundational_robustness/proj_cache_dir',
        help="The root directory for the datasets.",
    )

    parser.add_argument(
        "--train-datasets",
        default=None,
        type=list_of_strings,
        help="Sequence of training datasets",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="The type of model (e.g. RN50, ViT-B-32).",
    )

    parser.add_argument(
        "--model-checkpoint-path",
        type=str,
        default=None,
        help="Full path of the model checkpoint to load from",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate."
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=0.1,
        help="Weight decay"
    )

    # Loss functions to perform fine-tuning
    # ce: Cross entropy for full fine-tuning
    # ls: Cross entropy with smoothed labels
    # l2sp: Cross entropy with L2SP regularization (arxiv:1802.01483)
    # flyp_ce: FLYP with cross-entropy loss
    # flyp: FLYP with contrastive loss
    # ldifs: Cross entropy with LDIFS regularization
    parser.add_argument(
        "--finetune-loss",
        type=str,
        default='ce',
        help="Finetune loss function: [ce, ls, l2sp, flyp_ce, flyp, ldifs]"
    )

    parser.add_argument(
        "--ls-factor",
        type=float,
        default=0.0,
        help="Label smoothing."
    )

    parser.add_argument(
        "--l2sp-alpha",
        type=float,
        default=0.5,
        help="Alpha for L2SP regularization."
    )

    parser.add_argument(
        "--ldifs-alpha",
        type=float,
        default=10.0,
        help="Alpha for LDIFS regularization."
    )

    parser.add_argument(
        "--best-among-list",
        action="store_true",
        dest="best_among_list",
    )
    parser.set_defaults(best_among_list=False)

    parser.add_argument(
        "--weight-interpolate",
        action="store_true",
        dest="weight_interpolate",
    )
    parser.set_defaults(weight_interpolate=False)

    parser.add_argument(
        "--warmup_length",
        type=int,
        default=500,
    )

    # Linear layer initialization method:
    # zs_init = True will initialize the linear layer to ZS weights
    # zs_init = False will initialize the linear layer to LP weights
    parser.add_argument(
        "--zs-init",
        action="store_true",
        dest="zs_init",
    )
    parser.set_defaults(zs_init=False)

    parser.add_argument(
        "--freeze-head",
        action='store_true',
        dest="freeze_head"
    )
    parser.set_defaults(freeze_head=False)

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--save",
        type=str,
        default="/home/jishnu/Projects/foundational_robustness/proj_cache_dir",
        help="Optionally save a _classifier_, e.g. a zero shot classifier or probe.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default='/home/jishnu/Projects/foundational_robustness/proj_cache_dir',
        help="Directory for caching features and encoder",
    )
    parser.add_argument(
        "--openclip-cachedir",
        type=str,
        default='/home/jishnu/Projects/foundational_robustness/openclip_cache_dir',
        help='Directory for caching models from OpenCLIP'
    )

    parsed_args = parser.parse_args()
    parsed_args.device = "cuda" if torch.cuda.is_available() else "cpu"

    return parsed_args


def parse_eval_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-location",
        type=str,
        default='./',
        help="The root directory for the datasets.",
    )

    parser.add_argument(
        "--res-store-path",
        type=str,
        default='./',
        help="Path to store results",
    )

    parser.add_argument(
        "--train-dataset",
        default=None,
        type=str,
        help="Which dataset the model was trained on.",
    )

    parser.add_argument(
        "--eval-dataset",
        default=None,
        type=str,
        help="Which dataset the model is to be evaluated on.",
    )

    parser.add_argument(
        "--model-location",
        type=str,
        default=None,
        help='path of saved model checkpoints'
    )

    parser.add_argument(
        "--finetune-loss",
        type=str,
        default='ce',
        help="Finetune loss function: [ce, ls, l2sp, flyp_ce, flyp, ldifs]"
    )

    parser.add_argument(
        "--ldifs-alpha",
        type=float,
        default=10.0,
        help="Alpha for LDIFS regularization."
    )

    parser.add_argument(
        "--zs-init",
        action="store_true",
        dest="zs_init",
    )
    parser.set_defaults(zs_init=False)

    parser.add_argument(
        "--freeze-head",
        action='store_true',
        dest="freeze_head"
    )
    parser.set_defaults(freeze_head=False)

    parser.add_argument(
        "--it-index",
        type=int,
        dest="it_index",
        help="Iteration index: a value between [0, 1, 2, ... 100]"
    )

    parser.add_argument(
        "--zs",
        action='store_true',
        dest="zs",
        help="ZS evaluation?"
    )
    parser.set_defaults(zs=False)


    parser.add_argument(
        "--lp",
        action='store_true',
        dest="lp",
        help="LP evaluation?"
    )
    parser.set_defaults(lp=False)

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="The type of model (e.g. RN50, ViT-B-32).",
    )

    parser.add_argument(
        "--cache-dir",
        type=str,
        default='/home/jishnu/Projects/foundational_robustness/proj_cache_dir',
        help="Directory for caching features and encoder",
    )
    parser.add_argument(
        "--openclip-cachedir",
        type=str,
        default='/home/jishnu/Projects/foundational_robustness/openclip_cache_dir',
        help='Directory for caching models from OpenCLIP'
    )

    parsed_args = parser.parse_args()
    parsed_args.device = "cuda" if torch.cuda.is_available() else "cpu"

    return parsed_args
