# Generates positive movie reviews by tuning a pretrained model on IMDB dataset
# with a sentiment reward function

import os
import pathlib
from typing import List

import torch
import yaml
from datasets import load_dataset
from transformers import pipeline

import trlx
from trlx.data.configs import TRLConfig


def get_positive_score(scores):
    "Extract value associated with a positive sentiment from pipeline's output"
    return dict(map(lambda x: tuple(x.values()), scores))["POSITIVE"]


config_path = pathlib.Path(__file__).parent.joinpath("configs/ppo_config_summ_gptj.yml")
with config_path.open() as f:
    default_config = yaml.safe_load(f)


def train_test_split(data, test_size=0.1, random_state=42):
    import numpy as np

    np.random.seed(random_state)
    np.random.shuffle(data)
    split_index = int(len(data) * (1 - test_size))
    return data[:split_index], data[split_index:]


def main(hparams={}):
    config = TRLConfig.update(default_config, hparams)

    if torch.cuda.is_available():
        device = int(os.environ.get("LOCAL_RANK", 0))
    else:
        device = -1


    dataset = load_dataset("Dahoas/reward-labeled-static")

    train_set = [(sample["response"], sample['reward']) for sample in dataset["train"]]


    train_set, val_set = train_test_split(train_set, test_size=0.1, random_state=42)

    # shuffle val set, train set
    import random
    random.shuffle(train_set)
    random.shuffle(val_set)

    trlx.train(
        dataset=train_set,
        eval_prompts=val_set[
            0:10
        ],  # sampling 1000 validation prompts for evaluation speed in training
        config=config,
    )


if __name__ == "__main__":
    main()
