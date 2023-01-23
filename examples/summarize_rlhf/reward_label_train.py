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


    def reward_fn(samples: List[str], **kwargs):
        "Reward function for sentiment analysis"
        
        scores_list = []
        batch_size = 1
        for sample in samples:
            scores_list.append(reward_dict[sample])

        scores = torch.cat(scores_list, dim=0)
        return scores


    data = load_dataset("Dahoas/reward-labeled-static")

    train_set = [sample["response"] for sample in data["train"]]
    reward_set = [sample["reward"] for sample in data["train"]]

    # val_set is last 10 samples of train
    val_set = train_set[-40:]

    # delete the last 10 samples from train
    train_set = train_set[:-40]

    val_reward_set = reward_set[-40:]
    reward_set = reward_set[:-40]

    
    reward_dict = {}
    for i in range(len(train_set)):
        reward_dict[train_set[i]] = reward_set[i]
    for i in range(len(val_set)):
        reward_dict[val_set[i]] = val_reward_set[i]
    
    import code; code.interact(local=dict(globals(), **locals()))

    trlx.train(
        config.model.model_path,
        prompts=train_set,
        reward_fn = reward_fn,
        eval_prompts=val_set,  # sampling 1000 validation prompts for evaluation speed in training
        config=config,
    )


if __name__ == "__main__":
    main()
