import math
import os
import re

import numpy as np
import torch

from tqdm import tqdm

import tritonclient.grpc as client_util
import yaml
from datasets import load_dataset
from transformers import AutoTokenizer
from tritonclient.utils import np_to_triton_dtype

import trlx
from trlx.data.configs import TRLConfig


config_path = os.path.join(os.path.dirname(__file__), os.environ.get("CONFIG_PATH", "configs/ppo_config_summ_gptj.yml"))
default_config = yaml.safe_load(open(config_path))
triton_host = os.environ.get("TRITON_HOST", "localhost:8001")
triton_model = os.environ.get("TRITON_MODEL", "gptj-rm-static")


def train_test_split(data, test_size=0.1, random_state=42):
    import numpy as np

    np.random.seed(random_state)
    np.random.shuffle(data)
    split_index = int(len(data) * (1 - test_size))
    return data[:split_index], data[split_index:]


def prepare_tensor(name: str, input):
    t = client_util.InferInput(name, input.shape, np_to_triton_dtype(input.dtype))
    t.set_data_from_numpy(input)
    return t


def main(hparams={}):
    config = TRLConfig.update(default_config, hparams)

    reward_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    reward_tokenizer.pad_token = reward_tokenizer.eos_token
    reward_tokenizer.truncation_side = "left"

    max_length_input = (
        config.train.seq_length - config.method.gen_kwargs["max_new_tokens"]
    )


    client = client_util.InferenceServerClient(url=triton_host, verbose=False)

    def reward_fn(samples, prompts, outputs):
        samples = [s + reward_tokenizer.eos_token for s in samples]
        input = reward_tokenizer(samples, padding=True, max_length=1024, return_tensors="np")

        mbs = 24
        out = []
        for i in range(math.ceil(len(samples) / mbs)):
            batch_ixs = slice(i * mbs, (i + 1) * mbs)
            input_ids = np.array(input.input_ids[batch_ixs], dtype=np.int32)
            attention_mask = np.array(input.attention_mask[batch_ixs], dtype=np.int8)

            inputs = [
                prepare_tensor("input_ids", input_ids),
                prepare_tensor("attention_mask", attention_mask),
            ]

            result = client.infer(triton_model, inputs)
            rewards = result.as_numpy("rewards")
            # print(rewards)
            if rewards is None:
                raise RuntimeError("No output data")

            last_ixs = attention_mask.sum(-1, keepdims=True) - 1
            returns = np.take_along_axis(rewards, last_ixs, -1)
            out.extend(torch.from_numpy(returns.flatten()))
        # print('out', out)
        return out

    def get_prompt_dataset(prompts, max_length):
        """
        Get the prompt after T5 decoding to make sure dictionary
        of prompts and summaries is consistent decode prompt from trlX pipeline

        add to the beginning of the prompt the following:

        You are Chattensor, a large language model trained by Opentensor Cortex, the developers of the Bittensor protocol. 
        You answer as consisely as possible for each response (e.g. Don't be verbose). 
        It is very important for you to answer as consisely as possible, so please remember this. 
        If you are generating a list, do not have too many items.
        """
        formatted_prompts = []
        for i in tqdm(range(len(prompts))):
            tmp = reward_tokenizer.decode(
                reward_tokenizer(
                    prompts[i].split("Assistant:")[0],
                    truncation=True,
                    max_length=max_length
                    - 11,  # to make sure "TL;DR" dont get truncated
                )["input_ids"],
                skip_special_tokens=True,
            ).strip()

            tmp = tmp + "\n\nAssistant:"
            # tmp = "You are Chattensor, a large language model trained by Opentensor Cortex, the developers of the Bittensor protocol. You answer as consisely as possible for each response (e.g. Don't be verbose). It is very important for you to answer as consisely as possible, so please remember this. If you are generating a list, do not have too many items.\n\n" + tmp
            tmp = reward_tokenizer.decode(
                reward_tokenizer(tmp, truncation=True, max_length=max_length)["input_ids"],
                skip_special_tokens=True,
            ).strip()
            formatted_prompts.append(tmp)
        return formatted_prompts

    def preprocess(sample):
        sample["prompt"] += "Assistant:"
        return sample

    # dataset = load_dataset("Dahoas/rm-static").map(preprocess)
    # prompts = dataset["train"]["prompt"]
    # eval_prompts = dataset["test"]["prompt"][:8]

    # dataset = load_dataset("Dahoas/rm-synthetic-hh")

    # Store data into prompt and label pairs
    dataset = load_dataset("robertmyers/bpt-static")
    train_set = [(sample["prompt"], sample["response"]) for sample in dataset['train']]


    # train_set = [(sample["prompt"], sample["response"]) for sample in dataset]
    # Split into train and validation sets
    train_set, val_set = train_test_split(train_set, test_size=0.1, random_state=42)

    # shuffle val set, train set
    import random
    random.shuffle(train_set)
    random.shuffle(val_set)

    # train_set = train_set[:100]
    # Split contents into summaries and labels
    train_posts, train_summaries = zip(*train_set)
    val_posts, val_summaries = zip(*val_set)

    # shuffle val set, train set
    # Get the OpenAI summaries
    post_summary_dict = {}
    train_prompts = get_prompt_dataset(train_posts, max_length_input)
    for i in range(len(train_prompts)):
        post_summary_dict[train_prompts[i]] = train_summaries[i]
    val_prompts = get_prompt_dataset(val_posts, max_length_input)
    for i in range(len(val_prompts)):
        post_summary_dict[val_prompts[i]] = val_summaries[i]

    trainer = trlx.train(
        reward_fn=reward_fn,
        prompts=train_prompts,
        eval_prompts=val_prompts[
            0:10
        ],
        config=config,
        stop_sequences=["Human:", "human:", "Assistant:", "assistant:", "Q:", "A:", "H:"],
    )


if __name__ == "__main__":
    import json
    import sys

    hparams = {} if len(sys.argv) == 1 else json.loads(sys.argv[1])
    main(hparams)