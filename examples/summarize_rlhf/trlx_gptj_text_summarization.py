import os
import pathlib
from typing import List

import torch
from datasets import load_dataset
from reward_model.reward_model import GPTRewardModel
from tqdm import tqdm
from transformers import AutoTokenizer

import re
import trlx
from trlx.data.configs import TRLConfig

REWARD_CHECKPOINT_PATH = "reward_model/rm_checkpoint/hf_ckpt.pt"
# REWARD_CHECKPOINT_PATH = "reward_model/rm_checkpoint/checkpoint-500/pytorch_model.bin"
# if not os.path.exists(REWARD_CHECKPOINT_PATH):
#     os.makedirs("reward_model/rm_checkpoint", exist_ok=True)
#     os.system(
#         f"wget -O {REWARD_CHECKPOINT_PATH} \
#         https://huggingface.co/robertmyers/rm-1.3b/resolve/main/pytorch_model.bin"
#     )
SFT_MODEL_PATH = "robertmyers/bpt-sft"


if __name__ == "__main__":

    # Load the pre-trained reward model
    rw_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    rw_tokenizer.pad_token = rw_tokenizer.eos_token
    rw_model = GPTRewardModel("EleutherAI/gpt-j-6B", rw_tokenizer.pad_token)
    rw_model.load_state_dict(torch.load(REWARD_CHECKPOINT_PATH))
    rw_model.half()
    rw_model.eval()

    # find the current running gpu rank 
    # (if you are running on a single gpu, this will be 0)
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 1

    for i in range(8):
        rw_device = torch.device(f"cuda:{i}")
        rw_model.to(rw_device)
            

    def get_scores(samples: List[str], device=None):
        scores_list = []
        batch_size = 1
        # print('getting scores!')
        for i in range(0, len(samples), batch_size):
            sub_samples = samples[i : i + batch_size]
            sub_samples = [
                "<|startoftext|>" + chosen + "<|endoftext|>" for chosen in sub_samples
            ]
            encodings_dict = rw_tokenizer(
                sub_samples,
                truncation=True,
                max_length=config.train.seq_length,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = encodings_dict["input_ids"].to(device)
            attn_masks = encodings_dict["attention_mask"].to(device)
            input_ids = input_ids.repeat(2, 1)
            attn_masks = attn_masks.repeat(2, 1)
            with torch.no_grad():
                sub_scores = rw_model(input_ids=input_ids, attention_mask=attn_masks)
            scores_list.append(sub_scores["chosen_end_scores"])
        scores = torch.cat(scores_list, dim=0)
        # remove the input_ids and attention_mask from the gpu
        del input_ids
        del attn_masks

        return scores

    def get_prompt_dataset(prompts, max_length):
        """
        Get the prompt after T5 decoding to make sure dictionary
        of prompts and summaries is consistent decode prompt from trlX pipeline
        """
        formatted_prompts = []
        for i in tqdm(range(len(prompts))):
            tmp = tokenizer.decode(
                tokenizer(
                    prompts[i].split("Assistant:")[0],
                    truncation=True,
                    max_length=max_length
                    - 11,  # to make sure "TL;DR" dont get truncated
                )["input_ids"],
                skip_special_tokens=True,
            ).strip()
            tmp = tmp + "\nAssistant:"
            tmp = tokenizer.decode(
                tokenizer(tmp, truncation=True, max_length=max_length)["input_ids"],
                skip_special_tokens=True,
            ).strip()
            formatted_prompts.append(tmp)
        return formatted_prompts

    def reward_fn(samples: List[str], **kwargs):

        device = kwargs.get("device", "cuda:0")

        try:
            original_samples = [text.split("Assistant:")[0] + "Assistant: " for text in samples]
            original_samples = [
                text + post_summary_dict[text.strip()] for text in original_samples
            ]
            original_scores = get_scores(original_samples, device=device)
            scores = get_scores(samples, device=device)
            norms_scores = scores - original_scores
            # print('norms_scores', norms_scores)
            return norms_scores
        except Exception as e:
            print(e)
            return torch.zeros(len(samples))

    # recommend a error handler for the reward function
    def reward_fn_error_handler(e):
        print(e)
        return torch.zeros(1)

    

    def train_test_split(data, test_size=0.1, random_state=42):
        import numpy as np

        np.random.seed(random_state)
        np.random.shuffle(data)
        split_index = int(len(data) * (1 - test_size))
        return data[:split_index], data[split_index:]

    config_path = pathlib.Path(__file__).parent.joinpath(
        "configs/ppo_config_summ_gptj.yml"
    )
    config = TRLConfig.load_yaml(config_path)

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer.tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    max_length_input = (
        config.train.seq_length - config.method.gen_kwargs["max_new_tokens"]
    )

    dataset = load_dataset("Dahoas/rm-synthetic-hh")

    # Store data into prompt and label pairs
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
        config.model.model_path,
        reward_fn=reward_fn,
        prompts=train_prompts,
        stop_sequences=["Human:", "human:", "Assistant:", "assistant:"],
        eval_prompts=val_prompts[
            0:10
        ],  # sampling 1000 validation prompts for evaluation speed in training
        config=config,
    )

    trainer.save_pretrained("bpt-ppo")
