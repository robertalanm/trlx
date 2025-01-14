from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Model, PreTrainedModel, AutoModelForCausalLM, GPT2PreTrainedModel, GPT2Model, AutoTokenizer
from transformers.modeling_outputs import ModelOutput
from torch import nn
from torch.nn import Identity
import torch.nn.functional as F
import torch
from dataclasses import dataclass
from typing import Optional, Tuple


class GPTRewardModel(nn.Module):
    def __init__(self, config, PAD_ID):
        super().__init__()
        # TODO(dahoas): fix hacky fix
        self.model = AutoModelForCausalLM.from_pretrained(config)
        self.config = self.model.config
        self.neox = "neox" in self.config.model_type
        # gpt-neo models have hidden_size instead of n_embd
        self.config.n_embd = self.config.hidden_size if hasattr(self.config, "hidden_size") else self.config.n_embd
        self.transformer = self.model.gpt_neox if hasattr(self.model, "gpt_neox") else self.model.transformer
        dtype = self.config.torch_dtype if hasattr(self.config, "torch_dtype") is not None else torch.float32
        dtype = torch.float16 if dtype == "float16" else torch.float32
        self.v_head = nn.Linear(self.config.n_embd, 1, bias=False, dtype=torch.float16)
        self.PAD_ID = PAD_ID



    def gradient_checkpointing_enable(self):
        self.model.gradient_checkpointing_enable()


    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        mc_token_ids=None,
        lm_labels=None,
        mc_labels=None,
        return_dict=False,
        output_attentions=False,
        output_hidden_states=False,
    ):
        loss=None
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        ) if self.neox else self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        hidden_states = transformer_outputs[0]

        rewards = self.v_head(hidden_states).squeeze(-1)
        ends = torch.argmax((input_ids == self.PAD_ID).type(torch.float32), dim=1).view(-1, 1)
        rewards = torch.gather(rewards, 1, ends)
        return rewards


# import torch
# from torch import nn
# from transformers import AutoModelForCausalLM, AutoTokenizer


# class GPTRewardModel(nn.Module):
#     def __init__(self, model_path):
#         super().__init__()
#         model = AutoModelForCausalLM.from_pretrained(model_path)
#         self.config = model.config
#         self.neox = "neox" in self.config.model_type
#         # `gpt-neo(x)` models use `hidden_size` attribute names instead of `n_embd``
#         self.config.n_embd = (
#             self.config.hidden_size
#             if hasattr(self.config, "hidden_size")
#             else self.config.n_embd
#         )
#         tokenizer_name = "EleutherAI/gpt-j-6B" if model_path == "robertmyers/bpt-sft" else model_path
#         self.transformer = model.gpt_neox if hasattr(model, "gpt_neox") else model.transformer
#         self.v_head = nn.Linear(self.config.n_embd, 1, bias=False)
#         self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
#         self.tokenizer.pad_token = self.tokenizer.eos_token
#         self.PAD_ID = self.tokenizer(self.tokenizer.pad_token)["input_ids"][0]

#     def forward(
#         self,
#         input_ids=None,
#         past_key_values=None,
#         attention_mask=None,
#         token_type_ids=None,
#         position_ids=None,
#         head_mask=None,
#         inputs_embeds=None,
#         mc_token_ids=None,
#         labels=None,
#         return_dict=False,
#         output_attentions=False,
#         output_hidden_states=False,
#     ):
#         loss = None
#         transformer_outputs = self.transformer(
#             input_ids,
#             past_key_values=past_key_values,
#             attention_mask=attention_mask,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#         ) if self.neox else self.transformer(
#             input_ids,
#             past_key_values=past_key_values,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#         )

#         hidden_states = transformer_outputs[0]

#         rewards = self.v_head(hidden_states).squeeze(-1)
#         chosen_end_scores = []
#         rejected_end_scores = []

#         # Split the inputs and rewards into two parts, chosen and rejected
#         assert len(input_ids.shape) == 2
#         bs = input_ids.shape[0] // 2
#         chosen = input_ids[:bs]
#         rejected = input_ids[bs:]
#         chosen_rewards = rewards[:bs]
#         rejected_rewards = rewards[bs:]

#         # Compute pairwise loss. Only backprop on the last value before padding
#         loss = 0
#         inference = False
#         for i in range(bs):
#             if torch.all(torch.eq(chosen[i], rejected[i])).item():
#                 c_inds = (chosen[i] == self.PAD_ID).nonzero()
#                 c_ind = c_inds[0].item() if len(c_inds) > 0 else chosen.shape[1]
#                 chosen_end_scores.append(chosen_rewards[i, c_ind - 1])
#                 inference = True
#                 continue


#             # Check if there is any padding otherwise take length of sequence
#             c_inds = (chosen[i] == self.PAD_ID).nonzero()
#             c_ind = c_inds[0].item() if len(c_inds) > 0 else chosen.shape[1]
#             r_inds = (rejected[i] == self.PAD_ID).nonzero()
#             r_ind = r_inds[0].item() if len(r_inds) > 0 else rejected.shape[1]
#             end_ind = max(c_ind, r_ind)

#             # Retrieve first index where trajectories diverge
#             divergence_ind = (chosen[i] != rejected[i]).nonzero()[0]
#             assert divergence_ind > 0

#             # Index into the correct rewards
#             c_truncated_reward = chosen_rewards[i][divergence_ind:end_ind]
#             r_truncated_reward = rejected_rewards[i][divergence_ind:end_ind]

#             # Append the last rewards to the list of end scores
#             chosen_end_scores.append(c_truncated_reward[-1])
#             rejected_end_scores.append(r_truncated_reward[-1])

#             # Compute loss
#             loss += -torch.log(
#                 torch.sigmoid(c_truncated_reward - r_truncated_reward)
#             ).mean()
#         loss = loss / bs

#         if not inference:
#             chosen_end_scores = torch.stack(chosen_end_scores)
#             rejected_end_scores = torch.stack(rejected_end_scores)

#         if inference:
#             chosen_end_scores = torch.stack(chosen_end_scores)
#             return {"chosen_end_scores": chosen_end_scores}

#         return {
#             "loss": loss,
#             "chosen_end_scores": chosen_end_scores,
#             "rejected_end_scores": rejected_end_scores,
#         }