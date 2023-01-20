from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint
import os
import torch

def convert_deepspeed_checkpoint(model_path, model_name, model_ckpt):
    type_t = "causal"

    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-j-6B')
    fp32_model = load_state_dict_from_zero_checkpoint(model, os.path.join(model_path))

    pipe = pipeline('text-generation', model=fp32_model, tokenizer=tokenizer, framework='pt')
    import code; code.interact(local=dict(globals(), **locals()))
    # if type_t == "causal":
    #     torch.save(model.state_dict(), os.path.join(model_path, "hf_ckpt/hf_ckpt.pt"))
    # else:
    #     fp32_model.save_pretrained(os.path.join(model_path, "hf_ckpt"))


if __name__ == "__main__":
    model_path = "./ckpts/"
    model_name = "EleutherAI/gpt-j-6B"
    model_ckpt = "model-5000"
    convert_deepspeed_checkpoint(model_path, model_name, model_ckpt)