import torch
from reward_model import GPTRewardModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

premise = "The cat sat on the mat."

input_ids = tokenizer.encode(premise, return_tensors="pt", max_length=256)

mask = input_ids != 1
mask.long()


REWARD_CHECKPOINT_PATH = "rm_checkpoint/hf_ckpt.pt"
if not os.path.exists(REWARD_CHECKPOINT_PATH):
    os.makedirs("rm_checkpoint", exist_ok=True)
    os.system(
        f"wget -O {REWARD_CHECKPOINT_PATH} \
        https://huggingface.co/Dahoas/pythia-6b-rm-synthetic/resolve/main/hf_ckpt.pt"
    )


class Pytorch_to_Torchscript(torch.nn.Module):
    def __init__(self):
        super(Pytorch_to_Torchscript, self).__init__()

        # pre_model = AutoModelForCausalLM.from_pretrained('EleutherAI/pythia-6.9b')
        self.model = GPTRewardModel('EleutherAI/pythia-6.9b')
        self.model.load_state_dict(torch.load(REWARD_CHECKPOINT_PATH))
    
    def forward(self, data, attention_mask=None):
        return self.model(data.cuda(), attention_mask.cuda())


pt_model = Pytorch_to_Torchscript().eval()
traced_script_module = torch.jit.trace(pt_model, (input_ids, mask))
traced_script_module.save('model.pt')