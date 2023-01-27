import torch
from    reward_model import GPTRewardModel
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

premise = "The cat sat on the mat."

input_ids = tokenizer.encode(premise, return_tensors="pt", max_length=256)

mask = input_ids != 1
mask.long()



class Pytorch_to_Torchscript(torch.nn.Module):
    def __init__(self):
        super(Pytorch_to_Torchscript, self).__init__()
        self.model = GPTRewardModel('Dahoas/gptj-rm-static')
    
    def forward(self, data, attention_mask=None):
        return self.model(data.cuda(), attention_mask.cuda())


pt_model = Pytorch_to_Torchscript().eval()
traced_script_module = torch.jit.trace(pt_model, (input_ids, mask))
traced_script_module.save('model.pt')