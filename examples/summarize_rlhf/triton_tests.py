import os

import numpy as np

from tritonclient.utils import np_to_triton_dtype
import tritonclient.grpc as client_util

from transformers import AutoTokenizer

def prepare_tensor(name: str, input):
    t = client_util.InferInput(name, input.shape, np_to_triton_dtype(input.dtype))
    t.set_data_from_numpy(input)
    return t


reward_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
reward_tokenizer.pad_token = reward_tokenizer.eos_token
reward_tokenizer.truncation_side = "left"


triton_host = os.environ.get("TRITON_HOST", "localhost:8001")
triton_model = os.environ.get("TRITON_MODEL", "gptj-rm-static")

client = client_util.InferenceServerClient(url=triton_host, verbose=False)

input = reward_tokenizer("Assistant: Hello! How are you today", padding=True, max_length=1024)
input_ids = np.array(input.input_ids, dtype=np.int32)
attention_mask = np.array(input.attention_mask, dtype=np.int8).reshape(-1,-1)

inputs = [
    prepare_tensor("input_ids", input_ids),
    prepare_tensor("attention_mask", attention_mask),
]

result = client.infer(triton_model, inputs)
rewards = result.as_numpy("rewards")
print(rewards)

