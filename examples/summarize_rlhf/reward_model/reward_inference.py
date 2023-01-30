import mii
from transformers import AutoConfig

mii_config = {"dtype": "fp16"}

name = "robertmyers/bpt-rm-6b"

config = AutoConfig.from_pretrained(name)
model_hidden_size = config.n_embd

ds_config = {
    "fp16": {
        "enabled": True
    },
    "bf16": {
        "enabled": False
    },
    "aio": {
        "block_size": 262144,
        "queue_depth": 32,
        "thread_count": 1,
        "single_submit": False,
        "overlap_events": True
    },
    "zero_optimization": {
        "stage": 2,
        "offload_param": {
            "device": "cpu",
        },
        "overlap_comm": True,
        "contiguous_gradients": True
    },
    "train_micro_batch_size_per_gpu": 1,
}

mii.deploy(task='text-generation',
           model=name,
           deployment_name=name + "_deployment",
           model_path=".cache/models/" + name,
           mii_config=mii_config,
           enable_deepspeed=False,
           enable_zero=True,
           ds_config=ds_config)