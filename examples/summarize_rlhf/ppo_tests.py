from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint

import os



def generate(text):
    
    prompt = prompt + f"Human: {text}\nAssistant:"

    response = pipe(text, max_length=1024, min_length=32, do_sample=True, top_k=0, top_p=1.0, num_return_sequences=1,)[0]['generated_text']

    prompt = prompt + response + "\n"

    return response

# generate("Hello! How are you today?")

# build a chatbot cli

def run_chatbot():
    while True:
        text = input("HUMAN> ")
        response = generate(text)
        print(f"SYBIL> {response}")

if __name__ == "__main__":
    model_path = "./ckpts/"
    model_name = "EleutherAI/gpt-j-6B"
    model_ckpt = "model-5000"

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    # model = AutoModelForCausalLM.from_pretrained('./bpt-ppo-base')
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-j-6B')
    fp32_model = load_state_dict_from_zero_checkpoint(model, os.path.join(model_path))


    prompt = ""

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=1,
    )
    run_chatbot()