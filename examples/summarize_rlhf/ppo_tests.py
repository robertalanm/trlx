from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint

import os



def generate(text):
    
    prompt = f"Human: {text}\n\nAssistant:"

    inputs = tokenizer(text, return_tensors="pt")
    input_ids, attention_mask = inputs["input_ids"].to("cuda:1"), inputs["attention_mask"].to("cuda:1")

    response = fp32_model.generate(input_ids, attention_mask=attention_mask, max_length=1024, do_sample=False, top_k=0, top_p=1, num_return_sequences=1, early_stopping=True)

    # replace the prompt from the response
    response = response[0].tolist()
    response = response[len(tokenizer.encode(prompt)):]
    return tokenizer.decode(response, skip_special_tokens=True)

    # return tokenizer.decode(response[0], skip_special_tokens=True)
    # return response

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
    fp32_model.to("cuda:1")

    run_chatbot()