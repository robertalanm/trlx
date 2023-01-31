from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint

import os



def generate(text):
    
    prompt = f"Human: {text}\nAssistant:"

    response = pipe(
            prompt, 
            max_length=1024, 
            min_length=4, 
            do_sample=False, 
            top_k=512, 
            top_p=0.97, 
            num_beams=2, 
            early_stopping=True, 
            no_repeat_ngram_size=3, 
            num_return_sequences=1
        )[0]["generated_text"]


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


    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=1,
    )
    run_chatbot()