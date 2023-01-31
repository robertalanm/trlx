from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
model = AutoModelForCausalLM.from_pretrained('./bpt-ppo-base')

prompt = ""

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0,
)

def generate(text):
    
    prompt = prompt + f"Human: {text}\nAssistant:"

    response = pipe(text, max_length=1024, min_length=32, do_sample=True, top_k=0, top_p=1.0, num_return_sequences=1,)[0]['generated_text']

    prompt = prompt + response + "\n"

    return response

generate("Hello! How are you today?")

import code; code.interact(local=dict(globals(), **locals()))