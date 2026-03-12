import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load fine tuned model
model_path = "./model"

tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

prompt = "Once upon a time"

inputs = tokenizer.encode(prompt, return_tensors="pt")

output = model.generate(
    inputs,
    max_length=120,
    num_return_sequences=1,
    temperature=0.8,
    top_k=50,
    top_p=0.95,
    do_sample=True
)

story = tokenizer.decode(output[0], skip_special_tokens=True)

print("\nGenerated Story:\n")
print(story)