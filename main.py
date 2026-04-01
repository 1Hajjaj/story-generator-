from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

prompt = input("Enter your story idea: ")
inputs = tokenizer(prompt, return_tensors="pt")

output = model.generate(
    inputs["input_ids"],
    max_length=200,
    temperature=0.9,
    top_p=0.95,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id 
)

story = tokenizer.decode(output[0], skip_special_tokens=True)
print('\n' + story)  
