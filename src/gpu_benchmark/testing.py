from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import random
import numpy as np

def setup_qwen_model():
    # Set all random seeds for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    torch.cuda.manual_seed_all(42)
    # Enable deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    model_name = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    return model, tokenizer

def generate_response(model, tokenizer, prompt):
    # Prepare the input format
    messages = [{"role": "user", "content": prompt}]
    
    # Apply the chat template with thinking disabled
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    
    # Tokenize input
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    # Generate response with deterministic parameters
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=32768,
        temperature=0.0,  # Use greedy decoding
        top_p=1.0,        # Disable top-p sampling
        top_k=1,          # Disable top-k sampling
        do_sample=False,  # Disable sampling
        num_beams=1       # Use greedy search
    )
    
    # Decode and return the response
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    response = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
    
    return response

# Example usage
if __name__ == "__main__":
    # Setup the model
    model, tokenizer = setup_qwen_model()
    
    # Example prompt
    prompt = "Write a short story with 500 words"
    
    # Generate response
    response = generate_response(model, tokenizer, prompt)
    
    print("Response:", response)