import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model():
    """Load the DeepSeek LLM model and tokenizer."""
    model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    
    print(f"Loading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Ensure tokenizer has pad token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = model.config.eos_token_id
    
    print("Model loaded successfully!")
    return model, tokenizer

def generate_fixed_tokens(model, tokenizer, num_tokens=100):
    """Generate exactly num_tokens tokens."""
    # Simple prompt to start generation
    prompt = "The following is a test:"
    
    # Prepare input
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    
    # Generate with fixed parameters
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=num_tokens,
            min_new_tokens=num_tokens,  # Force exact number of tokens
            do_sample=False,            # Deterministic generation
            num_beams=1,                # No beam search
            temperature=1.0,            # No temperature scaling
            top_k=0,                    # No top-k filtering
            top_p=1.0,                  # No nucleus sampling
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=None           # Prevent early stopping
        )
    
    # Get generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

def main():
    # Load model
    model, tokenizer = load_model()
    
    # Generate fixed number of tokens
    TOKENS_TO_GENERATE = 100
    print(f"\nGenerating exactly {TOKENS_TO_GENERATE} tokens...")
    
    generated_text = generate_fixed_tokens(model, tokenizer, TOKENS_TO_GENERATE)
    print("\nGenerated text:")
    print(generated_text)

if __name__ == "__main__":
    main()