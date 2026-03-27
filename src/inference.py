import torch
from transformers import AutoTokenizer
from meanflow import MeanFlowLanguageModel

SEQ_LEN = 128

print(f"Inference time with SEQ_LEN={SEQ_LEN}...")

def generate_text(model, tokenizer, num_sequences=3, seq_len=32, device="cpu"):
    print("\n--- Generating New Text ---")
    model.eval() # Set the model to evaluation mode (turns off dropout, etc.)
    
    with torch.no_grad(): # We don't need gradients for inference
        # 1. Start with a block of pure Gaussian noise (x_0)
        # Shape: [Batch Size, Sequence Length, Embedding Dimension]
        x_0 = torch.randn(num_sequences, seq_len, model.d_model, device=device)
        
        # 2. Tell the model we are at time t = 0 (the starting line)
        t = torch.zeros(num_sequences, 1, device=device)
        
        # 3. The 1-Step Jump! Predict the final continuous state (x_1)
        pred_x1 = model.forward_net(x_0, t)
        
        # 4. The "Rounding" Step: Project continuous vectors to vocabulary logits
        logits = model.lm_head(pred_x1)
        
        # 5. Pick the most likely word for each position (Greedy Decoding)
        predicted_token_ids = torch.argmax(logits, dim=-1)
        
        # 6. Decode the token IDs back into readable English strings
        for i in range(num_sequences):
            # Extract the raw IDs for this specific sequence
            token_ids = predicted_token_ids[i].tolist()
            
            # Use the Hugging Face tokenizer to convert IDs to text
            # skip_special_tokens=True removes the padding tokens we added
            generated_text = tokenizer.decode(token_ids, skip_special_tokens=True)
            
            print(f"Sequence {i+1}: {generated_text}")

# --- How to call it ---
# generate_text(model, tokenizer, num_sequences=3, seq_len=SEQ_LEN, device=device)

# We use the standard GPT-2 tokenizer. 
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# GPT-2 doesn't have a default "padding" token, so we assign one.
# Padding is critical because parallel models need every sequence to be exactly SEQ_LEN.
tokenizer.pad_token = tokenizer.eos_token 

# Dynamically set the VOCAB_SIZE based on the tokenizer (GPT-2 is 50,257)
VOCAB_SIZE = len(tokenizer) 
print(f"Real Vocabulary Size: {VOCAB_SIZE}")
model = MeanFlowLanguageModel(
    vocab_size=VOCAB_SIZE,
    d_model=128,
    num_heads=4,
    num_layers=4,
    max_seq_len=SEQ_LEN
).to(torch.device("cpu"))
# Load the trained model weights (if you have them saved)
model.load_state_dict(torch.load("src/meanflow_language_model.pth", map_location=torch.device("cpu")))

generate_text(model, tokenizer, num_sequences=10, seq_len=SEQ_LEN, device=torch.device("cpu"))