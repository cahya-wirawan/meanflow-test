import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from meanflow import MeanFlowLanguageModel

# --- 1. Setup Parameters ---

# Check for GPU (Apple Silicon MPS or Nvidia CUDA), otherwise fallback to CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
    
print(f"Training on device: {device}")

SEQ_LEN = 128         # Length of our text sequences
BATCH_SIZE = 16      # Number of sequences processed at once
EPOCHS = 100           
LEARNING_RATE = 1e-4 

# --- 2. Load the Tokenizer ---
print("Loading Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token 
VOCAB_SIZE = len(tokenizer) 
print(f"Real Vocabulary Size: {VOCAB_SIZE}")

# --- 3. Load and Process the Dataset ---
print("Downloading/Loading Dataset...")
# We load a small sample for quick testing
raw_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:5%]")

# Filter out empty or very short lines
raw_dataset = raw_dataset.filter(lambda x: len(x["text"].strip()) > 0)

def tokenize_function(examples):
    return tokenizer(examples["text"])

print("Tokenizing data...")
tokenized_datasets = raw_dataset.map(
    tokenize_function, 
    batched=True, 
    remove_columns=["text"],
    num_proc=4
)

def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder
    total_length = (total_length // SEQ_LEN) * SEQ_LEN
    # Split by chunks of SEQ_LEN
    result = {
        k: [t[i : i + SEQ_LEN] for i in range(0, total_length, SEQ_LEN)]
        for k, t in concatenated_examples.items()
    }
    return result

print("Grouping text into blocks...")
lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    num_proc=4,
)

# Convert to PyTorch tensors
lm_datasets.set_format(type="torch", columns=["input_ids"])

# Wrap it in PyTorch DataLoader
dataloader = DataLoader(lm_datasets, batch_size=BATCH_SIZE, shuffle=True)

print(f"Dataset ready! Total blocks: {len(lm_datasets)}")

# --- 3. Initialize Model and Optimizer ---
model = MeanFlowLanguageModel(
    vocab_size=VOCAB_SIZE, 
    d_model=128,
    num_heads=4, 
    num_layers=4, 
    max_seq_len=SEQ_LEN
).to(device)

optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# --- 4. The Training Loop ---
print("\nStarting Training...")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        
        optimizer.zero_grad()
        
        # Pass the actual pad token ID to compute_loss
        loss = model.compute_loss(input_ids, pad_token_id=tokenizer.pad_token_id)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    avg_loss = total_loss / len(dataloader)
    
    if (epoch + 1) % 2 == 0 or epoch == 0:
        print(f"Epoch [{epoch + 1}/{EPOCHS}] | Average Loss: {avg_loss:.4f}")

print("\nTraining Complete!")
torch.save(model.state_dict(), "src/meanflow_language_model.pth")
