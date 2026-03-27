import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
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

# --- 1. Setup Parameters ---
SEQ_LEN = 128         # Length of our text sequences
BATCH_SIZE = 16      # Number of sequences processed at once
EPOCHS = 100           # Lowered epochs because real datasets are much larger
LEARNING_RATE = 1e-4 

# Device setup
if torch.cuda.is_available(): device = torch.device("cuda")
elif torch.backends.mps.is_available(): device = torch.device("mps")
else: device = torch.device("cpu")

# --- 2. Load the Tokenizer ---
print("Loading Tokenizer...")
# We use the standard GPT-2 tokenizer. 
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# GPT-2 doesn't have a default "padding" token, so we assign one.
# Padding is critical because parallel models need every sequence to be exactly SEQ_LEN.
tokenizer.pad_token = tokenizer.eos_token 

# Dynamically set the VOCAB_SIZE based on the tokenizer (GPT-2 is 50,257)
VOCAB_SIZE = len(tokenizer) 
print(f"Real Vocabulary Size: {VOCAB_SIZE}")

# --- 3. Load and Process the Dataset ---
print("Downloading/Loading Dataset...")
# We load just 1% of the WikiText-2 dataset for quick local testing
# raw_dataset = load_dataset("Apex-Datasets/German-Wikipedia-Cleaned-Sample", "wikitext-2-raw-v1", split="train[:1%]")
raw_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:5%]")

# Remove empty lines (Wikipedia has a lot of blank formatting lines)
raw_dataset = raw_dataset.filter(lambda x: len(x["text"].strip()) > 32)

def tokenize_function(examples):
    # This converts the text into integer IDs, truncates long sentences, 
    # and pads short sentences with 0s so every sequence is exactly 512 tokens long.
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=SEQ_LEN,
    )

print("Tokenizing data...")
tokenized_dataset = raw_dataset.map(tokenize_function, batched=True)

# Convert the dataset to PyTorch tensors and keep only the "input_ids" column
tokenized_dataset.set_format(type="torch", columns=["input_ids"])

# Wrap it in our PyTorch DataLoader
dataloader = DataLoader(tokenized_dataset, batch_size=BATCH_SIZE, shuffle=True)

print(f"Dataset ready! Total batches per epoch: {len(dataloader)}")

# --- 3. Initialize Model and Optimizer ---
# (Assuming the MeanFlowLanguageModel class from the previous step is defined here)
model = MeanFlowLanguageModel(
    vocab_size=VOCAB_SIZE, 
    d_model=128,      # Scaled down for quick local testing
    num_heads=4, 
    num_layers=4, 
    max_seq_len=SEQ_LEN
).to(device)

# AdamW is the standard optimizer for Transformer models
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# --- 4. The Training Loop ---
print("\nStarting Training...")

for epoch in range(EPOCHS):
    model.train() # Set the model to training mode
    total_loss = 0.0
    
    for batch in dataloader:
        # Move the batch to the GPU/CPU
        input_ids = batch["input_ids"].to(device)
        
        # Step A: Clear the old gradients from the previous step
        optimizer.zero_grad()
        
        # Step B: Compute the Mean Flow loss (MSE + Cross Entropy)
        loss = model.compute_loss(input_ids)
        
        # Step C: Backpropagation (calculate the gradients)
        loss.backward()
        
        # Step D: Update the model's weights
        optimizer.step()
        
        total_loss += loss.item()
        
    # Calculate the average loss for this epoch
    avg_loss = total_loss / len(dataloader)
    
    # Print the progress every 5 epochs
    if (epoch + 1) % 2 == 0 or epoch == 0:
        print(f"Epoch [{epoch + 1}/{EPOCHS}] | Average Loss: {avg_loss:.4f}")

# After training, the model should have learned to predict the continuous trajectories that correspond to the discrete token sequences in our dummy dataset.
print("\nTraining Complete! The model has learned the dummy continuous trajectories.")
# Save the trained model for later use (optional)
torch.save(model.state_dict(), "src/meanflow_language_model.pth")

# --- Proceed to Step 4 (Inference) from the previous script ---
