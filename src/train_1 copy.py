import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from meanflow import MeanFlowLanguageModel

# --- 1. Setup Parameters ---
VOCAB_SIZE = 1000    # A small vocabulary for testing
SEQ_LEN = 32         # Length of our text sequences
BATCH_SIZE = 16      # Number of sequences processed at once
NUM_SAMPLES = 320    # Total number of dummy text sequences
EPOCHS = 50          # How many times we loop over the data
LEARNING_RATE = 1e-4 # How fast the optimizer learns

# Check for GPU (Apple Silicon MPS or Nvidia CUDA), otherwise fallback to CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
    
print(f"Training on device: {device}")

# --- 2. Create Dummy Data ---
# Generate random integers between 0 and VOCAB_SIZE - 1 to simulate token IDs
dummy_input_ids = torch.randint(0, VOCAB_SIZE, (NUM_SAMPLES, SEQ_LEN), dtype=torch.long)

# Wrap it in a PyTorch DataLoader for easy batching
dataset = TensorDataset(dummy_input_ids)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

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
        input_ids = batch[0].to(device)
        
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
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch [{epoch + 1}/{EPOCHS}] | Average Loss: {avg_loss:.4f}")

# After training, the model should have learned to predict the continuous trajectories that correspond to the discrete token sequences in our dummy dataset.
print("\nTraining Complete! The model has learned the dummy continuous trajectories.")
# Save the trained model for later use (optional)
torch.save(model.state_dict(), "meanflow_language_model.pth")

# --- Proceed to Step 4 (Inference) from the previous script ---
