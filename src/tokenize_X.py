import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

# --- 1. Setup Parameters ---
SEQ_LEN = 32         # Length of our text sequences
BATCH_SIZE = 16      # Number of sequences processed at once
EPOCHS = 5           # Lowered epochs because real datasets are much larger
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
raw_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1%]")

# Remove empty lines (Wikipedia has a lot of blank formatting lines)
raw_dataset = raw_dataset.filter(lambda x: len(x["text"].strip()) > 0)

def tokenize_function(examples):
    # This converts the text into integer IDs, truncates long sentences, 
    # and pads short sentences with 0s so every sequence is exactly 32 tokens long.
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

# --- Proceed to Step 3 (Initialize Model) from the previous script ---
# IMPORTANT: Pass the new VOCAB_SIZE to your MeanFlowLanguageModel