import torch
import argparse

SEQ_LEN = 128
MODEL_PATH = "src/meanflow_language_model.pth"

print(f"Inference time with SEQ_LEN={SEQ_LEN}...")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run MeanFlow inference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=SEQ_LEN,
        help=(
            "Generated token length per sequence. "
            "Must match the model max_seq_len used for the loaded checkpoint architecture."
        ),
    )
    parser.add_argument(
        "--num-sequences",
        type=int,
        default=10,
        help=(
            "Number of sequences to generate in one call. "
            "Larger values increase memory usage linearly with batch size."
        ),
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=MODEL_PATH,
        help=(
            "Path to a trained checkpoint file (.pth). "
            "The model weights are loaded from this location before generation."
        ),
    )
    parser.add_argument(
        "--d-model",
        type=int,
        default=128,
        help=(
            "Transformer hidden size expected by the checkpoint architecture. "
            "Must match the value used during training."
        ),
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=4,
        help=(
            "Attention heads expected by the checkpoint architecture. "
            "Must match training-time configuration."
        ),
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=4,
        help=(
            "Transformer layer count expected by the checkpoint architecture. "
            "Must match training-time configuration."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help=(
            "Execution device. "
            "Use auto to pick CUDA, then MPS, then CPU; explicit values enforce a specific backend."
        ),
    )
    return parser.parse_args()


def pick_device(device_name="auto"):
    if device_name in {"cpu", "cuda", "mps"}:
        if device_name == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but is not available")
        if device_name == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS requested but is not available")
        return torch.device(device_name)

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def generate_text(model, tokenizer, num_sequences=3, seq_len=32, device=torch.device("cpu")):
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

if __name__ == "__main__":
    args = parse_args()
    if args.seq_len <= 0 or args.num_sequences <= 0:
        raise ValueError("--seq-len and --num-sequences must be > 0")

    from transformers import AutoTokenizer
    from meanflow import MeanFlowLanguageModel

    device = pick_device(args.device)
    print(f"Using device: {device}")

    # We use the standard GPT-2 tokenizer.
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Dynamically set the VOCAB_SIZE based on the tokenizer (GPT-2 is 50,257)
    vocab_size = len(tokenizer)
    print(f"Real Vocabulary Size: {vocab_size}")

    model = MeanFlowLanguageModel(
        vocab_size=vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        max_seq_len=args.seq_len,
    ).to(device)

    # Load the trained model weights (if you have them saved)
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    generate_text(
        model,
        tokenizer,
        num_sequences=args.num_sequences,
        seq_len=args.seq_len,
        device=device,
    )