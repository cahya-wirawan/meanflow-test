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
    parser.add_argument(
        "--sample",
        action="store_true",
        help=(
            "Enable stochastic decoding instead of greedy argmax. "
            "Recommended to reduce repetitive outputs such as 'the the the'."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help=(
            "Sampling temperature when --sample is enabled. "
            "Higher values increase diversity; lower values make outputs more deterministic."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help=(
            "Top-k truncation for sampling when --sample is enabled. "
            "Set 0 to sample from full vocabulary distribution."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Optional random seed for reproducible sampling. "
            "Ignored for deterministic greedy decoding unless stochastic ops are used."
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


def generate_text(
    model,
    tokenizer,
    num_sequences=3,
    seq_len=32,
    device="cpu",
    sample=False,
    temperature=1.0,
    top_k=50,
):
    print("\n--- Generating New Text ---")
    model.eval() # Set the model to evaluation mode (turns off dropout, etc.)
    
    with torch.no_grad(): # We don't need gradients for inference
        predicted_token_ids = model.generate_1_step(
            batch_size=num_sequences,
            seq_len=seq_len,
            device=str(device),
            sample=sample,
            temperature=temperature,
            top_k=top_k,
        )

        # Decode the token IDs back into readable English strings
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
    if args.temperature <= 0:
        raise ValueError("--temperature must be > 0")
    if args.top_k < 0:
        raise ValueError("--top-k must be >= 0")

    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

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
        device=str(device),
        sample=args.sample,
        temperature=args.temperature,
        top_k=args.top_k,
    )