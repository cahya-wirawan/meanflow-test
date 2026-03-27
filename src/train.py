import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
import random
from typing import Any
import argparse

SEQ_LEN = 128
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 1e-4
SEED = 42
DATASET_SPLIT = "train[:5%]"
VAL_RATIO = 0.1
MODEL_PATH = "src/meanflow_language_model.pth"
NUM_PROC = 4


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train MeanFlow language model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=SEQ_LEN,
        help=(
            "Token block length per sample. "
            "Longer blocks increase context but also memory and compute cost."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=(
            "Number of token blocks per optimization step. "
            "Reduce this value first if you run out of GPU/MPS memory."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help=(
            "Total passes over the grouped training dataset. "
            "Higher values may improve fit but increase training time and overfitting risk."
        ),
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=LEARNING_RATE,
        help=(
            "AdamW learning rate. "
            "Smaller values are usually more stable; larger values train faster but can diverge."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=(
            "Random seed for dataset split and weight/noise sampling. "
            "Use a fixed seed to improve reproducibility across runs."
        ),
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        default=DATASET_SPLIT,
        help=(
            "Hugging Face split selector passed to load_dataset, for example train[:5%%]. "
            "Use this to quickly scale experiments up or down."
        ),
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=VAL_RATIO,
        help=(
            "Fraction of filtered data reserved for validation before grouping. "
            "Used to track generalization and choose the best checkpoint."
        ),
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=MODEL_PATH,
        help=(
            "Output path for the best model checkpoint (.pth). "
            "The file is overwritten whenever validation loss improves."
        ),
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=NUM_PROC,
        help=(
            "Number of parallel workers for dataset map operations. "
            "Set to 1 on constrained environments or if multiprocessing causes issues."
        ),
    )
    parser.add_argument(
        "--d-model",
        type=int,
        default=128,
        help=(
            "Transformer hidden size (embedding dimension). "
            "Increasing this improves capacity but significantly raises memory usage."
        ),
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=4,
        help=(
            "Number of attention heads per transformer layer. "
            "Must be compatible with d_model for stable attention projections."
        ),
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=4,
        help=(
            "Number of transformer encoder layers in the backbone. "
            "More layers increase representational depth and runtime."
        ),
    )
    return parser.parse_args()


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def validate_args(args):
    if args.seq_len <= 0:
        raise ValueError("--seq-len must be > 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.epochs <= 0:
        raise ValueError("--epochs must be > 0")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be > 0")
    if not (0.0 < args.val_ratio < 1.0):
        raise ValueError("--val-ratio must be between 0 and 1 (exclusive)")
    if args.num_proc <= 0:
        raise ValueError("--num-proc must be > 0")
    if args.d_model <= 0 or args.num_heads <= 0 or args.num_layers <= 0:
        raise ValueError("--d-model, --num-heads, and --num-layers must be > 0")

def main():
    args = parse_args()
    validate_args(args)

    from transformers import AutoTokenizer
    from meanflow import MeanFlowLanguageModel

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = pick_device()
    print(f"Training on device: {device}")

    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = len(tokenizer)
    print(f"Real Vocabulary Size: {vocab_size}")

    print("Downloading/Loading Dataset...")
    raw_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=args.dataset_split)
    raw_dataset = raw_dataset.filter(lambda x: len(x["text"].strip()) > 0)
    split_dataset = raw_dataset.train_test_split(test_size=args.val_ratio, seed=args.seed)

    def tokenize_function(examples):
        return tokenizer(examples["text"])

    print("Tokenizing data...")
    tokenized_datasets = split_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        num_proc=args.num_proc,
    )

    def group_texts(examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // args.seq_len) * args.seq_len
        result = {
            k: [t[i : i + args.seq_len] for i in range(0, total_length, args.seq_len)]
            for k, t in concatenated_examples.items()
        }
        return result

    print("Grouping text into blocks...")
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=args.num_proc,
    )

    lm_datasets.set_format(type="torch", columns=["input_ids"])
    train_dataset: Any = lm_datasets["train"]
    val_dataset: Any = lm_datasets["test"]
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError(
            "Train/val dataset is empty after grouping. Try a larger --dataset-split or lower --seq-len."
        )

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    print(
        f"Dataset ready! Train blocks: {len(train_dataset)} | "
        f"Val blocks: {len(val_dataset)}"
    )

    model = MeanFlowLanguageModel(
        vocab_size=vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        max_seq_len=args.seq_len,
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    print("\nStarting Training...")
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0.0

        for batch in train_dataloader:
            input_ids = batch["input_ids"].to(device)
            optimizer.zero_grad()
            loss = model.compute_loss(input_ids, pad_token_id=None)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_dataloader)

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch["input_ids"].to(device)
                val_loss = model.compute_loss(input_ids, pad_token_id=None)
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(val_dataloader)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), args.model_path)

        if (epoch + 1) % 2 == 0 or epoch == 0:
            print(
                f"Epoch [{epoch + 1}/{args.epochs}] | "
                f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}"
            )

    print("\nTraining Complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best checkpoint saved to: {args.model_path}")


if __name__ == "__main__":
    main()
