import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
import random
from typing import Any
import argparse
import math

SEQ_LEN = 128
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 1e-4
SEED = 42
DATASET_SPLIT = "train[:5%]"
VAL_RATIO = 0.1
MODEL_PATH = "src/meanflow_language_model.pth"
NUM_PROC = 4
WANDB_PROJECT = "meanflow"


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
    parser.add_argument(
        "--t-sample-power",
        type=float,
        default=2.0,
        help=(
            "Power for sampling flow time t as U^p. "
            "Values > 1 bias training toward t near 0 to better match inference."
        ),
    )
    parser.add_argument(
        "--ce-weight-start",
        type=float,
        default=0.5,
        help=(
            "Initial CE coefficient in the combined loss. "
            "Higher values emphasize discrete token prediction early in training."
        ),
    )
    parser.add_argument(
        "--ce-weight-end",
        type=float,
        default=1.0,
        help=(
            "Final CE coefficient reached by linear schedule at last epoch. "
            "Useful to reduce collapse to frequent tokens."
        ),
    )
    parser.add_argument(
        "--diagnostic-samples",
        type=int,
        default=4,
        help=(
            "Number of sequences generated each epoch for collapse diagnostics. "
            "Used for entropy and distinct-n metrics logging."
        ),
    )
    parser.add_argument(
        "--diagnostic-temperature",
        type=float,
        default=1.0,
        help=(
            "Sampling temperature for diagnostic generation. "
            "Higher values increase diversity; lower values increase determinism."
        ),
    )
    parser.add_argument(
        "--diagnostic-top-k",
        type=int,
        default=50,
        help=(
            "Top-k truncation for diagnostic generation. "
            "Set to 0 to sample from full vocabulary distribution."
        ),
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help=(
            "Enable Weights & Biases experiment tracking. "
            "When set, epoch metrics and run config are logged to W&B."
        ),
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=WANDB_PROJECT,
        help=(
            "W&B project name to log runs into. "
            "Create a new project in your W&B workspace if it does not exist."
        ),
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help=(
            "W&B entity (username or team). "
            "Leave unset to use your default account/entity."
        ),
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help=(
            "Optional custom W&B run name. "
            "Useful for grouping experiment families by naming convention."
        ),
    )
    parser.add_argument(
        "--wandb-mode",
        type=str,
        choices=["online", "offline", "disabled"],
        default="online",
        help=(
            "W&B logging mode. "
            "Use offline to store logs locally for later sync."
        ),
    )
    parser.add_argument(
        "--wandb-log-interval",
        type=int,
        default=0,
        help=(
            "Batch-level W&B logging interval in optimizer steps. "
            "Set to 0 to disable batch logging; use values like 10 or 50 to reduce log volume."
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
    if args.wandb_log_interval < 0:
        raise ValueError("--wandb-log-interval must be >= 0")
    if args.t_sample_power <= 0:
        raise ValueError("--t-sample-power must be > 0")
    if args.ce_weight_start < 0 or args.ce_weight_end < 0:
        raise ValueError("--ce-weight-start and --ce-weight-end must be >= 0")
    if args.diagnostic_samples <= 0:
        raise ValueError("--diagnostic-samples must be > 0")
    if args.diagnostic_temperature <= 0:
        raise ValueError("--diagnostic-temperature must be > 0")
    if args.diagnostic_top_k < 0:
        raise ValueError("--diagnostic-top-k must be >= 0")


def compute_diversity_metrics(tokens, vocab_size):
    flat = tokens.reshape(-1)
    total_tokens = max(1, flat.numel())

    unigram_count = torch.unique(flat).numel()
    distinct_1 = unigram_count / total_tokens

    if tokens.size(1) > 1:
        bigrams = torch.stack([tokens[:, :-1], tokens[:, 1:]], dim=-1).reshape(-1, 2)
        distinct_2 = torch.unique(bigrams, dim=0).size(0) / max(1, bigrams.size(0))
    else:
        distinct_2 = 0.0

    counts = torch.bincount(flat, minlength=vocab_size).float()
    probs = counts / counts.sum().clamp_min(1.0)
    nz = probs > 0
    entropy = -(probs[nz] * probs[nz].log()).sum().item()
    norm_entropy = entropy / max(1e-8, math.log(vocab_size))

    return {
        "distinct_1": float(distinct_1),
        "distinct_2": float(distinct_2),
        "token_entropy": float(entropy),
        "token_entropy_norm": float(norm_entropy),
    }

def main():
    args = parse_args()
    validate_args(args)

    from transformers import AutoTokenizer
    from meanflow import MeanFlowLanguageModel

    wandb: Any = None
    wandb_run = None
    if args.wandb:
        try:
            import wandb  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "W&B tracking requested but 'wandb' is not installed. "
                "Install it with: pip install wandb"
            ) from exc

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

    if args.wandb:
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            mode=args.wandb_mode,
            config={
                "seq_len": args.seq_len,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "learning_rate": args.learning_rate,
                "seed": args.seed,
                "dataset_split": args.dataset_split,
                "val_ratio": args.val_ratio,
                "model_path": args.model_path,
                "num_proc": args.num_proc,
                "d_model": args.d_model,
                "num_heads": args.num_heads,
                "num_layers": args.num_layers,
                "wandb_log_interval": args.wandb_log_interval,
                "t_sample_power": args.t_sample_power,
                "ce_weight_start": args.ce_weight_start,
                "ce_weight_end": args.ce_weight_end,
                "diagnostic_samples": args.diagnostic_samples,
                "diagnostic_temperature": args.diagnostic_temperature,
                "diagnostic_top_k": args.diagnostic_top_k,
                "device": str(device),
                "train_blocks": len(train_dataset),
                "val_blocks": len(val_dataset),
            },
        )
        print(f"W&B enabled: project={args.wandb_project} mode={args.wandb_mode}")

    print("\nStarting Training...")
    best_val_loss = float("inf")
    global_step = 0

    try:
        for epoch in range(args.epochs):
            model.train()
            total_train_loss = 0.0
            if args.epochs == 1:
                epoch_progress = 1.0
            else:
                epoch_progress = epoch / (args.epochs - 1)
            ce_weight = args.ce_weight_start + (
                args.ce_weight_end - args.ce_weight_start
            ) * epoch_progress

            for batch in train_dataloader:
                input_ids = batch["input_ids"].to(device)
                optimizer.zero_grad()
                loss = model.compute_loss(
                    input_ids,
                    pad_token_id=None,
                    ce_weight=ce_weight,
                    t_sample_power=args.t_sample_power,
                )
                loss.backward()
                optimizer.step()
                global_step += 1
                total_train_loss += loss.item()

                if (
                    wandb_run is not None
                    and args.wandb_log_interval > 0
                    and global_step % args.wandb_log_interval == 0
                ):
                    wandb_run.log(
                        {
                            "batch_loss": loss.item(),
                            "epoch": epoch + 1,
                            "global_step": global_step,
                        },
                        step=global_step,
                    )

            avg_train_loss = total_train_loss / len(train_dataloader)

            model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for batch in val_dataloader:
                    input_ids = batch["input_ids"].to(device)
                    val_loss = model.compute_loss(
                        input_ids,
                        pad_token_id=None,
                        ce_weight=ce_weight,
                        t_sample_power=args.t_sample_power,
                    )
                    total_val_loss += val_loss.item()

            avg_val_loss = total_val_loss / len(val_dataloader)

            improved = avg_val_loss < best_val_loss
            if improved:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), args.model_path)

            if wandb_run is not None:
                diagnostic_tokens = model.generate_1_step(
                    batch_size=args.diagnostic_samples,
                    seq_len=args.seq_len,
                    device=str(device),
                    sample=True,
                    temperature=args.diagnostic_temperature,
                    top_k=args.diagnostic_top_k,
                )
                diversity = compute_diversity_metrics(diagnostic_tokens, vocab_size=vocab_size)
                wandb_run.log(
                    {
                        "epoch": epoch + 1,
                        "train_loss": avg_train_loss,
                        "val_loss": avg_val_loss,
                        "best_val_loss": best_val_loss,
                        "checkpoint_saved": int(improved),
                        "ce_weight": ce_weight,
                        "distinct_1": diversity["distinct_1"],
                        "distinct_2": diversity["distinct_2"],
                        "token_entropy": diversity["token_entropy"],
                        "token_entropy_norm": diversity["token_entropy_norm"],
                        "global_step": global_step,
                    },
                    step=global_step,
                )

            if (epoch + 1) % 2 == 0 or epoch == 0:
                print(
                    f"Epoch [{epoch + 1}/{args.epochs}] | "
                    f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}"
                )
    finally:
        if wandb_run is not None:
            wandb_run.finish()

    print("\nTraining Complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best checkpoint saved to: {args.model_path}")


if __name__ == "__main__":
    main()
