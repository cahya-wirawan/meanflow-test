import torch
import torch.optim as optim
import torch.nn.functional as F
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
DATASET_SPLIT = "train[:20%]"
VAL_RATIO = 0.1
MODEL_PATH = "src/meanflow_language_model.pth"
NUM_PROC = 4
WANDB_PROJECT = "meanflow"
MIN_TEXT_CHARS = 32


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
        "--min-text-chars",
        type=int,
        default=MIN_TEXT_CHARS,
        help=(
            "Minimum non-whitespace character length for keeping a raw text line. "
            "Higher values reduce noisy short fragments."
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
        "--prediction-target",
        type=str,
        choices=["x1", "v"],
        default="x1",
        help=(
            "Model prediction target for flow training. "
            "x1 predicts final clean embeddings directly, while v predicts flow velocity."
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
        "--t-zero-prob",
        type=float,
        default=0.1,
        help=(
            "Probability of forcing t=0 during training loss computation. "
            "This directly trains the exact inference condition."
        ),
    )
    parser.add_argument(
        "--eval-at-t0",
        action="store_true",
        help=(
            "Evaluate validation loss strictly at t=0. "
            "This makes validation more aligned with one-step inference quality."
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
        "--velocity-loss-weight",
        type=float,
        default=0.25,
        help=(
            "Weight applied to velocity MSE when --prediction-target v. "
            "Total flow loss becomes x1_mse + velocity_loss_weight * velocity_mse."
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
        "--sample-interval",
        type=int,
        default=10,
        help=(
            "Print generated text samples every N epochs using multi-step inference. "
            "Set to 0 to disable."
        ),
    )
    parser.add_argument(
        "--sample-steps",
        type=int,
        default=20,
        help=(
            "Number of integration steps for epoch-level text sampling. "
            "More steps generally produce more coherent text."
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
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=1.0,
        help=(
            "Max gradient norm for clipping before optimizer step. "
            "Set to 0 to disable clipping."
        ),
    )
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        choices=["none", "plateau"],
        default="plateau",
        help=(
            "Learning-rate scheduler strategy. "
            "plateau uses validation loss to reduce LR when progress stalls."
        ),
    )
    parser.add_argument(
        "--lr-patience",
        type=int,
        default=3,
        help=(
            "Number of epochs with no validation improvement before LR reduction "
            "when using the plateau scheduler."
        ),
    )
    parser.add_argument(
        "--lr-factor",
        type=float,
        default=0.5,
        help=(
            "Multiplicative LR decay factor for plateau scheduler. "
            "New LR = old LR * factor."
        ),
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=10,
        help=(
            "Stop training after this many epochs without meaningful validation improvement. "
            "Set to 0 to disable early stopping."
        ),
    )
    parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=1e-4,
        help=(
            "Minimum validation-loss decrease required to count as an improvement for early stopping."
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
    if args.prediction_target not in {"x1", "v"}:
        raise ValueError("--prediction-target must be one of: x1, v")
    if args.wandb_log_interval < 0:
        raise ValueError("--wandb-log-interval must be >= 0")
    if args.min_text_chars < 0:
        raise ValueError("--min-text-chars must be >= 0")
    if args.t_sample_power <= 0:
        raise ValueError("--t-sample-power must be > 0")
    if not (0.0 <= args.t_zero_prob <= 1.0):
        raise ValueError("--t-zero-prob must be between 0 and 1")
    if args.ce_weight_start < 0 or args.ce_weight_end < 0:
        raise ValueError("--ce-weight-start and --ce-weight-end must be >= 0")
    if args.velocity_loss_weight < 0:
        raise ValueError("--velocity-loss-weight must be >= 0")
    if args.diagnostic_samples <= 0:
        raise ValueError("--diagnostic-samples must be > 0")
    if args.diagnostic_temperature <= 0:
        raise ValueError("--diagnostic-temperature must be > 0")
    if args.diagnostic_top_k < 0:
        raise ValueError("--diagnostic-top-k must be >= 0")
    if args.grad_clip_norm < 0:
        raise ValueError("--grad-clip-norm must be >= 0")
    if args.lr_patience < 0:
        raise ValueError("--lr-patience must be >= 0")
    if not (0.0 < args.lr_factor < 1.0):
        raise ValueError("--lr-factor must be between 0 and 1 (exclusive)")
    if args.early_stop_patience < 0:
        raise ValueError("--early-stop-patience must be >= 0")
    if args.early_stop_min_delta < 0:
        raise ValueError("--early-stop-min-delta must be >= 0")


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


def generate_samples(model, tokenizer, num_sequences, seq_len, device, integration_steps, temperature, top_k):
    """Generate text via Heun integration and decode to strings."""
    model.eval()
    with torch.no_grad():
        x_t = torch.randn(num_sequences, seq_len, model.d_model, device=device)
        dt = 1.0 / integration_steps
        t_val = 0.0
        for _ in range(integration_steps):
            t = torch.full((num_sequences, 1), t_val, device=device)
            v = model.predict_velocity(x_t, t)
            t_next_val = t_val + dt
            if t_next_val >= 1.0 - 1e-6:
                # Last step: Euler only — querying velocity at t=1 causes 1/(1-t) singularity.
                x_t = x_t + dt * v
            else:
                x_pred = x_t + dt * v
                t_next = torch.full((num_sequences, 1), t_next_val, device=device)
                v_next = model.predict_velocity(x_pred, t_next)
                x_t = x_t + 0.5 * dt * (v + v_next)
            t_val += dt

        logits = model.lm_logits(x_t) / max(temperature, 1e-6)
        if top_k > 0:
            k = min(top_k, logits.size(-1))
            topk_logits, topk_indices = torch.topk(logits, k=k, dim=-1)
            probs = torch.softmax(topk_logits, dim=-1)
            sampled = torch.multinomial(probs.reshape(-1, k), num_samples=1)
            tokens = topk_indices.gather(-1, sampled.view(num_sequences, seq_len, 1)).squeeze(-1)
        else:
            probs = torch.softmax(logits, dim=-1)
            tokens = torch.multinomial(probs.reshape(-1, probs.size(-1)), num_samples=1).view(num_sequences, seq_len)

    return [tokenizer.decode(tokens[i].tolist(), skip_special_tokens=True) for i in range(num_sequences)]


def compute_loss_components(
    model,
    input_ids,
    pad_token_id=None,
    ce_weight=0.5,
    t_sample_power=1.0,
    t_zero_prob=0.0,
    eval_at_t0=False,
    velocity_loss_weight=0.25,
):
    batch_size, _ = input_ids.shape

    x_1 = model.embedding(input_ids)
    x_0 = torch.randn_like(x_1)

    if eval_at_t0:
        t = torch.zeros(batch_size, 1, device=x_1.device)
    else:
        t = torch.rand(batch_size, 1, device=x_1.device).pow(t_sample_power)
        if t_zero_prob > 0:
            zero_mask = (torch.rand(batch_size, 1, device=x_1.device) < t_zero_prob)
            t = torch.where(zero_mask, torch.zeros_like(t), t)
    t_expanded = t.unsqueeze(-1)
    x_t = t_expanded * x_1 + (1 - t_expanded) * x_0

    if pad_token_id is None:
        mask = torch.ones_like(input_ids, dtype=torch.float)
    else:
        mask = (input_ids != pad_token_id).float()
    mask_expanded = mask.unsqueeze(-1)

    if getattr(model, "prediction_target", "x1") == "v":
        # For linear bridges x_t = t*x1 + (1-t)*x0, the true velocity is constant: v = x1 - x0.
        target_v = x_1 - x_0
        pred_v = model.predict_velocity(x_t, t)
        pred_x1 = x_t + (1 - t_expanded) * pred_v
        x1_diff = (pred_x1 - x_1) * mask_expanded
        mse_loss = (x1_diff**2).sum() / (mask.sum() * model.d_model + 1e-6)
        vel_diff = (pred_v - target_v) * mask_expanded
        velocity_mse = (vel_diff**2).sum() / (mask.sum() * model.d_model + 1e-6)
        flow_loss = mse_loss + velocity_loss_weight * velocity_mse
    else:
        pred_x1 = model.forward_net(x_t, t)
        diff = (pred_x1 - x_1) * mask_expanded
        mse_loss = (diff**2).sum() / (mask.sum() * model.d_model + 1e-6)
        velocity_mse = torch.tensor(0.0, device=x_1.device)
        flow_loss = mse_loss

    logits = model.lm_logits(pred_x1)
    if pad_token_id is None:
        ce_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            input_ids.view(-1),
        )
    else:
        ce_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            input_ids.view(-1),
            ignore_index=pad_token_id,
        )

    total_loss = flow_loss + ce_weight * ce_loss
    return total_loss, mse_loss, ce_loss, velocity_mse

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
    raw_dataset = raw_dataset.filter(lambda x: len(x["text"].strip()) >= args.min_text_chars)
    split_dataset = raw_dataset.train_test_split(test_size=args.val_ratio, seed=args.seed)

    def tokenize_function(examples):
        # Add EOS separators between samples so grouped text keeps document boundaries.
        tokenized = tokenizer(examples["text"], add_special_tokens=False)
        eos_id = tokenizer.eos_token_id
        tokenized["input_ids"] = [ids + [eos_id] for ids in tokenized["input_ids"]]
        return tokenized

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
        prediction_target=args.prediction_target,
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = None
    if args.lr_scheduler == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.lr_factor,
            patience=args.lr_patience,
        )
    print("Num parameters:", sum(p.numel() for p in model.parameters()))
    print("Num trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

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
                "min_text_chars": args.min_text_chars,
                "val_ratio": args.val_ratio,
                "model_path": args.model_path,
                "num_proc": args.num_proc,
                "d_model": args.d_model,
                "num_heads": args.num_heads,
                "num_layers": args.num_layers,
                "prediction_target": args.prediction_target,
                "wandb_log_interval": args.wandb_log_interval,
                "t_sample_power": args.t_sample_power,
                "t_zero_prob": args.t_zero_prob,
                "eval_at_t0": args.eval_at_t0,
                "ce_weight_start": args.ce_weight_start,
                "ce_weight_end": args.ce_weight_end,
                "velocity_loss_weight": args.velocity_loss_weight,
                "diagnostic_samples": args.diagnostic_samples,
                "diagnostic_temperature": args.diagnostic_temperature,
                "diagnostic_top_k": args.diagnostic_top_k,
                "grad_clip_norm": args.grad_clip_norm,
                "lr_scheduler": args.lr_scheduler,
                "lr_patience": args.lr_patience,
                "lr_factor": args.lr_factor,
                "early_stop_patience": args.early_stop_patience,
                "early_stop_min_delta": args.early_stop_min_delta,
                "device": str(device),
                "train_blocks": len(train_dataset),
                "val_blocks": len(val_dataset),
            },
        )
        print(f"W&B enabled: project={args.wandb_project} mode={args.wandb_mode}")

    print("\nStarting Training...")
    best_val_loss = float("inf")
    global_step = 0
    epochs_without_improvement = 0

    try:
        for epoch in range(args.epochs):
            model.train()
            total_train_loss = 0.0
            total_train_mse = 0.0
            total_train_ce = 0.0
            total_train_velocity_mse = 0.0
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
                loss, mse_loss, ce_loss, velocity_mse = compute_loss_components(
                    model,
                    input_ids,
                    pad_token_id=None,
                    ce_weight=ce_weight,
                    t_sample_power=args.t_sample_power,
                    t_zero_prob=args.t_zero_prob,
                    eval_at_t0=False,
                    velocity_loss_weight=args.velocity_loss_weight,
                )
                loss.backward()
                if args.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
                optimizer.step()
                global_step += 1
                total_train_loss += loss.item()
                total_train_mse += mse_loss.item()
                total_train_ce += ce_loss.item()
                total_train_velocity_mse += velocity_mse.item()

                if (
                    wandb_run is not None
                    and args.wandb_log_interval > 0
                    and global_step % args.wandb_log_interval == 0
                ):
                    wandb_run.log(
                        {
                            "batch_loss": loss.item(),
                            "batch_mse_loss": mse_loss.item(),
                            "batch_ce_loss": ce_loss.item(),
                            "batch_velocity_mse": velocity_mse.item(),
                            "epoch": epoch + 1,
                            "global_step": global_step,
                        },
                        step=global_step,
                    )

            avg_train_loss = total_train_loss / len(train_dataloader)
            avg_train_mse = total_train_mse / len(train_dataloader)
            avg_train_ce = total_train_ce / len(train_dataloader)
            avg_train_velocity_mse = total_train_velocity_mse / len(train_dataloader)

            model.eval()
            total_val_loss = 0.0
            total_val_mse = 0.0
            total_val_ce = 0.0
            total_val_velocity_mse = 0.0
            with torch.no_grad():
                for batch in val_dataloader:
                    input_ids = batch["input_ids"].to(device)
                    val_loss, val_mse, val_ce, val_velocity_mse = compute_loss_components(
                        model,
                        input_ids,
                        pad_token_id=None,
                        ce_weight=ce_weight,
                        t_sample_power=args.t_sample_power,
                        t_zero_prob=0.0,
                        eval_at_t0=args.eval_at_t0,
                        velocity_loss_weight=args.velocity_loss_weight,
                    )
                    total_val_loss += val_loss.item()
                    total_val_mse += val_mse.item()
                    total_val_ce += val_ce.item()
                    total_val_velocity_mse += val_velocity_mse.item()

            avg_val_loss = total_val_loss / len(val_dataloader)
            avg_val_mse = total_val_mse / len(val_dataloader)
            avg_val_ce = total_val_ce / len(val_dataloader)
            avg_val_velocity_mse = total_val_velocity_mse / len(val_dataloader)
            val_ce_perplexity = math.exp(avg_val_ce)

            if scheduler is not None:
                # Step on MSE: CE at t=0 saturates early (hard task), causing plateau
                # scheduler to kill LR prematurely. MSE keeps a more stable signal.
                scheduler.step(avg_val_mse)

            current_lr = optimizer.param_groups[0]["lr"]

            improved = avg_val_mse < (best_val_loss - args.early_stop_min_delta)
            if improved:
                best_val_loss = avg_val_mse
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "model_config": {
                            "vocab_size": vocab_size,
                            "d_model": args.d_model,
                            "num_heads": args.num_heads,
                            "num_layers": args.num_layers,
                            "max_seq_len": args.seq_len,
                            "prediction_target": args.prediction_target,
                        },
                    },
                    args.model_path,
                )
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

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
                        "train_mse_loss": avg_train_mse,
                        "train_ce_loss": avg_train_ce,
                        "train_velocity_mse": avg_train_velocity_mse,
                        "val_loss": avg_val_loss,
                        "val_mse_loss": avg_val_mse,
                        "val_ce_loss": avg_val_ce,
                        "val_velocity_mse": avg_val_velocity_mse,
                        "val_ce_perplexity": val_ce_perplexity,
                        "best_val_loss": best_val_loss,
                        "checkpoint_saved": int(improved),
                        "ce_weight": ce_weight,
                        "learning_rate": current_lr,
                        "epochs_without_improvement": epochs_without_improvement,
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
                    f"Train Loss: {avg_train_loss:.4f} "
                    f"(MSE {avg_train_mse:.4f}, CE {avg_train_ce:.4f}, VelMSE {avg_train_velocity_mse:.4f}) | "
                    f"Val Loss: {avg_val_loss:.4f} "
                    f"(MSE {avg_val_mse:.4f}, CE {avg_val_ce:.4f}, VelMSE {avg_val_velocity_mse:.4f}, PPL {val_ce_perplexity:.2f}) | "
                    f"LR: {current_lr:.2e}"
                )

            if args.sample_interval > 0 and (epoch + 1) % args.sample_interval == 0:
                samples = generate_samples(
                    model, tokenizer,
                    num_sequences=args.diagnostic_samples,
                    seq_len=min(args.seq_len, 128),  # cap at 128 tokens for readable output
                    device=device,
                    integration_steps=args.sample_steps,
                    temperature=args.diagnostic_temperature,
                    top_k=args.diagnostic_top_k,
                )
                print(f"\n--- Generated samples (epoch {epoch + 1}, {args.sample_steps}-step Heun) ---")
                for i, text in enumerate(samples):
                    print(f"[{i+1}] {text}")
                print("---\n")

            if (
                args.early_stop_patience > 0
                and epochs_without_improvement >= args.early_stop_patience
            ):
                print(
                    "Early stopping triggered: "
                    f"no val improvement > {args.early_stop_min_delta} for "
                    f"{args.early_stop_patience} epoch(s)."
                )
                break
    finally:
        if wandb_run is not None:
            wandb_run.finish()

    print("\nTraining Complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best checkpoint saved to: {args.model_path}")


if __name__ == "__main__":
    main()
