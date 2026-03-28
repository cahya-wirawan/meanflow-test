# MeanFlow Language Model

MeanFlow is an experimental language model architecture that treats text generation as a continuous flow problem. Unlike traditional autoregressive models (like GPT) that generate text token-by-token, MeanFlow is designed to generate entire blocks of text simultaneously in a **single "1-step jump"** from pure Gaussian noise.

## Core Concepts

### Continuous Flow Matching
The model learns a linear trajectory (a "mean flow") between a starting distribution of pure Gaussian noise ($x_0$) and a target distribution of embedded text vectors ($x_1$). 

At training time, the model is presented with a noisy intermediate state $x_t$ at a random time $t \in [0, 1]$ and tasked with predicting the final clean state $x_1$.

### 1-Step Generation
Because the model learns the direct path to the data distribution, inference can be performed by:
1. Sampling random noise $x_0$.
2. Asking the model to predict the final state $x_1$ given $t=0$.
3. "Rounding" the continuous predicted vectors back into discrete text tokens using the language model head.

## Project Structure

- `src/meanflow.py`: The core model architecture, including the time-conditioned transformer and the flow-matching loss logic.
- `src/train.py`: The training script that loads the WikiText dataset, tokenizes it, and trains the model.
- `src/inference.py`: A script for loading a trained model and generating new text sequences.
- `src/tokenize_X.py`: Utility script demonstrating the tokenization and data loading pipeline.
- `meanflow_language_model.pth`: The saved weights of the trained model.

## Model Architecture

The `MeanFlowLanguageModel` consists of:
1. **Continuous Bridge**: Maps discrete tokens to a high-dimensional continuous space using learned embeddings and positional encodings.
2. **Time-Conditioned Transformer**: A standard Transformer Encoder backbone that is conditioned on the current "flow time" $t$ via a SiLU-based time embedding.
3. **Prediction Head**: Predicts the target continuous vector $x_1$.
4. **Rounding Head**: Maps the continuous predictions back to vocabulary logits (weight-tied with the input embeddings).

## Getting Started

### Installation

Ensure you have the required dependencies installed:

```bash
pip install torch transformers datasets
```

Optional experiment tracking with Weights & Biases:

```bash
pip install wandb
```

### Training

To train the model on a sample of the WikiText-2 dataset:

```bash
python src/train.py
```

You can override key run settings from the command line, for example:

```bash
python src/train.py --epochs 20 --batch-size 8 --dataset-split "train[:10%]" --model-path src/meanflow_language_model.pth
```

Recommended stable training profile:

```bash
python src/train.py \
	--dataset-split "train[:20%]" \
	--min-text-chars 32 \
	--learning-rate 5e-5 \
	--lr-scheduler plateau \
	--lr-patience 2 \
	--lr-factor 0.5 \
	--grad-clip-norm 1.0 \
	--early-stop-patience 8 \
	--early-stop-min-delta 1e-4 \
	--ce-weight-start 0.5 \
	--ce-weight-end 0.5 \
	--t-sample-power 1.2 \
	--t-zero-prob 0.2 \
	--eval-at-t0 \
	--wandb \
	--wandb-log-interval 50
```

Enable W&B logging during training:

```bash
python src/train.py --wandb --wandb-project meanflow --wandb-run-name "baseline-v1"
```

For local/offline logging:

```bash
python src/train.py --wandb --wandb-mode offline
```

The training script now also supports:
- Raw text filtering with `--min-text-chars`
- Gradient clipping with `--grad-clip-norm`
- Validation-driven LR scheduling with `--lr-scheduler plateau`
- Early stopping with `--early-stop-patience` and `--early-stop-min-delta`
- Time-bias control with `--t-sample-power` (biases training toward $t \approx 0$)
- Explicit $t=0$ alignment with `--t-zero-prob` and `--eval-at-t0`
- CE weighting schedule with `--ce-weight-start` and `--ce-weight-end`
- W&B batch metrics via `--wandb-log-interval`
- Separate MSE/CE logging for train and validation losses
- Diversity diagnostics in W&B (`distinct_1`, `distinct_2`, entropy)

The training pipeline uses **Text Grouping**, where all tokenized examples are concatenated and split into equal blocks of `SEQ_LEN`. This removes the bias caused by padding short sentences and ensures the model only learns from real data.

Training now uses a deterministic seed (`SEED=42`) and a train/validation split. The model checkpoint at `src/meanflow_language_model.pth` is updated when validation loss improves.

### Inference

To generate text using a trained model:

```bash
python src/inference.py
```

You can also configure inference without editing code:

```bash
python src/inference.py --model-path src/meanflow_language_model.pth --seq-len 128 --num-sequences 5 --device auto
```

To reduce repetitive outputs, use sampling instead of greedy decoding:

```bash
python src/inference.py \
	--model-path src/meanflow_language_model.pth \
	--sample \
	--temperature 1.0 \
	--top-k 50 \
	--seed 42
```

For iterative latent refinement, use few-step integration:

```bash
python src/inference.py \
	--model-path src/meanflow_language_model.pth \
	--integration-steps 8 \
	--integration-method heun \
	--sample \
	--temperature 1.0 \
	--top-k 50 \
	--seed 42
```

Supported integration methods:
- `euler`: first-order, fastest
- `heun`: second-order predictor-corrector, usually better quality/speed trade-off
- `rk4`: fourth-order, best accuracy per step but highest compute cost

### Smoke Test

Run a lightweight end-to-end smoke test (one training batch + one inference pass):

```bash
python src/smoke_test.py
```

## Troubleshooting

- Symptom: Generated text is repetitive (for example, "the the the...").
	- Try: `--sample --temperature 1.0 --top-k 50` in inference.
	- Try: Increase training diversity pressure with `--ce-weight-end 1.0` (or slightly higher).

- Symptom: Validation loss rises or oscillates late in training.
	- Try: Lower LR (for example, `--learning-rate 5e-5`).
	- Try: Enable plateau scheduler (`--lr-scheduler plateau --lr-patience 2 --lr-factor 0.5`).
	- Try: Enable early stopping (`--early-stop-patience 8`).

- Symptom: Training appears unstable (large jumps in loss).
	- Try: Enable gradient clipping (`--grad-clip-norm 1.0`).
	- Try: Use constant CE weighting (`--ce-weight-start 0.5 --ce-weight-end 0.5`) for easier diagnosis.

- Symptom: Generation quality is weak at inference time $t=0$.
	- Try: Bias training toward smaller $t$ with `--t-sample-power 1.2` to `2.0`.
	- Try: Add explicit $t=0$ supervision with `--t-zero-prob 0.1` to `0.3`.
	- Try: Evaluate with `--eval-at-t0` for inference-aligned model selection.

- Symptom: Few-step integration is too slow.
	- Try: Use `--integration-method euler` and fewer `--integration-steps`.
	- Try: Use `--integration-method heun` as a quality/speed middle ground.

- Symptom: W&B logging fails with import error.
	- Try: Install W&B with `pip install wandb`.
	- For air-gapped runs: use `--wandb --wandb-mode offline`.

- Symptom: Out-of-memory (CUDA/MPS) during training.
	- Try: Reduce `--batch-size` first.
	- Try: Reduce `--seq-len` or `--d-model` if memory is still insufficient.

- Symptom: Metrics are hard to interpret.
	- Use the logged component losses (`train_mse_loss`, `train_ce_loss`, `val_mse_loss`, `val_ce_loss`) instead of only total loss.
	- Watch diversity diagnostics (`distinct_1`, `distinct_2`, `token_entropy_norm`) to detect collapse early.

## Known Good Baselines

| Profile | Use Case | Command |
| --- | --- | --- |
| Fast Debug | Quick sanity check on small data with minimal runtime | `python src/train.py --dataset-split "train[:2%]" --epochs 5 --batch-size 8 --seq-len 64 --lr-scheduler none --early-stop-patience 0` |
| Stable Training | Main training run with anti-instability defaults | `python src/train.py --dataset-split "train[:20%]" --min-text-chars 32 --learning-rate 5e-5 --lr-scheduler plateau --lr-patience 2 --lr-factor 0.5 --grad-clip-norm 1.0 --early-stop-patience 8 --early-stop-min-delta 1e-4 --ce-weight-start 0.5 --ce-weight-end 0.5 --t-sample-power 1.2 --t-zero-prob 0.2 --eval-at-t0 --wandb --wandb-log-interval 50` |
| Diversity-Focused Inference | Reduce repetitive generation loops with stochastic decoding | `python src/inference.py --model-path src/meanflow_language_model.pth --sample --temperature 1.0 --top-k 50 --seed 42 --num-sequences 5` |
| Higher-Quality Few-Step Inference | Improve latent refinement quality using second- or fourth-order integration | `python src/inference.py --model-path src/meanflow_language_model.pth --integration-steps 8 --integration-method rk4 --sample --temperature 1.0 --top-k 50 --seed 42` |

## Loss Functions

The model uses a dual-objective loss with optional **Padding Masking**:
- **Masked MSE Loss**: Minimizes the distance between the predicted $x_1$ and the ground-truth text embeddings, ignoring padding tokens.
- **Masked CE Loss**: A cross-entropy loss (weighted at `0.5`) that ensures the predicted continuous vectors align perfectly with the discrete vocabulary. This auxiliary loss is critical for "rounding" accuracy.

When using grouped fixed-length blocks (the default training path), there is no padding and the loss is computed over all tokens.
