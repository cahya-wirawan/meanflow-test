# MeanFlow Language Model

*This is a Working in Progress (WIP) project, it is in very early state*

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
2. **Time-Conditioned Transformer**: A Transformer backbone conditioned on the current flow time $t$ via **sinusoidal Fourier time embeddings** (sin/cos features, then projected with SiLU). This gives the model a richer representation of scalar time than a plain linear layer.
3. **FiLM Conditioning**: Each transformer block applies per-block Feature-wise Linear Modulation (scale + shift) from the time embedding to both the attention and feed-forward sub-layers.
4. **Prediction Head**: Predicts the target — either $x_1$ directly (`--prediction-target x1`) or the flow velocity $v = x_1 - x_0$ (`--prediction-target v`).
5. **Rounding Head**: Maps the continuous predictions back to vocabulary logits via cosine-similarity with a learnable temperature, weight-tied with the input embeddings.

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
	--dataset-split "train[:100%]" \
	--min-text-chars 32 \
	--learning-rate 5e-5 \
	--lr-scheduler plateau \
	--lr-patience 5 \
	--lr-factor 0.5 \
	--grad-clip-norm 1.0 \
	--early-stop-patience 15 \
	--ce-weight-start 0.5 \
	--ce-weight-end 1.0 \
	--t-sample-power 3.0 \
	--t-zero-prob 0.5 \
	--eval-at-t0 \
	--d-model 512 \
	--num-heads 16 \
	--num-layers 8 \
	--batch-size 12 \
	--seq-len 1024 \
	--sample-interval 10 \
	--sample-steps 20 \
	--wandb \
	--wandb-log-interval 50
```

> **Note on `--t-sample-power`:** Values > 1 bias training toward small $t$ (closer to the $t=0$ inference condition). `t-sample-power 3.0` puts ~46% of training samples below $t=0.1$, vs only ~15% for `1.2`. Higher values make 1-step inference more reliable at the cost of slower learning at intermediate $t$.

> **Note on `--lr-patience`:** Use at least 5 epochs of patience. Val CE at $t=0$ converges to the unigram entropy floor (~7.2 nats for WikiText-2) and does not reflect model improvement; the scheduler now tracks val MSE instead.

Enable W&B logging during training:

```bash
python src/train.py --wandb --wandb-project meanflow --wandb-run-name "baseline-v1"
```

For local/offline logging:

```bash
python src/train.py --wandb --wandb-mode offline
```

The training script supports:
- Raw text filtering with `--min-text-chars`
- Gradient clipping with `--grad-clip-norm`
- Validation-driven LR scheduling with `--lr-scheduler plateau` (tracks val MSE, not val CE, since val CE saturates at the unigram entropy floor)
- Early stopping with `--early-stop-patience` and `--early-stop-min-delta`
- Time-bias control with `--t-sample-power` (biases training toward $t \approx 0$)
- Explicit $t=0$ alignment with `--t-zero-prob` and `--eval-at-t0`
- CE weighting schedule with `--ce-weight-start` and `--ce-weight-end`
- Epoch-level text sampling with `--sample-interval` and `--sample-steps` (prints decoded generated text during training using multi-step Heun integration, so you can monitor generation quality without running inference separately)
- W&B batch metrics via `--wandb-log-interval`
- Separate MSE/CE logging for train and validation losses
- Diversity diagnostics in W&B (`distinct_1`, `distinct_2`, entropy)

The training pipeline uses **Text Grouping**, where all tokenized examples are concatenated and split into equal blocks of `SEQ_LEN`. This removes the bias caused by padding short sentences and ensures the model only learns from real data.

Training uses a deterministic seed (`SEED=42`) and a train/validation split. The checkpoint saved to `--model-path` stores both the model weights and the architecture config (`d_model`, `num_heads`, `num_layers`, `max_seq_len`, `prediction_target`), so inference can auto-configure without requiring the user to repeat all architecture flags.

### Inference

To generate text using a trained model:

```bash
python src/inference.py --model-path src/meanflow_language_model.pth
```

Checkpoints saved by the current training script embed the architecture config, so `--d-model`, `--num-heads`, `--num-layers`, and `--prediction-target` are auto-loaded. You do not need to repeat them.

To reduce repetitive outputs, use sampling instead of greedy decoding:

```bash
python src/inference.py \
	--model-path src/meanflow_language_model.pth \
	--sample \
	--temperature 0.8 \
	--top-k 50 \
	--seed 42
```

For iterative latent refinement (recommended), use multi-step integration:

```bash
python src/inference.py \
	--model-path src/meanflow_language_model.pth \
	--integration-steps 20 \
	--integration-method heun \
	--sample \
	--temperature 0.8 \
	--top-k 50 \
	--seed 42
```

**How multi-step inference works for x1-prediction models:**
At each step, the model predicts $\hat{x}_1$ from the current noisy state $x_t$. Rather than integrating a velocity ODE (which amplifies errors near $t=1$ due to the $1/(1-t)$ denominator), the implementation uses an **x0-anchored update**:

$$x_{t+\Delta t} = (t + \Delta t)\,\hat{x}_1 + (1 - t - \Delta t)\,x_0$$

This keeps $x_0$ fixed as an anchor and places $x_{t+\Delta t}$ directly on the line between the original noise and the current best estimate of $x_1$. It is numerically stable and does not amplify prediction errors.

Supported integration methods (relevant for `--prediction-target v`):
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

- Symptom: Generated text is gibberish with multi-step inference.
	- Ensure you are using `--integration-steps N` with N > 1 (default is 1).
	- For x1-prediction models the x0-anchored update is used automatically — Heun/RK4 flags only affect v-prediction models.

- Symptom: Generation quality is weak at inference time $t=0$.
	- Try: Bias training toward smaller $t$ with `--t-sample-power 2.0` to `3.0`.
	- Try: Add explicit $t=0$ supervision with `--t-zero-prob 0.3` to `0.5`.
	- Try: Evaluate with `--eval-at-t0` for inference-aligned model selection.
	- Note: Val CE at $t=0$ will converge to the unigram entropy floor (~7.2 nats for WikiText-2) and will not decrease further regardless of model quality — this is expected and not a sign of failure. Watch val MSE instead.

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
| Stable Training | Main training run with recommended defaults | `python src/train.py --dataset-split "train[:100%]" --min-text-chars 32 --learning-rate 5e-5 --lr-scheduler plateau --lr-patience 5 --lr-factor 0.5 --grad-clip-norm 1.0 --early-stop-patience 15 --ce-weight-start 0.5 --ce-weight-end 1.0 --t-sample-power 3.0 --t-zero-prob 0.5 --eval-at-t0 --d-model 512 --num-heads 16 --num-layers 8 --batch-size 12 --seq-len 1024 --sample-interval 10 --wandb --wandb-log-interval 50` |
| Diversity-Focused Inference | Reduce repetitive generation with stochastic decoding | `python src/inference.py --model-path src/meanflow_language_model.pth --sample --temperature 0.8 --top-k 50 --seed 42 --num-sequences 5` |
| Multi-Step Inference | Better generation quality via iterative latent refinement | `python src/inference.py --model-path src/meanflow_language_model.pth --integration-steps 20 --sample --temperature 0.8 --top-k 50 --seed 42` |

## Loss Functions

The model uses a dual-objective loss:

- **MSE Loss**: Minimizes the distance between the predicted $\hat{x}_1$ and the ground-truth token embeddings. For `--prediction-target v`, this is computed by first converting the predicted velocity back to $\hat{x}_1$.
- **CE Loss** (auxiliary): Cross-entropy between the cosine-similarity logits of $\hat{x}_1$ and the ground-truth token ids. Weighted by `--ce-weight-start`/`--ce-weight-end` (linearly scheduled over training). Critical for aligning the continuous predictions with the discrete vocabulary.
- **Velocity MSE** (v-mode only): Additional loss on the predicted velocity $\hat{v}$ vs the true velocity $x_1 - x_0$, weighted by `--velocity-loss-weight`.

When using grouped fixed-length blocks (the default training path), there is no padding and the loss is computed over all tokens.

### Interpreting val CE

Val CE at `t=0` converges to the **unigram entropy** of the training corpus (≈7.2 nats for WikiText-2 with GPT-2 tokenizer) and does not decrease further — this is the information-theoretic floor for a model receiving pure noise as input with no conditioning signal. It is not a sign of failure. Track **val MSE** instead, which continues to decrease as the model learns the flow geometry.
