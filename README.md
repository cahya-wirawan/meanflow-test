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

Enable W&B logging during training:

```bash
python src/train.py --wandb --wandb-project meanflow --wandb-run-name "baseline-v1"
```

For local/offline logging:

```bash
python src/train.py --wandb --wandb-mode offline
```

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

### Smoke Test

Run a lightweight end-to-end smoke test (one training batch + one inference pass):

```bash
python src/smoke_test.py
```

## Loss Functions

The model uses a dual-objective loss with optional **Padding Masking**:
- **Masked MSE Loss**: Minimizes the distance between the predicted $x_1$ and the ground-truth text embeddings, ignoring padding tokens.
- **Masked CE Loss**: A cross-entropy loss (weighted at `0.5`) that ensures the predicted continuous vectors align perfectly with the discrete vocabulary. This auxiliary loss is critical for "rounding" accuracy.

When using grouped fixed-length blocks (the default training path), there is no padding and the loss is computed over all tokens.
