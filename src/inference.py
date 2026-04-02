import torch
import argparse

SEQ_LEN = 128
MODEL_PATH = "src/meanflow_language_model.pth"


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
        "--prediction-target",
        type=str,
        choices=["x1", "v"],
        default="x1",
        help=(
            "Prediction target used by the checkpoint architecture. "
            "Must match training configuration (x1 or v)."
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
    parser.add_argument(
        "--integration-steps",
        type=int,
        default=1,
        help=(
            "Number of integration steps from t=0 to t=1. "
            "Set to 1 for the original one-step jump, or >1 for iterative refinement."
        ),
    )
    parser.add_argument(
        "--integration-method",
        type=str,
        choices=["euler", "heun", "rk4"],
        default="euler",
        help=(
            "Numerical integration method for few-step refinement. "
            "Heun is a second-order predictor-corrector variant and RK4 is a fourth-order method."
        ),
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help=(
            "Text prompt for prefix-conditioned generation. "
            "The model generates a continuation conditioned on this prefix. "
            "Only works with checkpoints trained with --prefix-len > 0."
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
    integration_steps=1,
    integration_method="euler",
    local_to_orig=None,
    prefix_emb=None,
    prompt_text=None,
):
    if prompt_text:
        print(f"\n--- Generating Continuation of: \"{prompt_text}\" ---")
    else:
        print("\n--- Generating New Text ---")
    model.eval()

    with torch.no_grad():
        if integration_steps <= 1:
            predicted_token_ids = model.generate_1_step(
                batch_size=num_sequences,
                seq_len=seq_len,
                device=str(device),
                sample=sample,
                temperature=temperature,
                top_k=top_k,
                prefix_emb=prefix_emb,
            )
        else:
            x_0 = torch.randn(num_sequences, seq_len, model.d_model, device=device)
            x_t = x_0.clone()
            dt = 1.0 / integration_steps
            t_value = 0.0

            for step_idx in range(integration_steps):
                t = torch.full((num_sequences, 1), t_value, device=device)
                t_next_value = min(t_value + dt, 1.0)

                if model.prediction_target == "x1":
                    pred_x1 = model.forward_net(x_t, t, prefix_emb=prefix_emb)
                    t_next_exp = t_next_value
                    x_t = t_next_exp * pred_x1 + (1.0 - t_next_exp) * x_0
                else:
                    v_hat = model.predict_velocity(x_t, t, prefix_emb=prefix_emb)
                    is_last_step = t_next_value >= 1.0 - 1e-6

                    if integration_method == "heun" and not is_last_step:
                        x_pred = x_t + dt * v_hat
                        t_next = torch.full((num_sequences, 1), t_next_value, device=device)
                        v_hat_next = model.predict_velocity(x_pred, t_next, prefix_emb=prefix_emb)
                        x_t = x_t + 0.5 * dt * (v_hat + v_hat_next)
                    elif integration_method == "rk4" and not is_last_step:
                        k1 = v_hat
                        t2_value = t_value + 0.5 * dt
                        t2 = torch.full((num_sequences, 1), t2_value, device=device)
                        x2 = x_t + 0.5 * dt * k1
                        k2 = model.predict_velocity(x2, t2, prefix_emb=prefix_emb)
                        x3 = x_t + 0.5 * dt * k2
                        k3 = model.predict_velocity(x3, t2, prefix_emb=prefix_emb)
                        t4 = torch.full((num_sequences, 1), t_next_value, device=device)
                        x4 = x_t + dt * k3
                        k4 = model.predict_velocity(x4, t4, prefix_emb=prefix_emb)
                        x_t = x_t + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
                    else:
                        x_t = x_t + dt * v_hat

                t_value = t_next_value

            if model.prediction_target == "x1":
                logits = model.lm_logits(pred_x1)
            else:
                logits = model.lm_logits(x_t)
            if not sample:
                predicted_token_ids = torch.argmax(logits, dim=-1)
            else:
                logits = logits / temperature
                if top_k > 0:
                    k = min(top_k, logits.size(-1))
                    topk_logits, topk_indices = torch.topk(logits, k=k, dim=-1)
                    probs = torch.softmax(topk_logits, dim=-1)
                    sampled = torch.multinomial(probs.reshape(-1, k), num_samples=1)
                    sampled = sampled.view(num_sequences, seq_len, 1)
                    predicted_token_ids = topk_indices.gather(-1, sampled).squeeze(-1)
                else:
                    probs = torch.softmax(logits, dim=-1)
                    sampled = torch.multinomial(
                        probs.reshape(-1, probs.size(-1)), num_samples=1
                    )
                    predicted_token_ids = sampled.view(num_sequences, seq_len)

        if local_to_orig is not None:
            predicted_token_ids = local_to_orig[predicted_token_ids.cpu()]

        eos_id = tokenizer.eos_token_id
        for i in range(num_sequences):
            token_ids = predicted_token_ids[i].tolist()
            if eos_id in token_ids:
                token_ids = token_ids[:token_ids.index(eos_id)]
            generated_text = tokenizer.decode(token_ids, skip_special_tokens=True)
            if prompt_text:
                print(f"Sequence {i+1}: {prompt_text}{generated_text}")
            else:
                print(f"Sequence {i+1}: {generated_text}")

if __name__ == "__main__":
    args = parse_args()
    if args.seq_len <= 0 or args.num_sequences <= 0:
        raise ValueError("--seq-len and --num-sequences must be > 0")
    if args.temperature <= 0:
        raise ValueError("--temperature must be > 0")
    if args.top_k < 0:
        raise ValueError("--top-k must be >= 0")
    if args.integration_steps <= 0:
        raise ValueError("--integration-steps must be > 0")
    if args.integration_steps == 1 and args.integration_method != "euler":
        print(
            f"Warning: --integration-method {args.integration_method!r} has no effect "
            "when --integration-steps 1 (default). Use --integration-steps N with N > 1."
        )

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

    checkpoint = torch.load(args.model_path, map_location=device, weights_only=True)

    # Support both old (state_dict only) and new (dict with config) checkpoint formats.
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        cfg = checkpoint["model_config"]
        local_to_orig = checkpoint.get("local_to_orig", None)
        prefix_len = cfg.get("prefix_len", 0)
        print(
            f"Loaded checkpoint config: d_model={cfg['d_model']} num_heads={cfg['num_heads']} "
            f"num_layers={cfg['num_layers']} max_seq_len={cfg['max_seq_len']} "
            f"prediction_target={cfg['prediction_target']} vocab_size={cfg['vocab_size']}"
            + (f" prefix_len={prefix_len}" if prefix_len > 0 else "")
            + (f" (restricted from {vocab_size})" if local_to_orig is not None else "")
        )
        d_model = cfg["d_model"]
        num_heads = cfg["num_heads"]
        num_layers = cfg["num_layers"]
        max_seq_len = cfg["max_seq_len"]
        prediction_target = cfg["prediction_target"]
        effective_vocab_size = cfg["vocab_size"]
        state_dict = checkpoint["model_state_dict"]
    else:
        d_model, num_heads, num_layers = args.d_model, args.num_heads, args.num_layers
        max_seq_len = args.seq_len
        prediction_target = args.prediction_target
        effective_vocab_size = vocab_size
        local_to_orig = None
        prefix_len = 0
        state_dict = checkpoint

    prefix_mode = prefix_len > 0
    model = MeanFlowLanguageModel(
        vocab_size=effective_vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
        prediction_target=prediction_target,
        prefix_mode=prefix_mode,
    ).to(device)

    model.load_state_dict(state_dict)

    # Encode prompt as prefix if provided.
    prefix_emb = None
    prompt_text = args.prompt
    target_len = max_seq_len - prefix_len if prefix_len > 0 else max_seq_len
    if args.prompt is not None:
        if not prefix_mode:
            print(
                "Warning: --prompt provided but checkpoint was trained without prefix conditioning "
                "(prefix_len=0). The prompt will be ignored."
            )
        else:
            prompt_ids = tokenizer.encode(args.prompt, add_special_tokens=False)
            if len(prompt_ids) > prefix_len:
                print(f"Warning: prompt ({len(prompt_ids)} tokens) exceeds prefix_len ({prefix_len}), truncating.")
                prompt_ids = prompt_ids[:prefix_len]
            elif len(prompt_ids) < prefix_len:
                # Pad prompt to prefix_len with pad token.
                prompt_ids = prompt_ids + [tokenizer.pad_token_id] * (prefix_len - len(prompt_ids))
            # Remap to local vocab if restricted.
            if local_to_orig is not None:
                orig_to_local = {orig.item(): local for local, orig in enumerate(local_to_orig)}
                prompt_ids = [orig_to_local.get(t, 0) for t in prompt_ids]
            prompt_tensor = torch.tensor([prompt_ids] * args.num_sequences, dtype=torch.long, device=device)
            prefix_emb = model.encode_prefix(prompt_tensor)
    elif prefix_mode:
        print("Note: model was trained with prefix conditioning but no --prompt provided. Generating unconditionally.")

    generate_text(
        model,
        tokenizer,
        num_sequences=args.num_sequences,
        seq_len=target_len,
        device=str(device),
        sample=args.sample,
        temperature=args.temperature,
        top_k=args.top_k,
        integration_steps=args.integration_steps,
        integration_method=args.integration_method,
        local_to_orig=local_to_orig,
        prefix_emb=prefix_emb,
        prompt_text=prompt_text,
    )