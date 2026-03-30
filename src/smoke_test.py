import torch
import torch.optim as optim

from meanflow import MeanFlowLanguageModel
from train import compute_loss_components


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def smoke_one(prediction_target, device, vocab_size=128, seq_len=16, batch_size=4):
    model = MeanFlowLanguageModel(
        vocab_size=vocab_size,
        d_model=64,
        num_heads=4,
        num_layers=2,
        max_seq_len=seq_len,
        prediction_target=prediction_target,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    # Training step via the same path used by train.py.
    model.train()
    optimizer.zero_grad()
    loss, mse_loss, ce_loss, velocity_mse = compute_loss_components(
        model, input_ids, pad_token_id=None
    )
    loss.backward()
    optimizer.step()

    # Inference pass.
    model.eval()
    with torch.no_grad():
        generated = model.generate_1_step(batch_size=2, seq_len=seq_len, device=device)

    assert generated.shape == (2, seq_len), (
        f"[{prediction_target}] Unexpected generated shape: {tuple(generated.shape)}"
    )
    print(
        f"[{prediction_target}] train_loss={loss.item():.4f} "
        f"mse={mse_loss.item():.4f} ce={ce_loss.item():.4f} "
        f"vel_mse={velocity_mse.item():.4f} "
        f"generated={tuple(generated.shape)}"
    )


def main():
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = pick_device()
    print(f"Smoke test device: {device}")

    smoke_one("x1", device)
    smoke_one("v", device)
    print("Smoke test passed")


if __name__ == "__main__":
    main()
